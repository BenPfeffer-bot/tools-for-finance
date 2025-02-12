import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from collections import deque
import random
import logging
import torch.optim as optim
from collections import namedtuple

logger = logging.getLogger(__name__)

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class DQNTrader(nn.Module):
    """Deep Q-Network for forex trading."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        """Initialize the DQN network."""
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_pairs = action_dim // 3  # Each pair has 3 possible actions

        # Deeper feature extractor with batch normalization
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )

        # Dueling DQN architecture
        self.advantage_stream = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 3),
                )
                for _ in range(self.num_pairs)
            ]
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Ensure state has correct shape
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension if missing

        features = self.feature_extractor(state)

        # Compute value
        value = self.value_stream(features)

        # Compute advantages for each pair
        advantages = []
        for i in range(self.num_pairs):
            pair_advantage = self.advantage_stream[i](features)
            advantages.append(pair_advantage)

        # Combine advantages
        advantages = torch.cat(advantages, dim=-1)

        # Combine value and advantages (dueling architecture)
        q_values = value.unsqueeze(-1) + (
            advantages - advantages.mean(dim=-1, keepdim=True)
        )

        return q_values


class PPOTrader(nn.Module):
    """PPO model for dynamic trade sizing."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return both action probabilities and value estimate
        return F.softmax(self.actor(state), dim=-1), self.critic(state)


class TradingEnvironment:
    """Trading environment for RL agents."""

    def __init__(
        self,
        returns: pd.DataFrame,
        predictions: np.ndarray,
        initial_balance: float = 1e6,
        transaction_cost: float = 0.001,
        window_size: int = 50,
    ):
        self.returns = returns
        self.predictions = predictions
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size

        # Add tracking for signals and performance
        self.trading_history = {
            "positions": [],
            "portfolio_values": [],
            "returns": [],
            "actions": [],
            "signals": [],
            "timestamps": [],
        }

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.positions = np.zeros(len(self.returns.columns))
        self.done = False

        # Reset trading history
        self.trading_history = {
            "positions": [],
            "portfolio_values": [],
            "returns": [],
            "actions": [],
            "signals": [],
            "timestamps": [],
        }

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        if self.done:
            raise ValueError("Episode is done, please reset the environment")

        # Execute action and get reward
        reward = self._execute_action(action)

        # Record trading history
        self.trading_history["positions"].append(self.positions.copy())
        self.trading_history["portfolio_values"].append(self.balance)
        self.trading_history["returns"].append(
            float(np.sum(self.returns.iloc[self.current_step].values * self.positions))
        )
        self.trading_history["actions"].append(action)
        self.trading_history["signals"].append(self.predictions[self.current_step])
        self.trading_history["timestamps"].append(self.returns.index[self.current_step])

        # Update state
        self.current_step += 1
        if self.current_step >= len(self.returns) - 1:
            self.done = True

        # Get new state
        new_state = self._get_state()

        # Additional info
        info = {
            "balance": self.balance,
            "positions": self.positions,
            "step": self.current_step,
            "portfolio_value": self.balance,
            "current_return": self.trading_history["returns"][-1]
            if self.trading_history["returns"]
            else 0,
        }

        return new_state, reward, self.done, info

    def _get_state(self) -> np.ndarray:
        """Construct the state representation."""
        try:
            # Get window of returns data
            window_data = self.returns.iloc[
                self.current_step - self.window_size : self.current_step
            ]

            # Calculate features from window
            features = []

            # 1. Return features (per asset)
            returns_mean = window_data.mean()
            returns_std = window_data.std()
            returns_skew = window_data.skew()
            latest_return = window_data.iloc[-1]

            for asset in window_data.columns:
                features.extend(
                    [
                        returns_mean[asset],
                        returns_std[asset],
                        returns_skew[asset],
                        latest_return[asset],
                    ]
                )

            # 2. Portfolio features
            features.extend(
                [
                    self.balance / self.initial_balance - 1,  # Portfolio return
                    np.sum(np.abs(self.positions)),  # Total exposure
                    np.sum(self.positions),  # Net exposure
                ]
            )

            # 3. Position features
            features.extend(self.positions)  # Current positions

            # 4. Prediction features
            features.append(self.predictions[self.current_step])

            return np.array(features)

        except Exception as e:
            logger.error(f"Error in _get_state: {str(e)}")
            raise

    def _execute_action(self, action: int) -> float:
        """Execute trading action and calculate reward."""
        try:
            # Get current returns
            current_returns = self.returns.iloc[self.current_step].values

            # More conservative position sizing
            position_multiplier = float(self.balance) / float(self.initial_balance)
            position_multiplier = np.clip(
                position_multiplier, 0.1, 1.5
            )  # Reduced max leverage

            # Calculate position changes with stricter limits
            max_change = 0.02  # Reduced from 0.03 to 0.02
            if action == 0:  # Sell
                position_changes = (
                    -max_change * position_multiplier * np.ones_like(self.positions)
                )
            elif action == 1:  # Hold
                position_changes = np.zeros_like(self.positions)
            else:  # Buy
                position_changes = (
                    max_change * position_multiplier * np.ones_like(self.positions)
                )

            # Enhanced transaction costs
            base_cost = 0.002  # Increased from 0.001
            slippage = (
                0.0003 * np.abs(position_changes) * (1 + np.abs(self.positions))
            )  # Increased from 0.0002
            total_costs = float(
                np.sum((base_cost + slippage) * np.abs(position_changes))
            )

            # Update positions
            self.positions += position_changes

            # Calculate returns with higher cost penalty
            portfolio_return = (
                float(np.sum(current_returns * self.positions)) - total_costs * 1.5
            )
            self.balance *= 1 + portfolio_return

            # Stricter stop loss and take profit
            stop_loss_threshold = 0.85  # Increased from 0.8
            take_profit_threshold = 1.3  # Reduced from 1.5

            if float(self.balance) < float(self.initial_balance) * stop_loss_threshold:
                self.done = True
                return -2.0

            if (
                float(self.balance)
                > float(self.initial_balance) * take_profit_threshold
            ):
                self.done = True
                return 2.0

            # Enhanced reward components
            return_component = float(portfolio_return)

            # Stronger risk penalties
            volatility = float(np.std(current_returns * self.positions))
            risk_component = -2.0 * volatility  # Doubled volatility penalty

            # Enhanced position concentration penalty
            concentration = float(np.sum(np.abs(self.positions))) / len(self.positions)
            concentration_penalty = -0.3 * concentration  # Increased from -0.2

            # Stronger trading activity penalty
            activity_penalty = -0.8 * total_costs  # Increased from -0.5

            # Dynamic reward scaling based on market conditions
            market_volatility = float(np.std(current_returns))
            scaling_factor = 1.0 / (
                1.0 + market_volatility
            )  # Reduce rewards in high volatility

            # Combine reward components with dynamic weights
            reward = (
                float(
                    return_component
                    + risk_component
                    + concentration_penalty
                    + activity_penalty
                )
                * scaling_factor
            )

            # More conservative reward clipping
            reward = float(np.clip(reward, -0.8, 0.8))  # Reduced from -1.0, 1.0

            return reward

        except Exception as e:
            logger.error(f"Error in _execute_action: {str(e)}")
            raise

    def get_trading_signals(self) -> pd.DataFrame:
        """Return trading signals and performance metrics as a DataFrame."""
        if not self.trading_history["timestamps"]:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "timestamp": self.trading_history["timestamps"],
                "portfolio_value": self.trading_history["portfolio_values"],
                "returns": self.trading_history["returns"],
                "action": self.trading_history["actions"],
                "signal": self.trading_history["signals"],
            }
        )

        # Add derived metrics
        df["cumulative_returns"] = (1 + pd.Series(df["returns"])).cumprod()
        df["drawdown"] = df["portfolio_value"] / df["portfolio_value"].cummax() - 1

        return df


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays first for efficiency
        states = np.array(states)
        next_states = np.array(next_states)

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class RLTrader:
    """Main class for RL-based trading."""

    def __init__(
        self,
        env: TradingEnvironment,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the RL trader."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.device = device

        # Number of currency pairs
        self.num_pairs = action_dim // 3  # Each pair has 3 possible actions

        # Initialize networks
        self.policy_net = DQNTrader(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
        ).to(device)

        self.target_net = DQNTrader(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
        ).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()

        logger.info(f"RLTrader initialized successfully on device: {device}")

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """Select action using epsilon-greedy policy with balanced risk management."""
        try:
            # Ensure state is in correct format
            if not isinstance(state, np.ndarray):
                state = np.array(state)
            if len(state.shape) == 1:
                state = state.reshape(1, -1)

            # Convert to tensor
            state_tensor = torch.FloatTensor(state).to(self.device)

            # Select action
            if np.random.random() > epsilon:
                with torch.no_grad():
                    # Get Q-values
                    q_values = self.policy_net(state_tensor)

                    # Convert to numpy array
                    q_values = q_values.cpu().numpy()

                    # Get action for each pair with balanced risk management
                    actions = []
                    for i in range(self.num_pairs):
                        pair_q_values = q_values[0, i * 3 : (i + 1) * 3]

                        # Calculate action probabilities using softmax
                        exp_q_values = np.exp(pair_q_values - np.max(pair_q_values))
                        probs = exp_q_values / np.sum(exp_q_values)

                        # More balanced confidence threshold
                        max_prob = np.max(probs)
                        if max_prob < 0.35:  # Slightly lower threshold
                            pair_action = 0  # Hold if not confident
                        else:
                            # Check if the difference between best and second best is significant
                            sorted_probs = np.sort(probs)
                            if (
                                sorted_probs[-1] - sorted_probs[-2] < 0.15
                            ):  # Lower threshold
                                pair_action = 0  # Hold if decision is not clear
                            else:
                                pair_action = np.argmax(pair_q_values)

                        actions.append(pair_action)

                    return np.array(actions)
            else:
                # More balanced exploration
                actions = []
                for _ in range(self.num_pairs):
                    if np.random.random() < 0.5:  # 50% chance to hold (was 60%)
                        actions.append(0)
                    else:
                        actions.append(
                            np.random.randint(1, 3)
                        )  # 50% chance to buy or sell
                return np.array(actions)

        except Exception as e:
            logger.error(f"Error in select_action: {str(e)}")
            # Return safe default actions (hold)
            return np.zeros(self.num_pairs, dtype=np.int64)

    def train_dqn(self, batch_size: int = 64) -> float:
        """Train DQN using experience replay."""
        try:
            if len(self.replay_buffer) < batch_size:
                return 0.0

            # Sample from replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                batch_size
            )

            # Convert to tensors and move to device
            state_batch = states.to(self.device)
            action_batch = actions.to(self.device)
            reward_batch = rewards.to(self.device)
            next_state_batch = next_states.to(self.device)
            done_batch = dones.to(self.device)

            # Get current Q values
            current_q_values = self.policy_net(
                state_batch
            )  # [batch_size, num_pairs * 3]
            current_q_values = current_q_values.view(batch_size, self.num_pairs, 3)

            # Create index tensor for gathering
            batch_idx = torch.arange(batch_size).unsqueeze(1).to(self.device)
            pair_idx = torch.arange(self.num_pairs).unsqueeze(0).to(self.device)

            # Gather Q values for taken actions
            current_q_values = current_q_values[batch_idx, pair_idx, action_batch]
            current_q_values = current_q_values.mean(dim=1)  # Average across pairs

            # Compute next Q values
            with torch.no_grad():
                next_q_values = self.target_net(next_state_batch)
                next_q_values = next_q_values.view(batch_size, self.num_pairs, 3)
                next_q_values = next_q_values.max(2)[0].mean(
                    dim=1
                )  # Max action value per pair, then average
                next_q_values[done_batch.bool()] = 0.0
                expected_q_values = reward_batch + self.gamma * next_q_values

            # Compute loss and optimize
            loss = F.smooth_l1_loss(current_q_values, expected_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), 1.0
            )  # Add gradient clipping
            self.optimizer.step()

            return loss.item()

        except Exception as e:
            logger.error(f"Error in train_dqn: {str(e)}")
            return 0.0

    def update_target_network(self):
        """Update target network parameters."""
        try:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.info("Target network updated")
        except Exception as e:
            logger.error(f"Error updating target network: {str(e)}")

    def save_model(self, path: str):
        """Save model parameters."""
        try:
            torch.save(
                {
                    "policy_net_state_dict": self.policy_net.state_dict(),
                    "target_net_state_dict": self.target_net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                path,
            )
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, path: str):
        """Load model parameters."""
        try:
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    def train_rl_agent(self, env, episodes=100, max_steps=1000):
        logger.info(
            f"Starting RL training with {episodes} episodes and max {max_steps} steps per episode"
        )
        total_rewards = []
        episode_steps = []

        for episode in range(episodes):
            logger.info(f"Starting episode {episode + 1}/{episodes}")
            state = env.reset()
            episode_reward = 0
            steps = 0
            portfolio_values = []

            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                portfolio_values.append(info.get("portfolio_value", 0))

                if step % 100 == 0:
                    logger.info(
                        f"Episode {episode + 1}, Step {step + 1}: "
                        f"Action={action}, Reward={reward:.4f}, "
                        f"Portfolio Value={info.get('portfolio_value', 0):.2f}"
                    )

                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state

                if len(self.replay_buffer) > self.batch_size:
                    loss = self.train_dqn()
                    if step % 100 == 0:
                        logger.info(f"Step {step + 1}: Loss={loss:.4f}")

                steps += 1
                if done:
                    break

            total_rewards.append(episode_reward)
            episode_steps.append(steps)
            avg_portfolio_value = (
                sum(portfolio_values) / len(portfolio_values) if portfolio_values else 0
            )

            logger.info(
                f"Episode {episode + 1} finished: "
                f"Steps={steps}, Total Reward={episode_reward:.4f}, "
                f"Avg Portfolio Value={avg_portfolio_value:.2f}"
            )

            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                logger.info(f"Updated target network at episode {episode + 1}")

        logger.info(
            f"Training completed. Average reward over all episodes: {sum(total_rewards) / len(total_rewards):.4f}"
        )
        return total_rewards, episode_steps


# class ArbitrageDetector:
#     def __init__(
#         self,
#         returns: pd.DataFrame,
#         eigenportfolios: np.ndarray,
#         window: int = 50,
#         lag: int = 1,
#         te_threshold: float = 2.0,
#         vol_threshold: float = 1.5,
#         min_signal_strength: float = 0.3,
#     ):
#         # Add new parameters
#         self.min_signal_strength = min_signal_strength
#         self.correlation_threshold = 0.7
#         self.cointegration_window = 252  # One year of data

#         # Add statistical tests
#         self.statistical_tests = {
#             "adf_test": self._perform_adf_test,
#             "granger_test": self._perform_granger_test,
#             "johansen_test": self._perform_johansen_test,
#         }

#     def compute_advanced_features(self, returns: pd.DataFrame) -> pd.DataFrame:
#         """Enhanced feature computation with statistical arbitrage indicators"""
#         features = pd.DataFrame(index=returns.index)

#         # Eigenportfolio projections
#         projected_returns = self._compute_eigenportfolio_projections(returns)

#         # Statistical arbitrage features
#         pairs_features = self._compute_pairs_features(returns)

#         # Market regime features
#         regime_features = self._compute_regime_features(returns)

#         # Combine all features
#         features = pd.concat(
#             [projected_returns, pairs_features, regime_features], axis=1
#         )
#         return features

#     def _compute_eigenportfolio_projections(
#         self, returns: pd.DataFrame
#     ) -> pd.DataFrame:
#         """Compute advanced eigenportfolio projections"""
#         projections = pd.DataFrame(index=returns.index)

#         # Project returns onto eigenportfolios
#         for i, eigen_portfolio in enumerate(self.eigenportfolios):
#             projection = returns.dot(eigen_portfolio)

#             # Add rolling statistics
#             projections[f"eigen_{i}_return"] = projection
#             projections[f"eigen_{i}_vol"] = projection.rolling(window=20).std()
#             projections[f"eigen_{i}_zscore"] = (
#                 projection - projection.rolling(window=50).mean()
#             ) / projection.rolling(window=50).std()

#             # Add momentum indicators
#             projections[f"eigen_{i}_momentum"] = projection.rolling(window=10).mean()

#         return projections

#     def _compute_pairs_features(self, returns: pd.DataFrame) -> pd.DataFrame:
#         """Compute statistical arbitrage pairs features"""
#         pairs_features = pd.DataFrame(index=returns.index)

#         # Find cointegrated pairs
#         cointegrated_pairs = self._find_cointegrated_pairs(returns)

#         for pair in cointegrated_pairs:
#             spread = self._calculate_pair_spread(returns[pair[0]], returns[pair[1]])

#             # Add spread features
#             pairs_features[f"spread_{pair[0]}_{pair[1]}"] = spread
#             pairs_features[f"spread_zscore_{pair[0]}_{pair[1]}"] = (
#                 spread - spread.rolling(window=50).mean()
#             ) / spread.rolling(window=50).std()

#             # Add mean reversion strength
#             pairs_features[f"mean_rev_{pair[0]}_{pair[1]}"] = -spread.rolling(
#                 window=20
#             ).autocorr()

#         return pairs_features

#     def _compute_regime_features(self, returns: pd.DataFrame) -> pd.DataFrame:
#         """Compute market regime features"""
#         regime_features = pd.DataFrame(index=returns.index)

#         # Volatility regime
#         vol = returns.rolling(window=20).std()
#         regime_features["volatility_regime"] = (
#             vol > vol.rolling(window=60).mean()
#         ).astype(int)

#         # Correlation regime
#         corr = returns.rolling(window=60).corr()
#         regime_features["correlation_regime"] = (
#             corr.mean() > corr.rolling(window=120).mean()
#         ).astype(int)

#         # Trend regime
#         ma_fast = returns.rolling(window=20).mean()
#         ma_slow = returns.rolling(window=50).mean()
#         regime_features["trend_regime"] = (ma_fast > ma_slow).astype(int)

#         return regime_features

#     def generate_signals(self) -> pd.DataFrame:
#         """Generate enhanced trading signals"""
#         signals = pd.DataFrame(index=self.returns.index)

#         # Compute all features
#         features = self.compute_advanced_features(self.returns)

#         # Generate signals for each strategy
#         eigen_signals = self._generate_eigen_signals(features)
#         pairs_signals = self._generate_pairs_signals(features)
#         regime_signals = self._generate_regime_signals(features)

#         # Combine signals with regime-based weights
#         signals["combined_signal"] = (
#             eigen_signals * regime_signals["eigen_weight"]
#             + pairs_signals * regime_signals["pairs_weight"]
#         )

#         # Add confidence scores
#         signals["signal_confidence"] = self._compute_signal_confidence(features)

#         return signals

#     def _compute_signal_confidence(self, features: pd.DataFrame) -> pd.Series:
#         """Compute confidence scores for signals"""
#         confidence = pd.Series(index=features.index)

#         # Feature importance based confidence
#         feature_importance = self._compute_feature_importance(features)
#         weighted_features = features.multiply(feature_importance, axis=1)

#         # Signal strength
#         signal_strength = weighted_features.abs().mean(axis=1)

#         # Regime certainty
#         regime_certainty = self._compute_regime_certainty(features)

#         # Combine confidence metrics
#         confidence = signal_strength * 0.6 + regime_certainty * 0.4

#         return confidence.clip(0, 1)


# class Eigenportfolio:
#     def __init__(
#         self,
#         returns: pd.DataFrame,
#         n_components: int = 5,
#         min_explained_variance: float = 0.95,
#         regime_detection: bool = True,
#     ):
#         self.regime_detection = regime_detection
#         self.risk_factors = None
#         self.factor_exposures = None

#     def compute_advanced_eigenportfolios(self) -> Tuple[np.ndarray, pd.DataFrame]:
#         """Compute enhanced eigenportfolios with risk factor analysis"""
#         # Standard PCA computation
#         self.pca.fit(self.returns)
#         eigenportfolios = self.pca.components_

#         # Compute risk factor exposures
#         self.risk_factors = self._compute_risk_factors()
#         self.factor_exposures = self._compute_factor_exposures(eigenportfolios)

#         # Adjust eigenportfolios based on risk factors
#         adjusted_eigenportfolios = self._adjust_eigenportfolios(
#             eigenportfolios, self.factor_exposures
#         )

#         return adjusted_eigenportfolios, self.factor_exposures

#     def _compute_risk_factors(self) -> pd.DataFrame:
#         """Compute market risk factors"""
#         risk_factors = pd.DataFrame(index=self.returns.index)

#         # Market factor
#         risk_factors["market"] = self.returns.mean(axis=1)

#         # Volatility factor
#         risk_factors["volatility"] = self.returns.std(axis=1)

#         # Size factor (if applicable)
#         if hasattr(self, "market_caps"):
#             risk_factors["size"] = self._compute_size_factor()

#         # Momentum factor
#         risk_factors["momentum"] = self._compute_momentum_factor()

#         return risk_factors

#     def _compute_factor_exposures(self, eigenportfolios: np.ndarray) -> pd.DataFrame:
#         """Compute risk factor exposures for eigenportfolios"""
#         exposures = pd.DataFrame(
#             index=range(len(eigenportfolios)), columns=self.risk_factors.columns
#         )

#         for i, portfolio in enumerate(eigenportfolios):
#             portfolio_returns = self.returns.dot(portfolio)

#             # Compute factor betas
#             for factor in self.risk_factors.columns:
#                 exposures.loc[i, factor] = self._compute_beta(
#                     portfolio_returns, self.risk_factors[factor]
#                 )

#         return exposures

#     def analyze_portfolio_risk(self) -> Dict:
#         """Analyze portfolio risk characteristics"""
#         risk_metrics = {}

#         # Compute risk decomposition
#         risk_decomposition = self._compute_risk_decomposition()

#         # Compute factor contribution
#         factor_contribution = self._compute_factor_contribution()

#         # Compute diversification ratio
#         diversification_ratio = self._compute_diversification_ratio()

#         risk_metrics.update(
#             {
#                 "risk_decomposition": risk_decomposition,
#                 "factor_contribution": factor_contribution,
#                 "diversification_ratio": diversification_ratio,
#             }
#         )

#         return risk_metrics


# class ArbitragePortfolioStrategy:
#     """Integrates eigenportfolio and arbitrage detection for trading"""

#     def __init__(
#         self,
#         returns: pd.DataFrame,
#         lookback_window: int = 252,
#         rebalance_frequency: str = "W",
#         risk_target: float = 0.15,
#     ):
#         self.returns = returns
#         self.lookback_window = lookback_window
#         self.rebalance_frequency = rebalance_frequency
#         self.risk_target = risk_target

#         # Initialize components
#         self.eigenportfolio = Eigenportfolio(returns)
#         self.arbitrage_detector = ArbitrageDetector(
#             returns, self.eigenportfolio.eigenportfolios
#         )

#     def generate_portfolio_weights(self) -> pd.DataFrame:
#         """Generate portfolio weights combining both strategies"""
#         # Compute eigenportfolio weights
#         eigen_weights = self.eigenportfolio.get_eigenportfolio_weights()

#         # Get arbitrage signals
#         arb_signals = self.arbitrage_detector.generate_signals()

#         # Combine strategies
#         combined_weights = self._combine_strategies(eigen_weights, arb_signals)

#         # Apply risk targeting
#         final_weights = self._apply_risk_targeting(combined_weights)

#         return final_weights

#     def _combine_strategies(
#         self, eigen_weights: pd.DataFrame, arb_signals: pd.DataFrame
#     ) -> pd.DataFrame:
#         """Combine eigenportfolio and arbitrage signals"""
#         # Compute strategy allocations based on regime
#         regime_weights = self._compute_regime_weights()

#         # Combine weights
#         combined = (
#             eigen_weights * regime_weights["eigen"]
#             + arb_signals * regime_weights["arbitrage"]
#         )

#         return combined

#     def _apply_risk_targeting(self, weights: pd.DataFrame) -> pd.DataFrame:
#         """Apply risk targeting to portfolio weights"""
#         portfolio_vol = self._estimate_portfolio_volatility(weights)
#         scaling_factor = self.risk_target / portfolio_vol

#         return weights * scaling_factor
