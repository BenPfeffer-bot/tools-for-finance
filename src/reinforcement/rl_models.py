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
    """Deep Q-Network for trade execution optimization."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


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
            # Market features (window_size x num_assets)
            market_data = self.returns.iloc[
                self.current_step - self.window_size : self.current_step
            ].values

            # Position features
            position_features = np.array(
                [
                    self.positions,  # Current positions for each asset
                    np.full_like(
                        self.positions, self.balance / self.initial_balance
                    ),  # Normalized balance for each asset
                ]
            )

            # Prediction features (single value for current step)
            prediction_feature = np.array([self.predictions[self.current_step]])

            # Combine all features
            state = np.concatenate(
                [
                    market_data.flatten(),  # Flattened market data
                    position_features.flatten(),  # Flattened position features
                    prediction_feature.flatten(),  # Current prediction
                ]
            )

            return state
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
        """Initialize RLTrader."""
        logger.info(
            f"Initializing RLTrader with state_dim: {state_dim}, action_dim: {action_dim}"
        )
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device

        # Initialize networks
        self.policy_net = DQNTrader(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = DQNTrader(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(10000)
        logger.info(f"RLTrader initialized successfully on device: {device}")

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_dim)

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

            # Compute current Q values
            current_q_values = self.policy_net(state_batch).gather(
                1, action_batch.unsqueeze(1)
            )

            # Compute next Q values
            with torch.no_grad():
                next_q_values = self.target_net(next_state_batch).max(1)[0]
                next_q_values[done_batch.bool()] = 0.0
                expected_q_values = reward_batch + self.gamma * next_q_values

            # Compute loss and optimize
            loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values)

            self.optimizer.zero_grad()
            loss.backward()
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
