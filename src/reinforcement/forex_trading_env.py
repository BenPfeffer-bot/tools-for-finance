"""
Forex Trading Environment

This module implements a forex trading environment for reinforcement learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import gym
from gym import spaces
import logging

logger = logging.getLogger(__name__)


class ForexTradingEnvironment(gym.Env):
    """
    Forex trading environment for RL agents.

    Implements a custom gym environment for forex trading with:
    - Multiple currency pairs
    - Bid-ask spread consideration
    - Dynamic position sizing
    - Advanced risk management
    - Realistic transaction costs
    """

    def __init__(
        self,
        data: pd.DataFrame,
        pairs: List[str],
        initial_balance: float = 100000.0,
        max_position_size: float = 0.1,
        base_transaction_cost: float = 0.0001,
        slippage_factor: float = 0.0001,
        window_size: int = 50,
        reward_scaling: float = 1e-4,
        max_drawdown: float = 0.15,
        risk_per_trade: float = 0.02,
        volatility_lookback: int = 20,
    ):
        """Initialize forex trading environment."""
        super(ForexTradingEnvironment, self).__init__()

        if data.empty:
            raise ValueError("Data cannot be empty")

        self.data = data
        self.pairs = pairs
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.base_transaction_cost = base_transaction_cost
        self.slippage_factor = slippage_factor
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.max_drawdown = max_drawdown
        self.risk_per_trade = risk_per_trade
        self.volatility_lookback = volatility_lookback

        # Validate data has required columns
        required_columns = [
            "bid",
            "ask",
            "mid",
            "bid_size",
            "ask_size",
            "volume",
            "pair",
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Action space: For each pair [no_position, long, short]
        self.action_space = spaces.MultiDiscrete([3] * len(pairs))

        # Calculate observation space size
        n_features = self._calculate_n_features()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        # Initialize trading state
        self.balance = initial_balance
        self.positions = {pair: 0.0 for pair in pairs}
        self.entry_prices = {pair: 0.0 for pair in pairs}
        self.trades = []
        self.current_step = self.window_size

        # Risk management metrics
        self.drawdown_history = []
        self.volatility_history = {pair: [] for pair in pairs}
        self.position_concentration = 0.0

        logger.info("Forex trading environment initialized successfully")
        logger.info(f"Observation space shape: {self.observation_space.shape}")
        logger.info(f"Action space shape: {self.action_space.shape}")

    def _calculate_n_features(self) -> int:
        """Calculate number of features in the observation space."""
        try:
            # Price features per pair
            price_features = 3  # bid, ask, mid

            # Technical indicators per pair
            tech_features = 5  # MA5, MA10, MA20, volatility, RSI

            # Volume features per pair
            volume_features = 3  # volume, bid_size, ask_size

            # Position features per pair
            position_features = 2  # current_position, unrealized_pnl

            # Features per pair
            features_per_pair = (
                price_features + tech_features + volume_features + position_features
            )

            # Account features
            account_features = (
                5  # balance, equity, margin_used, max_drawdown, avg_volatility
            )

            total_features = len(self.pairs) * features_per_pair + account_features
            logger.debug(f"Total features in observation space: {total_features}")

            return total_features

        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")
            raise

    def _calculate_position_size(self, pair: str, action: int) -> float:
        """
        Calculate position size based on risk management rules.

        Args:
            pair: Currency pair
            action: Trading action

        Returns:
            Position size
        """
        if action == 0:  # No position
            return 0.0

        # Get current prices and volatility
        current_prices = self._get_current_prices()
        price_data = current_prices[pair]
        volatility = self._calculate_volatility(pair)

        # Calculate position size based on risk per trade
        risk_amount = self.balance * self.risk_per_trade
        stop_distance = volatility * 2  # 2 standard deviations for stop loss

        # Position size = Risk amount / (Stop distance * Exchange rate)
        position_size = risk_amount / (stop_distance * price_data["mid"])

        # Apply position limits
        max_size = self.balance * self.max_position_size / price_data["mid"]
        position_size = min(position_size, max_size)

        # Adjust for market volatility
        vol_scale = 1.0 / (1.0 + volatility)  # Reduce size in high volatility
        position_size *= vol_scale

        # Adjust for portfolio concentration
        total_positions = sum(abs(pos) for pos in self.positions.values())
        if total_positions > 0:
            concentration_scale = 1.0 / (1.0 + total_positions / len(self.pairs))
            position_size *= concentration_scale

        return position_size if action == 1 else -position_size

    def _calculate_volatility(self, pair: str) -> float:
        """Calculate volatility for a pair."""
        window_data = self.data.iloc[
            self.current_step - self.volatility_lookback : self.current_step
        ]
        pair_data = window_data[window_data["pair"] == pair]
        return pair_data["mid"].pct_change().std()

    def _calculate_transaction_cost(self, pair: str, position_change: float) -> float:
        """
        Calculate transaction cost including spread and slippage.

        Args:
            pair: Currency pair
            position_change: Change in position size

        Returns:
            Total transaction cost
        """
        current_prices = self._get_current_prices()
        price_data = current_prices[pair]

        # Base spread cost
        spread_cost = abs(position_change) * (price_data["ask"] - price_data["bid"])

        # Slippage cost based on position size and volatility
        volatility = self._calculate_volatility(pair)
        slippage = (
            abs(position_change)
            * price_data["mid"]
            * self.slippage_factor
            * (1 + volatility)
        )

        # Market impact cost
        market_impact = (
            abs(position_change)
            * price_data["mid"]
            * self.base_transaction_cost
            * (1 + abs(position_change) / 100000)
        )  # Increases with position size

        return spread_cost + slippage + market_impact

    def _calculate_reward(self) -> float:
        """
        Calculate reward with risk-adjusted metrics.

        Returns:
            Risk-adjusted reward
        """
        # Calculate basic PnL
        total_pnl = sum(self._calculate_unrealized_pnl(pair) for pair in self.pairs)

        # Calculate risk metrics
        current_drawdown = (
            self.balance / self.initial_balance - 1
        )  # Current drawdown from initial balance
        volatility_penalty = sum(
            self._calculate_volatility(pair) for pair in self.pairs
        ) / len(self.pairs)

        # Position concentration penalty
        total_exposure = sum(abs(pos) for pos in self.positions.values())
        concentration_penalty = (
            total_exposure / len(self.pairs)
        ) ** 2  # Quadratic penalty for concentration

        # Combine components into final reward
        reward = (
            total_pnl * self.reward_scaling
            - abs(current_drawdown) * 0.1  # Drawdown penalty
            - volatility_penalty * 0.05  # Volatility penalty
            - concentration_penalty * 0.05  # Concentration penalty
        )

        return float(reward)

    def _is_done(self) -> bool:
        """Check if episode should end."""
        # Check for max drawdown violation
        current_drawdown = self.balance / self.initial_balance - 1
        if abs(current_drawdown) > self.max_drawdown:
            logger.info(f"Episode ended due to max drawdown: {current_drawdown:.2%}")
            return True

        # Check for insufficient margin
        margin_used = sum(
            abs(pos) * self._get_current_prices()[pair]["mid"]
            for pair, pos in self.positions.items()
        )
        if margin_used > self.balance * 0.8:  # 80% margin threshold
            logger.info("Episode ended due to insufficient margin")
            return True

        # Check for excessive losses in open positions
        for pair in self.pairs:
            unrealized_pnl = self._calculate_unrealized_pnl(pair)
            if unrealized_pnl < -self.balance * 0.1:  # 10% loss threshold per position
                logger.info(f"Episode ended due to excessive loss in {pair}")
                return True

        return self.current_step >= len(self.data) - 1

    def _get_info(self) -> Dict:
        """Get enhanced trading information."""
        info = {
            "balance": self.balance,
            "equity": self.balance
            + sum(self._calculate_unrealized_pnl(pair) for pair in self.pairs),
            "positions": self.positions.copy(),
            "drawdown": self.balance / self.initial_balance - 1,
            "margin_used": sum(
                abs(pos) * self._get_current_prices()[pair]["mid"]
                for pair, pos in self.positions.items()
            ),
            "position_concentration": sum(abs(pos) for pos in self.positions.values())
            / len(self.pairs),
            "volatility": {
                pair: self._calculate_volatility(pair) for pair in self.pairs
            },
            "unrealized_pnl": {
                pair: self._calculate_unrealized_pnl(pair) for pair in self.pairs
            },
            "transaction_costs": sum(
                self._calculate_transaction_cost(pair, self.positions[pair])
                for pair in self.pairs
            ),
        }
        return info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, True, self._get_info()

        # Execute trades with position sizing and risk management
        self._execute_trades(action)

        # Update state and calculate reward
        reward = self._calculate_reward()
        self.current_step += 1

        # Update risk metrics
        self.drawdown_history.append(self.balance / self.initial_balance - 1)
        for pair in self.pairs:
            self.volatility_history[pair].append(self._calculate_volatility(pair))

        done = self._is_done()
        return self._get_observation(), reward, done, self._get_info()

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        try:
            # Reset trading state
            self.balance = self.initial_balance
            self.positions = {pair: 0.0 for pair in self.pairs}
            self.entry_prices = {pair: 0.0 for pair in self.pairs}
            self.trades = []
            self.current_step = self.window_size

            # Reset risk metrics
            self.drawdown_history = []
            self.volatility_history = {pair: [] for pair in self.pairs}
            self.position_concentration = 0.0

            # Get initial observation
            observation = self._get_observation()
            if observation is None:
                raise ValueError("Failed to get initial observation")

            logger.debug(
                f"Environment reset. Initial observation shape: {observation.shape}"
            )
            return observation

        except Exception as e:
            logger.error(f"Error resetting environment: {str(e)}")
            raise

    def render(self, mode="human"):
        """Render the environment state."""
        if mode == "human":
            info = self._get_info()
            print("\nTrading Environment State:")
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Equity: ${info['equity']:.2f}")
            print(f"Drawdown: {info['drawdown']:.2%}")
            print("\nPositions:")
            for pair, pos in self.positions.items():
                pnl = self._calculate_unrealized_pnl(pair)
                vol = self._calculate_volatility(pair)
                print(f"  {pair}: {pos:.4f} (PnL: ${pnl:.2f}, Vol: {vol:.4%})")

    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector."""
        try:
            if self.current_step < self.window_size:
                raise ValueError(
                    f"Current step {self.current_step} is less than window size {self.window_size}"
                )

            obs = []
            window_data = self.data.iloc[
                self.current_step - self.window_size : self.current_step
            ]

            for pair in self.pairs:
                pair_data = window_data[window_data["pair"] == pair]
                if len(pair_data) == 0:
                    # Use zeros for missing data
                    obs.extend([0.0] * 13)
                    continue

                try:
                    # Price features
                    obs.extend(
                        [
                            float(pair_data["mid"].iloc[-1]),
                            float(pair_data["spread"].iloc[-1]),
                            float(pair_data["spread_pct"].iloc[-1]),
                        ]
                    )

                    # Technical indicators
                    price_series = pair_data["mid"].astype(float)

                    # Moving averages
                    ma_5 = price_series.rolling(window=5, min_periods=1).mean().iloc[-1]
                    ma_10 = (
                        price_series.rolling(window=10, min_periods=1).mean().iloc[-1]
                    )
                    ma_20 = (
                        price_series.rolling(window=20, min_periods=1).mean().iloc[-1]
                    )
                    obs.extend([float(ma_5), float(ma_10), float(ma_20)])

                    # Volatility
                    volatility = (
                        price_series.rolling(window=20, min_periods=1).std().iloc[-1]
                    )
                    obs.append(float(volatility))

                    # RSI
                    delta = price_series.diff()
                    gain = (
                        (delta.where(delta > 0, 0))
                        .rolling(window=14, min_periods=1)
                        .mean()
                    )
                    loss = (
                        (-delta.where(delta < 0, 0))
                        .rolling(window=14, min_periods=1)
                        .mean()
                    )
                    rs = gain.iloc[-1] / (loss.iloc[-1] if loss.iloc[-1] != 0 else 1e-6)
                    rsi = float(100 - (100 / (1 + rs)) if not np.isnan(rs) else 50.0)
                    obs.append(rsi)

                    # Volume indicators
                    obs.extend(
                        [
                            float(pair_data["volume"].iloc[-1]),
                            float(pair_data["bid_size"].iloc[-1]),
                            float(pair_data["ask_size"].iloc[-1]),
                        ]
                    )

                    # Position and PnL
                    obs.extend(
                        [
                            float(self.positions[pair]),
                            float(self._calculate_unrealized_pnl(pair)),
                        ]
                    )

                except (IndexError, KeyError, ValueError) as e:
                    logger.error(f"Error processing pair {pair}: {str(e)}")
                    obs.extend([0.0] * 13)  # Use zeros for failed calculations

            # Account features
            try:
                equity = self.balance + sum(
                    self._calculate_unrealized_pnl(pair) for pair in self.pairs
                )
                margin_used = sum(
                    abs(pos) * self._get_current_prices()[pair]["mid"]
                    for pair, pos in self.positions.items()
                )

                obs.extend(
                    [
                        self.balance / self.initial_balance,
                        equity / self.initial_balance,
                        margin_used / self.balance if self.balance > 0 else 0.0,
                        min(self.drawdown_history or [0]),
                        float(
                            np.mean(
                                [
                                    self._calculate_volatility(pair)
                                    for pair in self.pairs
                                ]
                            )
                        ),
                    ]
                )
            except Exception as e:
                logger.error(f"Error calculating account features: {str(e)}")
                obs.extend([0.0] * 5)

            observation = np.array(obs, dtype=np.float32)
            if not np.all(np.isfinite(observation)):
                logger.warning(
                    "Non-finite values in observation. Replacing with zeros."
                )
                observation = np.nan_to_num(
                    observation, nan=0.0, posinf=0.0, neginf=0.0
                )

            if len(observation) != self.observation_space.shape[0]:
                raise ValueError(
                    f"Observation length {len(observation)} does not match "
                    f"observation space shape {self.observation_space.shape[0]}"
                )

            return observation

        except Exception as e:
            logger.error(f"Error constructing observation: {str(e)}")
            raise

    def _execute_trades(self, action: np.ndarray) -> float:
        """Execute trades based on actions."""
        total_reward = 0.0
        current_prices = self._get_current_prices()

        for i, pair in enumerate(self.pairs):
            current_action = action[i]
            current_position = self.positions[pair]

            # Get prices for the pair
            bid_price = current_prices[pair]["bid"]
            ask_price = current_prices[pair]["ask"]

            # Calculate target position
            target_position = 0.0
            if current_action == 1:  # Long
                target_position = self._calculate_position_size(pair, current_action)
            elif current_action == 2:  # Short
                target_position = self._calculate_position_size(pair, current_action)

            # Calculate position change
            position_change = target_position - current_position

            if position_change != 0:
                # Calculate trade size and costs
                trade_size = abs(position_change)
                trade_price = ask_price if position_change > 0 else bid_price
                cost = self._calculate_transaction_cost(pair, position_change)

                # Execute trade
                self.balance -= cost
                self.positions[pair] = target_position
                if target_position != 0:
                    self.entry_prices[pair] = trade_price

                # Record trade
                self.trades.append(
                    {
                        "timestamp": self.data.index[self.current_step],
                        "pair": pair,
                        "action": "buy" if position_change > 0 else "sell",
                        "size": trade_size,
                        "price": trade_price,
                        "cost": cost,
                    }
                )

            # Calculate reward for this pair
            pair_pnl = self._calculate_unrealized_pnl(pair)
            total_reward += pair_pnl * self.reward_scaling

        return total_reward

    def _get_current_prices(self) -> Dict:
        """Get current prices for all pairs."""
        current_data = self.data.iloc[self.current_step]
        prices = {}

        for pair in self.pairs:
            pair_data = current_data[current_data["pair"] == pair].iloc[0]
            prices[pair] = {
                "bid": pair_data["bid"],
                "ask": pair_data["ask"],
                "mid": pair_data["mid"],
            }

        return prices

    def _calculate_unrealized_pnl(self, pair: str) -> float:
        """Calculate unrealized PnL for a position."""
        if self.positions[pair] == 0:
            return 0.0

        current_prices = self._get_current_prices()
        current_price = current_prices[pair]["mid"]

        if self.positions[pair] > 0:
            return (current_price - self.entry_prices[pair]) * self.positions[pair]
        else:
            return (self.entry_prices[pair] - current_price) * abs(self.positions[pair])
