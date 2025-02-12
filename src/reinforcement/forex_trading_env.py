"""
Forex trading environment for reinforcement learning.

This module implements a custom OpenAI Gym environment for forex trading
with support for multiple currency pairs and realistic trading conditions.
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ForexTradingEnvironment(gym.Env):
    def __init__(
        self,
        data: pd.DataFrame,
        pairs: List[str],
        window_size: int = 20,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.0001,
    ):
        """Initialize the environment."""
        try:
            self.data = data
            self.pairs = pairs
            self.window_size = window_size
            self.initial_balance = initial_balance
            self.transaction_cost = transaction_cost

            # Initialize state variables
            self.current_step = 0
            self.balance = initial_balance
            self.positions = np.zeros(
                len(pairs)
            )  # 0 = no position, 1 = long, -1 = short
            self.prev_prices = np.zeros(len(pairs))
            self.done = False

            # Define action and observation spaces
            self.action_space = spaces.MultiDiscrete(
                [3] * len(pairs)
            )  # 0 = hold, 1 = buy, 2 = sell

            # Each pair has 6 features: close, volume, returns, ma5, ma20, std20
            n_features = 6 * len(pairs)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
            )

            logger.info("Forex trading environment initialized successfully")
            logger.info(f"Observation space shape: {self.observation_space.shape}")
            logger.info(f"Action space shape: {self.action_space.shape}")

        except Exception as e:
            logger.error(f"Error initializing environment: {str(e)}")
            raise

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        try:
            self.current_step = 0
            self.balance = self.initial_balance
            self.positions = np.zeros(len(self.pairs))
            self.prev_prices = np.zeros(len(self.pairs))
            self.done = False

            # Get initial prices
            current_prices = self._get_current_prices()
            for i, price_data in enumerate(current_prices):
                self.prev_prices[i] = price_data["close"]

            return self._get_observation()

        except Exception as e:
            logger.error(f"Error in reset: {str(e)}")
            raise

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        try:
            # Get current data
            current_data = self.data.iloc[self.current_step]

            # Build observation for each pair
            observations = []
            for pair in self.pairs:
                # Convert pair format
                formatted_pair = pair.lower().replace("/", "")

                # Get price data
                close = current_data[f"{formatted_pair}_close"]
                volume = current_data[f"{formatted_pair}_volume"]

                # Calculate returns
                if self.current_step > 0:
                    prev_close = self.data.iloc[self.current_step - 1][
                        f"{formatted_pair}_close"
                    ]
                    returns = (close - prev_close) / prev_close
                else:
                    returns = 0.0

                # Calculate technical indicators
                if self.current_step >= 5:
                    ma5 = self.data.iloc[self.current_step - 5 : self.current_step + 1][
                        f"{formatted_pair}_close"
                    ].mean()
                else:
                    ma5 = close

                if self.current_step >= 20:
                    ma20 = self.data.iloc[
                        self.current_step - 20 : self.current_step + 1
                    ][f"{formatted_pair}_close"].mean()
                    std20 = self.data.iloc[
                        self.current_step - 20 : self.current_step + 1
                    ][f"{formatted_pair}_close"].std()
                else:
                    ma20 = close
                    std20 = 0.0

                # Combine features
                pair_obs = [close, volume, returns, ma5, ma20, std20]
                observations.extend(pair_obs)

            return np.array(observations, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error in _get_observation: {str(e)}")
            raise

    def _get_current_prices(self) -> List[Dict[str, float]]:
        """Get current prices for all pairs."""
        try:
            current_data = self.data.iloc[self.current_step]
            prices = []

            for pair in self.pairs:
                formatted_pair = pair.lower().replace("/", "")
                prices.append(
                    {
                        "close": current_data[f"{formatted_pair}_close"],
                        "bid": current_data[f"{formatted_pair}_bid"],
                        "ask": current_data[f"{formatted_pair}_ask"],
                        "mid": current_data[f"{formatted_pair}_mid"],
                    }
                )

            return prices

        except Exception as e:
            logger.error(f"Error in _get_current_prices: {str(e)}")
            raise

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment."""
        try:
            if self.done:
                raise RuntimeError("Episode is done, please reset the environment")

            # Get current prices for all pairs
            current_prices = self._get_current_prices()

            # Process each action for each pair
            total_reward = 0
            total_pnl = 0
            position_changes = 0
            transaction_costs = 0

            for pair_idx, (pair, pair_action) in enumerate(zip(self.pairs, action)):
                # Get current position and price for this pair
                current_position = self.positions[pair_idx]
                current_price = current_prices[pair_idx]["close"]

                # Calculate position change
                new_position = pair_action - 1  # Convert [0,1,2] to [-1,0,1]
                position_change = abs(new_position - current_position)
                position_changes += position_change

                # Calculate transaction costs with moderate slippage
                base_cost = self.transaction_cost
                slippage = (
                    0.00008 * position_change * (1 + abs(current_position))
                )  # Reduced slippage
                pair_transaction_cost = (base_cost + slippage) * position_change
                transaction_costs += pair_transaction_cost

                # Calculate PnL
                price_change = (
                    current_price - self.prev_prices[pair_idx]
                ) / self.prev_prices[pair_idx]
                position_pnl = current_position * price_change
                total_pnl += position_pnl

                # Calculate reward components with more balanced weights
                pnl_reward = position_pnl * 1.2  # Slightly increase PnL importance
                cost_penalty = -1.5 * pair_transaction_cost  # Moderate cost penalty

                # Calculate volatility and risk-adjusted reward
                formatted_pair = pair.lower().replace("/", "")
                if self.current_step >= 20:
                    # Calculate volatility using 20-period window
                    price_window = self.data.iloc[
                        self.current_step - 20 : self.current_step + 1
                    ][f"{formatted_pair}_close"]
                    volatility = price_window.pct_change().std()

                    # More balanced volatility penalty
                    risk_penalty = -0.8 * abs(new_position) * volatility

                    # Add small positive reward for holding in uncertain conditions
                    if (
                        new_position == 0 and volatility > 0.001
                    ):  # Hold during high volatility
                        risk_penalty += 0.0001
                else:
                    risk_penalty = 0
                    volatility = 0

                # Combine reward components
                step_reward = pnl_reward + cost_penalty + risk_penalty
                total_reward += step_reward

                # Update position and price
                self.positions[pair_idx] = new_position
                self.prev_prices[pair_idx] = current_price

            # Update balance
            self.balance *= 1 + total_pnl

            # Add moderate position concentration penalty
            concentration = np.sum(np.abs(self.positions)) / len(self.pairs)
            if concentration > 1.8:  # Higher threshold
                total_reward -= 0.08 * (concentration - 1.8)  # Reduced penalty

            # Move to next timestep
            self.current_step += 1
            if self.current_step >= len(self.data) - 1:
                self.done = True

            # Get next observation
            next_observation = self._get_observation()

            # Prepare info dictionary
            info = {
                "timestamp": self.data.index[self.current_step],
                "positions": self.positions.tolist(),
                "current_prices": current_prices,
                "balance": self.balance,
                "total_pnl": total_pnl,
                "transaction_costs": transaction_costs,
                "concentration": concentration,
                "volatility": volatility,
            }

            return next_observation, total_reward, self.done, info

        except Exception as e:
            logger.error(f"Error in step: {str(e)}")
            return (
                self._get_observation(),
                0,
                True,
                {
                    "error": str(e),
                    "positions": self.positions.tolist(),
                    "current_prices": self._get_current_prices(),
                    "balance": self.balance,
                    "timestamp": self.data.index[self.current_step]
                    if self.current_step < len(self.data)
                    else None,
                },
            )
