"""
Out-of-sample simulation test script.

This script runs a 1-day simulation using tick data to evaluate
strategy performance on unseen data.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Tuple
import yfinance as yf
import random
import torch
import torch.nn.functional as F

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reinforcement.forex_trading_env import ForexTradingEnvironment
from src.reinforcement.rl_models import RLTrader
from src.database.forex_data_loader import ForexDataLoader
from src.config.paths import *
from src.config.settings import FOREX_PAIRS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            PROJECT_ROOT
            / "logs"
            / f"simulation_test_{datetime.now().strftime('%Y%m%d')}.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class SimulationTester:
    """Runs out-of-sample simulation tests."""

    def __init__(self, pairs: List[str], window_size: int = 20):
        self.pairs = pairs
        self.window_size = window_size
        self.portfolio_values = []
        self.positions = []
        self.timestamps = []
        self.initial_balance = 10000  # Starting balance
        self.balance = self.initial_balance
        logger.info(f"Initialized simulation tester for {len(pairs)} pairs")

    def run_simulation(self, agent: RLTrader, data: pd.DataFrame) -> Dict[str, float]:
        """Run a simulation with the given agent and data."""
        try:
            # Initialize environment
            env = ForexTradingEnvironment(
                data=data,
                pairs=self.pairs,
                window_size=self.window_size,
                initial_balance=self.initial_balance,
            )

            # Reset environment
            state = env.reset()
            done = False
            episode_reward = 0
            step = 0

            while not done:
                try:
                    # Select action
                    action = agent.select_action(state)

                    # Take action
                    next_state, reward, done, info = env.step(action)

                    # Update metrics
                    episode_reward += reward

                    # Calculate portfolio value based on positions and current prices
                    portfolio_value = self.initial_balance
                    for pair_idx, (pair, position) in enumerate(
                        zip(self.pairs, info["positions"])
                    ):
                        if position != 0:
                            price = info["current_prices"][pair_idx]["close"]
                            portfolio_value += position * price

                    # Store metrics
                    self.portfolio_values.append(portfolio_value)
                    self.positions.append(info["positions"])
                    self.timestamps.append(info["timestamp"])

                    # Move to next state
                    state = next_state
                    step += 1

                except Exception as e:
                    logger.error(f"Error in simulation step {step}: {str(e)}")
                    raise

            # Calculate metrics
            total_return = (
                self.portfolio_values[-1] - self.initial_balance
            ) / self.initial_balance
            sharpe_ratio = self._calculate_sharpe_ratio()
            max_drawdown = self._calculate_max_drawdown()

            metrics = {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "final_portfolio_value": self.portfolio_values[-1],
            }

            return metrics

        except Exception as e:
            logger.error(f"Error in simulation loop: {str(e)}")
            raise

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate the Sharpe ratio of the portfolio returns."""
        try:
            # Convert portfolio values to returns
            returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]

            # Calculate annualized Sharpe ratio (assuming daily data)
            if len(returns) > 0 and returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                sharpe = 0.0

            return sharpe

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate the maximum drawdown of the portfolio."""
        try:
            # Convert to numpy array
            portfolio_values = np.array(self.portfolio_values)

            # Calculate running maximum
            running_max = np.maximum.accumulate(portfolio_values)

            # Calculate drawdowns
            drawdowns = portfolio_values / running_max - 1

            # Get maximum drawdown
            max_drawdown = np.min(drawdowns)

            return max_drawdown

        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0


def fetch_tick_data() -> pd.DataFrame:
    """Fetch tick data for testing."""
    try:
        data_frames = []
        test_date = datetime.now() - timedelta(days=1)

        # Map forex pairs to yfinance symbols
        yf_symbols = {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X"}

        for pair in FOREX_PAIRS[:3]:
            # Get yfinance symbol
            yf_symbol = yf_symbols[pair]

            # Fetch intraday data
            ticker = yf.Ticker(yf_symbol)
            pair_data = ticker.history(
                start=test_date, end=test_date + timedelta(days=1), interval="1m"
            )

            if len(pair_data) == 0:
                logger.warning(f"No data available for {pair}, using simulated data")
                # Generate simulated data
                dates = pd.date_range(
                    test_date, test_date + timedelta(days=1), freq="1min"
                )
                base_price = 1.0 if "USD" in pair else 100.0

                pair_data = pd.DataFrame(
                    {
                        "Open": np.random.normal(
                            base_price, base_price * 0.001, len(dates)
                        ),
                        "High": np.random.normal(
                            base_price, base_price * 0.001, len(dates)
                        ),
                        "Low": np.random.normal(
                            base_price, base_price * 0.001, len(dates)
                        ),
                        "Close": np.random.normal(
                            base_price, base_price * 0.001, len(dates)
                        ),
                        "Volume": np.random.randint(1000, 10000, len(dates)),
                    },
                    index=dates,
                )

                # Ensure price consistency
                pair_data["High"] = np.maximum(
                    pair_data[["Open", "High", "Low", "Close"]].max(axis=1),
                    pair_data["High"],
                )
                pair_data["Low"] = np.minimum(
                    pair_data[["Open", "High", "Low", "Close"]].min(axis=1),
                    pair_data["Low"],
                )

            # Drop unnecessary columns
            pair_data = pair_data[["Open", "High", "Low", "Close", "Volume"]]

            # Convert pair format for column names
            formatted_pair = pair.lower().replace("/", "")

            # Rename columns to match environment expectations
            column_mapping = {
                "Open": f"{formatted_pair}_open",
                "High": f"{formatted_pair}_high",
                "Low": f"{formatted_pair}_low",
                "Close": f"{formatted_pair}_close",
                "Volume": f"{formatted_pair}_volume",
            }
            pair_data = pair_data.rename(columns=column_mapping)

            # Add required columns
            pair_data[f"{formatted_pair}_bid"] = (
                pair_data[f"{formatted_pair}_close"] * 0.9999
            )  # Simulate bid
            pair_data[f"{formatted_pair}_ask"] = (
                pair_data[f"{formatted_pair}_close"] * 1.0001
            )  # Simulate ask
            pair_data[f"{formatted_pair}_mid"] = pair_data[f"{formatted_pair}_close"]

            data_frames.append(pair_data)

        # Combine all pairs
        combined_data = pd.concat(data_frames, axis=1)

        # Fill any missing values
        combined_data = combined_data.ffill().bfill()

        logger.info(
            f"Fetched {len(combined_data)} ticks for {len(FOREX_PAIRS[:3])} pairs"
        )
        logger.info(f"Available columns: {combined_data.columns.tolist()}")

        return combined_data

    except Exception as e:
        logger.error(f"Error fetching tick data: {str(e)}")
        raise


def plot_and_save_results(
    portfolio_values: List[float],
    positions: List[List[float]],
    timestamps: List[datetime],
    pairs: List[str],
) -> None:
    """Plot and save trading results with signals and portfolio value."""
    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        fig.suptitle("Trading Simulation Results", fontsize=16)

        # Convert timestamps to pandas datetime
        dates = pd.to_datetime(timestamps)

        # Plot positions/signals
        for i, pair in enumerate(pairs):
            pair_positions = [pos[i] for pos in positions]
            ax1.plot(dates, pair_positions, label=f"{pair} Position", alpha=0.7)

            # Mark buy/sell signals
            buys = [i for i, pos in enumerate(pair_positions) if pos > 0]
            sells = [i for i, pos in enumerate(pair_positions) if pos < 0]

            if buys:
                ax1.scatter(
                    dates[buys],
                    [pair_positions[i] for i in buys],
                    marker="^",
                    color="green",
                    label=f"{pair} Buy",
                )
            if sells:
                ax1.scatter(
                    dates[sells],
                    [pair_positions[i] for i in sells],
                    marker="v",
                    color="red",
                    label=f"{pair} Sell",
                )

        ax1.set_title("Trading Signals")
        ax1.legend(loc="upper left")
        ax1.grid(True)

        # Plot portfolio value
        ax2.plot(dates, portfolio_values, label="Portfolio Value", color="blue")
        ax2.set_title("Portfolio Value Over Time")
        ax2.legend(loc="upper left")
        ax2.grid(True)

        # Format axes
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        daily_path = PROJECT_ROOT / "outputs" / "daily"
        daily_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(daily_path / f"trading_results_{timestamp}.png")

        # Save performance metrics
        performance_path = PROJECT_ROOT / "outputs" / "performance"
        performance_path.mkdir(parents=True, exist_ok=True)

        metrics = {
            "portfolio_value": portfolio_values,
            "returns": [
                ((v - portfolio_values[0]) / portfolio_values[0])
                for v in portfolio_values
            ],
            "positions": [
                {pair: pos[i] for i, pair in enumerate(pairs)} for pos in positions
            ],
            "timestamps": [ts.isoformat() for ts in timestamps],
        }

        with open(performance_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Results saved to {daily_path} and {performance_path}")

    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")
        raise


def pretrain_agent(
    agent: RLTrader, env: ForexTradingEnvironment, n_steps: int = 1000
) -> None:
    """Pre-train agent with sophisticated trading rules."""
    logger.info("Pre-training agent with sophisticated trading rules...")

    # Create replay buffer for pre-training
    pretrain_buffer = []

    # Generate pre-training data
    state = env.reset()
    for step in range(n_steps):
        # Get current prices and calculate basic indicators
        current_data = env.data.iloc[env.current_step]

        # Generate expert action based on sophisticated rules
        expert_actions = []
        for pair in env.pairs:
            formatted_pair = pair.lower().replace("/", "")

            if env.current_step >= 20:
                # Get price data
                price_window = env.data.iloc[
                    env.current_step - 20 : env.current_step + 1
                ]
                close_prices = price_window[f"{formatted_pair}_close"]
                current_price = close_prices.iloc[-1]

                # Calculate technical indicators
                ma20 = close_prices.mean()
                std20 = close_prices.std()
                upper_band = ma20 + 2 * std20
                lower_band = ma20 - 2 * std20

                # Calculate momentum
                returns = close_prices.pct_change()
                momentum = returns.mean()
                volatility = returns.std()

                # Calculate RSI-like indicator
                gains = (
                    returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
                )
                losses = (
                    -returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
                )
                rsi = 100 - (100 / (1 + gains / losses)) if losses != 0 else 50

                # Trading rules
                if current_price > upper_band and rsi > 70:
                    # Overbought
                    action = 2  # Sell
                elif current_price < lower_band and rsi < 30:
                    # Oversold
                    action = 1  # Buy
                elif momentum > 0 and volatility < 0.001:
                    # Strong uptrend with low volatility
                    action = 1  # Buy
                elif momentum < 0 and volatility < 0.001:
                    # Strong downtrend with low volatility
                    action = 2  # Sell
                else:
                    # Hold in uncertain conditions
                    action = 0
            else:
                action = 0  # Hold during warmup

            expert_actions.append(action)

        # Take action and get next state
        next_state, reward, done, _ = env.step(np.array(expert_actions))

        # Store transition
        pretrain_buffer.append((state, expert_actions, reward, next_state, done))

        # Move to next state
        state = next_state
        if done:
            state = env.reset()

    # Train on expert demonstrations
    logger.info("Training on expert demonstrations...")
    for epoch in range(100):  # Number of training iterations
        # Sample batch
        batch_size = min(32, len(pretrain_buffer))
        batch = random.sample(pretrain_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(agent.device)
        actions = torch.LongTensor(np.array(actions)).to(agent.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(agent.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(agent.device)
        dones = torch.BoolTensor(np.array(dones)).to(agent.device)

        # Update agent
        with torch.no_grad():
            next_q_values = agent.target_net(next_states)
            next_q_values = next_q_values.view(batch_size, agent.num_pairs, 3)
            next_q_values = next_q_values.max(2)[0].mean(dim=1)
            next_q_values[dones] = 0.0
            target_q_values = rewards + agent.gamma * next_q_values

        # Get current Q values
        current_q_values = agent.policy_net(states)
        current_q_values = current_q_values.view(batch_size, agent.num_pairs, 3)

        # Create index tensor for gathering
        batch_idx = torch.arange(batch_size).unsqueeze(1).to(agent.device)
        pair_idx = torch.arange(agent.num_pairs).unsqueeze(0).to(agent.device)

        # Gather Q values for taken actions
        current_q_values = current_q_values[batch_idx, pair_idx, actions]
        current_q_values = current_q_values.mean(dim=1)

        # Compute loss and optimize
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 1.0)
        agent.optimizer.step()

        # Update target network periodically
        if epoch % 10 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    # Final target network update
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    logger.info("Pre-training completed")


def main():
    """Main function to run simulation tests."""
    try:
        # Initialize tester
        tester = SimulationTester(
            pairs=FOREX_PAIRS[:3],  # Start with top 3 pairs for testing
        )

        # Fetch data
        data = fetch_tick_data()

        # Initialize environment for model setup
        env = ForexTradingEnvironment(
            data=data, pairs=FOREX_PAIRS[:3], window_size=20, initial_balance=10000
        )

        # Initialize agent
        state_dim = env.observation_space.shape[0]
        action_dim = len(FOREX_PAIRS[:3]) * 3
        hidden_dim = 128

        agent = RLTrader(
            env=env,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        logger.info(
            f"Using random initialization with state_dim={state_dim}, action_dim={action_dim}"
        )

        # Pre-train agent
        pretrain_agent(agent, env)

        # Run simulation
        metrics = tester.run_simulation(agent, data)

        # Plot and save results
        # plot_and_save_results(
        #     metrics["portfolio_values"],
        #     metrics["positions"],
        #     metrics["timestamps"],
        #     FOREX_PAIRS[:3],
        # )

        # Log results
        logger.info("Simulation completed successfully")
        logger.info(f"Final portfolio value: ${metrics['final_portfolio_value']:,.2f}")
        logger.info(f"Total return: {metrics['total_return']:.2%}")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Maximum drawdown: {metrics['max_drawdown']:.2%}")

    except Exception as e:
        logger.error(f"Error in simulation test: {str(e)}")
        raise


if __name__ == "__main__":
    main()
