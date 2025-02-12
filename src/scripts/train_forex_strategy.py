"""
Train and test forex trading strategy.

This script trains and evaluates the RL-based forex trading strategy.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.forex_data_loader import ForexDataLoader
from src.reinforcement.forex_trading_env import ForexTradingEnvironment
from src.reinforcement.rl_models import RLTrader
from src.deep_learning.dl_models import LSTMArbitrageDetector
from src.config.paths import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            PROJECT_ROOT
            / "logs"
            / f"forex_training_{datetime.now().strftime('%Y%m%d')}.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class ForexStrategyTrainer:
    """Trainer for forex trading strategy."""

    def __init__(
        self,
        pairs: List[str],
        initial_balance: float = 100000.0,
        window_size: int = 50,
        validation_split: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize forex strategy trainer.

        Args:
            pairs: List of forex pairs to trade
            initial_balance: Initial account balance
            window_size: Size of observation window
            validation_split: Fraction of data to use for validation
            device: Device to use for training
        """
        self.pairs = pairs
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.validation_split = validation_split
        self.device = device

        # Initialize data loader
        self.data_loader = ForexDataLoader(pairs=pairs)

        logger.info(f"Initialized ForexStrategyTrainer with {len(pairs)} pairs")

    def prepare_data(
        self, start_date: datetime, end_date: datetime
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training and validation data.

        Args:
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            tuple: (training_data, validation_data)
        """
        # Fetch historical data
        data = self.data_loader.fetch_historical_data(start_date, end_date)

        # Calculate features
        features = self.data_loader.prepare_features(data)

        # Split into training and validation
        split_idx = int(len(features) * (1 - self.validation_split))
        train_data = features.iloc[:split_idx]
        val_data = features.iloc[split_idx:]

        logger.info(
            f"Prepared data: {len(train_data)} training samples, {len(val_data)} validation samples"
        )
        return train_data, val_data

    def train_strategy(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        n_episodes: int = 1000,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
    ) -> Tuple[RLTrader, Dict]:
        """
        Train the trading strategy.

        Args:
            train_data: Training data
            val_data: Validation data
            n_episodes: Number of training episodes
            batch_size: Batch size for training
            learning_rate: Learning rate
            gamma: Discount factor

        Returns:
            tuple: (trained_agent, training_metrics)
        """
        # Create training environment
        train_env = ForexTradingEnvironment(
            data=train_data,
            pairs=self.pairs,
            initial_balance=self.initial_balance,
            window_size=self.window_size,
        )

        # Initialize RL agent
        state_dim = train_env.observation_space.shape[0]
        action_dim = len(self.pairs) * 3  # 3 actions per pair

        agent = RLTrader(
            env=train_env,
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            device=self.device,
        )

        # Training metrics
        metrics = {
            "episode_rewards": [],
            "portfolio_values": [],
            "n_trades": [],
            "validation_returns": [],
        }

        # Training loop
        for episode in range(n_episodes):
            logger.info(f"Starting episode {episode + 1}/{n_episodes}")

            # Train one episode
            episode_metrics = self._train_episode(agent, train_env, batch_size)

            # Validate strategy
            if (episode + 1) % 10 == 0:
                val_metrics = self.validate_strategy(agent, val_data)
                metrics["validation_returns"].append(val_metrics["total_return"])

                logger.info(
                    f"Episode {episode + 1} - "
                    f"Training Return: {episode_metrics['total_reward']:.2%}, "
                    f"Validation Return: {val_metrics['total_return']:.2%}"
                )

            # Update metrics
            metrics["episode_rewards"].append(episode_metrics["total_reward"])
            metrics["portfolio_values"].append(episode_metrics["final_portfolio_value"])
            metrics["n_trades"].append(episode_metrics["n_trades"])

            # Save model periodically
            if (episode + 1) % 100 == 0:
                self.save_model(agent, RL_MODELS / f"agent_episode_{episode + 1}.pth")

        # Plot training curves
        self._plot_training_curves(metrics)

        return agent, metrics

    def _train_episode(
        self, agent: RLTrader, env: ForexTradingEnvironment, batch_size: int
    ) -> Dict:
        """Train one episode."""
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Select action
            action = agent.select_action(state)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train if enough samples
            if len(agent.replay_buffer) > batch_size:
                loss = agent.train_dqn(batch_size)

            state = next_state
            total_reward += reward

        return {
            "total_reward": total_reward,
            "final_portfolio_value": info["balance"],
            "n_trades": info["n_trades"],
        }

    def validate_strategy(self, agent: RLTrader, val_data: pd.DataFrame) -> Dict:
        """
        Validate trading strategy.

        Args:
            agent: Trained RL agent
            val_data: Validation data

        Returns:
            Dictionary of validation metrics
        """
        # Create validation environment
        val_env = ForexTradingEnvironment(
            data=val_data,
            pairs=self.pairs,
            initial_balance=self.initial_balance,
            window_size=self.window_size,
        )

        # Run validation episode
        state = val_env.reset()
        done = False
        portfolio_values = [val_env.balance]

        while not done:
            action = agent.select_action(state, epsilon=0)  # No exploration
            state, _, done, info = val_env.step(action)
            portfolio_values.append(info["balance"])

        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        metrics = {
            "total_return": (portfolio_values[-1] / portfolio_values[0]) - 1,
            "sharpe_ratio": np.mean(returns) / np.std(returns) * np.sqrt(252),
            "max_drawdown": min(portfolio_values) / max(portfolio_values) - 1,
            "n_trades": len(val_env.trades),
        }

        return metrics

    def save_model(self, agent: RLTrader, path: str):
        """Save trained model."""
        agent.save_model(path)
        logger.info(f"Saved model to {path}")

    def load_model(self, path: str) -> RLTrader:
        """Load trained model."""
        agent = RLTrader(
            env=None, state_dim=self.state_dim, action_dim=len(self.pairs) * 3
        )
        agent.load_model(path)
        logger.info(f"Loaded model from {path}")
        return agent

    def _plot_training_curves(self, metrics: Dict):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot episode rewards
        axes[0, 0].plot(metrics["episode_rewards"])
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")

        # Plot portfolio values
        axes[0, 1].plot(metrics["portfolio_values"])
        axes[0, 1].set_title("Portfolio Value")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Portfolio Value")

        # Plot number of trades
        axes[1, 0].plot(metrics["n_trades"])
        axes[1, 0].set_title("Number of Trades")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Number of Trades")

        # Plot validation returns
        axes[1, 1].plot(metrics["validation_returns"])
        axes[1, 1].set_title("Validation Returns")
        axes[1, 1].set_xlabel("Validation Episode")
        axes[1, 1].set_ylabel("Total Return")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "training_curves.png")
        plt.close()


def main():
    """Main function to train forex strategy."""
    try:
        # Define forex pairs to trade
        pairs = ["eurusd", "gbpusd", "usdjpy", "audusd", "usdcad"]

        # Initialize trainer
        trainer = ForexStrategyTrainer(pairs=pairs)

        # Prepare data
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()

        train_data, val_data = trainer.prepare_data(start_date, end_date)

        # Train strategy
        agent, metrics = trainer.train_strategy(
            train_data=train_data, val_data=val_data, n_episodes=1000
        )

        # Save final model
        trainer.save_model(agent, RL_MODELS / "final_model.pth")

        # Final validation
        final_metrics = trainer.validate_strategy(agent, val_data)

        logger.info("\nFinal Validation Results:")
        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value:.2%}")

    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
