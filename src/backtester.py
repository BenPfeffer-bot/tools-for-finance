# backtester.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple
from scipy import stats
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtester class for evaluating trading strategies.
    """

    def __init__(self, returns: pd.DataFrame, predictions: np.ndarray):
        """
        Initialize backtester with returns data and model predictions.

        Args:
            returns: DataFrame of asset returns (dates x assets)
            predictions: Array of model predictions for trading signals
        """
        self.returns = returns
        self.predictions = predictions
        self.positions = None
        self.portfolio_value = None

        # Validate inputs
        if len(predictions) > len(returns):
            raise ValueError(
                "Length of predictions cannot exceed length of returns data"
            )

    def run_backtest(self):
        """
        Run backtest simulation.

        Returns:
            tuple: (portfolio_value, performance_metrics)
        """
        try:
            # Compute positions
            self.positions = self.compute_positions()

            # Calculate portfolio returns
            portfolio_returns = (self.returns * self.positions).sum(axis=1)

            # Apply transaction costs
            transaction_costs = self.compute_transaction_costs(self.positions)
            portfolio_returns = portfolio_returns - transaction_costs

            # Calculate cumulative portfolio value
            self.portfolio_value = (1 + portfolio_returns).cumprod()

            # Compute performance metrics
            metrics = self.compute_performance_metrics(portfolio_returns)

            return self.portfolio_value, metrics

        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise

    def compute_positions(self):
        """
        Compute positions based on model predictions and position sizing rules.
        Returns a Series of positions indexed by date.
        """
        # Initialize positions Series
        positions = pd.Series(0.0, index=self.returns.index)

        # Ensure predictions align with returns index
        predictions = pd.Series(
            self.predictions, index=self.returns.index[-len(self.predictions) :]
        )
        predictions = predictions.reindex(self.returns.index, fill_value=0)

        # Compute position sizes based on volatility
        volatility = self.returns.rolling(window=20).std()
        target_vol = 0.02  # Target 2% daily volatility
        position_sizes = target_vol / volatility.fillna(target_vol)
        position_sizes = position_sizes.clip(
            0.1, 1.0
        )  # Limit position sizes between 0.1 and 1.0

        # Apply positions based on predictions
        positions[predictions > 0] = position_sizes[predictions > 0]
        positions[predictions < 0] = -position_sizes[predictions < 0]

        # Apply stop-loss and take-profit rules
        cumulative_returns = (1 + self.returns * positions).cumprod()
        drawdown = cumulative_returns / cumulative_returns.cummax() - 1

        # Stop trading after significant drawdown
        stop_trading = drawdown < -0.2  # 20% drawdown threshold
        positions[stop_trading] = 0

        # Reduce position sizes in high volatility
        high_vol_mask = volatility > volatility.rolling(window=60).mean() * 1.5
        positions[high_vol_mask] *= 0.5

        return positions

    def compute_transaction_costs(self, positions: pd.Series) -> pd.Series:
        """
        Compute transaction costs based on position changes.

        Args:
            positions: Series of positions

        Returns:
            Series of transaction costs
        """
        # Define transaction cost parameters
        base_cost = 0.001  # 10 bps base cost
        market_impact = 0.0005  # 5 bps market impact

        # Calculate position changes
        position_changes = positions.diff().abs()

        # Compute total transaction costs
        costs = position_changes * (base_cost + market_impact)

        # Fill first value with initial position cost
        costs.iloc[0] = abs(positions.iloc[0]) * (base_cost + market_impact)

        return costs

    def compute_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Compute portfolio performance metrics.

        Args:
            returns: Series of portfolio returns

        Returns:
            Dictionary of performance metrics
        """
        # Annualization factor
        ann_factor = 252  # Trading days per year

        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1

        # Risk metrics
        ann_vol = returns.std() * np.sqrt(ann_factor)
        sharpe_ratio = ann_return / ann_vol if ann_vol != 0 else 0

        # Drawdown analysis
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()

        # Return metrics dictionary
        metrics = {
            "total_return": total_return,
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }

        return metrics

    def plot_backtest_results(self) -> None:
        """Plot comprehensive backtest results."""
        if self.portfolio_value is None:
            raise ValueError("Must run backtest first!")

        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Plot 1: Portfolio Value
        self.portfolio_value.plot(ax=axes[0, 0])
        axes[0, 0].set_title("Portfolio Value Over Time")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Portfolio Value")

        # Plot 2: Returns Distribution
        sns.histplot(self.portfolio_returns, kde=True, ax=axes[0, 1])
        axes[0, 1].set_title("Returns Distribution")
        axes[0, 1].set_xlabel("Return")
        axes[0, 1].set_ylabel("Frequency")

        # Plot 3: Drawdown
        drawdown = self.portfolio_value / self.portfolio_value.expanding().max() - 1
        drawdown.plot(ax=axes[1, 0])
        axes[1, 0].set_title("Portfolio Drawdown")
        axes[1, 0].set_xlabel("Date")
        axes[1, 0].set_ylabel("Drawdown")

        # Plot 4: Rolling Sharpe Ratio
        rolling_returns = self.portfolio_returns.rolling(window=252)
        rolling_sharpe = np.sqrt(252) * rolling_returns.mean() / rolling_returns.std()
        rolling_sharpe.plot(ax=axes[1, 1])
        axes[1, 1].set_title("Rolling Sharpe Ratio (1Y)")
        axes[1, 1].set_xlabel("Date")
        axes[1, 1].set_ylabel("Sharpe Ratio")

        # Plot 5: Position Sizes
        self.positions.plot(ax=axes[2, 0])
        axes[2, 0].set_title("Position Sizes Over Time")
        axes[2, 0].set_xlabel("Date")
        axes[2, 0].set_ylabel("Position Size")

        # Plot 6: Rolling Volatility
        rolling_vol = self.portfolio_returns.rolling(window=63).std() * np.sqrt(252)
        rolling_vol.plot(ax=axes[2, 1])
        axes[2, 1].set_title("Rolling Volatility (3M)")
        axes[2, 1].set_xlabel("Date")
        axes[2, 1].set_ylabel("Annualized Volatility")

        plt.tight_layout()
        plt.savefig("backtest_results.png")
        plt.close()

    def plot_position_analysis(self) -> None:
        """Plot analysis of trading positions."""
        if self.positions is None:
            raise ValueError("Must run backtest first!")

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Position Changes
        position_changes = self.positions.diff().abs()
        position_changes.plot(ax=axes[0])
        axes[0].set_title("Position Changes Over Time")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Absolute Position Change")

        # Plot 2: Position Distribution
        sns.histplot(self.positions, kde=True, ax=axes[1])
        axes[1].set_title("Position Distribution")
        axes[1].set_xlabel("Position Size")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()
