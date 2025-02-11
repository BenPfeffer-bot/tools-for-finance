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
    def __init__(
        self,
        returns: pd.DataFrame,
        model,
        features: np.ndarray,
        dates: pd.DatetimeIndex,
        capital: float = 10000,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize Backtester.

        Args:
            returns: DataFrame of asset returns
            model: Trained ML model
            features: Feature array for predictions
            dates: DatetimeIndex for the backtest
            capital: Initial capital
            transaction_cost: Transaction cost per trade
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.model = model
        self.features = features
        self.dates = dates
        self.capital = capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.positions = None
        self.portfolio_returns = None
        self.portfolio_value = None

    def compute_positions(self) -> pd.Series:
        """
        Compute trading positions based on model predictions.

        Returns:
            Series of position sizes (-1, 0, or 1)
        """
        try:
            # Convert features to DMatrix
            dtest = xgb.DMatrix(self.features)

            # Get predictions
            predictions = self.model.predict(dtest)

            # Convert to positions (-1 for short, 1 for long)
            positions = pd.Series(index=self.dates)
            positions[predictions > 0.5] = 1
            positions[predictions <= 0.5] = -1

            return positions

        except Exception as e:
            logger.error(f"Error computing positions: {str(e)}")
            raise

    def compute_transaction_costs(self, positions: pd.Series) -> pd.Series:
        """
        Compute transaction costs from position changes.

        Args:
            positions: Series of trading positions

        Returns:
            Series of transaction costs
        """
        position_changes = positions.diff().abs()
        return position_changes * self.transaction_cost

    def compute_portfolio_metrics(
        self, returns: pd.Series, risk_free_rate: float = None
    ) -> Dict[str, float]:
        """
        Compute portfolio performance metrics.

        Args:
            returns: Series of portfolio returns
            risk_free_rate: Annual risk-free rate (optional)

        Returns:
            Dictionary of performance metrics
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

        # Basic metrics
        total_return = (1 + returns).prod() - 1
        daily_returns_mean = returns.mean()
        daily_returns_std = returns.std()

        # Annualized metrics
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_volatility = daily_returns_std * np.sqrt(252)

        # Risk metrics
        sharpe_ratio = (
            (daily_returns_mean - daily_rf) / daily_returns_std * np.sqrt(252)
        )
        sortino_ratio = (daily_returns_mean - daily_rf) / (
            returns[returns < 0].std() * np.sqrt(252)
        )

        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()

        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "skewness": stats.skew(returns),
            "kurtosis": stats.kurtosis(returns),
        }

    def run_backtest(self) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Run backtest and compute performance metrics.

        Returns:
            Tuple of (portfolio value series, performance metrics dict)
        """
        # Compute positions
        self.positions = self.compute_positions()

        # Compute transaction costs
        transaction_costs = self.compute_transaction_costs(self.positions)

        # Compute strategy returns
        strategy_returns = (
            self.positions.shift(1) * self.returns.loc[self.dates].iloc[:, 0]
        )

        # Subtract transaction costs
        self.portfolio_returns = strategy_returns - transaction_costs
        self.portfolio_returns = self.portfolio_returns.dropna()

        # Compute cumulative portfolio value
        self.portfolio_value = self.capital * (1 + self.portfolio_returns).cumprod()

        # Compute performance metrics
        metrics = self.compute_portfolio_metrics(self.portfolio_returns)

        # Plot results
        self.plot_backtest_results()

        return self.portfolio_value, metrics

    def plot_backtest_results(self) -> None:
        """Plot comprehensive backtest results."""
        if self.portfolio_value is None:
            raise ValueError("Must run backtest first!")

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))

        # Plot 1: Portfolio Value
        self.portfolio_value.plot(ax=axes[0])
        axes[0].set_title("Portfolio Value Over Time")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Portfolio Value")

        # Plot 2: Returns Distribution
        sns.histplot(self.portfolio_returns, kde=True, ax=axes[1])
        axes[1].set_title("Returns Distribution")
        axes[1].set_xlabel("Return")
        axes[1].set_ylabel("Frequency")

        # Plot 3: Drawdown
        drawdown = self.portfolio_value / self.portfolio_value.expanding().max() - 1
        drawdown.plot(ax=axes[2])
        axes[2].set_title("Portfolio Drawdown")
        axes[2].set_xlabel("Date")
        axes[2].set_ylabel("Drawdown")

        plt.tight_layout()
        plt.show()

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
