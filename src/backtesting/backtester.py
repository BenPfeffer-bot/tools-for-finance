# backtester.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List
from scipy import stats
import xgboost as xgb
import logging
from datetime import datetime, timedelta
from src.reinforcement.rl_models import RLTrader, TradingEnvironment

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

            # Calculate portfolio returns (element-wise multiplication and sum across assets)
            portfolio_returns = (self.returns * self.positions).sum(axis=1)

            # Apply transaction costs
            transaction_costs = self.compute_transaction_costs(self.positions)
            portfolio_returns = portfolio_returns - transaction_costs

            # Calculate cumulative portfolio value
            self.portfolio_value = (1 + portfolio_returns).cumprod()

            # Compute performance metrics
            metrics = self.compute_performance_metrics(portfolio_returns)

            # Log key metrics
            logger.info("\nBacktest Results:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.2%}")

            return self.portfolio_value, metrics

        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise

    def compute_positions(self):
        """
        Compute positions based on model predictions and position sizing rules.
        Returns a DataFrame of positions indexed by date.
        """
        # Initialize positions DataFrame with same structure as returns
        positions = pd.DataFrame(
            0.0, index=self.returns.index, columns=self.returns.columns
        )

        # Ensure predictions align with returns index
        pred_series = pd.Series(
            self.predictions, index=self.returns.index[-len(self.predictions) :]
        )
        pred_series = pred_series.reindex(self.returns.index, fill_value=0)

        # Compute rolling volatility and correlation
        volatility = self.returns.rolling(window=20).std()
        correlation = self.returns.rolling(window=60).corr()

        # Compute position sizes based on volatility targeting
        target_portfolio_vol = 0.15  # Target 15% annualized portfolio volatility
        daily_vol_target = target_portfolio_vol / np.sqrt(252)

        # Scale positions by volatility
        position_sizes = daily_vol_target / (volatility * np.sqrt(252))
        position_sizes = position_sizes.clip(
            -0.5, 0.5
        )  # Limit individual position sizes

        # Apply positions based on predictions and confidence
        confidence_threshold = 0.6
        for col in positions.columns:
            # Only take positions when prediction confidence is high
            positions[col] = np.where(
                abs(pred_series) > confidence_threshold,
                position_sizes[col] * np.sign(pred_series),
                0,
            )

        # Risk management rules
        # 1. Stop trading after significant drawdown
        portfolio_returns = (self.returns * positions).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        drawdown = cumulative_returns / cumulative_returns.cummax() - 1
        stop_trading = drawdown < -0.15  # 15% drawdown threshold
        positions.loc[stop_trading] = 0

        # 2. Reduce exposure in high volatility regimes
        vol_ma = volatility.mean(axis=1).rolling(window=60).mean()
        high_vol_mask = volatility.mean(axis=1) > vol_ma * 1.5
        positions.loc[high_vol_mask] *= 0.5

        # 3. Portfolio-level position limits
        gross_exposure = positions.abs().sum(axis=1)
        net_exposure = positions.sum(axis=1)

        # Limit gross exposure to 200%
        gross_scaling = pd.Series(1.0, index=positions.index)
        gross_scaling[gross_exposure > 2.0] = 2.0 / gross_exposure[gross_exposure > 2.0]

        # Limit net exposure to ±100%
        net_scaling = pd.Series(1.0, index=positions.index)
        net_scaling[net_exposure > 1.0] = 1.0 / net_exposure[net_exposure > 1.0]
        net_scaling[net_exposure < -1.0] = -1.0 / net_exposure[net_exposure < -1.0]

        # Apply scaling
        final_scaling = pd.concat([gross_scaling, net_scaling], axis=1).min(axis=1)
        for col in positions.columns:
            positions[col] *= final_scaling

        return positions

    def compute_transaction_costs(self, positions: pd.DataFrame) -> pd.Series:
        """
        Compute transaction costs based on position changes.

        Args:
            positions: DataFrame of positions

        Returns:
            Series of transaction costs
        """
        # Define transaction cost parameters
        base_cost = 0.001  # 10 bps base cost
        market_impact = 0.0005  # 5 bps market impact

        # Calculate absolute position changes for each asset
        position_changes = positions.diff().abs()

        # Fill first row with initial position costs
        position_changes.iloc[0] = positions.iloc[0].abs()

        # Compute total transaction costs across all assets
        total_costs = position_changes.sum(axis=1) * (base_cost + market_impact)

        return total_costs

    def compute_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Compute portfolio performance metrics.

        Args:
            returns: Series of portfolio returns

        Returns:
            Dictionary of performance metrics
        """
        # Convert returns to series if DataFrame
        if isinstance(returns, pd.DataFrame):
            returns = returns.mean(axis=1)

        # Annualization factor
        ann_factor = 252  # Trading days per year

        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1

        # Risk metrics
        ann_vol = returns.std() * np.sqrt(ann_factor)
        sharpe_ratio = ann_return / ann_vol if ann_vol > 0 else 0

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
        portfolio_returns = self.portfolio_value.pct_change()
        sns.histplot(portfolio_returns, kde=True, ax=axes[0, 1])
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
        rolling_returns = portfolio_returns.rolling(window=252)
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
        rolling_vol = portfolio_returns.rolling(window=63).std() * np.sqrt(252)
        rolling_vol.plot(ax=axes[2, 1])
        axes[2, 1].set_title("Rolling Volatility (3M)")
        axes[2, 1].set_xlabel("Date")
        axes[2, 1].set_ylabel("Annualized Volatility")

        plt.tight_layout()
        plt.savefig("outputs/plots/backtest_results.png")
        plt.close()

    def plot_position_analysis(self) -> None:
        """Plot analysis of trading positions."""
        if self.positions is None:
            raise ValueError("Must run backtest first!")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Position Changes
        position_changes = self.positions.diff().abs()
        position_changes.plot(ax=axes[0, 0])
        axes[0, 0].set_title("Position Changes Over Time")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Absolute Position Change")

        # Plot 2: Position Distribution
        sns.histplot(self.positions.values.flatten(), kde=True, ax=axes[0, 1])
        axes[0, 1].set_title("Position Distribution")
        axes[0, 1].set_xlabel("Position Size")
        axes[0, 1].set_ylabel("Frequency")

        # Plot 3: Gross Exposure
        gross_exposure = self.positions.abs().sum(axis=1)
        gross_exposure.plot(ax=axes[1, 0])
        axes[1, 0].set_title("Gross Exposure Over Time")
        axes[1, 0].set_xlabel("Date")
        axes[1, 0].set_ylabel("Gross Exposure")

        # Plot 4: Net Exposure
        net_exposure = self.positions.sum(axis=1)
        net_exposure.plot(ax=axes[1, 1])
        axes[1, 1].set_title("Net Exposure Over Time")
        axes[1, 1].set_xlabel("Date")
        axes[1, 1].set_ylabel("Net Exposure")

        plt.tight_layout()
        plt.savefig("outputs/plots/position_analysis.png")
        plt.close()

    def plot_signal_analysis(self) -> None:
        """Plot analysis of trading signals and predictions."""
        if self.predictions is None:
            raise ValueError("Must have predictions to analyze signals!")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Signal Distribution
        sns.histplot(self.predictions, kde=True, ax=axes[0, 0])
        axes[0, 0].set_title("Signal Distribution")
        axes[0, 0].set_xlabel("Signal Value")
        axes[0, 0].set_ylabel("Frequency")

        # Plot 2: Signal Over Time
        pd.Series(
            self.predictions, index=self.returns.index[-len(self.predictions) :]
        ).plot(ax=axes[0, 1])
        axes[0, 1].set_title("Signal Evolution Over Time")
        axes[0, 1].set_xlabel("Date")
        axes[0, 1].set_ylabel("Signal Value")

        # Plot 3: Signal Autocorrelation
        pd.plotting.autocorrelation_plot(self.predictions, ax=axes[1, 0])
        axes[1, 0].set_title("Signal Autocorrelation")

        # Plot 4: Signal vs Returns
        portfolio_returns = (self.returns * self.positions).sum(axis=1)
        signal_series = pd.Series(
            self.predictions, index=self.returns.index[-len(self.predictions) :]
        )
        aligned_returns = portfolio_returns[signal_series.index]
        axes[1, 1].scatter(signal_series, aligned_returns)
        axes[1, 1].set_title("Signal vs Next-Day Returns")
        axes[1, 1].set_xlabel("Signal Value")
        axes[1, 1].set_ylabel("Next-Day Return")

        plt.tight_layout()
        plt.savefig("outputs/plots/signal_analysis.png")
        plt.close()

    def save_performance_report(self) -> None:
        """Save performance metrics and analysis to a JSON file."""
        if self.portfolio_value is None:
            raise ValueError("Must run backtest first!")

        portfolio_returns = self.portfolio_value.pct_change()

        report = {
            "performance_metrics": self.compute_performance_metrics(portfolio_returns),
            "position_metrics": {
                "avg_gross_exposure": float(self.positions.abs().sum(axis=1).mean()),
                "avg_net_exposure": float(self.positions.sum(axis=1).mean()),
                "max_position_size": float(self.positions.abs().max().max()),
                "avg_position_count": float((self.positions != 0).sum(axis=1).mean()),
            },
            "signal_metrics": {
                "signal_mean": float(np.mean(self.predictions)),
                "signal_std": float(np.std(self.predictions)),
                "signal_skew": float(stats.skew(self.predictions)),
                "signal_kurtosis": float(stats.kurtosis(self.predictions)),
            },
        }

        import json

        with open("outputs/reports/backtest_report.json", "w") as f:
            json.dump(report, f, indent=4)


class StrategyBacktester:
    """Backtester for RL trading strategy across multiple periods."""

    def __init__(
        self,
        returns: pd.DataFrame,
        predictions: np.ndarray,
        rl_agent: RLTrader,
        initial_balance: float = 1e6,
        window_size: int = 50,
    ):
        self.returns = returns
        self.predictions = predictions
        self.rl_agent = rl_agent
        self.initial_balance = initial_balance
        self.window_size = window_size

    def run_backtest(
        self, start_date: datetime, end_date: datetime, epsilon: float = 0.0
    ) -> Tuple[pd.DataFrame, Dict]:
        """Run backtest for a specific period."""

        # Filter data for the period
        mask = (self.returns.index >= start_date) & (self.returns.index <= end_date)
        period_returns = self.returns[mask]
        period_predictions = self.predictions[mask.values]

        if len(period_returns) < self.window_size:
            raise ValueError("Period too short for the given window size")

        # Create environment for the period
        env = TradingEnvironment(
            returns=period_returns,
            predictions=period_predictions,
            initial_balance=self.initial_balance,
            window_size=self.window_size,
        )

        # Run simulation
        state = env.reset()
        done = False

        while not done:
            action = self.rl_agent.select_action(state, epsilon)
            state, reward, done, info = env.step(action)

        # Get trading signals and metrics
        results_df = env.get_trading_signals()

        # Calculate performance metrics
        metrics = self._calculate_metrics(results_df)

        return results_df, metrics

    def run_rolling_backtests(
        self,
        period_length: int = 252,  # Default 1 year
        stride: int = 126,  # Default 6 months
        epsilon: float = 0.0,
    ) -> List[Tuple[pd.DataFrame, Dict]]:
        """Run multiple backtests using rolling windows."""

        all_results = []
        start_idx = 0

        while start_idx + period_length <= len(self.returns):
            start_date = self.returns.index[start_idx]
            end_date = self.returns.index[start_idx + period_length - 1]

            logger.info(f"Running backtest for period {start_date} to {end_date}")

            try:
                results = self.run_backtest(start_date, end_date, epsilon)
                all_results.append((start_date, end_date, results))
            except Exception as e:
                logger.error(
                    f"Error in backtest for period {start_date} to {end_date}: {str(e)}"
                )

            start_idx += stride

        return all_results

    def _calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics for a backtest period."""

        if results_df.empty:
            return {}

        returns = pd.Series(results_df["returns"])

        metrics = {
            "total_return": float((1 + returns).prod() - 1),
            "annual_return": float((1 + returns).prod() ** (252 / len(returns)) - 1),
            "annual_volatility": float(returns.std() * np.sqrt(252)),
            "sharpe_ratio": float(returns.mean() / returns.std() * np.sqrt(252))
            if returns.std() > 0
            else 0,
            "max_drawdown": float(results_df["drawdown"].min()),
            "win_rate": float((returns > 0).mean()),
            "avg_win": float(returns[returns > 0].mean())
            if len(returns[returns > 0]) > 0
            else 0,
            "avg_loss": float(returns[returns < 0].mean())
            if len(returns[returns < 0]) > 0
            else 0,
        }

        return metrics

    def plot_backtest_results(
        self, results_df: pd.DataFrame, metrics: Dict, title: str = "Backtest Results"
    ):
        """Create visualization of backtest results."""

        plt.style.use("seaborn")
        fig = plt.figure(figsize=(15, 10))

        # Plot 1: Portfolio Value and Drawdown
        ax1 = plt.subplot(311)
        ax1.plot(
            results_df["timestamp"],
            results_df["portfolio_value"],
            label="Portfolio Value",
        )
        ax1.set_title(f"{title}\nPortfolio Value and Drawdown")
        ax1.set_ylabel("Portfolio Value")

        ax1_twin = ax1.twinx()
        ax1_twin.fill_between(
            results_df["timestamp"],
            results_df["drawdown"] * 100,
            0,
            alpha=0.3,
            color="red",
            label="Drawdown %",
        )
        ax1_twin.set_ylabel("Drawdown %")

        # Plot 2: Cumulative Returns
        ax2 = plt.subplot(312)
        ax2.plot(
            results_df["timestamp"],
            results_df["cumulative_returns"],
            label="Cumulative Returns",
        )
        ax2.set_title("Cumulative Returns")
        ax2.set_ylabel("Cumulative Returns")

        # Plot 3: Trading Signals and Actions
        ax3 = plt.subplot(313)
        ax3.plot(
            results_df["timestamp"], results_df["signal"], label="Signal", alpha=0.5
        )

        # Plot buy/sell markers
        buys = results_df[results_df["action"] == 2]
        sells = results_df[results_df["action"] == 0]
        holds = results_df[results_df["action"] == 1]

        ax3.scatter(
            buys["timestamp"], buys["signal"], color="green", marker="^", label="Buy"
        )
        ax3.scatter(
            sells["timestamp"], sells["signal"], color="red", marker="v", label="Sell"
        )
        ax3.scatter(
            holds["timestamp"],
            holds["signal"],
            color="gray",
            marker="o",
            alpha=0.3,
            label="Hold",
        )

        ax3.set_title("Trading Signals and Actions")
        ax3.set_ylabel("Signal Strength")

        # Add metrics as text
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2%}\n"
            f"Annual Return: {metrics['annual_return']:.2%}\n"
            f"Annual Vol: {metrics['annual_volatility']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Win Rate: {metrics['win_rate']:.2%}"
        )

        plt.figtext(
            0.02,
            0.02,
            metrics_text,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        return fig
