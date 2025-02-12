"""
Run paper trading simulation and analyze results.

This script runs the paper trading simulation and generates
comprehensive performance analysis reports.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import signal
from typing import Dict, List

from paper_trading_simulation import PaperTradingSimulation
from config import TICKERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/simulation_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class SimulationManager:
    """Manages paper trading simulation and results analysis."""

    def __init__(self, tickers: List[str]):
        """Initialize simulation manager."""
        self.tickers = tickers
        self.simulation = None
        self.results_dir = Path("outputs")

        # Create directories
        self._setup_directories()

        logger.info("Simulation manager initialized")

    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            "logs",
            "outputs/reports/daily",
            "outputs/reports/weekly",
            "outputs/reports/monthly",
            "outputs/plots/daily",
            "outputs/plots/weekly",
            "outputs/plots/monthly",
            "outputs/signals",
            "outputs/performance",
            "outputs/analysis",
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def start_simulation(self):
        """Start paper trading simulation."""
        try:
            # Initialize simulation
            self.simulation = PaperTradingSimulation(
                tickers=self.tickers,
                initial_capital=1000000.0,
                polling_interval=5,
                window_size=50,
            )

            # Register signal handlers
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)

            # Start simulation
            self.simulation.start()

            logger.info("Simulation started")

            # Keep running until interrupted
            signal.pause()

        except Exception as e:
            logger.error(f"Error starting simulation: {str(e)}")
            self._handle_shutdown()

    def _handle_shutdown(self, signum=None, frame=None):
        """Handle graceful shutdown."""
        try:
            if self.simulation:
                logger.info("Stopping simulation...")
                self.simulation.stop()

                # Analyze results
                self.analyze_results()

            logger.info("Simulation stopped")
            sys.exit(0)

        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            sys.exit(1)

    def analyze_results(self):
        """Analyze simulation results."""
        try:
            # Load all reports
            daily_reports = self._load_reports("daily")
            weekly_reports = self._load_reports("weekly")
            monthly_reports = self._load_reports("monthly")

            # Generate comprehensive analysis
            self._analyze_performance(daily_reports, weekly_reports, monthly_reports)
            self._analyze_trading_patterns()
            self._analyze_signals()

            logger.info("Results analysis completed")

        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")

    def _load_reports(self, period: str) -> List[Dict]:
        """Load reports for given period."""
        reports = []
        report_dir = self.results_dir / "reports" / period

        for file in report_dir.glob("*.json"):
            try:
                with open(file, "r") as f:
                    report = json.load(f)
                    reports.append(report)
            except Exception as e:
                logger.error(f"Error loading report {file}: {str(e)}")

        return sorted(reports, key=lambda x: x["date"])

    def _analyze_performance(
        self,
        daily_reports: List[Dict],
        weekly_reports: List[Dict],
        monthly_reports: List[Dict],
    ):
        """Analyze performance metrics."""
        try:
            # Create performance summary
            summary = {
                "total_return": self._calculate_total_return(daily_reports),
                "annualized_return": self._calculate_annualized_return(monthly_reports),
                "sharpe_ratio": self._calculate_sharpe_ratio(daily_reports),
                "max_drawdown": self._calculate_max_drawdown(daily_reports),
                "win_rate": np.mean([r["win_rate"] for r in monthly_reports]),
                "avg_trade_duration": np.mean(
                    [r["avg_trade_duration"] for r in monthly_reports]
                ),
            }

            # Save summary
            with open("outputs/analysis/performance_summary.json", "w") as f:
                json.dump(summary, f, indent=4)

            # Generate performance plots
            self._plot_performance_metrics(
                daily_reports, weekly_reports, monthly_reports
            )

        except Exception as e:
            logger.error(f"Error analyzing performance: {str(e)}")

    def _calculate_total_return(self, daily_reports: List[Dict]) -> float:
        """Calculate total return."""
        returns = [r["daily_return"] for r in daily_reports]
        return float(np.prod(1 + np.array(returns)) - 1)

    def _calculate_annualized_return(self, monthly_reports: List[Dict]) -> float:
        """Calculate annualized return."""
        monthly_returns = [r["monthly_return"] for r in monthly_reports]
        total_return = np.prod(1 + np.array(monthly_returns)) - 1
        years = len(monthly_reports) / 12
        return float((1 + total_return) ** (1 / years) - 1)

    def _calculate_sharpe_ratio(self, daily_reports: List[Dict]) -> float:
        """Calculate Sharpe ratio."""
        returns = np.array([r["daily_return"] for r in daily_reports])
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    def _calculate_max_drawdown(self, daily_reports: List[Dict]) -> float:
        """Calculate maximum drawdown."""
        portfolio_values = [r["portfolio_value"] for r in daily_reports]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return float(np.min(drawdown))

    def _plot_performance_metrics(
        self,
        daily_reports: List[Dict],
        weekly_reports: List[Dict],
        monthly_reports: List[Dict],
    ):
        """Generate performance metrics plots."""
        try:
            # Create figure
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))

            # Plot 1: Cumulative returns
            daily_values = [r["portfolio_value"] for r in daily_reports]
            dates = [r["date"] for r in daily_reports]
            cumulative_returns = np.array(daily_values) / daily_values[0] - 1

            axes[0, 0].plot(dates, cumulative_returns)
            axes[0, 0].set_title("Cumulative Returns")
            axes[0, 0].set_xlabel("Date")
            axes[0, 0].set_ylabel("Return")

            # Plot 2: Rolling Sharpe ratio
            daily_returns = [r["daily_return"] for r in daily_reports]
            rolling_sharpe = (
                pd.Series(daily_returns)
                .rolling(60)
                .apply(lambda x: np.mean(x) / np.std(x) * np.sqrt(252))
            )

            axes[0, 1].plot(dates, rolling_sharpe)
            axes[0, 1].set_title("Rolling Sharpe Ratio (60-day)")
            axes[0, 1].set_xlabel("Date")
            axes[0, 1].set_ylabel("Sharpe Ratio")

            # Plot 3: Monthly returns distribution
            monthly_returns = [r["monthly_return"] for r in monthly_reports]
            sns.histplot(monthly_returns, kde=True, ax=axes[1, 0])
            axes[1, 0].set_title("Monthly Returns Distribution")
            axes[1, 0].set_xlabel("Return")

            # Plot 4: Drawdown
            drawdown = self._calculate_drawdown_series(daily_values)
            axes[1, 1].plot(dates, drawdown)
            axes[1, 1].set_title("Portfolio Drawdown")
            axes[1, 1].set_xlabel("Date")
            axes[1, 1].set_ylabel("Drawdown")

            # Plot 5: Rolling volatility
            rolling_vol = pd.Series(daily_returns).rolling(21).std() * np.sqrt(252)

            axes[2, 0].plot(dates, rolling_vol)
            axes[2, 0].set_title("Rolling Volatility (21-day)")
            axes[2, 0].set_xlabel("Date")
            axes[2, 0].set_ylabel("Annualized Volatility")

            # Plot 6: Win rate over time
            win_rates = [r["win_rate"] for r in weekly_reports]
            weekly_dates = [r["date"] for r in weekly_reports]

            axes[2, 1].plot(weekly_dates, win_rates)
            axes[2, 1].set_title("Weekly Win Rate")
            axes[2, 1].set_xlabel("Date")
            axes[2, 1].set_ylabel("Win Rate")

            plt.tight_layout()
            plt.savefig("outputs/analysis/performance_metrics.png")
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting performance metrics: {str(e)}")

    def _calculate_drawdown_series(self, values: List[float]) -> np.ndarray:
        """Calculate drawdown series."""
        peak = np.maximum.accumulate(values)
        return (values - peak) / peak

    def _analyze_trading_patterns(self):
        """Analyze trading patterns and behavior."""
        try:
            # Load trading history
            with open("trading_history.json", "r") as f:
                history = json.load(f)

            trades = pd.DataFrame(history["trades"])

            # Calculate trade metrics
            trade_metrics = {
                "total_trades": len(trades),
                "avg_trade_return": float(trades["realized_pnl"].mean()),
                "best_trade": float(trades["realized_pnl"].max()),
                "worst_trade": float(trades["realized_pnl"].min()),
                "avg_holding_period": float(
                    pd.to_timedelta(trades["duration"]).mean().total_seconds() / 3600
                ),
                "trade_frequency": len(trades) / len(pd.unique(trades["date"])),
            }

            # Save metrics
            with open("outputs/analysis/trading_patterns.json", "w") as f:
                json.dump(trade_metrics, f, indent=4)

            # Generate trade analysis plots
            self._plot_trade_analysis(trades)

        except Exception as e:
            logger.error(f"Error analyzing trading patterns: {str(e)}")

    def _plot_trade_analysis(self, trades: pd.DataFrame):
        """Generate trade analysis plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Plot 1: Trade P&L distribution
            sns.histplot(trades["realized_pnl"], kde=True, ax=axes[0, 0])
            axes[0, 0].set_title("Trade P&L Distribution")
            axes[0, 0].set_xlabel("P&L")

            # Plot 2: Trade duration vs return
            axes[0, 1].scatter(
                pd.to_timedelta(trades["duration"]).dt.total_seconds() / 3600,
                trades["realized_pnl"],
            )
            axes[0, 1].set_title("Trade Duration vs P&L")
            axes[0, 1].set_xlabel("Duration (hours)")
            axes[0, 1].set_ylabel("P&L")

            # Plot 3: Trade frequency over time
            daily_trades = trades.groupby("date").size()
            axes[1, 0].plot(daily_trades.index, daily_trades.values)
            axes[1, 0].set_title("Daily Trade Frequency")
            axes[1, 0].set_xlabel("Date")
            axes[1, 0].set_ylabel("Number of Trades")

            # Plot 4: Cumulative P&L
            cumulative_pnl = trades["realized_pnl"].cumsum()
            axes[1, 1].plot(trades.index, cumulative_pnl)
            axes[1, 1].set_title("Cumulative P&L")
            axes[1, 1].set_xlabel("Trade Number")
            axes[1, 1].set_ylabel("Cumulative P&L")

            plt.tight_layout()
            plt.savefig("outputs/analysis/trade_analysis.png")
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting trade analysis: {str(e)}")

    def _analyze_signals(self):
        """Analyze trading signals and their effectiveness."""
        try:
            # Load signal history
            signals_df = pd.read_csv("outputs/signals/signal_history.csv")

            # Calculate signal metrics
            signal_metrics = {
                "signal_accuracy": float(
                    np.mean(signals_df["realized_return"] * signals_df["signal"] > 0)
                ),
                "avg_signal_strength": float(np.mean(np.abs(signals_df["signal"]))),
                "signal_correlation": float(
                    np.corrcoef(signals_df["signal"], signals_df["realized_return"])[
                        0, 1
                    ]
                ),
            }

            # Save metrics
            with open("outputs/analysis/signal_metrics.json", "w") as f:
                json.dump(signal_metrics, f, indent=4)

            # Generate signal analysis plots
            self._plot_signal_analysis(signals_df)

        except Exception as e:
            logger.error(f"Error analyzing signals: {str(e)}")

    def _plot_signal_analysis(self, signals_df: pd.DataFrame):
        """Generate signal analysis plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Plot 1: Signal strength distribution
            sns.histplot(signals_df["signal"], kde=True, ax=axes[0, 0])
            axes[0, 0].set_title("Signal Strength Distribution")
            axes[0, 0].set_xlabel("Signal Strength")

            # Plot 2: Signal vs realized return
            axes[0, 1].scatter(signals_df["signal"], signals_df["realized_return"])
            axes[0, 1].set_title("Signal vs Realized Return")
            axes[0, 1].set_xlabel("Signal Strength")
            axes[0, 1].set_ylabel("Realized Return")

            # Plot 3: Signal accuracy over time
            rolling_accuracy = (
                (signals_df["realized_return"] * signals_df["signal"] > 0)
                .rolling(50)
                .mean()
            )
            axes[1, 0].plot(signals_df.index, rolling_accuracy)
            axes[1, 0].set_title("Rolling Signal Accuracy (50-period)")
            axes[1, 0].set_xlabel("Signal Number")
            axes[1, 0].set_ylabel("Accuracy")

            # Plot 4: Signal strength over time
            rolling_strength = signals_df["signal"].abs().rolling(50).mean()
            axes[1, 1].plot(signals_df.index, rolling_strength)
            axes[1, 1].set_title("Rolling Signal Strength (50-period)")
            axes[1, 1].set_xlabel("Signal Number")
            axes[1, 1].set_ylabel("Average Signal Strength")

            plt.tight_layout()
            plt.savefig("outputs/analysis/signal_analysis.png")
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting signal analysis: {str(e)}")


def main():
    """Main function to run paper trading simulation."""
    try:
        # Initialize simulation manager
        manager = SimulationManager(tickers=TICKERS[:5])  # Start with subset of tickers

        # Start simulation
        manager.start_simulation()

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
