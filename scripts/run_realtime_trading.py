"""
Script to run real-time trading strategy.

This script initializes and runs the real-time trading strategy
with proper configuration and error handling.
"""

import os
import sys
import logging
import json
from datetime import datetime
import signal
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import TICKERS, MARKET_HOURS
from src.config.paths import *
from src import RealTimeStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            PROJECT_ROOT / "logs" / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class TradingSession:
    """Manages the trading session and handles graceful shutdown."""

    def __init__(self):
        self.strategy = None
        self.is_running = False

    def initialize(self):
        """Initialize trading strategy and resources."""
        try:
            # Load configuration
            config = self._load_config()

            # Initialize strategy
            self.strategy = RealTimeStrategy(
                tickers=TICKERS,
                initial_capital=config["initial_capital"],
                position_size=config["position_size"],
                stop_loss=config["stop_loss"],
                take_profit=config["take_profit"],
            )

            # Register signal handlers
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)

            logger.info("Trading session initialized")

        except Exception as e:
            logger.error(f"Error initializing trading session: {str(e)}")
            raise

    def _load_config(self) -> dict:
        """
        Load trading configuration.

        Returns:
            Dictionary of configuration parameters
        """
        try:
            config_path = PROJECT_ROOT / "config" / "trading_config.json"

            if not config_path.exists():
                # Create default config if not exists
                default_config = {
                    "initial_capital": 1000000.0,
                    "position_size": 0.1,
                    "stop_loss": 0.02,
                    "take_profit": 0.05,
                    "risk_free_rate": 0.02,
                    "market_impact": 0.001,
                    "transaction_cost": 0.001,
                }

                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, "w") as f:
                    json.dump(default_config, f, indent=4)

                return default_config

            with open(config_path, "r") as f:
                config = json.load(f)

            logger.info("Configuration loaded successfully")
            return config

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def start(self):
        """Start the trading session."""
        try:
            if self.is_running:
                logger.warning("Trading session already running")
                return

            logger.info("Starting trading session")
            self.is_running = True

            # Start strategy
            self.strategy.start()

            # Keep session alive
            while self.is_running:
                signal.pause()

        except Exception as e:
            logger.error(f"Error in trading session: {str(e)}")
            self._handle_shutdown()

    def _handle_shutdown(self, signum=None, frame=None):
        """Handle graceful shutdown of trading session."""
        try:
            if not self.is_running:
                return

            logger.info("Initiating graceful shutdown")
            self.is_running = False

            if self.strategy:
                # Stop strategy and save results
                self.strategy.stop()

                # Generate and save performance report
                self._save_performance_report()

            logger.info("Trading session shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
        finally:
            sys.exit(0)

    def _save_performance_report(self):
        """Generate and save performance report."""
        try:
            # Load trading history
            with open(TRADES_DIR / "trading_history.json", "r") as f:
                history = json.load(f)

            # Convert to DataFrame
            trades_df = pd.DataFrame(history["trades"])
            portfolio_df = pd.DataFrame(history["portfolio_history"])

            # Calculate performance metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df["realized_pnl"] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            avg_win = trades_df[trades_df["realized_pnl"] > 0]["realized_pnl"].mean()
            avg_loss = trades_df[trades_df["realized_pnl"] < 0]["realized_pnl"].mean()
            profit_factor = abs(avg_win / avg_loss) if avg_loss else float("inf")

            # Create report
            report = {
                "session_end": datetime.now().isoformat(),
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "final_equity": portfolio_df["equity"].iloc[-1],
                "max_drawdown": portfolio_df["drawdown"].max(),
                "sharpe_ratio": self._calculate_sharpe_ratio(portfolio_df),
                "trade_statistics": {
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "largest_win": trades_df["realized_pnl"].max(),
                    "largest_loss": trades_df["realized_pnl"].min(),
                },
            }

            # Save report
            report_path = (
                REPORTS_DIR
                / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4, default=str)

            logger.info(f"Performance report saved to {report_path}")

        except Exception as e:
            logger.error(f"Error saving performance report: {str(e)}")

    def _calculate_sharpe_ratio(self, portfolio_df: pd.DataFrame) -> float:
        """
        Calculate Sharpe ratio from portfolio history.

        Args:
            portfolio_df: DataFrame of portfolio history

        Returns:
            Sharpe ratio
        """
        try:
            # Calculate returns
            portfolio_df["returns"] = portfolio_df["equity"].pct_change()

            # Annualize metrics
            annual_return = portfolio_df["returns"].mean() * 252
            annual_vol = portfolio_df["returns"].std() * np.sqrt(252)
            risk_free_rate = 0.02  # Assumed annual risk-free rate

            # Calculate Sharpe ratio
            sharpe_ratio = (annual_return - risk_free_rate) / annual_vol
            return sharpe_ratio

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0


def main():
    """Main entry point for running trading session."""
    try:
        # Create and start trading session
        session = TradingSession()
        session.initialize()
        session.start()

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
