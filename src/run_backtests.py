import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reinforcement.rl_models import RLTrader, TradingEnvironment
from src.backtester import StrategyBacktester

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # Create output directories if they don't exist
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    # Load the trained agent and data
    logger.info("Loading data and trained agent...")

    try:
        returns = pd.read_csv(
            "data/processed/returns.csv", index_col=0, parse_dates=True
        )
        predictions = np.load("outputs/predictions/combined_predictions.npy")
    except FileNotFoundError as e:
        logger.error(f"Required data files not found: {str(e)}")
        logger.info("Please ensure you have run the training script first")
        return

    # Load the trained agent
    try:
        env = TradingEnvironment(
            returns=returns, predictions=predictions, window_size=50
        )

        agent = RLTrader(
            env=env, state_dim=env._get_state().shape[0], action_dim=3, hidden_dim=128
        )
        agent.load_model("outputs/models/rl_model.pth")
    except Exception as e:
        logger.error(f"Error loading the trained agent: {str(e)}")
        return

    # Create backtester
    backtester = StrategyBacktester(
        returns=returns, predictions=predictions, rl_agent=agent, window_size=50
    )

    # Run full period backtest
    logger.info("Running full period backtest...")
    try:
        full_results_df, full_metrics = backtester.run_backtest(
            start_date=returns.index[0], end_date=returns.index[-1]
        )

        # Plot and save full period results
        fig = backtester.plot_backtest_results(
            full_results_df, full_metrics, "Full Period Backtest Results"
        )
        plt.savefig("outputs/plots/full_period_backtest.png")
        plt.close()

        # Save full period metrics
        pd.DataFrame([full_metrics]).to_csv(
            "outputs/reports/full_period_metrics.csv", index=False
        )
    except Exception as e:
        logger.error(f"Error in full period backtest: {str(e)}")
        return

    # Run rolling backtests
    logger.info("Running rolling backtests...")
    periods = [
        (252, 126),  # 1 year periods, 6 month stride
        (504, 252),  # 2 year periods, 1 year stride
        (756, 378),  # 3 year periods, 18 month stride
    ]

    for period_length, stride in periods:
        logger.info(f"Running rolling backtests with {period_length} day periods...")
        try:
            results = backtester.run_rolling_backtests(
                period_length=period_length, stride=stride
            )

            # Create summary of rolling backtest results
            summary = []
            for start_date, end_date, (results_df, metrics) in results:
                summary.append(
                    {"start_date": start_date, "end_date": end_date, **metrics}
                )

                # Plot and save individual period results
                fig = backtester.plot_backtest_results(
                    results_df,
                    metrics,
                    f"Backtest Results ({start_date.date()} to {end_date.date()})",
                )
                plt.savefig(
                    f"outputs/plots/backtest_{start_date.date()}_{end_date.date()}.png"
                )
                plt.close()

            # Save summary to CSV
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(
                f"outputs/reports/rolling_backtest_summary_{period_length}d.csv",
                index=False,
            )

            # Plot rolling metrics
            plt.figure(figsize=(15, 10))

            plt.subplot(311)
            plt.plot(summary_df["start_date"], summary_df["annual_return"])
            plt.title(f"Rolling {period_length}-day Annual Returns")
            plt.ylabel("Annual Return")

            plt.subplot(312)
            plt.plot(summary_df["start_date"], summary_df["sharpe_ratio"])
            plt.title(f"Rolling {period_length}-day Sharpe Ratios")
            plt.ylabel("Sharpe Ratio")

            plt.subplot(313)
            plt.plot(summary_df["start_date"], summary_df["max_drawdown"])
            plt.title(f"Rolling {period_length}-day Maximum Drawdowns")
            plt.ylabel("Max Drawdown")

            plt.tight_layout()
            plt.savefig(f"outputs/plots/rolling_metrics_{period_length}d.png")
            plt.close()

        except Exception as e:
            logger.error(
                f"Error in rolling backtest for period length {period_length}: {str(e)}"
            )
            continue

    logger.info("Backtesting completed successfully")


if __name__ == "__main__":
    main()
