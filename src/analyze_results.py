import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy import stats
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_returns_distribution(
    returns, save_path="outputs/plots/returns_distribution.png"
):
    """Plot the distribution of returns to check for normality and outliers."""
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    sns.histplot(returns, kde=True)
    plt.title("Returns Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")

    plt.subplot(122)
    stats.probplot(returns, dist="norm", plot=plt)
    plt.title("Q-Q Plot")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_rolling_metrics(
    returns, window=30, save_path="outputs/plots/rolling_metrics.png"
):
    """Plot rolling Sharpe ratio and volatility."""
    rolling_sharpe = (
        returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
    )
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)

    plt.figure(figsize=(12, 8))

    plt.subplot(211)
    rolling_sharpe.plot()
    plt.title(f"Rolling {window}-day Sharpe Ratio")
    plt.ylabel("Sharpe Ratio")

    plt.subplot(212)
    rolling_vol.plot()
    plt.title(f"Rolling {window}-day Volatility")
    plt.ylabel("Volatility")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_drawdown_analysis(returns, save_path="outputs/plots/drawdown_analysis.png"):
    """Plot drawdown analysis."""
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1

    plt.figure(figsize=(12, 6))
    drawdowns.plot()
    plt.title("Drawdown Analysis")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_trading_signals(
    positions, returns, predictions, save_path="outputs/plots/trading_signals.png"
):
    """Plot trading signals and actual returns."""
    plt.figure(figsize=(15, 10))

    plt.subplot(311)
    plt.plot(positions, label="Position Size")
    plt.title("Position Sizes Over Time")
    plt.ylabel("Position Size")
    plt.legend()

    plt.subplot(312)
    plt.plot(returns.cumsum(), label="Cumulative Returns")
    plt.title("Cumulative Returns")
    plt.ylabel("Returns")
    plt.legend()

    plt.subplot(313)
    plt.plot(predictions, label="Model Predictions")
    plt.title("Model Predictions")
    plt.ylabel("Prediction")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def analyze_strategy_robustness():
    """Analyze the robustness of the trading strategy."""
    try:
        # Load trading history
        with open("outputs/trading_history.json", "r") as f:
            history = json.load(f)

        # Convert to pandas DataFrame
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Calculate returns
        returns = pd.Series(df["returns"].values, index=df.index)

        # Generate plots
        plot_returns_distribution(returns)
        plot_rolling_metrics(returns)
        plot_drawdown_analysis(returns)
        plot_trading_signals(
            positions=df["positions"].values,
            returns=returns,
            predictions=df["signals"].values,
        )

        # Calculate additional metrics
        daily_returns = returns.resample("D").sum()
        monthly_returns = returns.resample("M").sum()

        # Calculate hit ratio
        profitable_trades = (returns > 0).sum()
        total_trades = len(returns)
        hit_ratio = profitable_trades / total_trades if total_trades > 0 else 0

        # Calculate additional risk metrics
        var_95 = np.percentile(returns, 5)  # 95% VaR
        cvar_95 = returns[returns <= var_95].mean()  # 95% CVaR

        # Print results
        logger.info("\nDetailed Strategy Analysis:")
        logger.info(f"Hit Ratio: {hit_ratio:.2%}")
        logger.info(f"95% VaR: {var_95:.2%}")
        logger.info(f"95% CVaR: {cvar_95:.2%}")
        logger.info("\nMonthly Returns Statistics:")
        logger.info(monthly_returns.describe())

        # Check for serial correlation
        autocorr = pd.Series(returns).autocorr()
        logger.info(f"\nReturn Autocorrelation: {autocorr:.3f}")

        return True

    except Exception as e:
        logger.error(f"Error in strategy analysis: {str(e)}")
        return False


if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    # Run analysis
    success = analyze_strategy_robustness()
    if success:
        logger.info("Strategy analysis completed successfully")
    else:
        logger.error("Strategy analysis failed")
