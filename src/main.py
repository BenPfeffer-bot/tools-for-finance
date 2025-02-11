# main.py
import numpy as np
import pandas as pd
import sys, os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import MultiDataLoader
from src.eigenportfolio_builder import EigenportfolioBuilder
from src.arbitrage_signal_detector import ArbitrageSignalDetector
from src.ml_model_trainer import MLModelTrainer
from src.backtester import Backtester
from src.config import TICKERS
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_eta_strategy(
    tickers: list = TICKERS,
    n_eigenportfolios: int = 5,
    window: int = 50,
    lag: int = 1,
    initial_capital: float = 100000,
    min_tickers: int = 30,  # Minimum number of tickers required
    batch_size: int = 10,  # Batch size for data loading
    lookback_years: int = 5,  # Number of years of historical data
    missing_threshold: float = 0.1,  # Maximum allowed missing data percentage
):
    """
    Run the Eigenportfolio Transfer Entropy Arbitrage (ETA) strategy.

    Args:
        tickers: List of stock tickers
        n_eigenportfolios: Number of eigenportfolios to construct
        window: Rolling window size for computations
        lag: Time lag for signal detection
        initial_capital: Initial capital for backtesting
        min_tickers: Minimum number of tickers required
        batch_size: Number of tickers to download in each batch
        lookback_years: Number of years of historical data
        missing_threshold: Maximum allowed missing data percentage
    """
    try:
        # Step 1: Load and prepare data
        logger.info(f"Loading market data for {len(tickers)} tickers...")

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365)
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

        data_loader = MultiDataLoader(tickers)
        df = data_loader.run(
            batch_size=batch_size,
            start_date=start_date,
            end_date=end_date,
            progress=True,
            auto_adjust=True,
        )

        if df is None or df.empty:
            logger.error("Failed to load any data")
            return None

        # Check if we have enough tickers
        unique_tickers = df["Ticker"].unique()
        if len(unique_tickers) < min_tickers:
            logger.error(
                f"Insufficient tickers: {len(unique_tickers)}/{min_tickers} required"
            )
            return None

        logger.info(f"Successfully loaded data for {len(unique_tickers)} tickers")

        # Prepare returns data
        logger.info("Preparing returns data...")
        df_pivot = df.pivot(index="Date", columns="Ticker", values="Close")

        # Analyze missing data
        missing_pct = df_pivot.isnull().mean()
        logger.info("\nMissing data analysis:")
        logger.info(f"Average missing: {missing_pct.mean():.1%}")
        logger.info(
            f"Worst ticker: {missing_pct.idxmax()} ({missing_pct.max():.1%} missing)"
        )
        logger.info(f"Number of tickers with >10% missing: {(missing_pct > 0.1).sum()}")

        # Remove columns (tickers) with too many missing values
        valid_tickers = missing_pct[missing_pct < missing_threshold].index
        if len(valid_tickers) < min_tickers:
            logger.error(
                f"Insufficient tickers after removing those with too many missing values: {len(valid_tickers)}/{min_tickers}"
            )
            return None

        df_pivot = df_pivot[valid_tickers]
        logger.info(f"Retained {len(valid_tickers)} tickers after filtering")

        # Forward fill missing values (at most 5 days)
        df_pivot = df_pivot.ffill(limit=5)

        # Remove any remaining rows with missing values
        initial_rows = len(df_pivot)
        df_pivot = df_pivot.dropna()
        rows_removed = initial_rows - len(df_pivot)
        logger.info(
            f"Removed {rows_removed} rows with missing values ({rows_removed / initial_rows:.1%})"
        )

        # Compute returns
        returns = df_pivot.pct_change().dropna()
        logger.info(f"Final returns matrix shape: {returns.shape}")

        # Verify minimum data requirements
        min_samples = window * 2
        if len(returns) < min_samples:
            logger.error(
                f"Insufficient samples: {len(returns)} < {min_samples} required"
            )
            return None

        if returns.empty:
            logger.error("No valid returns data available")
            return None

        # Step 2: Construct eigenportfolios
        logger.info("\nComputing eigenportfolios...")
        n_components = min(
            n_eigenportfolios, len(valid_tickers) - 1
        )  # Ensure we don't exceed n-1 components
        eigen_builder = EigenportfolioBuilder(
            returns=returns, n_components=n_components
        )
        eigenportfolios, explained_variance = eigen_builder.compute_eigenportfolios()

        # Project returns onto eigenportfolios
        factor_returns = eigen_builder.project_returns_to_factors()

        # Log eigenportfolio analysis
        logger.info("\nEigenportfolio Analysis:")
        cumulative_var = np.cumsum(explained_variance)
        for i, (var_ratio, cum_var) in enumerate(
            zip(explained_variance, cumulative_var)
        ):
            logger.info(
                f"PC{i + 1}: {var_ratio:.1%} explained ({cum_var:.1%} cumulative)"
            )

        # Verify we have enough data for ML
        if len(factor_returns) < window * 2:
            logger.error(f"Insufficient time series data: {len(factor_returns)} points")
            return None

        # Step 3: Detect arbitrage signals
        logger.info("\nDetecting arbitrage signals...")
        signal_detector = ArbitrageSignalDetector(
            returns=factor_returns, window=window, lag=lag
        )
        features, labels, dates = signal_detector.detect_opportunities()

        if len(features) < 100:  # Minimum samples for ML
            logger.error(f"Insufficient samples for ML: {len(features)}")
            return None

        # Step 4: Train ML model
        logger.info("\nTraining ML model...")
        model_trainer = MLModelTrainer(n_estimators=100, learning_rate=0.1, max_depth=5)

        # Train model and perform cross-validation
        model = model_trainer.train_model(features, labels)
        cv_results = model_trainer.cross_validate(features, labels)

        # Plot model diagnostics
        logger.info("Plotting model diagnostics...")
        model_trainer.plot_feature_importance()
        model_trainer.plot_learning_curves()

        # Step 5: Backtest strategy
        logger.info("\nRunning backtest...")
        backtester = Backtester(
            returns=factor_returns,
            model=model,
            features=features,
            dates=dates,
            capital=initial_capital,
        )

        # Run backtest and get results
        portfolio_value, metrics = backtester.run_backtest()

        # Plot detailed analysis
        backtester.plot_position_analysis()

        # Print performance metrics
        logger.info("\nStrategy Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        return {
            "eigenportfolios": eigenportfolios,
            "explained_variance": explained_variance,
            "model": model,
            "cv_results": cv_results,
            "portfolio_value": portfolio_value,
            "metrics": metrics,
            "valid_tickers": list(valid_tickers),
        }

    except Exception as e:
        logger.error(f"Error running ETA strategy: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the complete ETA strategy pipeline
    results = run_eta_strategy(
        n_eigenportfolios=5,
        window=50,
        lag=1,
        initial_capital=100000,
        min_tickers=30,
        batch_size=10,
        lookback_years=5,
        missing_threshold=0.1,
    )

    if results:
        logger.info("\nStrategy execution completed successfully!")
        logger.info(f"Number of valid tickers used: {len(results['valid_tickers'])}")
        logger.info(
            f"Explained variance ratios: {[f'{v:.1%}' for v in results['explained_variance']]}"
        )
        logger.info(f"Sharpe ratio: {results['metrics']['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {results['metrics']['max_drawdown']:.1%}")
