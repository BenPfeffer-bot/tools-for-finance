# main.py
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict
import logging

from src.data_loader import DataLoader
from src.eigenportfolio_analyzer import EigenportfolioAnalyzer
from src.arbitrage_signal_detector import ArbitrageSignalDetector
from src.ml_model_trainer import MLModelTrainer
from src.backtester import Backtester
from src.config import TICKERS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_eta_strategy(
    returns: pd.DataFrame,
    model_trainer: MLModelTrainer,
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Run the ETA strategy with the trained model.

    Args:
        returns: DataFrame of asset returns
        model_trainer: Trained MLModelTrainer instance
        features: Feature array
        labels: Label array
        test_size: Size of test set

    Returns:
        tuple: (portfolio_value, performance_metrics)
    """
    try:
        # Split data into train/test
        split_idx = int(len(features) * (1 - test_size))
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]

        # Train models on training data
        model_trainer.train_models(X_train, y_train)

        # Generate predictions on test set
        predictions = model_trainer.predict_ensemble(X_test)

        # Initialize backtester with test set data
        test_returns = returns.iloc[split_idx:]
        backtester = Backtester(returns=test_returns, predictions=predictions)

        # Run backtest
        portfolio_value, metrics = backtester.run_backtest()

        # Generate and save all plots
        backtester.plot_backtest_results()
        backtester.plot_position_analysis()
        backtester.plot_signal_analysis()

        # Save performance report
        backtester.save_performance_report()

        # Log results
        logger.info("\nBacktest Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.2%}")

        logger.info(
            "\nAll analysis completed. Check the outputs directory for results."
        )

        return portfolio_value, metrics

    except Exception as e:
        logger.error(f"Error running ETA strategy: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Load and prepare data
        data_loader = DataLoader(TICKERS)
        market_data = data_loader.load_data()
        logger.info(
            f"Successfully loaded data for {len(data_loader.valid_tickers)} tickers"
        )

        # Prepare returns data
        logger.info("Preparing returns data...")
        returns = data_loader.prepare_returns_matrix()

        # Analyze missing data
        logger.info("\nMissing data analysis:")
        data_loader.analyze_missing_data()

        # Compute eigenportfolios
        logger.info("\nComputing eigenportfolios...")
        eigen_analyzer = EigenportfolioAnalyzer(returns)
        eigenportfolios = eigen_analyzer.compute_eigenportfolios(n_components=5)
        eigen_analyzer.analyze_variance_explained()

        # Detect arbitrage signals
        logger.info("\nDetecting arbitrage signals...")
        signal_detector = ArbitrageSignalDetector(returns, eigenportfolios)
        features, labels = signal_detector.generate_features_and_labels()

        # Train ML model
        logger.info("\nTraining ML model...")
        model_trainer = MLModelTrainer(n_estimators=100, learning_rate=0.1, max_depth=5)

        # Run strategy
        results = run_eta_strategy(
            returns=returns,
            model_trainer=model_trainer,
            features=features,
            labels=labels,
            test_size=0.2,
        )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
