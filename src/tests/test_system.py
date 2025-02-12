"""
System test script for forex trading implementation.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import torch

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.market_data.tiingo_client import TiingoForexClient
from src.database.forex_data_loader import ForexDataLoader
from src.reinforcement.forex_trading_env import ForexTradingEnvironment
from src.config.settingss.trading_config import *

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"{LOG_DIR}/system_test_{datetime.now().strftime('%Y%m%d')}.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def test_api_connection():
    """Test Tiingo API connection."""
    try:
        pairs = FOREX_PAIRS[:2]  # Test with first two pairs
        client = TiingoForexClient(pairs=pairs)
        client.start()
        logger.info("API connection test successful")
        client.stop()
        return True
    except Exception as e:
        logger.error(f"API connection test failed: {str(e)}")
        return False


def test_data_loading():
    """Test historical data loading."""
    try:
        loader = ForexDataLoader(pairs=FOREX_PAIRS)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        data = loader.fetch_historical_data(start_date, end_date)
        if data.empty:
            logger.error("No historical data loaded")
            return False

        logger.info(f"Successfully loaded {len(data)} data points")
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"Sample data:\n{data.head()}")
        return True
    except Exception as e:
        logger.error(f"Data loading test failed: {str(e)}")
        return False


def test_environment():
    """Test trading environment."""
    try:
        # Load data
        loader = ForexDataLoader(pairs=FOREX_PAIRS)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        data = loader.fetch_historical_data(start_date, end_date)

        if data.empty:
            logger.error("No data available for trading environment test")
            return False

        # Initialize environment
        env = ForexTradingEnvironment(
            data=data,
            pairs=FOREX_PAIRS,
            initial_balance=INITIAL_BALANCE,
            max_position_size=MAX_POSITION_SIZE,
            risk_per_trade=RISK_PER_TRADE,
            window_size=WINDOW_SIZE,
        )

        # Test reset
        try:
            state = env.reset()
            logger.info(f"Environment reset successful. State shape: {state.shape}")
        except Exception as e:
            logger.error(f"Environment reset failed: {str(e)}")
            return False

        # Test step with random actions
        try:
            for i in range(5):
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)

                logger.info(f"\nStep {i + 1}:")
                logger.info(f"Action: {action}")
                logger.info(f"State shape: {next_state.shape}")
                logger.info(f"Reward: {reward:.4f}")
                logger.info(f"Done: {done}")
                logger.info(f"Balance: ${info['balance']:.2f}")
                logger.info(f"Equity: ${info['equity']:.2f}")
                logger.info(f"Positions: {info['positions']}")

                if done:
                    break
        except Exception as e:
            logger.error(f"Environment step failed: {str(e)}")
            return False

        return True

    except Exception as e:
        logger.error(f"Environment test failed: {str(e)}")
        return False


def test_model_training():
    """Test model training."""
    try:
        # Prepare data
        loader = ForexDataLoader(pairs=FOREX_PAIRS)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        data = loader.fetch_historical_data(start_date, end_date)

        if data.empty:
            logger.error("No data available for model training test")
            return False

        # Create environment
        try:
            env = ForexTradingEnvironment(
                data=data,
                pairs=FOREX_PAIRS,
                initial_balance=INITIAL_BALANCE,
                max_position_size=MAX_POSITION_SIZE,
                risk_per_trade=RISK_PER_TRADE,
                window_size=WINDOW_SIZE,
            )
        except Exception as e:
            logger.error(f"Failed to create environment: {str(e)}")
            return False

        # Test environment interactions
        try:
            state = env.reset()
            logger.info(f"Initial state shape: {state.shape}")

            for i in range(5):
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)

                logger.info(f"\nTraining Step {i + 1}:")
                logger.info(f"Action: {action}")
                logger.info(f"State shape: {next_state.shape}")
                logger.info(f"Reward: {reward:.4f}")
                logger.info(f"Done: {done}")
                logger.info(f"Balance: ${info['balance']:.2f}")

                if done:
                    break

            logger.info("Model training test successful")
            return True

        except Exception as e:
            logger.error(f"Failed during training steps: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"Model training test failed: {str(e)}")
        return False


def main():
    """Run all system tests."""
    try:
        logger.info("Starting system tests...")

        # Create necessary directories
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

        # Run tests
        tests = {
            "API Connection": test_api_connection,
            "Data Loading": test_data_loading,
            "Trading Environment": test_environment,
            "Model Training": test_model_training,
        }

        results = {}
        for test_name, test_func in tests.items():
            logger.info(f"\nRunning {test_name} test...")
            results[test_name] = test_func()

        # Print summary
        logger.info("\nTest Results Summary:")
        for test_name, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            logger.info(f"{test_name}: {status}")

        # Check if all tests passed
        if all(results.values()):
            logger.info("\nAll system tests passed successfully!")
            return 0
        else:
            logger.error("\nSome tests failed. Please check the logs for details.")
            return 1

    except Exception as e:
        logger.error(f"System test failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
