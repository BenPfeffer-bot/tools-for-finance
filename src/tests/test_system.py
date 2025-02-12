"""
System test script for forex trading implementation.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import torch
import numpy as np

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.market_data.tiingo_client import TiingoForexClient
from src.database.forex_data_loader import ForexDataLoader
from src.reinforcement.forex_trading_env import ForexTradingEnvironment
from src.config.trading_config import *

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

        # Test connection without starting websocket
        test_data = client.get_latest_prices()
        if not test_data:
            logger.error("Failed to get latest prices")
            return False

        logger.info(f"Successfully retrieved prices: {test_data}")
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

        # Test with cache first
        data = loader.fetch_historical_data(start_date, end_date, use_cache=True)
        if data.empty:
            logger.warning("No cached data found, fetching from API")
            data = loader.fetch_historical_data(start_date, end_date, use_cache=False)

        if data.empty:
            logger.error("No historical data loaded")
            return False

        # Verify data structure
        required_columns = ["bid", "ask", "mid", "volume", "pair"]
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False

        logger.info(f"Successfully loaded {len(data)} data points")
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"Sample data:\n{data.head()}")
        return True

    except Exception as e:
        logger.error(f"Data loading test failed: {str(e)}")
        return False


def prepare_test_data() -> pd.DataFrame:
    """Prepare test data for the trading environment."""
    # Generate sample data
    dates = pd.date_range(start="2025-01-01", end="2025-01-07", freq="5min")
    pairs = FOREX_PAIRS
    data = []

    for pair in pairs:
        pair_upper = pair.upper()
        base_price = (
            1.0
            if pair.startswith("eur")
            else (
                1.3
                if pair.startswith("gbp")
                else (110.0 if pair.startswith("jpy") else 1.0)
            )
        )

        # Generate random prices
        close_prices = np.random.normal(base_price, base_price * 0.001, size=len(dates))
        spreads = base_price * np.random.normal(0.0001, 0.00001, size=len(dates))

        pair_data = pd.DataFrame(
            {
                "timestamp": dates,
                f"{pair_upper}_close": close_prices,
                f"{pair_upper}_bid": close_prices - spreads / 2,
                f"{pair_upper}_ask": close_prices + spreads / 2,
                f"{pair_upper}_mid": close_prices,
                f"{pair_upper}_volume": np.random.randint(
                    1000000, 10000000, size=len(dates)
                ),
                f"{pair_upper}_bid_size": np.random.randint(
                    100000, 1000000, size=len(dates)
                ),
                f"{pair_upper}_ask_size": np.random.randint(
                    100000, 1000000, size=len(dates)
                ),
                f"{pair_upper}_spread": spreads,
                f"{pair_upper}_spread_pct": spreads / close_prices,
                "pair": pair_upper,
            }
        )
        data.append(pair_data)

    # Combine all data
    combined_data = pd.concat(data, axis=0)
    combined_data.set_index("timestamp", inplace=True)

    return combined_data


def test_environment():
    """Test trading environment."""
    try:
        # Use prepared test data
        data = prepare_test_data()
        logger.info(f"Test data shape: {data.shape}")
        logger.info(f"Test data columns: {data.columns.tolist()}")
        logger.info(f"Sample test data:\n{data.head()}")

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
            logger.info(f"State values: {state}")

            # Verify state dimensions
            expected_features = (
                len(FOREX_PAIRS)
                * 14  # Price, volume, technical, and position features per pair
                + 4  # Account features: balance, equity, margin_used, avg_volatility
            )
            logger.info(f"Expected features calculation:")
            logger.info(
                f"  - Features per pair: 14 (3 price + 3 volume + 6 technical + 2 position)"
            )
            logger.info(f"  - Number of pairs: {len(FOREX_PAIRS)}")
            logger.info(f"  - Account features: 4")
            logger.info(f"  - Total expected: {expected_features}")

            if state.shape[0] != expected_features:
                logger.error(
                    f"Unexpected state dimension: got {state.shape[0]}, expected {expected_features}"
                )
                return False

        except Exception as e:
            logger.error(f"Environment reset failed: {str(e)}")
            return False

        # Test step with random actions
        try:
            for i in range(5):
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)

                # Verify step outputs
                if not isinstance(reward, (int, float)):
                    logger.error(f"Invalid reward type: {type(reward)}")
                    return False

                if not isinstance(info, dict):
                    logger.error(f"Invalid info type: {type(info)}")
                    return False

                logger.info(f"\nStep {i + 1}:")
                logger.info(f"Action: {action}")
                logger.info(f"State shape: {next_state.shape}")
                logger.info(f"State values: {next_state}")
                logger.info(f"Reward: {reward:.4f}")
                logger.info(f"Done: {done}")
                logger.info(f"Balance: ${info['balance']:.2f}")
                logger.info(f"Equity: ${info['equity']:.2f}")
                logger.info(f"Positions: {info['positions']}")

                if done:
                    break

            return True

        except Exception as e:
            logger.error(f"Environment step failed: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"Environment test failed: {str(e)}")
        return False


def test_model_training():
    """Test model training."""
    try:
        # Use prepared test data
        data = prepare_test_data()

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

            # Verify environment properties
            if not hasattr(env, "observation_space") or not hasattr(
                env, "action_space"
            ):
                logger.error("Environment missing required spaces")
                return False

        except Exception as e:
            logger.error(f"Failed to create environment: {str(e)}")
            return False

        # Test environment interactions
        try:
            state = env.reset()
            logger.info(f"Initial state shape: {state.shape}")

            # Run short training loop
            total_reward = 0
            n_steps = 0

            for i in range(5):
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                n_steps += 1

                logger.info(f"\nTraining Step {i + 1}:")
                logger.info(f"Action: {action}")
                logger.info(f"State shape: {next_state.shape}")
                logger.info(f"Reward: {reward:.4f}")
                logger.info(f"Done: {done}")
                logger.info(f"Balance: ${info['balance']:.2f}")

                if done:
                    break

            logger.info(
                f"Training loop completed: {n_steps} steps, avg reward: {total_reward / n_steps:.4f}"
            )
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
            try:
                results[test_name] = test_func()
            except Exception as e:
                logger.error(f"Unexpected error in {test_name}: {str(e)}")
                results[test_name] = False

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
