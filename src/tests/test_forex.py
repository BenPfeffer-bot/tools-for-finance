"""
Test script for forex trading system.
"""

import os
import sys
import time
import logging
from datetime import datetime
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.transfer_entropy import TransferEntropyCalculator
from src.market_data.tiingo_client import TiingoForexClient
from src.database.forex_data_loader import ForexDataLoader
from src.reinforcement.forex_trading_env import ForexTradingEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/forex_test_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def test_websocket():
    """Test WebSocket connection and data streaming."""
    try:
        # Test with major forex pairs
        pairs = ["eurusd", "gbpusd", "usdjpy", "audusd", "usdcad"]

        def on_forex_data(data):
            logger.info(f"Received forex data: {data}")

        # Initialize client
        client = TiingoForexClient(pairs=pairs, on_message_callback=on_forex_data)

        # Start client
        logger.info("Starting WebSocket client...")
        client.start()

        # Keep running for a while
        time.sleep(30)  # Run for 30 seconds

        # Stop client
        logger.info("Stopping WebSocket client...")
        client.stop()

    except Exception as e:
        logger.error(f"Error in WebSocket test: {str(e)}")
        raise


def test_data_loader():
    """Test forex data loading and processing."""
    try:
        # Initialize data loader
        pairs = ["eurusd", "gbpusd", "usdjpy"]
        loader = ForexDataLoader(pairs=pairs)

        # Test historical data loading
        start_date = pd.Timestamp.now() - pd.Timedelta(days=30)
        end_date = pd.Timestamp.now()

        logger.info("Loading historical data...")
        data = loader.fetch_historical_data(start_date, end_date)

        if not data.empty:
            logger.info(f"Loaded {len(data)} data points")
            logger.info(f"Data columns: {data.columns.tolist()}")
            logger.info(f"Sample data:\n{data.head()}")
        else:
            logger.warning("No historical data loaded")

        # Test real-time data
        logger.info("Testing real-time data...")
        loader.start_realtime_feed()

        # Wait for some data
        time.sleep(30)

        # Get latest data
        for pair in pairs:
            realtime_data = loader.get_realtime_data(pair)
            if not realtime_data.empty:
                logger.info(f"\nReal-time data for {pair}:")
                logger.info(f"Latest data:\n{realtime_data.tail()}")

        loader.stop_realtime_feed()

    except Exception as e:
        logger.error(f"Error in data loader test: {str(e)}")
        raise


def test_trading_env():
    """Test forex trading environment."""
    try:
        # Load some historical data first
        pairs = ["eurusd", "gbpusd"]
        loader = ForexDataLoader(pairs=pairs)

        start_date = pd.Timestamp.now() - pd.Timedelta(days=30)
        end_date = pd.Timestamp.now()
        data = loader.fetch_historical_data(start_date, end_date)

        if data.empty:
            logger.error("No data available for trading environment test")
            return

        # Initialize trading environment
        env = ForexTradingEnvironment(
            data=data,
            pairs=pairs,
            initial_balance=100000.0,
            max_position_size=0.1,
            risk_per_trade=0.02,
        )

        # Test environment
        logger.info("Testing trading environment...")

        # Reset environment
        state = env.reset()
        logger.info(f"Initial state shape: {state.shape}")

        # Try a few random actions
        for i in range(5):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)

            logger.info(f"\nStep {i + 1}:")
            logger.info(f"Action: {action}")
            logger.info(f"Reward: {reward:.4f}")
            logger.info(f"Done: {done}")
            logger.info(f"Info: {info}")

            if done:
                break

        logger.info("Trading environment test completed")

    except Exception as e:
        logger.error(f"Error in trading environment test: {str(e)}")
        raise


def main():
    """Run all tests."""
    try:
        logger.info("Starting forex system tests...")

        # Test WebSocket
        logger.info("\n=== Testing WebSocket ===")
        test_websocket()

        # Test data loader
        logger.info("\n=== Testing Data Loader ===")
        test_data_loader()

        # Test trading environment
        logger.info("\n=== Testing Trading Environment ===")
        test_trading_env()

        logger.info("\nAll tests completed successfully")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
