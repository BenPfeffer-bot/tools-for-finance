"""
Test script for market data client.

This script tests the real-time market data client functionality
by subscribing to a few tickers and printing the received data.
"""

import logging
import signal
import sys
import os
from typing import Dict

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def on_tick_data(tick_data: Dict):
    """Callback function to process market data updates."""
    logger.info(f"\nReceived market data for {tick_data['symbol']}:")
    logger.info(f"Price: ${tick_data['price']:.2f}")
    logger.info(f"Volume: {tick_data['volume']:,}")
    logger.info(f"Timestamp: {tick_data['timestamp']}")


def main():
    """Main function to test the market data client."""
    from src.market_data.websocket_client import MarketDataClient

    client = None
    try:
        # Test with a few tickers
        tickers = ["AAPL", "MSFT", "GOOGL"]
        logger.info(f"Testing market data client with tickers: {tickers}")

        # Create and start client
        logger.info("Initializing market data client...")
        client = MarketDataClient(
            tickers=tickers, callback=on_tick_data, polling_interval=5
        )

        logger.info("Starting market data client...")
        client.start()
        logger.info("Market data client started successfully")
        logger.info("Press Ctrl+C to stop")

        # Handle graceful shutdown
        def signal_handler(signum, frame):
            logger.info("\nStopping market data client...")
            if client:
                client.stop()
            logger.info("Market data client stopped")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.pause()

    except Exception as e:
        logger.error(f"Error in market data test: {str(e)}")
        if client:
            client.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
