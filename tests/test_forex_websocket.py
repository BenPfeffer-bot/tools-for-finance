"""
Test script for Tiingo forex WebSocket client.
"""

import logging
import signal
import sys
from datetime import datetime
from typing import Dict
import pandas as pd
from tiingo_forex_client import TiingoForexClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"logs/forex_websocket_{datetime.now().strftime('%Y%m%d')}.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ForexDataTester:
    def __init__(self, pairs: list):
        self.pairs = pairs
        self.data_buffer = {pair: [] for pair in pairs}
        self.client = None

    def on_forex_data(self, data: Dict):
        """Handle incoming forex data."""
        try:
            pair = data.get("ticker", "").lower()
            if pair in self.pairs:
                # Extract and format data
                timestamp = pd.to_datetime(data["timestamp"])
                bid_price = data.get("bidPrice")
                ask_price = data.get("askPrice")
                mid_price = (
                    (bid_price + ask_price) / 2 if bid_price and ask_price else None
                )

                # Store data
                tick_data = {
                    "timestamp": timestamp,
                    "bid": bid_price,
                    "ask": ask_price,
                    "mid": mid_price,
                    "spread": ask_price - bid_price
                    if bid_price and ask_price
                    else None,
                }

                self.data_buffer[pair].append(tick_data)

                # Log data
                logger.info(f"\nReceived {pair.upper()} data:")
                logger.info(f"Time: {timestamp}")
                logger.info(f"Bid: ${bid_price:.5f}")
                logger.info(f"Ask: ${ask_price:.5f}")
                logger.info(f"Mid: ${mid_price:.5f}")
                logger.info(f"Spread: ${tick_data['spread']:.6f}")

                # Calculate basic analytics
                if len(self.data_buffer[pair]) > 1:
                    prev_mid = self.data_buffer[pair][-2]["mid"]
                    if prev_mid and mid_price:
                        pct_change = (mid_price - prev_mid) / prev_mid * 100
                        logger.info(f"Price Change: {pct_change:.4f}%")

        except Exception as e:
            logger.error(f"Error processing forex data: {str(e)}")

    def start(self):
        """Start forex data testing."""
        try:
            logger.info(f"Starting forex data test with pairs: {self.pairs}")

            # Initialize and start client
            self.client = TiingoForexClient(
                pairs=self.pairs, on_message_callback=self.on_forex_data
            )
            self.client.start()

            # Register signal handlers
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)

            logger.info("Forex data test started. Press Ctrl+C to stop.")
            signal.pause()

        except Exception as e:
            logger.error(f"Error in forex data test: {str(e)}")
            self.stop()

    def stop(self):
        """Stop forex data testing."""
        if self.client:
            self.client.stop()

        # Save collected data
        self._save_data()
        logger.info("Forex data test stopped")

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("\nInitiating graceful shutdown...")
        self.stop()
        sys.exit(0)

    def _save_data(self):
        """Save collected data to file."""
        try:
            for pair, data in self.data_buffer.items():
                if data:
                    df = pd.DataFrame(data)
                    filename = f"data/forex/test_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                    df.to_parquet(filename)
                    logger.info(f"Saved {pair} data to {filename}")

                    # Calculate and log statistics
                    stats = {
                        "n_ticks": len(df),
                        "avg_spread": df["spread"].mean(),
                        "max_spread": df["spread"].max(),
                        "price_volatility": df["mid"].pct_change().std(),
                        "time_span": df["timestamp"].max() - df["timestamp"].min(),
                    }

                    logger.info(f"\nStatistics for {pair.upper()}:")
                    for metric, value in stats.items():
                        logger.info(f"{metric}: {value}")

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")


def main():
    """Main function to run forex WebSocket test."""
    try:
        # Test with major forex pairs
        pairs = ["eurusd", "gbpusd", "usdjpy", "audusd", "usdcad"]

        # Create and start tester
        tester = ForexDataTester(pairs=pairs)
        tester.start()

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
