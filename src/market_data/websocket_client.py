"""
Real-time market data client using yfinance.

This module handles market data connections and processing through polling.
"""

import json
import threading
import logging
from typing import Callable, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
import queue
from collections import defaultdict
import yfinance as yf
import time

logger = logging.getLogger(__name__)


class MarketDataClient:
    """
    Market data client for real-time data using yfinance.

    Handles market data polling and processing.
    Uses polling to simulate real-time data feed.
    """

    def __init__(self, tickers, callback=None, polling_interval=5, max_retries=3):
        """Initialize the market data client.

        Args:
            tickers (list): List of ticker symbols to subscribe to
            callback (callable): Callback function for tick data
            polling_interval (int): Polling interval in seconds
            max_retries (int): Maximum number of retries for failed requests
        """
        self.tickers = tickers
        self.callback = callback
        self.polling_interval = polling_interval
        self.max_retries = max_retries
        self.running = False
        self.lock = threading.Lock()

        # Initialize yfinance tickers
        self.yf_tickers = {ticker: yf.Ticker(ticker) for ticker in tickers}

        # Initialize data structures
        self.tick_buffer = {symbol: queue.Queue(maxsize=100) for symbol in tickers}
        self.last_prices = {symbol: None for symbol in tickers}
        self.error_counts = defaultdict(int)

        logger.info(f"Market data client initialized with {len(tickers)} tickers")

    def start(self):
        """Start the market data client."""
        if self.running:
            logger.warning("Market data client is already running")
            return

        self.running = True
        self.poll_thread = threading.Thread(target=self._poll_loop)
        self.poll_thread.daemon = True
        self.poll_thread.start()

        logger.info("Market data client started")

    def stop(self):
        """Stop the market data client."""
        if not self.running:
            logger.warning("Market data client is not running")
            return

        logger.info("Stopping market data client...")
        self.running = False
        if hasattr(self, "poll_thread"):
            self.poll_thread.join(timeout=5)  # Wait up to 5 seconds
            if self.poll_thread.is_alive():
                logger.warning("Poll thread did not stop gracefully")
        logger.info("Market data client stopped")

    def _poll_loop(self):
        """Main polling loop."""
        while self.running:
            try:
                self._poll_market_data()
                time.sleep(self.polling_interval)
            except Exception as e:
                logger.error(f"Error in polling loop: {str(e)}")
                time.sleep(self.polling_interval)

    def _poll_market_data(self):
        """Poll market data for all tickers."""
        logger.debug("Fetching market data...")
        try:
            for symbol in self.yf_tickers:
                if not self.running:  # Check if we should stop
                    break

                tick_data = self._get_latest_market_data_with_retry(symbol)
                if tick_data:
                    with self.lock:
                        if self.tick_buffer[symbol].full():
                            self.tick_buffer[symbol].get()
                        self.tick_buffer[symbol].put(tick_data)
                        self.last_prices[symbol] = tick_data["price"]
                        self.error_counts[symbol] = 0  # Reset error count on success

                    if self.callback:
                        try:
                            self.callback(tick_data)
                        except Exception as e:
                            logger.error(f"Error in callback for {symbol}: {str(e)}")
                else:
                    self.error_counts[symbol] += 1
                    if self.error_counts[symbol] >= self.max_retries:
                        logger.error(f"Max retries exceeded for {symbol}")

        except Exception as e:
            logger.error(f"Error in _poll_market_data: {str(e)}")

    def _get_latest_market_data_with_retry(
        self, symbol: str, retries: int = 3
    ) -> Optional[Dict]:
        """Get latest market data with retry logic."""
        for attempt in range(retries):
            try:
                data = self._get_latest_market_data(symbol)
                if data:
                    return data
                time.sleep(1)  # Wait before retry
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{retries} failed for {symbol}: {str(e)}"
                )
                if attempt < retries - 1:  # Don't sleep on the last attempt
                    time.sleep(1)
        return None

    def _get_latest_market_data(self, symbol: str) -> Optional[Dict]:
        """Get latest market data for a symbol."""
        try:
            ticker = self.yf_tickers[symbol]
            hist = ticker.history(period="1d", interval="1m", prepost=True)

            if hist.empty:
                logger.warning(f"No historical data available for {symbol}")
                return None

            latest = hist.iloc[-1]
            info = ticker.get_info()

            if info:
                logger.debug(f"Got info for {symbol}: volume={info.get('volume', 0)}")
            else:
                logger.warning(f"No info available for {symbol}")

            tick_data = {
                "symbol": symbol,
                "price": float(latest["Close"]),
                "volume": info.get("volume", 0) if info else 0,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"Updated {symbol} price: ${tick_data['price']:.2f}, volume: {tick_data['volume']:,}"
            )
            return tick_data

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None
