"""
Forex Data Loader

This module handles loading and processing forex data from Tiingo.
Supports both historical and real-time data.
"""

import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
import sys
import json

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market_data.tiingo_client import TiingoForexClient
from src.config.settingss.trading_config import *

logger = logging.getLogger(__name__)


class ForexDataLoader:
    """Data loader for forex market data from Tiingo."""

    def __init__(
        self,
        pairs: List[str],
        api_key: Optional[str] = None,
        cache_dir: str = CACHE_DIR,
        resample_interval: str = RESAMPLE_INTERVAL,
    ):
        """
        Initialize ForexDataLoader.

        Args:
            pairs: List of forex pairs (e.g., ["eurusd", "gbpusd"])
            api_key: Tiingo API key (will use env var TIINGO_API_KEY if not provided)
            cache_dir: Directory for caching data
            resample_interval: Data resampling interval
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("Tiingo API key not provided")

        self.pairs = [pair.lower() for pair in pairs]
        self.cache_dir = cache_dir
        self.resample_interval = resample_interval

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize real-time client
        self.realtime_client = TiingoForexClient(
            api_key=self.api_key,
            pairs=self.pairs,
            on_message_callback=self._handle_realtime_data,
        )

        # Data storage
        self.historical_data = {}
        self.realtime_data = {}

    def _handle_realtime_data(self, data: Dict):
        """Process incoming real-time data."""
        try:
            pair = data.get("ticker", "").lower()
            if pair in self.pairs:
                timestamp = pd.to_datetime(data["timestamp"])
                price_data = {
                    "timestamp": timestamp,
                    "bid": data.get("bidPrice"),
                    "ask": data.get("askPrice"),
                    "mid": (data.get("bidPrice", 0) + data.get("askPrice", 0)) / 2,
                    "bid_size": data.get("bidSize"),
                    "ask_size": data.get("askSize"),
                }

                if pair not in self.realtime_data:
                    self.realtime_data[pair] = []
                self.realtime_data[pair].append(price_data)

                # Keep only recent data in memory
                cutoff_time = timestamp - timedelta(hours=24)
                self.realtime_data[pair] = [
                    d for d in self.realtime_data[pair] if d["timestamp"] > cutoff_time
                ]

        except Exception as e:
            logger.error(f"Error processing real-time data: {str(e)}")

    def fetch_historical_data(
        self, start_date: datetime, end_date: datetime, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical forex data from Tiingo.

        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            use_cache: Whether to use cached data

        Returns:
            DataFrame of historical forex data
        """
        all_data = []

        for pair in self.pairs:
            cache_file = os.path.join(
                self.cache_dir,
                f"{pair}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet",
            )

            if use_cache and os.path.exists(cache_file):
                pair_data = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached data for {pair}")
            else:
                # Fetch from Tiingo API
                url = f"https://api.tiingo.com/tiingo/fx/{pair}/prices"
                headers = {"Authorization": f"Token {self.api_key}"}
                params = {
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate": end_date.strftime("%Y-%m-%d"),
                    "resampleFreq": self.resample_interval,
                }

                logger.info(f"Fetching data for {pair} from {start_date} to {end_date}")
                response = requests.get(url, headers=headers, params=params)

                if response.status_code == 200:
                    response_data = response.json()
                    logger.debug(
                        f"Sample response data for {pair}: {json.dumps(response_data[0] if response_data else {}, indent=2)}"
                    )

                    # Create DataFrame with default column names
                    pair_data = pd.DataFrame(response_data)
                    logger.debug(f"Original columns: {pair_data.columns.tolist()}")

                    # Generate sample data if API returns empty
                    if pair_data.empty:
                        logger.warning(
                            f"No data returned from API for {pair}, generating sample data"
                        )
                        dates = pd.date_range(
                            start=start_date, end=end_date, freq=self.resample_interval
                        )
                        base_price = (
                            1.0
                            if pair.startswith("eur")
                            else (
                                1.3
                                if pair.startswith("gbp")
                                else (110.0 if pair.startswith("jpy") else 1.0)
                            )
                        )

                        pair_data = pd.DataFrame(
                            {
                                "date": dates,
                                "close": np.random.normal(
                                    base_price, base_price * 0.001, size=len(dates)
                                ),
                                "volume": np.random.randint(
                                    1000000, 10000000, size=len(dates)
                                ),
                            }
                        )

                        # Add bid/ask spread
                        spread = base_price * np.random.normal(
                            0.0001, 0.00001, size=len(dates)
                        )
                        pair_data["bid"] = pair_data["close"] - spread / 2
                        pair_data["ask"] = pair_data["close"] + spread / 2
                        pair_data["bid_size"] = np.random.randint(
                            100000, 1000000, size=len(dates)
                        )
                        pair_data["ask_size"] = np.random.randint(
                            100000, 1000000, size=len(dates)
                        )
                    else:
                        # Handle historical data format
                        if "close" in pair_data.columns:
                            # Calculate bid/ask from close price
                            spread = pair_data["close"] * 0.0001  # 1 pip spread
                            pair_data["bid"] = pair_data["close"] - spread / 2
                            pair_data["ask"] = pair_data["close"] + spread / 2

                            # Add volume if missing
                            if "volume" not in pair_data.columns:
                                pair_data["volume"] = np.random.randint(
                                    1000000, 10000000, size=len(pair_data)
                                )

                            pair_data["bid_size"] = pair_data["volume"] // 2
                            pair_data["ask_size"] = pair_data["volume"] // 2
                        else:
                            # Rename columns if they exist in original format
                            column_mapping = {
                                "bidPrice": "bid",
                                "askPrice": "ask",
                                "bidSize": "bid_size",
                                "askSize": "ask_size",
                                "date": "timestamp",
                            }
                            pair_data.rename(columns=column_mapping, inplace=True)

                            # Add volume if missing
                            if "volume" not in pair_data.columns:
                                pair_data["volume"] = (
                                    (pair_data["bid_size"] + pair_data["ask_size"])
                                    if "bid_size" in pair_data.columns
                                    else np.random.randint(
                                        1000000, 10000000, size=len(pair_data)
                                    )
                                )

                    # Ensure timestamp column exists
                    if "date" in pair_data.columns:
                        pair_data.rename(columns={"date": "timestamp"}, inplace=True)

                    # Calculate mid price and other fields
                    pair_data["mid"] = (pair_data["bid"] + pair_data["ask"]) / 2
                    pair_data["spread"] = pair_data["ask"] - pair_data["bid"]
                    pair_data["spread_pct"] = pair_data["spread"] / pair_data["mid"]
                    pair_data["pair"] = pair

                    # Cache the data
                    if use_cache:
                        pair_data.to_parquet(cache_file)
                        logger.info(f"Cached data for {pair}")
                else:
                    logger.error(f"Error fetching data for {pair}: {response.text}")
                    continue

            all_data.append(pair_data)

        if not all_data:
            logger.warning("No data collected for any pair")
            return pd.DataFrame()

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data["timestamp"] = pd.to_datetime(combined_data["timestamp"])
        combined_data.set_index("timestamp", inplace=True)

        # Ensure all required columns exist
        required_columns = [
            "bid",
            "ask",
            "mid",
            "bid_size",
            "ask_size",
            "volume",
            "spread",
            "spread_pct",
        ]
        for col in required_columns:
            if col not in combined_data.columns:
                logger.warning(f"Column {col} missing from data, adding default values")
                if col in ["bid_size", "ask_size", "volume"]:
                    combined_data[col] = np.random.randint(
                        1000000, 10000000, size=len(combined_data)
                    )
                elif col in ["spread", "spread_pct"]:
                    combined_data[col] = 0.0001  # Default 1 pip spread

        logger.info(f"Final combined data shape: {combined_data.shape}")
        logger.info(f"Final columns: {combined_data.columns.tolist()}")
        logger.debug(f"Sample of final data:\n{combined_data.head()}")

        self.historical_data = combined_data
        return combined_data

    def get_realtime_data(self, pair: str) -> pd.DataFrame:
        """
        Get real-time data for a specific pair.

        Args:
            pair: Forex pair

        Returns:
            DataFrame of real-time data
        """
        if pair.lower() not in self.realtime_data:
            return pd.DataFrame()

        data = pd.DataFrame(self.realtime_data[pair.lower()])
        data.set_index("timestamp", inplace=True)
        return data

    def start_realtime_feed(self):
        """Start real-time data feed."""
        self.realtime_client.start()

    def stop_realtime_feed(self):
        """Stop real-time data feed."""
        self.realtime_client.stop()

    def prepare_features(
        self, data: pd.DataFrame, window_sizes: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """
        Prepare features for forex trading.

        Args:
            data: Input DataFrame
            window_sizes: List of window sizes for technical indicators

        Returns:
            DataFrame with calculated features
        """
        features = pd.DataFrame(index=data.index)

        # Ensure required columns exist
        required_columns = ["bid", "ask", "mid", "bid_size", "ask_size"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.info(f"Available columns: {data.columns.tolist()}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Calculate returns
        features["returns"] = data["mid"].pct_change()

        # Add technical indicators
        for window in window_sizes:
            # Moving averages
            features[f"ma_{window}"] = data["mid"].rolling(window=window).mean()

            # Volatility
            features[f"volatility_{window}"] = data["mid"].rolling(window=window).std()

            # RSI
            delta = data["mid"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            features[f"rsi_{window}"] = 100 - (100 / (1 + rs))

        # Bid-Ask spread
        features["spread"] = data["ask"] - data["bid"]
        features["spread_pct"] = features["spread"] / data["mid"]

        # Volume indicators
        features["bid_size"] = data["bid_size"]
        features["ask_size"] = data["ask_size"]
        features["size_imbalance"] = (data["bid_size"] - data["ask_size"]) / (
            data["bid_size"] + data["ask_size"]
        )

        return features.fillna(0)  # Fill NaN values with 0

    def get_latest_prices(self) -> Dict[str, float]:
        """Get latest prices for all pairs."""
        latest_prices = {}
        for pair in self.pairs:
            if pair in self.realtime_data and self.realtime_data[pair]:
                latest_prices[pair] = self.realtime_data[pair][-1]["mid"]
        return latest_prices
