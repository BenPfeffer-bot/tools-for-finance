import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import yfinance as yf
import logging
from typing import Optional, Union, List, Tuple, Dict
from datetime import datetime, timedelta
import requests
from alpha_vantage.timeseries import TimeSeries
from concurrent.futures import ThreadPoolExecutor
import time

# sys.path.append("/Users/benpfeffer/Desktop/CODE/tools-for-finance/src/")

from config import (
    TICKERS,
    DB,
    RAW,
    PROCESSED,
    COMPILED,
    HISTORY,
    FUNDAMENTAL_DATA_DIR,
    MARKET_DATA_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataProvider:
    """Base class for data providers"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def fetch_intraday_data(
        self, ticker: str, start_date: datetime, end_date: datetime, interval: str
    ) -> pd.DataFrame:
        raise NotImplementedError


class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.ts = TimeSeries(key=api_key, output_format="pandas")
        self.rate_limit = 5  # calls per minute for free tier
        self.last_call_time = 0

    def _rate_limit_handler(self):
        """Handle API rate limiting"""
        current_time = time.time()
        time_passed = current_time - self.last_call_time
        if time_passed < 60 / self.rate_limit:
            time.sleep((60 / self.rate_limit) - time_passed)
        self.last_call_time = time.time()

    def fetch_intraday_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5min",
    ) -> pd.DataFrame:
        try:
            self._rate_limit_handler()
            data, meta_data = self.ts.get_intraday(
                symbol=ticker, interval=interval, outputsize="full"
            )
            data = data.loc[start_date:end_date]
            return data
        except Exception as e:
            logger.error(
                f"Error fetching data from Alpha Vantage for {ticker}: {str(e)}"
            )
            return pd.DataFrame()


class EuronextProvider(DataProvider):
    """Euronext data provider"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.base_url = "https://live.euronext.com/ajax/getHistoricalPriceData"

    def fetch_intraday_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5min",
    ) -> pd.DataFrame:
        try:
            params = {
                "instrumentId": ticker,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "resolution": interval,
            }
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data["data"])
            else:
                logger.error(
                    f"Error fetching data from Euronext for {ticker}: {response.status_code}"
                )
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data from Euronext for {ticker}: {str(e)}")
            return pd.DataFrame()


class IntradayDataLoader:
    """Enhanced data loader for intraday data"""

    def __init__(
        self,
        tickers: List[str],
        provider: str = "alpha_vantage",
        api_key: Optional[str] = None,
        data_dir: Union[str, Path] = MARKET_DATA_DIR,
    ):
        self.tickers = tickers
        self.data_dir = Path(data_dir)
        self.provider = self._initialize_provider(provider, api_key)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_provider(
        self, provider: str, api_key: Optional[str]
    ) -> DataProvider:
        """Initialize the specified data provider"""
        providers = {
            "alpha_vantage": AlphaVantageProvider,
            "euronext": EuronextProvider,
        }

        if provider not in providers:
            raise ValueError(f"Unsupported provider: {provider}")

        return providers[provider](api_key)

    def _get_cache_path(
        self, ticker: str, start_date: datetime, end_date: datetime, interval: str
    ) -> Path:
        """Generate cache file path"""
        cache_file = f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{interval}.parquet"
        return self.cache_dir / cache_file

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5min",
        use_cache: bool = True,
        batch_size: int = 5,
    ) -> pd.DataFrame:
        """
        Fetch intraday data for multiple tickers

        Args:
            start_date: Start date for data fetching
            end_date: End date for data fetching
            interval: Data interval (e.g., '1min', '5min', '15min', '30min', '60min')
            use_cache: Whether to use cached data
            batch_size: Number of tickers to process in parallel

        Returns:
            DataFrame containing intraday data for all tickers
        """
        all_data = []

        def process_ticker(ticker: str) -> pd.DataFrame:
            cache_path = self._get_cache_path(ticker, start_date, end_date, interval)

            # Try to load from cache first
            if use_cache and cache_path.exists():
                try:
                    return pd.read_parquet(cache_path)
                except Exception as e:
                    logger.warning(f"Error reading cache for {ticker}: {str(e)}")

            # Fetch fresh data
            df = self.provider.fetch_intraday_data(
                ticker, start_date, end_date, interval
            )

            if not df.empty:
                df["Ticker"] = ticker
                # Save to cache
                if use_cache:
                    try:
                        df.to_parquet(cache_path)
                    except Exception as e:
                        logger.warning(f"Error saving cache for {ticker}: {str(e)}")

            return df

        # Process tickers in parallel batches
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(executor.map(process_ticker, self.tickers))

        # Combine all data
        all_data = pd.concat([df for df in results if not df.empty], axis=0)

        if all_data.empty:
            logger.warning("No data retrieved for any tickers")
            return pd.DataFrame()

        logger.info(
            f"Successfully loaded data for {len(results)}/{len(self.tickers)} tickers"
        )
        return all_data

    def process_intraday_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process intraday data with additional features

        Args:
            df: Raw intraday data

        Returns:
            Processed DataFrame with additional features
        """
        if df.empty:
            return df

        df = df.copy()

        try:
            # Calculate returns
            df["intraday_return"] = df.groupby("Ticker")["Close"].pct_change()
            df["log_return"] = np.log(df.groupby("Ticker")["Close"].pct_change() + 1)

            # Calculate volatility using Garman-Klass estimator
            df["GK_volatility"] = np.sqrt(
                0.5 * np.log(df["High"] / df["Low"]) ** 2
                - (2 * np.log(2) - 1) * np.log(df["Close"] / df["Open"]) ** 2
            )

            # Volume analysis
            df["volume_ma"] = df.groupby("Ticker")["Volume"].transform(
                lambda x: x.rolling(window=12, min_periods=1).mean()
            )
            df["relative_volume"] = df["Volume"] / df["volume_ma"]

            # Price momentum
            for window in [12, 26, 50]:
                df[f"momentum_{window}"] = df.groupby("Ticker")["Close"].transform(
                    lambda x: x.pct_change(periods=window)
                )

            # VWAP
            df["VWAP"] = (df["Volume"] * df["Close"]).groupby(
                df["Ticker"]
            ).cumsum() / df["Volume"].groupby(df["Ticker"]).cumsum()

            logger.info(
                f"Successfully processed intraday data. Final shape: {df.shape}"
            )

        except Exception as e:
            logger.error(f"Error processing intraday data: {str(e)}")
            raise

        return df


class DataLoader:
    """
    Class for loading and preparing market data.
    """

    def __init__(self, tickers: List[str], lookback_years: int = 5):
        """
        Initialize DataLoader.

        Args:
            tickers: List of stock tickers
            lookback_years: Number of years of historical data to load
        """
        self.tickers = tickers
        self.lookback_years = lookback_years
        self.valid_tickers = []
        self.market_data = None
        self.returns_matrix = None

    def load_data(self) -> pd.DataFrame:
        """
        Load market data for all tickers.

        Returns:
            DataFrame with market data
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_years * 365)

        # Download data in batches
        batch_size = 10
        all_data = []

        for i in range(0, len(self.tickers), batch_size):
            batch_tickers = self.tickers[i : i + batch_size]
            logger.info(f"Downloading batch {i // batch_size + 1}: {batch_tickers}")

            try:
                data = yf.download(
                    batch_tickers,
                    start=start_date,
                    end=end_date,
                    progress=True,
                    auto_adjust=True,
                )

                # Reshape data for multi-ticker download
                if len(batch_tickers) > 1:
                    # Reorder columns to get price data first
                    cols_order = ["Close", "Open", "High", "Low", "Volume", "Adj Close"]
                    data = data.reindex(
                        columns=[
                            (col, ticker)
                            for col in cols_order
                            for ticker in batch_tickers
                        ]
                    )

                    # Stack and reset index
                    data = data.stack(level=1, future_stack=True)
                    data.index.names = ["Date", "Ticker"]
                else:
                    # For single ticker, create proper column names
                    data.columns = pd.MultiIndex.from_product(
                        [
                            ["Close", "Open", "High", "Low", "Volume", "Adj Close"],
                            [batch_tickers[0]],
                        ]
                    )
                    data = data.stack(level=1, future_stack=True)
                    data.index.names = ["Date", "Ticker"]

                all_data.append(data)

            except Exception as e:
                logger.error(f"Error downloading batch: {str(e)}")
                continue

        if not all_data:
            raise ValueError("Failed to load any data")

        # Combine all batches and reset index
        self.market_data = pd.concat(all_data).reset_index()
        self.valid_tickers = self.market_data["Ticker"].unique().tolist()

        logger.info(
            f"Successfully loaded data for {len(self.valid_tickers)}/{len(self.tickers)} tickers"
        )

        return self.market_data

    def prepare_returns_matrix(self) -> pd.DataFrame:
        """
        Prepare returns matrix from market data.

        Returns:
            DataFrame with daily returns (dates x tickers)
        """
        if self.market_data is None:
            raise ValueError("Must load data first")

        # Pivot data to wide format
        prices = self.market_data.pivot(index="Date", columns="Ticker", values="Close")

        # Forward fill missing values (limit to 5 days)
        prices = prices.ffill(limit=5)

        # Remove tickers with too many missing values (>10%)
        missing_pct = prices.isnull().mean()
        valid_tickers = missing_pct[missing_pct < 0.1].index
        prices = prices[valid_tickers]

        # Remove any remaining rows with missing values
        initial_rows = len(prices)
        prices = prices.dropna()
        rows_removed = initial_rows - len(prices)
        logger.info(
            f"Removed {rows_removed} rows with missing values ({rows_removed / initial_rows:.1%})"
        )

        # Compute returns
        self.returns_matrix = prices.pct_change().dropna()
        logger.info(f"Final returns matrix shape: {self.returns_matrix.shape}")

        # Save returns matrix to file
        processed_dir = PROCESSED
        processed_dir.mkdir(parents=True, exist_ok=True)
        self.returns_matrix.to_csv(processed_dir / "returns.csv")

        return self.returns_matrix

    def analyze_missing_data(self) -> None:
        """
        Analyze and log missing data statistics.
        """
        if self.returns_matrix is None:
            raise ValueError("Must prepare returns matrix first")

        missing_pct = self.returns_matrix.isnull().mean()
        logger.info(f"Average missing: {missing_pct.mean():.1%}")

        if not missing_pct.empty and not missing_pct.isna().all():
            worst_ticker = missing_pct.idxmax()
            worst_pct = missing_pct.max()
            logger.info(f"Worst ticker: {worst_ticker} ({worst_pct:.1%} missing)")
        else:
            logger.info("No valid missing data statistics available")

        logger.info(f"Number of tickers with >10% missing: {(missing_pct > 0.1).sum()}")


class MultiDataLoader:
    """
    A class for loading, processing, and saving financial market data for multiple tickers.
    """

    def __init__(
        self,
        tickers: Union[str, List[str]],
        df: Optional[pd.DataFrame] = None,
        raw_data_dir: Union[str, Path] = MARKET_DATA_DIR,
    ):
        """
        Initialize the MultiDataLoader instance.

        Args:
            tickers: Stock ticker symbol or list of symbols
            df: Optional pre-loaded DataFrame
            raw_data_dir: Directory path for storing raw data
        """
        if isinstance(tickers, str):
            self.tickers = [tickers.upper()]
        else:
            self.tickers = [ticker.upper() for ticker in tickers]

        self.raw_data_dir = Path(raw_data_dir)
        self.data_dir = MARKET_DATA_DIR
        self.df = df

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_data(
        self,
        interval: str = "1d",
        period: str = "5y",
        progress: bool = False,
        auto_adjust: bool = True,
        actions: bool = False,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 10,
    ) -> Optional[pd.DataFrame]:
        """
        Load market data from Yahoo Finance for multiple tickers.

        Args:
            interval: Data frequency ('1d', '1h', etc.)
            period: Time period to download ('5y', '2y', etc.)
            progress: Show download progress
            auto_adjust: Adjust prices for splits and dividends
            actions: Include dividend and split data
            start_date: Custom start date (overrides period)
            end_date: Custom end date
            batch_size: Number of tickers to download in each batch

        Returns:
            DataFrame containing market data or None if download fails
        """
        try:
            if end_date is None:
                end_date = pd.Timestamp.today()
            elif isinstance(end_date, datetime):
                end_date = pd.Timestamp(end_date)

            if start_date is None and period:
                years = int(period.strip("y"))
                start_date = end_date - timedelta(days=years * 365)
            elif isinstance(start_date, datetime):
                start_date = pd.Timestamp(start_date)

            # Split tickers into batches
            all_data = []
            failed_tickers = []

            for i in range(0, len(self.tickers), batch_size):
                batch_tickers = self.tickers[i : i + batch_size]
                logger.info(f"Downloading batch {i // batch_size + 1}: {batch_tickers}")

                try:
                    df_batch = yf.download(
                        tickers=batch_tickers,
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        interval=interval,
                        progress=progress,
                        auto_adjust=auto_adjust,
                        actions=actions,
                        group_by="ticker",
                        threads=True,
                    )

                    if not df_batch.empty:
                        # Handle the case when only one ticker is downloaded
                        if isinstance(df_batch.columns, pd.MultiIndex):
                            # Multiple tickers
                            df_batch = df_batch.stack(level=0)
                            df_batch.index.names = ["Date", "Ticker"]
                            df_batch = df_batch.reset_index()
                        else:
                            # Single ticker
                            df_batch["Ticker"] = batch_tickers[0]
                            df_batch = df_batch.reset_index()

                        all_data.append(df_batch)
                    else:
                        failed_tickers.extend(batch_tickers)

                except Exception as e:
                    logger.warning(f"Error downloading batch {batch_tickers}: {str(e)}")
                    failed_tickers.extend(batch_tickers)
                    continue

            if not all_data:
                logger.warning("No data retrieved for any tickers")
                return None

            # Combine all batches
            df = pd.concat(all_data, axis=0)

            # Log failed tickers
            if failed_tickers:
                logger.warning(f"Failed to download data for tickers: {failed_tickers}")

            successful_tickers = df["Ticker"].nunique()
            logger.info(
                f"Successfully loaded data for {successful_tickers}/{len(self.tickers)} tickers"
            )

            # Ensure Date column is datetime
            df["Date"] = pd.to_datetime(df["Date"])

            self.df = df
            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None

    def save_data(
        self,
        df: Optional[pd.DataFrame] = None,
        file_name: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> bool:
        """
        Save the DataFrame to a CSV file.

        Args:
            df: DataFrame to save (uses self.df if None)
            file_name: Name of output file
            output_dir: Directory to save file (uses self.data_dir if None)

        Returns:
            bool indicating success or failure
        """
        try:
            if df is None:
                df = self.df

            if df is None or df.empty:
                logger.error("No data to save")
                return False

            if output_dir is None:
                output_dir = self.data_dir
            else:
                output_dir = Path(output_dir)

            if file_name is None:
                file_name = (
                    f"{self.tickers[0]}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
                )

            output_path = output_dir / file_name
            df.to_csv(output_path, index=False)
            logger.info(f"Successfully saved data to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False

    def clean_data(
        self,
        df: Optional[pd.DataFrame] = None,
        remove_outliers: bool = True,
        zscore_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Clean the market data by removing invalid entries and outliers.

        Args:
            df: DataFrame to clean (uses self.df if None)
            remove_outliers: Whether to remove statistical outliers
            zscore_threshold: Z-score threshold for outlier removal

        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.df

        if df is None or df.empty:
            logger.error("No data to clean")
            return pd.DataFrame()

        initial_rows = len(df)
        logger.info(f"Starting data cleaning. Initial shape: {df.shape}")

        # Create a copy to avoid modifying the original data
        df = df.copy()

        # Remove rows with NaN values in essential columns
        essential_columns = ["Open", "High", "Low", "Close", "Volume"]
        df = df.dropna(subset=essential_columns)

        # Remove zero or negative values in essential columns
        for col in essential_columns:
            df = df[df[col] > 0]

        # Remove outliers if requested (per ticker)
        if remove_outliers:
            clean_dfs = []
            for ticker in df["Ticker"].unique():
                ticker_data = df[df["Ticker"] == ticker].copy()

                # Only apply outlier removal if we have enough data
                if len(ticker_data) > 30:  # Minimum sample size
                    for column in essential_columns:
                        z_scores = np.abs(
                            (ticker_data[column] - ticker_data[column].mean())
                            / ticker_data[column].std()
                        )
                        ticker_data = ticker_data[z_scores < zscore_threshold]

                clean_dfs.append(ticker_data)

            df = pd.concat(clean_dfs, axis=0)

        rows_removed = initial_rows - len(df)
        logger.info(
            f"Cleaned data. Rows removed: {rows_removed} ({rows_removed / initial_rows:.1%}). New shape: {df.shape}"
        )

        # Verify we still have data for most tickers
        remaining_tickers = df["Ticker"].nunique()
        if remaining_tickers < len(self.tickers) * 0.8:  # 80% threshold
            logger.warning(
                f"Only {remaining_tickers}/{len(self.tickers)} tickers remain after cleaning"
            )

        return df

    def process_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Process the market data by calculating additional features.

        Args:
            df: DataFrame to process (uses self.df if None)

        Returns:
            Processed DataFrame with additional features
        """
        if df is None:
            df = self.df

        if df.empty:
            logger.error("No data to process")
            return df

        # Create a copy to avoid modifying the original data
        df = df.copy()

        try:
            # Calculate returns
            df["daily_return"] = df.groupby("Ticker")["Close"].pct_change()
            df["log_return"] = np.log(df.groupby("Ticker")["Close"].pct_change() + 1)

            # Calculate moving averages
            for window in [5, 10, 20, 50, 200]:
                df[f"MA_{window}"] = df.groupby("Ticker")["Close"].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )

            # Calculate volatility (20-day rolling)
            df["volatility_20d"] = df.groupby("Ticker")["log_return"].transform(
                lambda x: x.rolling(window=20, min_periods=1).std().mul(np.sqrt(252))
            )

            # Trading volume indicators
            df["volume_ma_20"] = df.groupby("Ticker")["Volume"].transform(
                lambda x: x.rolling(window=20, min_periods=1).mean()
            )
            df["volume_ratio"] = df["Volume"] / df["volume_ma_20"]

            # Replace deprecated fillna methods
            df = df.ffill()
            df = df.bfill()

            logger.info(f"Successfully processed data. Final shape: {df.shape}")
            logger.info(f"Final columns: {df.columns.tolist()}")

        except Exception as e:
            logger.error(f"Error during data processing: {str(e)}")
            raise

        return df

    def run(
        self,
        interval: str = "1d",
        period: str = "5y",
        save: bool = True,
        batch_size: int = 10,
        progress: bool = False,
        auto_adjust: bool = True,
        actions: bool = False,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Execute the complete data pipeline: load, clean, and process data.

        Args:
            interval: Data frequency
            period: Time period to download
            save: Whether to save the processed data
            batch_size: Number of tickers to download in each batch
            progress: Show download progress
            auto_adjust: Adjust prices for splits and dividends
            actions: Include dividend and split data
            start_date: Custom start date
            end_date: Custom end date

        Returns:
            Processed DataFrame or None if pipeline fails
        """
        logger.info(f"Starting data pipeline for {self.tickers}")

        # Load data with batch processing
        df = self.load_data(
            interval=interval,
            period=period,
            batch_size=batch_size,
            progress=progress,
            auto_adjust=auto_adjust,
            actions=actions,
            start_date=start_date,
            end_date=end_date,
        )

        if df is None:
            return None

        # First clean the data
        df = self.clean_data(df)
        if df.empty:
            return None

        # Process the cleaned data
        df = self.process_data(df)

        # Save if requested
        if save:
            file_name = f"market_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
            self.save_data(df, file_name=file_name)

        self.df = df
        logger.info(f"Completed data pipeline for {self.tickers}")

        return df


if __name__ == "__main__":
    # Example usage
    for ticker in TICKERS:
        loader = DataLoader(tickers=[ticker])
        df = loader.load_data()

        if df is not None:
            print("\nData Summary:")
            print(df.describe())
            print("\nAvailable Features:", df.columns.tolist())
