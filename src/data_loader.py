import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import yfinance as yf
import logging
from typing import Optional, Union, List
from datetime import datetime, timedelta

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


class DataLoader:
    """
    A class for loading, processing, and saving financial market data.

    This class handles the retrieval of market data from Yahoo Finance,
    performs basic data cleaning and processing, and manages data storage.

    Attributes:
        ticker (str): The stock ticker symbol
        raw_data_dir (Path): Directory for storing raw data
        data_dir (Path): Specific directory for the ticker's data
        df (pd.DataFrame): DataFrame containing the loaded data
    """

    def __init__(
        self,
        ticker: str,
        df: Optional[pd.DataFrame] = None,
        raw_data_dir: Union[str, Path] = MARKET_DATA_DIR,
    ):
        """
        Initialize the DataLoader instance.

        Args:
            ticker: Stock ticker symbol
            df: Optional pre-loaded DataFrame
            raw_data_dir: Directory path for storing raw data
        """
        self.ticker = ticker.upper()
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
        actions: bool = True,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load market data from Yahoo Finance.

        Args:
            interval: Data frequency ('1d', '1h', etc.)
            period: Time period to download ('5y', '2y', etc.)
            progress: Show download progress
            auto_adjust: Adjust prices for splits and dividends
            actions: Include dividend and split data
            start_date: Custom start date (overrides period)
            end_date: Custom end date

        Returns:
            DataFrame containing market data or None if download fails
        """
        try:
            if end_date is None:
                end_date = pd.Timestamp.today()
            elif isinstance(end_date, datetime):
                end_date = pd.Timestamp(end_date)

            if start_date is None and period:
                years = int(period.split("y")[0])
                start_date = end_date - timedelta(days=years * 365)
            elif isinstance(start_date, datetime):
                start_date = pd.Timestamp(start_date)

            logger.info(
                f"Downloading data for {self.ticker} from {start_date} to {end_date}"
            )

            df = yf.download(
                self.ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval,
                progress=progress,
                auto_adjust=auto_adjust,
                actions=actions,
            )

            if df.empty:
                logger.warning(f"No data retrieved for {self.ticker}")
                return None

            df = df.reset_index()
            logger.info(f"Successfully loaded {len(df)} rows for {self.ticker}")
            return df

        except Exception as e:
            logger.error(f"Error loading data for {self.ticker}: {str(e)}")
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
                file_name = f"{self.ticker}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"

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

        # Remove rows with NaN values
        df = df.dropna()

        # Remove zero volume rows
        df = df[df["Volume"] > 0]

        # Remove outliers if requested
        if remove_outliers:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                if column in ["Open", "High", "Low", "Close", "Volume"]:
                    z_scores = np.abs(
                        (df[column] - df[column].mean()) / df[column].std()
                    )
                    df = df[z_scores < zscore_threshold]

        rows_removed = initial_rows - len(df)
        logger.info(
            f"Cleaned data. Rows removed: {rows_removed}. New shape: {df.shape}"
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

        # Handle MultiIndex columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Converting MultiIndex columns to single level")
            df.columns = [col[0] for col in df.columns]  # Remove the _AAPL suffix

        # Ensure data is sorted by date
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        logger.info(f"Columns in DataFrame: {df.columns.tolist()}")

        try:
            # Calculate returns
            df.loc[:, "daily_return"] = df["Close"].pct_change()
            df.loc[:, "log_return"] = np.log(df["Close"] / df["Close"].shift(1))

            # Calculate moving averages
            for window in [5, 10, 20, 50, 200]:
                df.loc[:, f"MA_{window}"] = (
                    df["Close"].rolling(window=window, min_periods=1).mean()
                )

            # Calculate volatility (20-day rolling)
            df.loc[:, "volatility_20d"] = (
                df["log_return"]
                .rolling(window=20, min_periods=1)
                .std()
                .mul(np.sqrt(252))
            )

            # Trading volume indicators
            df.loc[:, "volume_ma_20"] = (
                df["Volume"].rolling(window=20, min_periods=1).mean()
            )
            df.loc[:, "volume_ratio"] = df["Volume"].div(df.loc[:, "volume_ma_20"])

            # Forward fill any NaN values created by the calculations
            df = df.fillna(method="ffill")

            # Backward fill any remaining NaN values at the beginning
            df = df.fillna(method="bfill")

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
        logger.info(f"Starting data pipeline for {self.ticker}")

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
            file_name = f"{self.ticker}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
            self.save_data(df, file_name=file_name)

        self.df = df
        logger.info(f"Completed data pipeline for {self.ticker}")

        # Verify the processed columns exist and have valid data
        expected_columns = [
            "daily_return",
            "log_return",
            "MA_20",
            "volatility_20d",
            "volume_ratio",
        ]
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")
        else:
            logger.info("All expected processed columns are present")

        return df


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
        loader = DataLoader(ticker=ticker)
        df = loader.run()

        if df is not None:
            print("\nData Summary:")
            print(df.describe())
            print("\nAvailable Features:", df.columns.tolist())
