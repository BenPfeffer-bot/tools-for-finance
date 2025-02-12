import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.data_loader import IntradayDataLoader
from src.config.settings import EURONEXT_TICKERS, DATA_PROVIDERS, INTRADAY_SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # Load environment variables
    load_dotenv()

    # Get API keys from environment variables
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not alpha_vantage_key:
        logger.error("ALPHA_VANTAGE_API_KEY not found in environment variables")
        return

    # Initialize data loader
    data_loader = IntradayDataLoader(
        tickers=EURONEXT_TICKERS[:5],  # Start with a small subset for testing
        provider="alpha_vantage",
        api_key=alpha_vantage_key,
    )

    # Set date range for fetching data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)  # Get 5 days of intraday data

    try:
        # Fetch intraday data
        logger.info(f"Fetching intraday data from {start_date} to {end_date}")
        df = data_loader.fetch_data(
            start_date=start_date,
            end_date=end_date,
            interval=INTRADAY_SETTINGS["default_interval"],
            use_cache=True,
            batch_size=INTRADAY_SETTINGS["batch_size"],
        )

        if df.empty:
            logger.error("No data retrieved")
            return

        # Process the data
        logger.info("Processing intraday data")
        processed_df = data_loader.process_intraday_data(df)

        # Save processed data
        output_file = Path("data/processed/intraday_sample.parquet")
        processed_df.to_parquet(output_file)
        logger.info(f"Saved processed data to {output_file}")

        # Print summary statistics
        logger.info("\nData Summary:")
        logger.info(f"Total rows: {len(processed_df)}")
        logger.info(
            f"Date range: {processed_df.index.min()} to {processed_df.index.max()}"
        )
        logger.info(f"Tickers: {processed_df['Ticker'].nunique()}")
        logger.info("\nFeatures:")
        for column in processed_df.columns:
            logger.info(f"- {column}")

        # Calculate basic statistics
        stats = processed_df.groupby("Ticker").agg(
            {
                "intraday_return": ["mean", "std"],
                "GK_volatility": "mean",
                "Volume": "sum",
            }
        )
        logger.info("\nTicker Statistics:")
        logger.info(stats)

    except Exception as e:
        logger.error(f"Error in data fetching process: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
