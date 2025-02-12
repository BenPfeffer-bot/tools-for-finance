"""
Configuration Module

This module handles project-wide configuration settings.
"""

from .paths import *

# Market data parameters
MIN_HISTORY_DAYS = 252  # Minimum days of history required
MAX_MISSING_PCTS = 0.1  # Maximum allowed percentage of missing data
PRICE_DECIMAL_PLACES = 4  # Number of decimal places for price data

# Feature generation parameters
VOLATILITY_WINDOW = 20
CORRELATION_WINDOW = 50
MOMENTUM_WINDOWS = [5, 10, 20]
RSI_WINDOW = 14
TRANSFER_ENTROPY_WINDOW = 50

# Model parameters
CV_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Trading parameters
POSITION_SIZE = 0.1  # Maximum position size as fraction of portfolio
STOP_LOSS = 0.02  # Stop loss threshold
TAKE_PROFIT = 0.05  # Take profit threshold

# List of EuroStoxx50 tickers to analyze (updated to remove delisted)
TICKERS = [
    "ADYEN.AS",  # Adyen
    "AIR.PA",  # Airbus
    "ALV.DE",  # Allianz
    "ASML.AS",  # ASML Holding
    "BAS.DE",  # BASF
    "BAYN.DE",  # Bayer
    "BBVA.MC",  # BBVA
    "BMW.DE",  # BMW
    "BNP.PA",  # BNP Paribas
    "CS.PA",  # AXA
    "DTE.DE",  # Deutsche Telekom
    "ENEL.MI",  # Enel
    "ENI.MI",  # Eni
    "IBE.MC",  # Iberdrola
    "INGA.AS",  # ING Group
    "ISP.MI",  # Intesa Sanpaolo
    "MC.PA",  # LVMH
    "MUV2.DE",  # Munich Re
    "OR.PA",  # L'Oreal
    "PHIA.AS",  # Philips
    "SAF.PA",  # Safran
    "SAN.MC",  # Banco Santander
    "SAP.DE",  # SAP
    "SIE.DE",  # Siemens
    "TTE.PA",  # TotalEnergies
    "VOW3.DE",  # Volkswagen
]

# Data Provider Settings
DATA_PROVIDERS = {
    "alpha_vantage": {
        "api_key": None,  # Set your Alpha Vantage API key here
        "rate_limit": 5,  # Calls per minute (free tier)
        "intervals": ["1min", "5min", "15min", "30min", "60min"],
    },
    "euronext": {
        "api_key": None,  # Optional Euronext API key
        "intervals": ["5min", "15min", "30min", "60min"],
    },
}

# Market Hours (UTC)
MARKET_HOURS = {
    "Euronext": {
        "open": "07:00",
        "close": "15:30",
    },
    "Deutsche Börse": {
        "open": "07:00",
        "close": "15:30",
    },
}


# Error handling
class DataError(Exception):
    """Base class for data-related errors."""

    pass


class InsufficientDataError(DataError):
    """Raised when there is insufficient historical data."""

    pass


class DataQualityError(DataError):
    """Raised when data quality does not meet requirements."""

    pass


def validate_market_data(data: pd.DataFrame) -> None:
    """
    Validate market data meets requirements.

    Args:
        data: DataFrame of market data

    Raises:
        InsufficientDataError: If insufficient history
        DataQualityError: If data quality issues detected
    """
    if len(data) < MIN_HISTORY_DAYS:
        raise InsufficientDataError(
            f"Insufficient history: {len(data)} days < {MIN_HISTORY_DAYS} required"
        )

    missing_pcts = data.isnull().mean()
    if (missing_pcts > MAX_MISSING_PCTS).any():
        bad_cols = missing_pcts[missing_pcts > MAX_MISSING_PCTS].index.tolist()
        raise DataQualityError(f"Excessive missing data in columns: {bad_cols}")


# Intraday Data Settings
INTRADAY_SETTINGS = {
    "default_interval": "5min",
    "cache_expiry_days": 1,  # How long to keep cached data
    "batch_size": 5,  # Number of parallel requests
    "retries": 3,  # Number of retry attempts
    "retry_delay": 5,  # Seconds between retries
}
