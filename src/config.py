"""
Configuration Module

This module handles project-wide configuration and paths.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Market data
DB = DATA_DIR
RAW = DB / "raw"
PROCESSED = DB / "processed"
COMPILED = DB / "compiled"

# Fetching
MARKET_DATA_DIR = RAW / "market_data"
FUNDAMENTAL_DATA_DIR = RAW / "fundamental_data"
NEWS_DATA_DIR = RAW / "news_data"

HISTORY = DATA_DIR / "history"
# Backtesting
# BACKTESTING_DIR = DATA_DIR / "backtesting"

# List of EuroStoxx50 tickers to analyze
TICKERS = [
    "ABI.BR",
    "ADYEN.AS",
    "AIR.PA",
    "AI.PA",
    "ALV.DE",
    "ASML.AS",
    "AZA.MI",
    "BAS.DE",
    "BAYN.DE",
    "BBVA.MC",
    "BMW.DE",
    "BNP.PA",
    "CRG.IR",
    "CS.PA",
    "DAI.DE",
    "DTE.DE",
    "DG.PA",
    "ENEL.MI",
    "ENI.MI",
    "EN.PA",
    "EL.PA",
    "ENGI.PA",
    "FRE.DE",
    "IBE.MC",
    "INGA.AS",
    "ISP.MI",
    "KER.PA",
    "LIN.DE",
    "MC.PA",
    "MUV2.DE",
    "OR.PA",
    "PHIA.AS",
    "PRX.AS",
    "RI.PA",
    "RMS.PA",
    "RNO.PA",
    "SAF.PA",
    "SAN.PA",
    "SAN.MC",
    "SAP.DE",
    "SGO.PA",
    "SIE.DE",
    "SU.PA",
    "TEF.MC",
    "TTE.PA",
    "UCG.MI",
    "VIV.PA",
    "VOW3.DE",
    "ZAL.DE",
]

# Trading parameters
# INITIAL_CAPITAL = 100000
# TRANSACTION_COSTS = 0.001
# RISK_FREE_RATE = 0.02  # Annual risk-free rate

# # Technical analysis parameters
# LOOKBACK_PERIODS = {"short": 20, "medium": 50, "long": 200}

# # Risk management parameters
# POSITION_SIZE_LIMITS = {"min": 0.01, "max": 0.3}

# STOP_LOSS_LIMITS = {"min": 0.01, "max": 0.05}

# # Market regime parameters
# REGIME_DETECTION = {"n_regimes": 4, "lookback_period": 60, "min_samples": 252}

# # Optimization parameters
# OPTIMIZATION = {"n_initial_points": 10, "n_iterations": 50, "exploration_weight": 0.1}

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

# Updated EuroStoxx50 tickers with exchange suffixes
EURONEXT_TICKERS = [
    # France - Euronext Paris
    "AI.PA",  # Air Liquide
    "AIR.PA",  # Airbus
    "ALO.PA",  # Alstom
    "BN.PA",  # Danone
    "BNP.PA",  # BNP Paribas
    "CA.PA",  # Carrefour
    "CAP.PA",  # Capgemini
    "CS.PA",  # AXA
    "DG.PA",  # Vinci
    "ENGI.PA",  # Engie
    "KER.PA",  # Kering
    "MC.PA",  # LVMH
    "OR.PA",  # L'Oreal
    "RI.PA",  # Pernod Ricard
    "SAN.PA",  # Sanofi
    "SGO.PA",  # Saint Gobain
    "VIE.PA",  # Veolia
    # Netherlands - Euronext Amsterdam
    "AD.AS",  # Ahold Delhaize
    "ADYEN.AS",  # Adyen
    "ASML.AS",  # ASML Holding
    "INGA.AS",  # ING Group
    "PHIA.AS",  # Philips
    "PRX.AS",  # Prosus
    # Belgium - Euronext Brussels
    "ABI.BR",  # AB InBev
    "SOLB.BR",  # Solvay
    "UCB.BR",  # UCB
    # Germany - Deutsche Börse
    "ADS.DE",  # Adidas
    "ALV.DE",  # Allianz
    "BAS.DE",  # BASF
    "BAYN.DE",  # Bayer
    "BMW.DE",  # BMW
    "DTG.DE",  # Daimler Truck
    "DTE.DE",  # Deutsche Telekom
    "HEI.DE",  # HeidelbergCement
    "IFX.DE",  # Infineon
    "SIE.DE",  # Siemens
    "VOW3.DE",  # Volkswagen
]

# Intraday Data Settings
INTRADAY_SETTINGS = {
    "default_interval": "5min",
    "cache_expiry_days": 1,  # How long to keep cached data
    "batch_size": 5,  # Number of parallel requests
    "retries": 3,  # Number of retry attempts
    "retry_delay": 5,  # Seconds between retries
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

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    OUTPUTS_DIR,
    DB,
    RAW,
    PROCESSED,
    COMPILED,
    MARKET_DATA_DIR,
    FUNDAMENTAL_DATA_DIR,
    NEWS_DATA_DIR,
    HISTORY,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Create additional necessary directories
CACHE_DIR = DATA_DIR / "cache"
INTRADAY_DATA_DIR = DATA_DIR / "intraday"

for directory in [CACHE_DIR, INTRADAY_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
