"""
Configuration Module

This module handles project-wide configuration and paths.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent

# Data directories
CACHE_DIR = PROJECT_ROOT / "cache"
FIGURES_DIR = CACHE_DIR / "fig"

# Market data
DB = CACHE_DIR / "db"
RAW = DB / "raw"
PROCESSED = DB / "processed"
COMPILED = DB / "compiled"

# Fetching
MARKET_DATA_DIR = RAW / "market_data"
FUNDAMENTAL_DATA_DIR = RAW / "fundamental_data"
NEWS_DATA_DIR = RAW / "news_data"

HISTORY = CACHE_DIR / "history"
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

# Create directories if they don't exist
for directory in [
    CACHE_DIR,
    FIGURES_DIR,
    DB,
    PROCESSED,
    COMPILED,
    MARKET_DATA_DIR,
    FUNDAMENTAL_DATA_DIR,
    NEWS_DATA_DIR,
    HISTORY,
]:
    directory.mkdir(parents=True, exist_ok=True)
