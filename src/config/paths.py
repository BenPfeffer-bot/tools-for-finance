"""
Path configuration for data and outputs.

This module defines all paths used in the project for data storage and outputs.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Data directories
DATA_DIR = PROJECT_ROOT / "data"

# Raw data
RAW_DATA = DATA_DIR / "raw"
RAW_FOREX = RAW_DATA / "forex"
RAW_EQUITIES = RAW_DATA / "equities"
RAW_INTRADAY = RAW_DATA / "intraday"

# Processed data
PROCESSED_DATA = DATA_DIR / "processed"
PROCESSED_FOREX = PROCESSED_DATA / "forex"
PROCESSED_EQUITIES = PROCESSED_DATA / "equities"
PROCESSED_FEATURES = PROCESSED_DATA / "features"

# Cache
CACHE_DIR = DATA_DIR / "cache"
MARKET_DATA_CACHE = CACHE_DIR / "market_data"
FEATURES_CACHE = CACHE_DIR / "features"
MODELS_CACHE = CACHE_DIR / "models"

# History
HISTORY_DIR = DATA_DIR / "history"
FOREX_HISTORY = HISTORY_DIR / "forex"
EQUITIES_HISTORY = HISTORY_DIR / "equities"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Models
MODELS_DIR = OUTPUTS_DIR / "models"
ML_MODELS = MODELS_DIR / "ml"
RL_MODELS = MODELS_DIR / "rl"

# Analysis
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
EIGENPORTFOLIO_ANALYSIS = ANALYSIS_DIR / "eigenportfolio"
ARBITRAGE_ANALYSIS = ANALYSIS_DIR / "arbitrage"
SIGNALS_DIR = ANALYSIS_DIR / "signals"

# Backtest
BACKTEST_DIR = OUTPUTS_DIR / "backtest"
PERFORMANCE_DIR = BACKTEST_DIR / "performance"
TRADES_DIR = BACKTEST_DIR / "trades"
PLOTS_DIR = BACKTEST_DIR / "plots"

# Reports
REPORTS_DIR = OUTPUTS_DIR / "reports"
DAILY_REPORTS = REPORTS_DIR / "daily"
WEEKLY_REPORTS = REPORTS_DIR / "weekly"
MONTHLY_REPORTS = REPORTS_DIR / "monthly"

# Create all directories
DIRS = [
    RAW_FOREX,
    RAW_EQUITIES,
    RAW_INTRADAY,
    PROCESSED_FOREX,
    PROCESSED_EQUITIES,
    PROCESSED_FEATURES,
    MARKET_DATA_CACHE,
    FEATURES_CACHE,
    MODELS_CACHE,
    FOREX_HISTORY,
    EQUITIES_HISTORY,
    ML_MODELS,
    RL_MODELS,
    EIGENPORTFOLIO_ANALYSIS,
    ARBITRAGE_ANALYSIS,
    SIGNALS_DIR,
    PERFORMANCE_DIR,
    TRADES_DIR,
    PLOTS_DIR,
    DAILY_REPORTS,
    WEEKLY_REPORTS,
    MONTHLY_REPORTS,
]


def create_directories():
    """Create all required directories if they don't exist."""
    for directory in DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# Create directories when module is imported
create_directories()
