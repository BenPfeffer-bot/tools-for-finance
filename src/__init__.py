"""
Tools for Finance package initialization.
"""

from .market_data.websocket_client import MarketDataClient
from .market_data.tiingo_client import TiingoForexClient
from .analysis.eigenportfolio import Eigenportfolio
from .analysis.arbitrage_detector import ArbitrageDetector
from .trading.paper_trading import PaperTradingEngine
from .trading.realtime_strategy import RealTimeStrategy
from .backtesting.backtester import Backtester, StrategyBacktester
from .models.ml_model import MLModelTrainer
from .models.rl_model import RLTrader

__version__ = "0.1.0"
__author__ = "Ben Pfeffer"
