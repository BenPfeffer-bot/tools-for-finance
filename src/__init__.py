"""
Tools for Finance package.

This package provides tools for financial data analysis,
machine learning model training, and real-time trading.
"""

from .websocket_client import MarketDataClient
from .paper_trading import PaperTradingEngine, Order, OrderType, OrderSide
from .arbitrage_signal_detector import ArbitrageSignalDetector
from .ml_model_trainer import MLModelTrainer
from .realtime_strategy import RealTimeStrategy

__version__ = "0.1.0"
__author__ = "Ben Pfeffer"
