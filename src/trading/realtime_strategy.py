"""
Real-time trading strategy implementation.

This module combines ML predictions with paper trading execution
for real-time trading simulation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import queue
from collections import defaultdict

from src.market_data.websocket_client import MarketDataClient
from src.trading.paper_trading import PaperTradingEngine, Order, OrderType, OrderSide
from src.analysis.arbitrage_detector import ArbitrageDetector
from src.models.ml_model import MLModelTrainer

logger = logging.getLogger(__name__)


class RealTimeStrategy:
    """
    Real-time trading strategy implementation.

    Combines ML predictions with paper trading execution
    for real-time trading simulation.
    """

    def __init__(
        self,
        tickers: List[str],
        initial_capital: float = 1000000.0,
        position_size: float = 0.1,
        stop_loss: float = 0.02,
        take_profit: float = 0.05,
        buffer_size: int = 1000,
    ):
        """
        Initialize real-time trading strategy.

        Args:
            tickers: List of tickers to trade
            initial_capital: Starting capital
            position_size: Maximum position size as fraction of capital
            stop_loss: Stop loss threshold
            take_profit: Take profit threshold
            buffer_size: Size of market data buffer
        """
        self.tickers = tickers
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # Initialize components
        self.trading_engine = PaperTradingEngine(initial_capital=initial_capital)
        self.websocket = MarketDataClient(
            tickers=tickers, callback=self._on_market_data, polling_interval=5
        )

        # Data management
        self.market_data_buffer = defaultdict(lambda: queue.Queue(maxsize=buffer_size))
        self.last_signal_time = defaultdict(lambda: datetime.min)
        self.signal_cooldown = timedelta(minutes=5)

        # Load ML model
        self.model_trainer = MLModelTrainer()
        self.model = None
        self.signal_detector = None

        logger.info("Real-time strategy initialized")

    def start(self):
        """Start the trading strategy."""
        try:
            # Load and initialize models
            self._initialize_models()

            # Start market data client
            self.websocket.start()
            logger.info("Strategy started")

        except Exception as e:
            logger.error(f"Error starting strategy: {str(e)}")
            raise

    def stop(self):
        """Stop the trading strategy."""
        try:
            # Stop market data client
            self.websocket.stop()

            # Save trading history
            self.trading_engine.save_trading_history("trading_history.json")
            logger.info("Strategy stopped")

        except Exception as e:
            logger.error(f"Error stopping strategy: {str(e)}")

    def _initialize_models(self):
        """Initialize ML models and signal detector."""
        try:
            # Initialize signal detector
            returns = pd.DataFrame()  # Initialize with historical data
            eigenportfolios = np.array(
                []
            )  # Initialize with pre-computed eigenportfolios

            self.signal_detector = ArbitrageDetector(
                returns=returns, eigenportfolios=eigenportfolios
            )

            # Load trained model
            self.model = self.model_trainer.load_model("model.json")
            logger.info("Models initialized")

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def _on_market_data(self, market_data: pd.DataFrame):
        """
        Process incoming market data.

        Args:
            market_data: DataFrame of current market data
        """
        try:
            # Update market data buffers
            self._update_market_data(market_data)

            # Generate trading signals
            signals = self._generate_signals(market_data)

            # Execute trading decisions
            self._execute_signals(signals, market_data)

            # Update portfolio
            self.trading_engine.process_market_data(market_data)

            # Log portfolio status periodically
            self._log_portfolio_status()

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")

    def _update_market_data(self, market_data: pd.DataFrame):
        """
        Update market data buffers.

        Args:
            market_data: New market data
        """
        for symbol, data in market_data.iterrows():
            if self.market_data_buffer[symbol].full():
                self.market_data_buffer[symbol].get()
            self.market_data_buffer[symbol].put(data)

    def _generate_signals(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signals using ML model.

        Args:
            market_data: Current market data

        Returns:
            Dictionary of trading signals by symbol
        """
        try:
            signals = {}
            current_time = datetime.now()

            # Convert market data buffers to features
            features = self._prepare_features(market_data)

            if features is not None and not features.empty:
                # Generate predictions
                predictions = self.model.predict_proba(features)[:, 1]

                # Process predictions into signals
                for symbol, pred in zip(features.index, predictions):
                    # Check signal cooldown
                    if (
                        current_time - self.last_signal_time[symbol]
                    ) < self.signal_cooldown:
                        continue

                    # Convert prediction to signal (-1, 0, 1)
                    if pred > 0.7:  # Strong buy signal
                        signals[symbol] = 1
                        self.last_signal_time[symbol] = current_time
                    elif pred < 0.3:  # Strong sell signal
                        signals[symbol] = -1
                        self.last_signal_time[symbol] = current_time
                    else:
                        signals[symbol] = 0

            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return {}

    def _prepare_features(self, market_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare features for ML model.

        Args:
            market_data: Current market data

        Returns:
            DataFrame of features or None if insufficient data
        """
        try:
            # Convert market data buffers to DataFrame
            data_dict = {}
            for symbol in self.tickers:
                if not self.market_data_buffer[symbol].empty():
                    buffer_data = list(self.market_data_buffer[symbol].queue)
                    if len(buffer_data) >= 20:  # Minimum required history
                        data_dict[symbol] = pd.DataFrame(buffer_data)

            if not data_dict:
                return None

            # Generate features using signal detector
            features = self.signal_detector.generate_features(data_dict)
            return features

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None

    def _execute_signals(self, signals: Dict[str, float], market_data: pd.DataFrame):
        """
        Execute trading signals.

        Args:
            signals: Dictionary of trading signals by symbol
            market_data: Current market data
        """
        try:
            for symbol, signal in signals.items():
                if signal == 0:
                    continue

                current_price = market_data.loc[symbol, "price"]
                position = self.trading_engine.positions.get(symbol)

                if signal > 0 and (position is None or position.quantity == 0):
                    # Buy signal
                    self._place_buy_order(symbol, current_price)

                elif signal < 0 and position is not None and position.quantity > 0:
                    # Sell signal
                    self._place_sell_order(symbol, current_price, position.quantity)

        except Exception as e:
            logger.error(f"Error executing signals: {str(e)}")

    def _place_buy_order(self, symbol: str, current_price: float):
        """
        Place buy order with position sizing and risk management.

        Args:
            symbol: Symbol to trade
            current_price: Current market price
        """
        try:
            # Calculate position size
            portfolio_value = self.trading_engine.equity
            max_position_value = portfolio_value * self.position_size
            quantity = max_position_value / current_price

            # Place main order
            main_order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )

            # Place stop loss order
            stop_price = current_price * (1 - self.stop_loss)
            stop_order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.STOP,
                stop_price=stop_price,
            )

            # Place take profit order
            limit_price = current_price * (1 + self.take_profit)
            limit_order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
            )

            # Submit orders
            self.trading_engine.place_order(main_order)
            self.trading_engine.place_order(stop_order)
            self.trading_engine.place_order(limit_order)

        except Exception as e:
            logger.error(f"Error placing buy orders: {str(e)}")

    def _place_sell_order(self, symbol: str, current_price: float, quantity: float):
        """
        Place sell order to close position.

        Args:
            symbol: Symbol to trade
            current_price: Current market price
            quantity: Quantity to sell
        """
        try:
            # Place market sell order
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )

            self.trading_engine.place_order(order)

        except Exception as e:
            logger.error(f"Error placing sell order: {str(e)}")

    def _log_portfolio_status(self):
        """Log current portfolio status."""
        try:
            # Get portfolio summary
            summary = self.trading_engine.get_portfolio_summary()

            # Log status every 5 minutes
            current_time = datetime.now()
            if current_time.minute % 5 == 0 and current_time.second == 0:
                logger.info("Portfolio Status:")
                logger.info(f"Equity: ${summary['current_equity']:,.2f}")
                logger.info(f"Returns: {summary['returns']:.2%}")
                logger.info(f"Max Drawdown: {summary['max_drawdown']:.2%}")
                logger.info(f"Active Positions: {summary['positions']}")
                logger.info(f"Total Trades: {summary['total_trades']}")

        except Exception as e:
            logger.error(f"Error logging portfolio status: {str(e)}")
