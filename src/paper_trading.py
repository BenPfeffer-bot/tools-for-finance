"""
Paper trading engine for simulated trading with real-time market data.

This module implements paper trading functionality including order execution,
position tracking, and portfolio management.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Position:
    """Represents a trading position."""

    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0
    realized_pnl: float = 0


class PaperTradingEngine:
    """
    Paper trading engine for simulated trading.

    Handles order execution, position tracking, and portfolio management
    using real-time market data.
    """

    def __init__(
        self, initial_capital: float = 1000000.0, transaction_cost: float = 0.001
    ):
        """
        Initialize paper trading engine.

        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as percentage
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.transaction_cost = transaction_cost

        # Portfolio tracking
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Dict] = []
        self.portfolio_history: List[Dict] = []

        # Performance tracking
        self.equity = initial_capital
        self.high_watermark = initial_capital
        self.max_drawdown = 0.0

        logger.info(f"Paper trading engine initialized with ${initial_capital:,.2f}")

    def place_order(self, order: Order) -> bool:
        """
        Place a new order.

        Args:
            order: Order to place

        Returns:
            bool: Whether order was successfully placed
        """
        try:
            # Validate order
            if not self._validate_order(order):
                return False

            # Add to order book
            self.orders.append(order)
            logger.info(f"Order placed: {order}")
            return True

        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return False

    def _validate_order(self, order: Order) -> bool:
        """
        Validate order parameters.

        Args:
            order: Order to validate

        Returns:
            bool: Whether order is valid
        """
        try:
            # Check for required fields
            if not all([order.symbol, order.side, order.quantity, order.order_type]):
                logger.error("Missing required order fields")
                return False

            # Validate quantity
            if order.quantity <= 0:
                logger.error("Order quantity must be positive")
                return False

            # Validate prices for limit and stop orders
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if order.limit_price is None or order.limit_price <= 0:
                    logger.error("Invalid limit price")
                    return False

            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                if order.stop_price is None or order.stop_price <= 0:
                    logger.error("Invalid stop price")
                    return False

            return True

        except Exception as e:
            logger.error(f"Order validation error: {str(e)}")
            return False

    def process_market_data(self, market_data: pd.DataFrame) -> None:
        """
        Process incoming market data and update positions.

        Args:
            market_data: DataFrame of current market data
        """
        try:
            # Update positions with new prices
            for symbol, data in market_data.iterrows():
                if symbol in self.positions:
                    position = self.positions[symbol]
                    current_price = data["price"]
                    position.unrealized_pnl = position.quantity * (
                        current_price - position.avg_price
                    )

            # Process pending orders
            self._process_orders(market_data)

            # Update portfolio metrics
            self._update_portfolio_metrics(market_data)

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")

    def _process_orders(self, market_data: pd.DataFrame) -> None:
        """
        Process pending orders with current market data.

        Args:
            market_data: DataFrame of current market data
        """
        remaining_orders = []

        for order in self.orders:
            if self._should_execute_order(order, market_data):
                if self._execute_order(order, market_data):
                    # Order executed successfully
                    continue
            # Keep order in queue if not executed
            remaining_orders.append(order)

        self.orders = remaining_orders

    def _should_execute_order(self, order: Order, market_data: pd.DataFrame) -> bool:
        """
        Determine if order should be executed based on current market data.

        Args:
            order: Order to check
            market_data: Current market data

        Returns:
            bool: Whether order should be executed
        """
        if order.symbol not in market_data.index:
            return False

        current_price = market_data.loc[order.symbol, "price"]

        if order.order_type == OrderType.MARKET:
            return True

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                return current_price <= order.limit_price
            else:
                return current_price >= order.limit_price

        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                return current_price >= order.stop_price
            else:
                return current_price <= order.stop_price

        elif order.order_type == OrderType.STOP_LIMIT:
            if order.side == OrderSide.BUY:
                return (
                    current_price >= order.stop_price
                    and current_price <= order.limit_price
                )
            else:
                return (
                    current_price <= order.stop_price
                    and current_price >= order.limit_price
                )

        return False

    def _execute_order(self, order: Order, market_data: pd.DataFrame) -> bool:
        """
        Execute an order at current market price.

        Args:
            order: Order to execute
            market_data: Current market data

        Returns:
            bool: Whether order was executed successfully
        """
        try:
            current_price = market_data.loc[order.symbol, "price"]
            transaction_value = order.quantity * current_price
            transaction_cost = transaction_value * self.transaction_cost

            # Check if we have enough cash for buy orders
            if order.side == OrderSide.BUY:
                total_cost = transaction_value + transaction_cost
                if total_cost > self.cash:
                    logger.warning(f"Insufficient funds for order: {order}")
                    return False

            # Update position
            if order.symbol in self.positions:
                position = self.positions[order.symbol]
                if order.side == OrderSide.BUY:
                    # Add to position
                    new_quantity = position.quantity + order.quantity
                    new_avg_price = (
                        position.quantity * position.avg_price
                        + order.quantity * current_price
                    ) / new_quantity
                    position.quantity = new_quantity
                    position.avg_price = new_avg_price
                else:
                    # Reduce or close position
                    if order.quantity > position.quantity:
                        logger.warning(f"Insufficient position for order: {order}")
                        return False
                    position.realized_pnl += order.quantity * (
                        current_price - position.avg_price
                    )
                    position.quantity -= order.quantity
                    if position.quantity == 0:
                        del self.positions[order.symbol]
            else:
                if order.side == OrderSide.BUY:
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        avg_price=current_price,
                    )
                else:
                    logger.warning(f"No position to sell: {order}")
                    return False

            # Update cash
            if order.side == OrderSide.BUY:
                self.cash -= transaction_value + transaction_cost
            else:
                self.cash += transaction_value - transaction_cost

            # Record trade
            self.trades.append(
                {
                    "timestamp": datetime.now(),
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "price": current_price,
                    "transaction_cost": transaction_cost,
                }
            )

            logger.info(f"Order executed: {order}")
            return True

        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            return False

    def _update_portfolio_metrics(self, market_data: pd.DataFrame) -> None:
        """
        Update portfolio metrics with current market data.

        Args:
            market_data: Current market data
        """
        # Calculate total equity
        position_value = sum(
            pos.quantity * market_data.loc[pos.symbol, "price"]
            for pos in self.positions.values()
            if pos.symbol in market_data.index
        )
        self.equity = self.cash + position_value

        # Update high watermark and drawdown
        self.high_watermark = max(self.high_watermark, self.equity)
        current_drawdown = (self.high_watermark - self.equity) / self.high_watermark
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Record portfolio state
        self.portfolio_history.append(
            {
                "timestamp": datetime.now(),
                "cash": self.cash,
                "position_value": position_value,
                "equity": self.equity,
                "drawdown": current_drawdown,
            }
        )

    def get_portfolio_summary(self) -> Dict:
        """
        Get current portfolio summary.

        Returns:
            Dictionary with portfolio metrics
        """
        return {
            "timestamp": datetime.now(),
            "initial_capital": self.initial_capital,
            "current_equity": self.equity,
            "cash": self.cash,
            "positions": len(self.positions),
            "total_trades": len(self.trades),
            "returns": (self.equity - self.initial_capital) / self.initial_capital,
            "max_drawdown": self.max_drawdown,
        }

    def save_trading_history(self, filepath: str) -> None:
        """
        Save trading history to file.

        Args:
            filepath: Path to save history
        """
        history = {"trades": self.trades, "portfolio_history": self.portfolio_history}

        with open(filepath, "w") as f:
            json.dump(history, f, default=str)
        logger.info(f"Trading history saved to {filepath}")
