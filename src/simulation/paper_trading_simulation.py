"""
Paper trading simulation with real-time market data.

This script runs a paper trading simulation using real-time market data,
combining ML predictions and RL-based execution.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import threading
from typing import Dict, List, Optional
import queue
import yfinance as yf
import time

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.market_data.websocket_client import MarketDataClient
from src.trading.paper_trading import PaperTradingEngine, Order, OrderType, OrderSide
from src.analysis.arbitrage_detector import ArbitrageDetector
from src.models.ml_model import MLModelTrainer
from src.models.rl_model import RLTrader, TradingEnvironment
from src.config.settings import TICKERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"logs/paper_trading_{datetime.now().strftime('%Y%m%d')}.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class PaperTradingSimulation:
    """Paper trading simulation with real-time data and ML/RL models."""

    def __init__(
        self,
        tickers: List[str],
        initial_capital: float = 1000000.0,
        polling_interval: int = 5,
        window_size: int = 50,
    ):
        """Initialize paper trading simulation.

        Args:
            tickers: List of tickers to trade
            initial_capital: Starting capital
            polling_interval: Market data polling interval
            window_size: Window size for feature generation
        """
        self.tickers = tickers
        self.window_size = window_size

        # Initialize components
        self.market_data_client = MarketDataClient(
            tickers=tickers,
            callback=self._on_market_data,
            polling_interval=polling_interval,
        )

        self.trading_engine = PaperTradingEngine(initial_capital=initial_capital)

        # Data management
        self.market_data_buffer = {
            symbol: queue.Queue(maxsize=1000) for symbol in tickers
        }
        self.signal_buffer = queue.Queue(maxsize=1000)

        # Performance tracking
        self.performance_metrics = {
            "portfolio_value": [],
            "returns": [],
            "positions": [],
            "signals": [],
            "timestamps": [],
        }

        # Initialize models
        self._initialize_models()

        # Create output directories
        self._setup_directories()

        logger.info("Paper trading simulation initialized")

    def _setup_directories(self):
        """Create necessary output directories."""
        directories = [
            "outputs/reports/daily",
            "outputs/reports/weekly",
            "outputs/reports/monthly",
            "outputs/plots/daily",
            "outputs/plots/weekly",
            "outputs/plots/monthly",
            "outputs/signals",
            "outputs/performance",
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _initialize_models(self):
        """Initialize ML and RL models."""
        try:
            # Load historical data for initial training
            historical_data = self._load_historical_data()

            # Initialize signal detector
            self.signal_detector = ArbitrageDetector(
                returns=historical_data,
                eigenportfolios=np.array([]),  # Will be computed from data
            )

            # Initialize ML model trainer
            self.model_trainer = MLModelTrainer()

            # Initialize RL environment and agent
            self.trading_env = TradingEnvironment(
                returns=historical_data,
                predictions=np.zeros(len(historical_data)),  # Placeholder
                window_size=self.window_size,
            )

            state_dim = self.trading_env._get_state().shape[0]
            self.rl_agent = RLTrader(
                env=self.trading_env,
                state_dim=state_dim,
                action_dim=3,  # Buy, Hold, Sell
                hidden_dim=128,
            )

            logger.info("Models initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def _load_historical_data(self) -> pd.DataFrame:
        """Load historical data for initial model training."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data

            # Download data
            data = yf.download(
                self.tickers, start=start_date, end=end_date, interval="1d"
            )

            logger.info(f"Downloaded data shape: {data.shape}")
            logger.info(f"Data columns: {data.columns}")
            logger.info(f"Data index: {data.index[:5]}")  # Show first 5 dates

            # Handle data structure
            if isinstance(data.columns, pd.MultiIndex):
                logger.info("Multi-index columns detected")
                close_prices = data.loc[:, ("Close", slice(None))]
                close_prices.columns = close_prices.columns.get_level_values(1)
            else:
                logger.info("Single-level columns detected")
                close_prices = pd.DataFrame(data["Close"])
                if len(self.tickers) == 1:
                    close_prices.columns = self.tickers

            logger.info(f"Close prices shape: {close_prices.shape}")
            logger.info(f"Close prices columns: {close_prices.columns}")

            # Calculate returns
            returns = close_prices.pct_change().dropna()

            logger.info(f"Loaded historical data: {len(returns)} days")
            return returns

        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            logger.error(f"Data structure: {type(data)}")
            if isinstance(data, pd.DataFrame):
                logger.error(f"Data head:\n{data.head()}")
            raise

    def _on_market_data(self, tick_data: Dict):
        """Process incoming market data updates."""
        try:
            symbol = tick_data["symbol"]

            # Update market data buffer
            if self.market_data_buffer[symbol].full():
                self.market_data_buffer[symbol].get()
            self.market_data_buffer[symbol].put(tick_data)

            # Log buffer size
            buffer_size = self.market_data_buffer[symbol].qsize()
            logger.info(f"Market data buffer size for {symbol}: {buffer_size}")

            # Generate trading signals if we have at least 5 data points
            min_data_points = 5
            if all(
                self.market_data_buffer[s].qsize() >= min_data_points
                for s in self.tickers
            ):
                # Generate trading signals
                signals = self._generate_signals()

                if signals:
                    # Save signals to history
                    self._save_signals(signals)

                    # Execute trading decisions
                    self._execute_signals(signals)

            # Update performance metrics
            self._update_performance_metrics(tick_data)

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")

    def _save_signals(self, signals: Dict[str, float]):
        """Save signals to history file."""
        try:
            # Create signals directory if it doesn't exist
            os.makedirs("outputs/signals", exist_ok=True)

            # Prepare signal data
            signal_data = {"timestamp": datetime.now().isoformat(), "signals": signals}

            # Append to signal history
            with open("outputs/signals/signal_history.jsonl", "a") as f:
                f.write(json.dumps(signal_data) + "\n")

        except Exception as e:
            logger.error(f"Error saving signals: {str(e)}")

    def _prepare_features(self) -> Optional[np.ndarray]:
        """Prepare features for ML models."""
        try:
            features = []

            # Process each symbol
            for symbol in self.tickers:
                # Get market data from buffer
                market_data = list(self.market_data_buffer[symbol].queue)
                data_points = len(market_data)

                if data_points < 5:  # Minimum required data points
                    return None

                # Extract price and volume data
                prices = np.array([d["price"] for d in market_data])
                volumes = np.array([d["volume"] for d in market_data])

                # Calculate basic features
                returns = np.diff(np.log(prices)) if len(prices) > 1 else np.array([0])
                volatility = np.std(returns) if len(returns) > 1 else 0
                momentum = prices[-1] / prices[0] - 1 if len(prices) > 1 else 0

                # Price features
                ma = np.mean(prices)
                std = np.std(prices)

                price_features = [
                    prices[-1] / ma - 1,  # Price relative to MA
                    std / ma,  # Normalized volatility
                ]

                # Volume features
                volume_ma = np.mean(volumes)
                volume_features = [
                    volumes[-1] / volume_ma - 1,  # Volume relative to MA
                    np.corrcoef(prices, volumes)[0, 1]
                    if len(prices) > 1
                    else 0,  # Price-volume correlation
                ]

                # Combine all features
                symbol_features = [
                    returns[-1],  # Latest return
                    volatility,
                    momentum,
                    *price_features,
                    *volume_features,
                ]

                features.extend(symbol_features)

            return np.array(features)

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None

    def _generate_signals(self) -> Dict[str, float]:
        """Generate trading signals using ML/RL models."""
        try:
            # Convert market data to features
            features = self._prepare_features()

            if features is not None:
                signals = {}
                for i, symbol in enumerate(self.tickers):
                    # Get market data from buffer
                    market_data = list(self.market_data_buffer[symbol].queue)
                    prices = np.array([d["price"] for d in market_data])

                    # Calculate simple moving averages
                    if len(prices) >= 5:
                        # Use exponential moving averages for smoother signals
                        alpha_short = 0.4  # Higher weight for recent prices
                        alpha_long = 0.2  # Lower weight for recent prices

                        # Calculate EMAs
                        ema_short = prices[-1]
                        ema_long = prices[-1]
                        for i in range(2, min(len(prices), 10)):
                            price = prices[-i]
                            ema_short = (
                                alpha_short * price + (1 - alpha_short) * ema_short
                            )
                            ema_long = alpha_long * price + (1 - alpha_long) * ema_long

                        # Generate signal based on EMA crossover
                        if ema_long != 0:  # Avoid division by zero
                            signal = (ema_short / ema_long - 1) * 5  # Scale to [-5, 5]
                        else:
                            signal = 0

                        # Add momentum component
                        if len(prices) >= 3:
                            momentum = (prices[-1] / prices[-3] - 1) * 2
                            signal = signal * 0.7 + momentum * 0.3  # Combine signals
                    else:
                        signal = 0
                        ema_short = prices[-1] if len(prices) > 0 else 0
                        ema_long = prices[-1] if len(prices) > 0 else 0

                    signals[symbol] = signal

                    # Log signal generation with more details
                    logger.info(
                        f"Generated signal for {symbol}: "
                        f"Signal={signal:.3f}, "
                        f"Price=${prices[-1]:.2f}, "
                        f"EMA_Short=${ema_short:.2f}, "
                        f"EMA_Long=${ema_long:.2f}"
                    )

                return signals

        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return {}

    def _execute_signals(self, signals: Dict[str, float]):
        """Execute trading signals."""
        try:
            # Calculate total portfolio risk allocation
            active_positions = len(self.trading_engine.positions)
            max_positions = min(len(self.tickers), 10)  # Cap at 10 positions
            available_positions = max_positions - active_positions

            if available_positions <= 0:
                logger.info("Maximum number of positions reached, skipping new trades")
                return

            # Sort signals by absolute strength
            sorted_signals = sorted(
                signals.items(), key=lambda x: abs(x[1]), reverse=True
            )[:available_positions]

            for symbol, signal in sorted_signals:
                # Get current position
                position = self.trading_engine.positions.get(symbol)
                current_position = position.quantity if position else 0

                # Get current price
                tick_data = list(self.market_data_buffer[symbol].queue)[-1]
                current_price = tick_data["price"]

                # Calculate position size (dynamic based on number of positions)
                portfolio_value = self.trading_engine.equity
                max_position_value = portfolio_value * 0.15  # Max 15% per position
                risk_factor = (
                    1.0 / (active_positions + 1) if active_positions > 0 else 1.0
                )
                target_position_value = max_position_value * risk_factor
                target_quantity = target_position_value / current_price

                # Execute trades based on signal strength with adaptive thresholds
                signal_threshold = 0.05 * (
                    1 + active_positions * 0.1
                )  # Increase threshold with more positions

                if signal > signal_threshold:  # Buy signal
                    if current_position < target_quantity:
                        # Calculate quantity to buy with position scaling
                        buy_quantity = min(
                            target_quantity - current_position,
                            target_quantity
                            * (abs(signal) / 0.2),  # Scale by signal strength
                        )
                        logger.info(
                            f"BUY signal for {symbol} "
                            f"(signal={signal:.3f}, threshold={signal_threshold:.3f}, "
                            f"price=${current_price:.2f}, "
                            f"quantity={buy_quantity:.2f}, "
                            f"target_value=${target_position_value:.2f}, "
                            f"active_positions={active_positions})"
                        )
                        self._place_buy_order(symbol, buy_quantity, current_price)

                elif signal < -signal_threshold:  # Sell signal
                    if current_position > 0:
                        # Calculate sell quantity based on signal strength
                        sell_quantity = current_position * min(1.0, abs(signal) / 0.2)
                        logger.info(
                            f"SELL signal for {symbol} "
                            f"(signal={signal:.3f}, threshold={signal_threshold:.3f}, "
                            f"price=${current_price:.2f}, "
                            f"quantity={sell_quantity:.2f}, "
                            f"position_value=${current_position * current_price:.2f}, "
                            f"active_positions={active_positions})"
                        )
                        self._place_sell_order(symbol, sell_quantity, current_price)
                else:
                    logger.info(
                        f"HOLD signal for {symbol} "
                        f"(signal={signal:.3f}, threshold={signal_threshold:.3f}, "
                        f"price=${current_price:.2f}, "
                        f"position={current_position:.2f})"
                    )

        except Exception as e:
            logger.error(f"Error executing signals: {str(e)}")

    def _place_buy_order(self, symbol: str, quantity: float, current_price: float):
        """Place buy order."""
        try:
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )

            # Log order placement
            logger.info(
                f"Placing BUY order for {symbol}: "
                f"{quantity:.2f} shares @ ${current_price:.2f} "
                f"(${quantity * current_price:,.2f} total)"
            )

            self.trading_engine.place_order(order)

        except Exception as e:
            logger.error(f"Error placing buy order: {str(e)}")

    def _place_sell_order(self, symbol: str, quantity: float, current_price: float):
        """Place sell order."""
        try:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )

            # Log order placement
            logger.info(
                f"Placing SELL order for {symbol}: "
                f"{quantity:.2f} shares @ ${current_price:.2f} "
                f"(${quantity * current_price:,.2f} total)"
            )

            self.trading_engine.place_order(order)

        except Exception as e:
            logger.error(f"Error placing sell order: {str(e)}")

    def _update_performance_metrics(self, tick_data: Dict):
        """Update performance tracking metrics."""
        try:
            current_time = datetime.now()

            # Update metrics
            current_equity = self.trading_engine.equity
            self.performance_metrics["portfolio_value"].append(current_equity)
            self.performance_metrics["timestamps"].append(current_time)

            # Calculate returns
            if len(self.performance_metrics["portfolio_value"]) > 1:
                returns = (
                    current_equity / self.performance_metrics["portfolio_value"][-2] - 1
                )
            else:
                returns = 0

            self.performance_metrics["returns"].append(returns)

            # Save current positions
            positions = {
                k: v.quantity for k, v in self.trading_engine.positions.items()
            }
            self.performance_metrics["positions"].append(positions)

            # Log performance update
            logger.info(
                f"Portfolio Update: "
                f"Equity=${current_equity:,.2f}, "
                f"Return={returns:.2%}, "
                f"Positions={positions}"
            )

            # Save metrics to file periodically (every 5 minutes)
            if not hasattr(
                self, "_last_save_time"
            ) or current_time - self._last_save_time >= timedelta(minutes=5):
                self._save_performance_metrics()
                self._last_save_time = current_time

        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")

    def _save_performance_metrics(self):
        """Save performance metrics to file."""
        try:
            # Create performance directory if it doesn't exist
            os.makedirs("outputs/performance", exist_ok=True)

            # Save metrics to JSON file
            metrics = {
                "portfolio_value": self.performance_metrics["portfolio_value"],
                "returns": self.performance_metrics["returns"],
                "positions": self.performance_metrics["positions"],
                "timestamps": [
                    t.isoformat() for t in self.performance_metrics["timestamps"]
                ],
            }

            with open("outputs/performance/metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            logger.info("Performance metrics saved to outputs/performance/metrics.json")

        except Exception as e:
            logger.error(f"Error saving performance metrics: {str(e)}")

    def _generate_reports(self):
        """Generate performance reports and plots."""
        try:
            current_time = datetime.now()

            # Generate daily reports at end of day
            if current_time.hour == 16 and current_time.minute == 0:
                self._generate_daily_report()
                self._generate_daily_plots()

            # Generate weekly reports on Friday
            if current_time.weekday() == 4 and current_time.hour == 16:
                self._generate_weekly_report()
                self._generate_weekly_plots()

            # Generate monthly reports at month end
            if current_time.day == 1 and current_time.hour == 0:
                self._generate_monthly_report()
                self._generate_monthly_plots()

        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")

    def _generate_daily_report(self):
        """Generate daily performance report."""
        try:
            if not self.performance_metrics["portfolio_value"]:
                logger.warning("No performance data available for report")
                return

            report = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "portfolio_value": self.trading_engine.equity,
                "daily_return": self.performance_metrics["returns"][-1],
                "positions": self.performance_metrics["positions"][-1],
                "trades": len(self.trading_engine.trades),
                "signals": len(self.signal_buffer.queue),
            }

            # Save report
            filename = (
                f"outputs/reports/daily/report_{datetime.now().strftime('%Y%m%d')}.json"
            )
            with open(filename, "w") as f:
                json.dump(report, f, indent=4)

        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")

    def _generate_daily_plots(self):
        """Generate daily performance plots."""
        try:
            if not self.performance_metrics["portfolio_value"]:
                logger.warning("No performance data available for plots")
                return

            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))

            # Plot portfolio value
            df = pd.DataFrame(
                {
                    "timestamp": self.performance_metrics["timestamps"],
                    "portfolio_value": self.performance_metrics["portfolio_value"],
                }
            )
            df.set_index("timestamp")["portfolio_value"].plot(ax=axes[0])
            axes[0].set_title("Portfolio Value")

            # Plot returns distribution
            returns = pd.Series(self.performance_metrics["returns"])
            sns.histplot(returns, kde=True, ax=axes[1])
            axes[1].set_title("Returns Distribution")

            # Plot positions
            positions_df = pd.DataFrame(self.performance_metrics["positions"])
            if not positions_df.empty:
                positions_df.plot(ax=axes[2])
                axes[2].set_title("Positions Over Time")

            # Save plot
            filename = f"outputs/plots/daily/performance_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename)
            plt.close()

        except Exception as e:
            logger.error(f"Error generating daily plots: {str(e)}")

    def start(self):
        """Start paper trading simulation."""
        try:
            logger.info("Starting paper trading simulation")
            self.market_data_client.start()

        except Exception as e:
            logger.error(f"Error starting simulation: {str(e)}")
            self.stop()

    def stop(self):
        """Stop paper trading simulation."""
        try:
            logger.info("Stopping paper trading simulation")
            self.market_data_client.stop()

            # Generate final reports
            self._generate_daily_report()
            self._generate_daily_plots()

        except Exception as e:
            logger.error(f"Error stopping simulation: {str(e)}")

    def _analyze_signals(self):
        """Analyze trading signals and their effectiveness."""
        try:
            # Load signal history from JSONL file
            signals = []
            with open("outputs/signals/signal_history.jsonl", "r") as f:
                for line in f:
                    signals.append(json.loads(line))

            # Convert to DataFrame
            signals_df = pd.DataFrame(signals)
            signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])
            signals_df.set_index("timestamp", inplace=True)

            # Calculate signal metrics
            signal_metrics = {
                "total_signals": len(signals_df),
                "buy_signals": len(signals_df[signals_df["signals"] > 0.1]),
                "sell_signals": len(signals_df[signals_df["signals"] < -0.1]),
                "hold_signals": len(signals_df[abs(signals_df["signals"]) <= 0.1]),
            }

            # Save metrics
            with open("outputs/analysis/signal_metrics.json", "w") as f:
                json.dump(signal_metrics, f, indent=4)

            # Generate signal analysis plots
            self._plot_signal_analysis(signals_df)

        except Exception as e:
            logger.error(f"Error analyzing signals: {str(e)}")

    def _plot_signal_analysis(self, signals_df: pd.DataFrame):
        """Generate signal analysis plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Plot 1: Signal strength distribution
            sns.histplot(signals_df["signals"], kde=True, ax=axes[0, 0])
            axes[0, 0].set_title("Signal Strength Distribution")
            axes[0, 0].set_xlabel("Signal Strength")

            # Plot 2: Signal over time
            for symbol in self.tickers:
                if symbol in signals_df.columns:
                    axes[0, 1].plot(signals_df.index, signals_df[symbol], label=symbol)
            axes[0, 1].set_title("Signals Over Time")
            axes[0, 1].set_xlabel("Time")
            axes[0, 1].set_ylabel("Signal Strength")
            axes[0, 1].legend()

            # Plot 3: Buy/Sell/Hold signal distribution
            signal_types = pd.cut(
                signals_df["signals"],
                bins=[-np.inf, -0.1, 0.1, np.inf],
                labels=["Sell", "Hold", "Buy"],
            )
            signal_counts = signal_types.value_counts()
            axes[1, 0].bar(signal_counts.index, signal_counts.values)
            axes[1, 0].set_title("Signal Type Distribution")
            axes[1, 0].set_ylabel("Count")

            # Plot 4: Signal strength over time
            rolling_strength = abs(signals_df["signals"]).rolling(50).mean()
            axes[1, 1].plot(signals_df.index, rolling_strength)
            axes[1, 1].set_title("Rolling Signal Strength (50-period)")
            axes[1, 1].set_xlabel("Time")
            axes[1, 1].set_ylabel("Average Signal Strength")

            plt.tight_layout()
            plt.savefig("outputs/analysis/signal_analysis.png")
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting signal analysis: {str(e)}")


def main():
    """Main function to run paper trading simulation."""
    try:
        # Initialize simulation with all tickers
        logger.info(f"Starting simulation with {len(TICKERS)} tickers")

        # Create batches of tickers to avoid overwhelming the data feed
        batch_size = 10
        ticker_batches = [
            TICKERS[i : i + batch_size] for i in range(0, len(TICKERS), batch_size)
        ]

        simulations = []
        for i, ticker_batch in enumerate(ticker_batches):
            logger.info(
                f"Initializing batch {i + 1}/{len(ticker_batches)} with tickers: {ticker_batch}"
            )

            # Initialize simulation for this batch
            simulation = PaperTradingSimulation(
                tickers=ticker_batch,
                initial_capital=1000000.0
                / len(ticker_batches),  # Split capital across batches
                polling_interval=5,
                window_size=50,
            )

            # Start simulation
            simulation.start()
            simulations.append(simulation)

            # Wait briefly between batch starts to stagger data requests
            time.sleep(2)

        logger.info("All simulation batches started")

        # Keep running until interrupted
        try:
            while True:
                # Monitor all simulations
                total_equity = sum(sim.trading_engine.equity for sim in simulations)
                positions_count = sum(
                    len(sim.trading_engine.positions) for sim in simulations
                )

                logger.info(f"Total portfolio value: ${total_equity:,.2f}")
                logger.info(f"Total active positions: {positions_count}")

                time.sleep(60)  # Update status every minute

        except KeyboardInterrupt:
            logger.info("Stopping all simulations...")
            for sim in simulations:
                sim.stop()

            # Combine and save final results
            _combine_simulation_results(simulations)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


def _combine_simulation_results(simulations: List[PaperTradingSimulation]):
    """Combine results from multiple simulation batches."""
    try:
        # Combine performance metrics
        combined_metrics = {
            "portfolio_value": [],
            "returns": [],
            "positions": {},
            "timestamps": [],
        }

        for sim in simulations:
            # Add portfolio values
            combined_metrics["portfolio_value"].extend(
                sim.performance_metrics["portfolio_value"]
            )
            combined_metrics["returns"].extend(sim.performance_metrics["returns"])
            combined_metrics["timestamps"].extend(sim.performance_metrics["timestamps"])

            # Combine positions
            for positions in sim.performance_metrics["positions"]:
                for symbol, quantity in positions.items():
                    if symbol not in combined_metrics["positions"]:
                        combined_metrics["positions"][symbol] = []
                    combined_metrics["positions"][symbol].append(quantity)

        # Save combined metrics
        os.makedirs("outputs/combined_results", exist_ok=True)
        with open("outputs/combined_results/combined_metrics.json", "w") as f:
            json.dump(combined_metrics, f, indent=4, default=str)

        logger.info(
            "Combined simulation results saved to outputs/combined_results/combined_metrics.json"
        )

        # Generate combined performance plots
        _generate_combined_plots(combined_metrics)

    except Exception as e:
        logger.error(f"Error combining simulation results: {str(e)}")


def _generate_combined_plots(metrics: Dict):
    """Generate plots for combined simulation results."""
    try:
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))

        # Plot 1: Total portfolio value over time
        df = pd.DataFrame(
            {
                "timestamp": metrics["timestamps"],
                "portfolio_value": metrics["portfolio_value"],
            }
        )
        df.set_index("timestamp")["portfolio_value"].plot(ax=axes[0])
        axes[0].set_title("Combined Portfolio Value")
        axes[0].set_ylabel("Portfolio Value ($)")

        # Plot 2: Returns distribution
        returns = pd.Series(metrics["returns"])
        sns.histplot(returns, kde=True, ax=axes[1])
        axes[1].set_title("Combined Returns Distribution")
        axes[1].set_xlabel("Return")

        # Plot 3: Position values by symbol
        positions_df = pd.DataFrame(metrics["positions"])
        if not positions_df.empty:
            positions_df.plot(ax=axes[2])
            axes[2].set_title("Positions by Symbol")
            axes[2].set_xlabel("Time")
            axes[2].set_ylabel("Position Size")

        plt.tight_layout()
        plt.savefig("outputs/combined_results/combined_performance.png")
        plt.close()

        logger.info(
            "Combined performance plots saved to outputs/combined_results/combined_performance.png"
        )

    except Exception as e:
        logger.error(f"Error generating combined plots: {str(e)}")


if __name__ == "__main__":
    main()
