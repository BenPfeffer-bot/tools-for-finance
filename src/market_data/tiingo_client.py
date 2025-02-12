"""
Tiingo Forex WebSocket Client

This module implements a WebSocket client for real-time forex data from Tiingo.
"""

import websocket
import json
import logging
import threading
import time
from typing import Callable, Dict, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class TiingoForexClient:
    """Real-time forex data client using Tiingo WebSocket API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        pairs: list = None,
        on_message_callback: Optional[Callable] = None,
        reconnect_timeout: int = 5,
    ):
        """
        Initialize Tiingo Forex client.

        Args:
            api_key: Tiingo API key (will use env var TIINGO_API_KEY if not provided)
            pairs: List of forex pairs to subscribe to (e.g., ["eurusd", "gbpusd"])
            on_message_callback: Callback function for handling messages
            reconnect_timeout: Reconnection timeout in seconds
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("Tiingo API key not provided")

        self.pairs = [pair.lower() for pair in (pairs or [])]
        self.on_message_callback = on_message_callback
        self.reconnect_timeout = reconnect_timeout
        self.ws = None
        self.running = False
        self.ws_thread = None
        self.connected = False
        self.connection_event = threading.Event()

    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            if self.on_message_callback:
                self.on_message_callback(data)

            logger.debug(f"Received data: {data}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {str(error)}")
        self.connected = False
        self.connection_event.clear()

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        logger.info("WebSocket connection closed")
        self.connected = False
        self.connection_event.clear()

        if self.running:
            logger.info(
                f"Attempting to reconnect in {self.reconnect_timeout} seconds..."
            )
            threading.Timer(self.reconnect_timeout, self.connect).start()

    def _on_open(self, ws):
        """Handle WebSocket connection open."""
        logger.info("WebSocket connection established")
        self.connected = True
        self.connection_event.set()

        # Subscribe to forex pairs
        subscribe_message = {
            "eventName": "subscribe",
            "authorization": self.api_key,
            "eventData": {
                "thresholdLevel": 5,  # Update frequency level
                "tickers": self.pairs,
            },
        }

        try:
            ws.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to pairs: {self.pairs}")
        except Exception as e:
            logger.error(f"Error subscribing to pairs: {str(e)}")
            self.stop()

    def connect(self):
        """Establish WebSocket connection."""
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            "wss://api.tiingo.com/fx",
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )

    def start(self):
        """Start the WebSocket client."""
        if self.running:
            logger.warning("Client is already running")
            return

        self.running = True
        self.connect()

        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        # Wait for connection to be established
        if not self.connection_event.wait(timeout=10):
            logger.error("Timeout waiting for WebSocket connection")
            self.stop()
            return

        logger.info("Tiingo Forex client started")

    def stop(self):
        """Stop the WebSocket client."""
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {str(e)}")

        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)

        self.connected = False
        self.connection_event.clear()
        logger.info("Tiingo Forex client stopped")

    def subscribe_pairs(self, pairs: list):
        """Subscribe to additional forex pairs."""
        new_pairs = [pair.lower() for pair in pairs]
        self.pairs.extend(new_pairs)

        if self.connected and self.ws and self.ws.sock and self.ws.sock.connected:
            subscribe_message = {
                "eventName": "subscribe",
                "authorization": self.api_key,
                "eventData": {"thresholdLevel": 5, "tickers": new_pairs},
            }
            try:
                self.ws.send(json.dumps(subscribe_message))
                logger.info(f"Subscribed to additional pairs: {new_pairs}")
            except Exception as e:
                logger.error(f"Error subscribing to pairs: {str(e)}")

    def unsubscribe_pairs(self, pairs: list):
        """Unsubscribe from forex pairs."""
        pairs_to_remove = [pair.lower() for pair in pairs]

        if self.connected and self.ws and self.ws.sock and self.ws.sock.connected:
            unsubscribe_message = {
                "eventName": "unsubscribe",
                "authorization": self.api_key,
                "eventData": {"tickers": pairs_to_remove},
            }
            try:
                self.ws.send(json.dumps(unsubscribe_message))
                # Update local pairs list
                self.pairs = [
                    pair for pair in self.pairs if pair not in pairs_to_remove
                ]
                logger.info(f"Unsubscribed from pairs: {pairs_to_remove}")
            except Exception as e:
                logger.error(f"Error unsubscribing from pairs: {str(e)}")
