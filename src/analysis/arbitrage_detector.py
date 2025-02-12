"""
This module implements arbitrage signal detection using various market features and metrics.

The ArbitrageDetector class analyzes market data to identify potential arbitrage
opportunities by combining multiple features like volatility, correlation, momentum and
transfer entropy. It generates features and labels for machine learning models.

Key components:
- Feature generation from market data
- Label generation based on forward returns
- Regime detection and feature interactions
- Signal detection combining multiple metrics

Author: Ben Pfeffer
Created: 2024-01-15
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from src.utils.transfer_entropy import TransferEntropyCalculator
import logging

logger = logging.getLogger(__name__)


class ArbitrageDetector:
    """Rest of the class implementation remains the same..."""
