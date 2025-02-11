"""
This module implements arbitrage signal detection using various market features and metrics.

The ArbitrageSignalDetector class analyzes market data to identify potential arbitrage
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
from transfer_entropy import TransferEntropyCalculator
import logging

logger = logging.getLogger(__name__)


class ArbitrageSignalDetector:
    """
    Detects arbitrage opportunities using market features and machine learning.

    Analyzes market data to identify potential arbitrage opportunities by combining
    multiple features like volatility, correlation, momentum and transfer entropy.
    Generates features and labels for training ML models.

    Attributes:
        returns (pd.DataFrame): Asset returns data
        eigenportfolios (np.ndarray): Eigenportfolio weights
        window (int): Rolling window size for computations
        lag (int): Time lag for signal detection
        te_threshold (float): Z-score threshold for transfer entropy signals
        vol_threshold (float): Volatility threshold for signal filtering
        te_calculator (TransferEntropyCalculator): For computing transfer entropy
        scaler (StandardScaler): For feature scaling
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        eigenportfolios: np.ndarray,
        window: int = 50,
        lag: int = 1,
        te_threshold: float = 2.0,
        vol_threshold: float = 1.5,
    ):
        """
        Initialize ArbitrageSignalDetector.

        Args:
            returns: DataFrame of asset returns
            eigenportfolios: Array of eigenportfolio weights
            window: Rolling window size for computations
            lag: Time lag for signal detection
            te_threshold: Z-score threshold for transfer entropy signals
            vol_threshold: Volatility threshold for signal filtering
        """
        self.returns = returns
        self.eigenportfolios = eigenportfolios
        self.window = window
        self.lag = lag
        self.te_threshold = te_threshold
        self.vol_threshold = vol_threshold
        self.te_calculator = TransferEntropyCalculator()
        self.scaler = StandardScaler()

    def compute_volatility_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volatility-based features.

        Calculates various volatility metrics including:
        - Rolling volatility
        - Volatility regime indicators
        - Cross-sectional volatility features

        Args:
            returns: DataFrame of returns

        Returns:
            DataFrame of volatility features
        """
        features = pd.DataFrame(index=returns.index)

        # Rolling volatility
        vol = returns.rolling(window=20).std()
        mean_vol = vol.mean(axis=1)
        features["mean_volatility"] = mean_vol

        # Volatility regime
        vol_ma = mean_vol.rolling(window=60).mean()
        features["vol_regime"] = (mean_vol > vol_ma).astype(int)

        # Cross-sectional features
        features["vol_dispersion"] = vol.std(axis=1)
        features["vol_skew"] = vol.skew(axis=1)

        return features

    def compute_correlation_features(self) -> pd.DataFrame:
        """
        Compute correlation-based features from returns data.

        Calculates:
        - Pairwise correlations between eigenportfolios
        - Aggregate correlation measures
        - Correlation dispersion metrics

        Returns:
            DataFrame containing correlation features
        """
        features = pd.DataFrame(index=self.returns.index)

        # Compute rolling correlations between eigenportfolios
        for i, col1 in enumerate(self.returns.columns):
            for j, col2 in enumerate(self.returns.columns):
                if i < j:  # Only compute upper triangle to avoid redundancy
                    corr = (
                        self.returns[col1]
                        .rolling(window=self.window)
                        .corr(self.returns[col2])
                    )
                    features[f"corr_{col1}_{col2}"] = corr

        # Add aggregate correlation measures
        features["mean_correlation"] = features.mean(axis=1)
        features["correlation_dispersion"] = features.std(axis=1)

        return features

    def compute_te_features(self) -> pd.DataFrame:
        """
        Compute transfer entropy based features.

        Calculates:
        - Pairwise transfer entropy between assets
        - Causality shift detection
        - Transfer entropy z-scores

        Returns:
            DataFrame with transfer entropy features
        """
        features = pd.DataFrame(index=self.returns.index)

        if isinstance(self.returns, pd.DataFrame) and self.returns.shape[1] > 1:
            for i in range(self.returns.shape[1]):
                for j in range(i + 1, self.returns.shape[1]):
                    col_i = self.returns.iloc[:, i]
                    col_j = self.returns.iloc[:, j]

                    # Compute rolling transfer entropy
                    te_ij = self.te_calculator.compute_rolling_te(
                        col_i.values, col_j.values, window=self.window, lag=self.lag
                    )

                    # Store features
                    pair_name = f"TE_{i}_{j}"
                    features[pair_name] = te_ij

                    # Detect causality shifts
                    shift_detected, z_score = self.te_calculator.detect_causality_shift(
                        col_i.values,
                        col_j.values,
                        window=self.window,
                        threshold=self.te_threshold,
                    )
                    features[f"{pair_name}_shift"] = shift_detected
                    features[f"{pair_name}_zscore"] = z_score

        return features

    def compute_momentum_features(
        self, returns: pd.DataFrame, windows: list = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Compute momentum features for each eigenportfolio.

        Calculates rolling mean returns over multiple lookback windows
        to capture momentum effects at different time scales.

        Args:
            returns: DataFrame of returns
            windows: List of lookback windows

        Returns:
            DataFrame of momentum features
        """
        features = pd.DataFrame(index=returns.index)

        for window in windows:
            # Compute rolling mean returns
            rolling_means = returns.rolling(window=window).mean()

            # Add features for each factor
            for i in range(returns.shape[1]):
                features[f"momentum_{window}d_Factor_{i + 1}"] = rolling_means.iloc[
                    :, i
                ]

        return features

    def compute_rsi(self, returns: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Compute RSI for each eigenportfolio.

        Calculates the Relative Strength Index technical indicator
        to identify overbought/oversold conditions.

        Args:
            returns: DataFrame of returns
            window: Lookback window

        Returns:
            DataFrame of RSI values
        """
        features = pd.DataFrame(index=returns.index)

        for i in range(returns.shape[1]):
            # Separate gains and losses
            gains = returns.iloc[:, i].copy()
            losses = returns.iloc[:, i].copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)

            # Compute RSI
            avg_gain = gains.rolling(window=window).mean()
            avg_loss = losses.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            features[f"rsi_Factor_{i + 1}"] = rsi

        return features

    def generate_labels(
        self, returns: pd.DataFrame, threshold: float = 0.02
    ) -> pd.Series:
        """
        Generate trading signals based on future returns.

        Creates binary labels by comparing forward returns
        to a threshold value.

        Args:
            returns: DataFrame of returns
            threshold: Return threshold for positive label

        Returns:
            Series of binary labels
        """
        # First shift returns to get future values
        future_returns = returns.shift(-5)

        # Then compute rolling sum of the shifted returns
        forward_returns = future_returns.rolling(window=5).sum()

        # Generate labels based on mean forward return
        mean_forward_returns = forward_returns.mean(axis=1)
        labels = (mean_forward_returns > threshold).astype(int)

        # Remove labels that use future data
        labels = labels[:-5]

        n_positive = labels.sum()
        logger.info(
            f"Generated {len(labels)} labels with {n_positive} positive cases ({n_positive / len(labels):.1%} positive rate)"
        )

        return labels

    def generate_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate features and labels for machine learning.

        Combines multiple feature sets and aligns them with labels
        for training ML models.

        Returns:
            tuple: (features array, labels array)
        """
        # Project returns onto eigenportfolios
        factor_returns = pd.DataFrame(
            np.dot(self.returns, self.eigenportfolios.T), index=self.returns.index
        )

        # Compute features
        momentum_features = self.compute_momentum_features(factor_returns)
        rsi_features = self.compute_rsi(factor_returns)
        vol_features = self.compute_volatility_features(factor_returns)

        # Combine all features
        features = pd.concat([momentum_features, rsi_features, vol_features], axis=1)

        # Generate labels
        labels = self.generate_labels(factor_returns)

        # Align features and labels
        features = features[:-5]  # Remove last 5 rows that don't have labels
        features = features.dropna()  # Remove any rows with missing features
        labels = labels[features.index]  # Align labels with features

        logger.info(
            f"Generated {len(features)} samples with {features.shape[1]} features"
        )

        return features.values, labels.values

    def detect_opportunities(self) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        """
        Detect arbitrage opportunities using various features and metrics.

        Combines multiple feature sets and market regime indicators
        to identify potential arbitrage opportunities.

        Returns:
            tuple: (features DataFrame, labels Series, timestamps Index)
        """
        try:
            # Generate labels first
            labels = self.generate_labels(self.returns)
            if labels.empty:
                logger.error("No valid labels generated")
                return pd.DataFrame(), pd.Series(), pd.DatetimeIndex([])

            # Compute features
            vol_features = self.compute_volatility_features(self.returns)
            corr_features = self.compute_correlation_features()
            momentum_features = self.compute_momentum_features(self.returns)

            # Add market regime features
            regime_features = pd.DataFrame(index=self.returns.index)

            # Market trend regime
            returns_ma = self.returns.mean(axis=1).rolling(window=50).mean()
            regime_features["market_trend"] = (returns_ma > 0).astype(int)

            # Market volatility regime
            vol_ma = vol_features["mean_volatility"].rolling(window=50).mean()
            regime_features["volatility_trend"] = (
                vol_features["mean_volatility"] > vol_ma
            ).astype(int)

            # Market correlation regime
            corr_ma = corr_features["mean_correlation"].rolling(window=50).mean()
            regime_features["correlation_regime"] = (
                corr_features["mean_correlation"] > corr_ma
            ).astype(int)

            # Combine all features
            features = pd.concat(
                [
                    vol_features,
                    corr_features,
                    momentum_features,
                    regime_features,
                ],
                axis=1,
            )

            # Add feature interactions
            features["vol_momentum"] = (
                features["mean_volatility"] * features["momentum_5d_Factor_1"]
            )
            features["vol_correlation"] = (
                features["mean_volatility"] * features["mean_correlation"]
            )
            features["momentum_regime"] = (
                features["momentum_20d_Factor_1"] * features["market_trend"]
            )

            # Align features with labels
            common_index = features.index.intersection(labels.index)
            features = features.loc[common_index]
            labels = labels.loc[common_index]

            # Drop any rows with NaN values
            valid_mask = ~(features.isna().any(axis=1) | labels.isna())
            features = features[valid_mask]
            labels = labels[valid_mask]
            common_index = common_index[valid_mask]

            if features.empty or labels.empty:
                logger.error("No valid samples after cleaning")
                return pd.DataFrame(), pd.Series(), pd.DatetimeIndex([])

            logger.info(
                f"Generated {len(features)} samples with {features.shape[1]} features"
            )

            return features, labels, common_index

        except Exception as e:
            logger.error(f"Error detecting opportunities: {str(e)}")
            raise
