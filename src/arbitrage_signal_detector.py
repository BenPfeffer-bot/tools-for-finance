# arbitrage_signal_detector.py
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from src.transfer_entropy import TransferEntropyCalculator
import logging

logger = logging.getLogger(__name__)


class ArbitrageSignalDetector:
    def __init__(
        self,
        returns: pd.DataFrame,
        window: int = 50,
        lag: int = 1,
        te_threshold: float = 2.0,
        vol_threshold: float = 1.5,
    ):
        """
        Initialize ArbitrageSignalDetector.

        Args:
            returns: DataFrame of asset/factor returns
            window: Rolling window size for computations
            lag: Time lag for signal detection
            te_threshold: Z-score threshold for transfer entropy signals
            vol_threshold: Volatility threshold for signal filtering
        """
        self.returns = returns
        self.window = window
        self.lag = lag
        self.te_threshold = te_threshold
        self.vol_threshold = vol_threshold
        self.te_calculator = TransferEntropyCalculator()
        self.scaler = StandardScaler()

    def compute_volatility_features(self) -> pd.DataFrame:
        """
        Compute volatility-based features from returns data.

        Returns:
            DataFrame containing volatility features
        """
        features = pd.DataFrame(index=self.returns.index)

        # Compute rolling volatility for each eigenportfolio
        for col in self.returns.columns:
            features[f"volatility_{col}"] = (
                self.returns[col].rolling(window=self.window, min_periods=1).std()
            )

        # Add aggregate volatility measures
        features["mean_volatility"] = features.mean(axis=1)
        features["max_volatility"] = features.max(axis=1)
        features["volatility_dispersion"] = features.std(axis=1)

        # Add volatility regime indicators
        vol_ma = features["mean_volatility"].rolling(window=20).mean()
        vol_std = features["mean_volatility"].rolling(window=20).std()
        features["vol_regime"] = (features["mean_volatility"] - vol_ma) / vol_std
        features["high_vol_regime"] = (features["vol_regime"] > 1.0).astype(int)
        features["low_vol_regime"] = (features["vol_regime"] < -1.0).astype(int)

        return features

    def compute_correlation_features(self) -> pd.DataFrame:
        """
        Compute correlation-based features from returns data.

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
        """Compute transfer entropy based features."""
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

    def compute_momentum_features(self) -> pd.DataFrame:
        """
        Compute momentum-based features.
        """
        features = pd.DataFrame(index=self.returns.index)

        # Compute various momentum indicators
        for col in self.returns.columns:
            # Short-term momentum (5, 10 days)
            for window in [5, 10]:
                features[f"momentum_{window}d_{col}"] = (
                    self.returns[col].rolling(window=window).mean()
                )

            # Medium-term momentum (20, 50 days)
            for window in [20, 50]:
                features[f"momentum_{window}d_{col}"] = (
                    self.returns[col].rolling(window=window).mean()
                )

            # Relative strength index
            delta = self.returns[col]
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features[f"rsi_{col}"] = 100 - (100 / (1 + rs))

        return features

    def generate_labels(self, returns: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate binary labels based on future returns.

        Args:
            returns: DataFrame of returns (uses self.returns if None)

        Returns:
            Series of binary labels
        """
        if returns is None:
            returns = self.returns

        if returns.empty:
            logger.error("No returns data for label generation")
            return pd.Series()

        try:
            # Calculate future returns (mean across all eigenportfolios)
            future_returns = returns.mean(axis=1).shift(-self.lag)

            # Calculate rolling volatility with shorter window
            volatility = returns.mean(axis=1).rolling(window=20).std()

            # Generate labels with adaptive threshold
            # Signal if return exceeds 1.0 * volatility (more signals)
            # or if return is less than -1.0 * volatility (catch downside moves)
            labels = (
                (future_returns.abs() > 1.0 * volatility)
                & (future_returns.abs() > 0.001)
            ).astype(int)  # Minimum return threshold

            # Remove NaN values
            labels = labels.dropna()

            # Log label statistics
            positive_labels = labels.sum()
            total_labels = len(labels)
            logger.info(
                f"Generated {total_labels} labels with {positive_labels} positive cases "
                f"({positive_labels / total_labels:.1%} positive rate)"
            )

            return labels

        except Exception as e:
            logger.error(f"Error generating labels: {str(e)}")
            raise

    def detect_opportunities(self) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        """
        Detect arbitrage opportunities using various features and metrics.
        """
        try:
            # Generate labels first
            labels = self.generate_labels()
            if labels.empty:
                logger.error("No valid labels generated")
                return pd.DataFrame(), pd.Series(), pd.DatetimeIndex([])

            # Compute features
            vol_features = self.compute_volatility_features()
            corr_features = self.compute_correlation_features()
            momentum_features = self.compute_momentum_features()

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
