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

            # Calculate rolling volatility
            volatility = returns.mean(axis=1).rolling(window=self.window).std()

            # Generate labels: 1 if absolute return > 1.5 * volatility
            labels = (future_returns.abs() > 1.5 * volatility).astype(int)

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

        Returns:
            Tuple containing:
            - DataFrame of features
            - Series of binary labels
            - DatetimeIndex of dates
        """
        try:
            # Generate labels first (they will have fewer points due to forward-looking nature)
            labels = self.generate_labels()
            if labels.empty:
                logger.error("No valid labels generated")
                return pd.DataFrame(), pd.Series(), pd.DatetimeIndex([])

            # Compute features
            vol_features = self.compute_volatility_features()
            corr_features = self.compute_correlation_features()

            # Combine all features
            features = pd.concat([vol_features, corr_features], axis=1)

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
