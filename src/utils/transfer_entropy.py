# transfer_entropy.py
import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import Tuple, Optional
from sklearn.neighbors import KernelDensity


class TransferEntropyCalculator:
    def __init__(self, bins: int = 10, kde_bandwidth: float = 0.1):
        """
        Initialize TransferEntropyCalculator.

        Args:
            bins: Number of bins for histogram-based estimation
            kde_bandwidth: Bandwidth for kernel density estimation
        """
        self.bins = bins
        self.kde_bandwidth = kde_bandwidth

    def estimate_entropy(self, X: np.ndarray) -> float:
        """
        Estimate Shannon entropy using kernel density estimation.

        Args:
            X: Input time series
        """
        kde = KernelDensity(bandwidth=self.kde_bandwidth)
        kde.fit(X.reshape(-1, 1))
        log_dens = kde.score_samples(X.reshape(-1, 1))
        return -np.mean(log_dens)

    @staticmethod
    def compute_transfer_entropy(
        X: np.ndarray, Y: np.ndarray, bins: int = 10, lag: int = 1
    ) -> float:
        """
        Compute transfer entropy from X to Y.

        Args:
            X: Source time series
            Y: Target time series
            bins: Number of bins for discretization
            lag: Time lag for causality detection

        Returns:
            Transfer entropy value
        """
        # Ensure arrays are the same length and shift by lag
        X = X[:-lag]
        Y_future = Y[lag:]
        Y_past = Y[:-lag]

        # Compute joint and marginal histograms
        joint_hist, _, _ = np.histogram2d(X, Y_future, bins=bins)
        joint_prob = joint_hist / np.sum(joint_hist)

        X_hist, _ = np.histogram(X, bins=bins)
        Y_hist, _ = np.histogram(Y_future, bins=bins)

        P_X = X_hist / np.sum(X_hist)
        P_Y = Y_hist / np.sum(Y_hist)

        # Compute transfer entropy
        TE = entropy(joint_prob.flatten()) - entropy(P_X) - entropy(P_Y)
        return TE

    def compute_bidirectional_te(
        self, X: np.ndarray, Y: np.ndarray, lag: int = 1
    ) -> Tuple[float, float]:
        """
        Compute transfer entropy in both directions.

        Args:
            X: First time series
            Y: Second time series
            lag: Time lag for causality detection

        Returns:
            Tuple of (TE from X to Y, TE from Y to X)
        """
        te_xy = self.compute_transfer_entropy(X, Y, self.bins, lag)
        te_yx = self.compute_transfer_entropy(Y, X, self.bins, lag)
        return te_xy, te_yx

    def compute_net_transfer_entropy(
        self, X: np.ndarray, Y: np.ndarray, lag: int = 1
    ) -> float:
        """
        Compute net transfer entropy between X and Y.

        Args:
            X: First time series
            Y: Second time series
            lag: Time lag for causality detection

        Returns:
            Net transfer entropy (TE_XY - TE_YX)
        """
        te_xy, te_yx = self.compute_bidirectional_te(X, Y, lag)
        return te_xy - te_yx

    def compute_rolling_te(
        self, X: np.ndarray, Y: np.ndarray, window: int = 50, lag: int = 1
    ) -> pd.Series:
        """
        Compute rolling transfer entropy over time.

        Args:
            X: Source time series
            Y: Target time series
            window: Rolling window size
            lag: Time lag for causality detection

        Returns:
            Series of rolling transfer entropy values
        """
        te_values = []
        for i in range(window, len(X)):
            x_window = X[i - window : i]
            y_window = Y[i - window : i]
            te = self.compute_transfer_entropy(x_window, y_window, self.bins, lag)
            te_values.append(te)

        return pd.Series(te_values)

    def detect_causality_shift(
        self, X: np.ndarray, Y: np.ndarray, window: int = 50, threshold: float = 2.0
    ) -> Tuple[bool, float]:
        """
        Detect significant shifts in causality structure.

        Args:
            X: First time series
            Y: Second time series
            window: Rolling window size
            threshold: Z-score threshold for shift detection

        Returns:
            Tuple of (shift_detected, z_score)
        """
        rolling_te = self.compute_rolling_te(X, Y, window)
        if len(rolling_te) < 2:
            return False, 0.0

        # Compute z-score of latest TE value
        te_mean = rolling_te[:-1].mean()
        te_std = rolling_te[:-1].std()
        latest_zscore = (rolling_te.iloc[-1] - te_mean) / te_std

        shift_detected = abs(latest_zscore) > threshold
        return shift_detected, latest_zscore
