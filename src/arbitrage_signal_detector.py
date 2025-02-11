# arbitrage_signal_detector.py
import numpy as np
import pandas as pd
from src.transfer_entropy import TransferEntropyCalculator


class ArbitrageSignalDetector:
    def __init__(self, returns, window=50, lag=1):
        self.returns = returns
        self.window = window
        self.lag = lag

    def detect_opportunities(self):
        features = []
        labels = []

        # Ensure returns are in a DataFrame
        returns_df = pd.DataFrame(self.returns)
        dates = returns_df.index[self.window + self.lag : -self.lag]

        for i in range(self.window + self.lag, len(returns_df) - self.lag):
            X = returns_df.iloc[i - self.window - self.lag : i - self.lag, 0]
            Y = returns_df.iloc[i - self.window : i, 0]

            # Compute transfer entropy from X to Y and Y to X
            TE_XY = TransferEntropyCalculator.compute_transfer_entropy(
                X.values, Y.values
            )
            TE_YX = TransferEntropyCalculator.compute_transfer_entropy(
                Y.values, X.values
            )

            # TE difference as feature
            TE_diff = TE_XY - TE_YX

            # Store features
            features.append([TE_XY, TE_YX, TE_diff])

            # Generate label based on future return
            future_return = returns_df.iloc[i + self.lag, 0]
            label = 1 if future_return > 0 else 0
            labels.append(label)

        features = np.array(features)
        labels = np.array(labels)

        return features, labels, dates
