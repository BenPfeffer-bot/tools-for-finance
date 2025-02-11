# backtester.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Backtester:
    def __init__(self, returns, model, features, dates, capital=10000):
        self.returns = returns
        self.model = model
        self.features = features
        self.dates = dates
        self.capital = capital

    def run_backtest(self):
        # Predict positions using the ML model
        positions = self.model.predict(self.features)

        # Align positions with returns dates
        positions_series = pd.Series(positions, index=self.dates)

        # Align returns
        strategy_returns = (
            positions_series.shift(1) * self.returns.loc[self.dates].iloc[:, 0]
        )
        strategy_returns = strategy_returns.dropna()

        # Compute cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod() * self.capital

        # Plot results
        plt.figure(figsize=(10, 5))
        plt.plot(
            cumulative_returns.index, cumulative_returns.values, label="ETA Strategy"
        )
        plt.title("Backtest Results")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.show()

        return cumulative_returns
