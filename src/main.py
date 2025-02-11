# main.py
import numpy as np
import pandas as pd
from src.data_loader import DataLoader
from src.eigenportfolio_builder import EigenportfolioBuilder
from src.arbitrage_signal_detector import ArbitrageSignalDetector
from src.ml_model_trainer import MLModelTrainer
from src.backtester import Backtester
from src.config import TICKERS

if __name__ == "__main__":
    # Step 1: Data Loading
    data_loader = DataLoader(tickers=TICKERS)
    df = data_loader.load_data()

    if df is not None:
        # Preprocess data to get returns
        # Pivot the data to have dates as index and tickers as columns
        df_pivot = df.pivot(index="Date", columns="Ticker", values="Adj Close")
        returns = df_pivot.pct_change().dropna()

        # Step 2: Eigenportfolio Construction
        eigen_builder = EigenportfolioBuilder(returns, n_components=5)
        eigenportfolios, _ = eigen_builder.compute_eigenportfolios()
        eigen_returns = pd.DataFrame(
            np.dot(returns, eigenportfolios.T), index=returns.index
        )

        # Step 3: Arbitrage Signal Detection
        signal_detector = ArbitrageSignalDetector(
            eigen_returns.iloc[:, :1]
        )  # Using the first eigenportfolio
        features, labels, dates = signal_detector.detect_opportunities()

        # Step 4: Machine Learning Model Training
        model_trainer = MLModelTrainer()
        model = model_trainer.train_model(features, labels)

        # Step 5: Backtesting
        backtester = Backtester(eigen_returns, model, features, dates)
        portfolio = backtester.run_backtest()

        # Print Portfolio Performance Metrics
        print("\nFinal Portfolio Value:", portfolio.iloc[-1])
        returns = portfolio.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        print("Sharpe Ratio:", sharpe_ratio)

    else:
        print("Data loading failed.")
