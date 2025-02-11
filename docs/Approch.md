# Strategy

## Fundamentals Approach

1. Smart Beta Approach
2. Value Approach
3. Top-Down Approach 
4. Bottom-Up Approach

We're working with 50 stocks present in EUROSTOX50, the idea is first understanding what type of position we want to have.
I believe their are multiples approach we can follow, the first as private fund should orientate over long/short equity -- for this approach, we will look at fundamental analysis, economics analysis and news analysis to select a selection of 10 interesting stocks to implement this strategy.

On the other sides, they are others opportunities to implement on more active basis / daytrading. For this approach we will include technical analysis to take opportunites on short-term movements, we can also include news sentiments analysis. For the datasets, we need data with different timeframes, and different interval.  A first approach is taking into consideration atleast 3year of historical data on different timeframes (1m, 5m, 10m, 1d, 1w). 


+------------------------------+
|  1️⃣ Data Acquisition Layer  |
|  (data_fetcher.py)           |
+------------------------------+
            │
            ▼
+------------------------------+
|  2️⃣ Eigenportfolio Builder  |
|  (eigenportfolio_builder.py) |
|  - Computes PCA on returns   |
|  - Extracts market factors   |
+------------------------------+
            │
            ▼
+------------------------------+
|  3️⃣ Transfer Entropy Calc   |
|  (transfer_entropy.py)       |
|  - Computes TE shifts        |
|  - Identifies info asymmetry |
+------------------------------+
            │
            ▼
+------------------------------+
|  4️⃣ ML Signal Detection     |
|  (arbitrage_signal_ml.py)    |
|  - Feature engineering       |
|  - ML model prediction       |
|  - Reinforcement learning    |
+------------------------------+
            │
            ▼
+------------------------------+
|  5️⃣ Trade Execution Engine  |
|  (execution_engine.py)       |
|  - Order types (Market, Limit)|
|  - Smart Order Routing (SOR) |
|  - Risk management           |
+------------------------------+
            │
            ▼
+------------------------------+
|  6️⃣ Backtesting Engine      |
|  (backtester.py)             |
|  - Simulates past trading    |
|  - Computes Sharpe Ratio     |
|  - Evaluates strategy        |
+------------------------------+
            │
            ▼
+------------------------------+
|  7️⃣ Live Trading & Monitor  |
|  (live_trader.py)            |
|  - Fetch real-time data      |
|  - Automated execution       |
|  - Sends trade alerts        |
+------------------------------+
