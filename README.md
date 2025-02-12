# Tools for Finance

A comprehensive Python toolkit for financial analysis and algorithmic trading.

## Overview

This project provides a collection of tools and modules for financial market analysis, algorithmic trading, and portfolio optimization. It includes functionality for:

- Market data collection and processing
- Statistical arbitrage detection
- Portfolio optimization using eigenportfolios
- Real-time trading simulation
- Machine learning and reinforcement learning models
- Backtesting and performance analysis

## Project Structure

```
tools-for-finance/
├── src/                    # Source code
│   ├── data/              # Data loading and processing
│   ├── analysis/          # Market analysis tools
│   ├── trading/           # Trading execution
│   ├── models/            # ML/RL models
│   ├── market_data/       # Market data clients
│   ├── backtesting/       # Backtesting framework
│   ├── simulation/        # Trading simulation
│   ├── config/            # Configuration
│   └── utils/             # Utilities
├── scripts/               # Running scripts
├── tests/                 # Test files
├── notebooks/             # Jupyter notebooks
├── data/                  # Data storage
├── outputs/               # Results and outputs
└── logs/                  # Log files
```

## Key Components

### Eigenportfolio Analysis

The `analysis.eigenportfolio` module provides tools for portfolio optimization using Principal Component Analysis (PCA). Key features include:

- Eigenportfolio computation and analysis
- Portfolio decomposition and reconstruction
- Variance analysis and visualization
- Optimal component selection

### Arbitrage Detection

The `analysis.arbitrage_detector` module implements statistical arbitrage detection using multiple approaches:

- Feature generation from market data
- Volatility and correlation analysis
- Transfer entropy computation
- Machine learning signal generation

### Trading Engine

The trading components provide:

- Paper trading simulation
- Real-time strategy execution
- Order management
- Position tracking
- Risk management

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tools-for-finance.git
cd tools-for-finance

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Usage Examples

### Eigenportfolio Analysis

```python
from src.analysis.eigenportfolio import Eigenportfolio
import pandas as pd

# Load returns data
returns = pd.read_csv("data/returns.csv", index_col=0)

# Initialize and compute eigenportfolios
eigen = Eigenportfolio(returns, n_components=5)
eigenportfolios, variance_ratios = eigen.compute_eigenportfolios()

# Analyze variance explained
eigen.analyze_variance_explained()

# Plot results
eigen.plot_variance_explained("outputs/plots/variance_explained.png")
```

### Arbitrage Detection

```python
from src.analysis.arbitrage_detector import ArbitrageDetector

# Initialize detector
detector = ArbitrageDetector(returns, eigenportfolios)

# Generate features and labels
features, labels = detector.generate_features_and_labels()

# Compute specific feature sets
vol_features = detector.compute_volatility_features(returns)
corr_features = detector.compute_correlation_features()
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Documentation

For detailed documentation of each module, please refer to the `docs/` directory or the docstrings in the source code.