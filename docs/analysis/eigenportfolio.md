# Eigenportfolio Analysis

The eigenportfolio analysis module provides tools for portfolio optimization using Principal Component Analysis (PCA). It combines analysis and construction of eigenportfolios, providing a complete toolkit for portfolio decomposition and optimization.

## Overview

Eigenportfolios are portfolios constructed from the principal components of asset returns. They represent uncorrelated risk factors that can be used for:

- Portfolio optimization
- Risk decomposition
- Statistical arbitrage
- Factor investing

## Key Features

- Eigenportfolio computation using PCA
- Portfolio decomposition and reconstruction
- Variance analysis and visualization
- Optimal component selection
- Portfolio weight optimization

## Usage Example

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

## Key Methods

### compute_eigenportfolios()

Computes eigenportfolios using PCA. Returns a tuple of:
- eigenportfolios: Array of eigenportfolio weights
- explained_variance_ratio: Explained variance ratios

### analyze_variance_explained()

Analyzes and logs the variance explained by each component, both individually and cumulatively.

### project_returns()

Projects returns onto eigenportfolios, effectively decomposing returns into factor exposures.

### get_eigenportfolio_weights(component_idx)

Gets the normalized weights of a specific eigenportfolio.

### reconstruct_returns(n_components)

Reconstructs returns from top n_components eigenportfolios, which can be used for noise reduction.

## Visualization Methods

### plot_variance_explained(save_path)

Plots cumulative variance explained vs number of components.

### plot_eigenportfolio_weights(component_idx, save_path)

Plots weights of a specific eigenportfolio.

## Mathematical Background

Eigenportfolios are computed using the following steps:

1. Calculate the covariance matrix of asset returns
2. Perform eigendecomposition using PCA
3. Extract eigenvectors (eigenportfolios) and eigenvalues (explained variance)
4. Normalize weights to create tradeable portfolios

The eigenportfolios are ordered by explained variance, with the first eigenportfolio explaining the most variance in the returns data.

## References

- Avellaneda, M., & Lee, J. H. (2010). Statistical arbitrage in the US equities market. Quantitative Finance, 10(7), 761-782.
- Partovi, M. H., & Caputo, M. (2004). Principal portfolios: Recasting the efficient frontier. Economics Bulletin, 7(3), 1-10.