# Transfer Entropy Analysis

The transfer entropy module implements information flow analysis between time series using transfer entropy measures. This is particularly useful for detecting lead-lag relationships and causality in financial markets.

## Overview

Transfer entropy is a non-parametric measure of directed information flow between time series. It can detect:

- Lead-lag relationships
- Causality shifts
- Information transfer
- Market inefficiencies

## Key Features

- Transfer entropy computation
- Rolling transfer entropy analysis
- Causality shift detection
- Information flow metrics
- Kernel density estimation

## Usage Example

```python
from src.utils.transfer_entropy import TransferEntropyCalculator
import numpy as np

# Initialize calculator
calculator = TransferEntropyCalculator(bins=10, kde_bandwidth=0.1)

# Compute transfer entropy between two time series
te_xy = calculator.compute_transfer_entropy(X, Y, lag=1)

# Compute bidirectional transfer entropy
te_xy, te_yx = calculator.compute_bidirectional_te(X, Y, lag=1)

# Compute net transfer entropy
net_te = calculator.compute_net_transfer_entropy(X, Y, lag=1)

# Compute rolling transfer entropy
rolling_te = calculator.compute_rolling_te(X, Y, window=50, lag=1)

# Detect causality shifts
shift_detected, z_score = calculator.detect_causality_shift(
    X, Y, window=50, threshold=2.0
)
```

## Methods

### compute_transfer_entropy(X, Y, bins, lag)

Computes transfer entropy from X to Y using binned probability estimation.

Parameters:
- X: Source time series
- Y: Target time series
- bins: Number of bins for discretization
- lag: Time lag for causality detection

### compute_bidirectional_te(X, Y, lag)

Computes transfer entropy in both directions between X and Y.

Returns:
- Tuple of (TE from X to Y, TE from Y to X)

### compute_net_transfer_entropy(X, Y, lag)

Computes net transfer entropy (TE_XY - TE_YX) to determine dominant direction.

### compute_rolling_te(X, Y, window, lag)

Computes rolling transfer entropy over time using a sliding window.

### detect_causality_shift(X, Y, window, threshold)

Detects significant shifts in causality structure using z-scores.

## Mathematical Background

### Transfer Entropy Formula

Transfer entropy from X to Y is defined as:

```
TE(X→Y) = H(Y|Y_past) - H(Y|Y_past,X_past)
```

where:
- H(Y|Y_past) is the conditional entropy of Y given its past
- H(Y|Y_past,X_past) is the conditional entropy of Y given both its past and X's past

### Kernel Density Estimation

For continuous variables, probabilities are estimated using kernel density estimation:

```
p(x) = (1/nh) Σ K((x - xᵢ)/h)
```

where:
- K is the kernel function
- h is the bandwidth
- n is the number of samples

### Causality Shift Detection

Shifts are detected by:
1. Computing rolling transfer entropy
2. Calculating z-scores of recent values
3. Comparing against threshold

## Applications

Transfer entropy analysis can be used for:

1. Lead-lag relationship detection
2. Market efficiency analysis
3. Information flow networks
4. Risk spillover detection
5. Trading signal generation

## References

- Schreiber, T. (2000). Measuring information transfer. Physical Review Letters, 85(2), 461.
- Dimpfl, T., & Peter, F. J. (2013). Using transfer entropy to measure information flows between financial markets. Studies in Nonlinear Dynamics and Econometrics, 17(1), 85-102.
- Kwon, O., & Yang, J. S. (2008). Information flow between stock indices. EPL (Europhysics Letters), 82(6), 68003.