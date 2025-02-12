# eigenportfolio_builder.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Tuple, List


class EigenportfolioBuilder:
    def __init__(
        self,
        returns: pd.DataFrame,
        n_components: int = 5,
        min_explained_variance: float = 0.95,
    ):
        """
        Initialize the EigenportfolioBuilder.

        Args:
            returns: DataFrame of asset returns
            n_components: Number of eigenportfolios to construct
            min_explained_variance: Minimum cumulative variance to explain
        """
        self.returns = returns
        self.n_components = min(n_components, len(returns.columns))
        self.min_explained_variance = min_explained_variance
        self.pca = None
        self.eigenportfolios = None
        self.explained_variance_ratio = None

    def compute_covariance_matrix(self) -> np.ndarray:
        """Compute the covariance matrix of returns."""
        return np.cov(self.returns.T)

    def compute_eigenportfolios(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenportfolios using PCA.

        Returns:
            Tuple of (eigenportfolios, explained_variance_ratio)
        """
        # Initialize PCA
        self.pca = PCA(n_components=self.n_components)

        # Fit PCA to returns
        self.pca.fit(self.returns)

        # Store results
        self.eigenportfolios = self.pca.components_
        self.explained_variance_ratio = self.pca.explained_variance_ratio_

        # Print variance explained
        cumulative_variance = np.cumsum(self.explained_variance_ratio)
        print(
            f"Cumulative variance explained by {self.n_components} components: {cumulative_variance[-1]:.4f}"
        )

        return self.eigenportfolios, self.explained_variance_ratio

    def get_optimal_n_components(self) -> int:
        """Determine optimal number of components based on explained variance."""
        cumsum = np.cumsum(self.explained_variance_ratio)
        n_optimal = np.argmax(cumsum >= self.min_explained_variance) + 1
        return n_optimal

    def project_returns_to_factors(self) -> pd.DataFrame:
        """Project original returns onto eigenportfolio factors."""
        if self.pca is None:
            raise ValueError("Must compute eigenportfolios first!")

        factor_returns = self.pca.transform(self.returns)
        return pd.DataFrame(
            factor_returns,
            index=self.returns.index,
            columns=[f"Factor_{i + 1}" for i in range(self.n_components)],
        )

    def reconstruct_returns(self, n_components: int = None) -> pd.DataFrame:
        """
        Reconstruct returns from top n_components eigenportfolios.

        Args:
            n_components: Number of components to use (default: all)
        """
        if self.pca is None:
            raise ValueError("Must compute eigenportfolios first!")

        if n_components is None:
            n_components = self.n_components

        # Get factor returns and limit to n_components
        factor_returns = self.project_returns_to_factors().iloc[:, :n_components]

        # Reconstruct using inverse transform
        reconstructed = self.pca.inverse_transform(factor_returns)
        return pd.DataFrame(
            reconstructed, index=self.returns.index, columns=self.returns.columns
        )

    def get_eigenportfolio_weights(self, component_idx: int) -> pd.Series:
        """
        Get the weights of a specific eigenportfolio.

        Args:
            component_idx: Index of the eigenportfolio (0-based)
        """
        if self.eigenportfolios is None:
            raise ValueError("Must compute eigenportfolios first!")

        weights = pd.Series(
            self.eigenportfolios[component_idx], index=self.returns.columns
        )
        return weights / np.abs(weights).sum()  # Normalize to sum to 1
