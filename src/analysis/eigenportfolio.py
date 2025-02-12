"""
Eigenportfolio Analysis Module

This module implements eigenportfolio analysis using Principal Component Analysis (PCA).
It provides functionality for both analyzing and constructing eigenportfolios from asset returns.

Key Features:
- Eigenportfolio computation using PCA
- Variance analysis and component selection
- Portfolio reconstruction and projection
- Portfolio weight optimization
- Performance analysis and visualization

Author: Ben Pfeffer
Created: 2024-01-15
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import logging
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Eigenportfolio:
    """
    A class for analyzing and constructing eigenportfolios using PCA.

    This class combines analysis and construction of eigenportfolios,
    providing a complete toolkit for portfolio decomposition and optimization.

    Attributes:
        returns (pd.DataFrame): Asset returns data (dates x assets)
        n_components (int): Number of eigenportfolios to construct
        min_explained_variance (float): Minimum cumulative variance to explain
        pca (PCA): Fitted PCA model
        eigenportfolios (np.ndarray): Computed eigenportfolio weights
        explained_variance_ratio (np.ndarray): Explained variance ratios
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        n_components: int = 5,
        min_explained_variance: float = 0.95,
    ):
        """
        Initialize Eigenportfolio.

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
        self.feature_names = returns.columns.tolist()

    def compute_covariance_matrix(self) -> np.ndarray:
        """
        Compute the covariance matrix of returns.

        Returns:
            Covariance matrix of asset returns
        """
        return np.cov(self.returns.T)

    def compute_eigenportfolios(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenportfolios using PCA.

        This method performs PCA on the returns data to extract eigenportfolios
        and their corresponding explained variance ratios.

        Returns:
            tuple: (eigenportfolios, explained_variance_ratio)
        """
        # Initialize PCA
        self.pca = PCA(n_components=self.n_components)

        # Fit PCA to returns
        self.pca.fit(self.returns)

        # Store results
        self.eigenportfolios = self.pca.components_
        self.explained_variance_ratio = self.pca.explained_variance_ratio_

        # Log cumulative variance explained
        cumulative_variance = np.cumsum(self.explained_variance_ratio)
        logger.info(
            f"Cumulative variance explained by {self.n_components} components: {cumulative_variance[-1]:.4f}"
        )

        return self.eigenportfolios, self.explained_variance_ratio

    def analyze_variance_explained(self) -> None:
        """
        Analyze and log the variance explained by each component.

        This method provides detailed analysis of how much variance
        each eigenportfolio explains, both individually and cumulatively.
        """
        if self.explained_variance_ratio is None:
            raise ValueError("Must compute eigenportfolios first")

        cumulative_var = np.cumsum(self.explained_variance_ratio)

        logger.info("\nEigenportfolio Analysis:")
        for i, (var_ratio, cum_var) in enumerate(
            zip(self.explained_variance_ratio, cumulative_var)
        ):
            logger.info(
                f"PC{i + 1}: {var_ratio:.1%} explained ({cum_var:.1%} cumulative)"
            )

    def get_optimal_n_components(self) -> int:
        """
        Determine optimal number of components based on explained variance.

        Returns:
            Optimal number of components
        """
        if self.explained_variance_ratio is None:
            raise ValueError("Must compute eigenportfolios first")

        cumsum = np.cumsum(self.explained_variance_ratio)
        n_optimal = np.argmax(cumsum >= self.min_explained_variance) + 1
        return n_optimal

    def project_returns(self) -> pd.DataFrame:
        """
        Project returns onto eigenportfolios.

        This method projects the original returns onto the eigenportfolio
        basis, effectively decomposing returns into factor exposures.

        Returns:
            DataFrame of projected returns with meaningful column names
        """
        if self.pca is None:
            raise ValueError("Must compute eigenportfolios first")

        # Project returns onto principal components
        projected_returns = self.pca.transform(self.returns)

        # Create DataFrame with meaningful column names
        columns = [f"PC{i + 1}" for i in range(projected_returns.shape[1])]
        return pd.DataFrame(
            projected_returns, index=self.returns.index, columns=columns
        )

    def get_eigenportfolio_weights(self, component_idx: int) -> pd.Series:
        """
        Get the weights of a specific eigenportfolio.

        Args:
            component_idx: Index of the eigenportfolio (0-based)

        Returns:
            Series of normalized portfolio weights
        """
        if self.eigenportfolios is None:
            raise ValueError("Must compute eigenportfolios first")

        weights = pd.Series(
            self.eigenportfolios[component_idx], index=self.feature_names
        )
        return weights / np.abs(weights).sum()  # Normalize to sum to 1

    def reconstruct_returns(self, n_components: Optional[int] = None) -> pd.DataFrame:
        """
        Reconstruct returns from top n_components eigenportfolios.

        This method reconstructs the original returns using a subset of
        eigenportfolios, which can be used for noise reduction.

        Args:
            n_components: Number of components to use (default: all)

        Returns:
            DataFrame of reconstructed returns
        """
        if self.pca is None:
            raise ValueError("Must compute eigenportfolios first")

        if n_components is None:
            n_components = self.n_components

        # Get factor returns and limit to n_components
        factor_returns = self.project_returns().iloc[:, :n_components]

        # Reconstruct using inverse transform
        reconstructed = self.pca.inverse_transform(factor_returns)
        return pd.DataFrame(
            reconstructed, index=self.returns.index, columns=self.returns.columns
        )

    def plot_variance_explained(self, save_path: Optional[str] = None) -> None:
        """
        Plot cumulative variance explained.

        Args:
            save_path: Path to save the plot (optional)
        """
        if self.explained_variance_ratio is None:
            raise ValueError("Must compute eigenportfolios first")

        plt.figure(figsize=(10, 6))
        cumulative_var = np.cumsum(self.explained_variance_ratio)
        plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, "bo-")
        plt.axhline(y=self.min_explained_variance, color="r", linestyle="--")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.title("Explained Variance vs Number of Components")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_eigenportfolio_weights(
        self, component_idx: int, save_path: Optional[str] = None
    ) -> None:
        """
        Plot weights of a specific eigenportfolio.

        Args:
            component_idx: Index of the eigenportfolio to plot
            save_path: Path to save the plot (optional)
        """
        weights = self.get_eigenportfolio_weights(component_idx)

        plt.figure(figsize=(12, 6))
        weights.plot(kind="bar")
        plt.title(f"Eigenportfolio {component_idx + 1} Weights")
        plt.xlabel("Assets")
        plt.ylabel("Weight")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
