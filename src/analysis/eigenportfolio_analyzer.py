import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class EigenportfolioAnalyzer:
    """
    Class for analyzing eigenportfolios using PCA.
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize EigenportfolioAnalyzer.

        Args:
            returns: DataFrame of asset returns (dates x assets)
        """
        self.returns = returns
        self.pca = None
        self.eigenportfolios = None
        self.explained_variance_ratio = None

    def compute_eigenportfolios(self, n_components: int = 5) -> np.ndarray:
        """
        Compute eigenportfolios using PCA.

        Args:
            n_components: Number of components to compute

        Returns:
            Array of eigenportfolio weights
        """
        # Ensure n_components doesn't exceed number of assets
        n_components = min(n_components, self.returns.shape[1] - 1)

        # Fit PCA
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.returns)

        # Store results
        self.eigenportfolios = self.pca.components_
        self.explained_variance_ratio = self.pca.explained_variance_ratio_

        # Log cumulative variance explained
        cumulative_var = np.cumsum(self.explained_variance_ratio)
        print(
            f"Cumulative variance explained by {n_components} components: {cumulative_var[-1]:.4f}"
        )

        return self.eigenportfolios

    def analyze_variance_explained(self) -> None:
        """
        Analyze and log the variance explained by each component.
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

    def project_returns(self) -> pd.DataFrame:
        """
        Project returns onto eigenportfolios.

        Returns:
            DataFrame of projected returns
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
