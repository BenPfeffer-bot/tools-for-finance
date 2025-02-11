# eigenportfolio_builder.py
import numpy as np
from sklearn.decomposition import PCA


class EigenportfolioBuilder:
    def __init__(self, returns, n_components=5):
        self.returns = returns
        self.n_components = n_components

    def compute_eigenportfolios(self):
        pca = PCA(n_components=self.n_components)
        pca.fit(self.returns)
        eigenportfolios = pca.components_
        return eigenportfolios, pca.explained_variance_ratio_
