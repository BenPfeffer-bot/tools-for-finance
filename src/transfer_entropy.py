# transfer_entropy.py
import numpy as np
from scipy.stats import entropy


class TransferEntropyCalculator:
    @staticmethod
    def compute_transfer_entropy(X, Y, bins=10):
        joint_hist, _, _ = np.histogram2d(X[:-1], Y[1:], bins=bins)
        joint_prob = joint_hist / np.sum(joint_hist)

        X_hist, _ = np.histogram(X[:-1], bins=bins)
        Y_hist, _ = np.histogram(Y[1:], bins=bins)
        P_X = X_hist / np.sum(X_hist)
        P_Y = Y_hist / np.sum(Y_hist)

        TE = entropy(joint_prob.flatten()) - entropy(P_X) - entropy(P_Y)
        return TE
