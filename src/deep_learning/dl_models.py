import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class LSTMArbitrageDetector(nn.Module):
    """LSTM-based model for arbitrage signal detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = "gpu" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)

        # Output layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

        # Move model to device
        self.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention
        attn_out, attn_weights = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
        )
        attn_out = attn_out.transpose(0, 1)

        # Dense layers
        x = self.dropout(F.relu(self.fc1(attn_out[:, -1, :])))
        signal = torch.sigmoid(self.fc2(x))

        return signal, attn_weights


class ArbitrageAutoencoder(nn.Module):
    """Autoencoder for anomaly detection in arbitrage signals."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 50,
        encoding_dim: int = 128,
        latent_dim: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.input_dim = input_dim * sequence_length
        self.device = device

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, self.input_dim),
            nn.Sigmoid(),
        )

        # Move model to device
        self.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        return encoded, decoded

    def detect_anomalies(
        self, x: torch.Tensor, threshold: float = 2.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect anomalies based on reconstruction error."""
        # Flatten input if necessary
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        encoded, decoded = self.forward(x)
        reconstruction_error = F.mse_loss(decoded, x, reduction="none")
        mean_error = reconstruction_error.mean(dim=1)
        threshold = mean_error.mean() + threshold * mean_error.std()
        anomalies = mean_error > threshold
        return anomalies, reconstruction_error


class FeatureExtractor:
    """Extract and process features for deep learning models."""

    def __init__(
        self,
        window_size: int = 50,
        n_components: int = 5,
    ):
        self.window_size = window_size
        self.n_components = n_components

    def create_sequences(
        self,
        data: np.ndarray,
        window_size: Optional[int] = None,
    ) -> np.ndarray:
        """Create sequences for LSTM input."""
        if window_size is None:
            window_size = self.window_size

        sequences = []
        for i in range(len(data) - window_size + 1):
            sequences.append(data[i : i + window_size])
        return np.array(sequences)

    def extract_features(
        self,
        returns: pd.DataFrame,
        te_values: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract combined features for deep learning."""
        features = []
        feature_names = []

        # 1. Statistical features
        for col in returns.columns:
            # Volatility
            vol = returns[col].rolling(window=20).std()
            features.append(vol.values.reshape(-1, 1))
            feature_names.append(f"vol_{col}")

            # Momentum
            mom = returns[col].rolling(window=10).mean()
            features.append(mom.values.reshape(-1, 1))
            feature_names.append(f"mom_{col}")

            # Skewness
            skew = returns[col].rolling(window=50).skew()
            features.append(skew.values.reshape(-1, 1))
            feature_names.append(f"skew_{col}")

        # 2. Transfer Entropy features
        features.append(te_values)
        feature_names.extend([f"te_{i}" for i in range(te_values.shape[1])])

        # Combine features
        combined_features = np.hstack(features)

        return combined_features, feature_names
