import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Tuple, Dict
import json
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.deep_learning.dl_models import (
    LSTMArbitrageDetector,
    ArbitrageAutoencoder,
    FeatureExtractor,
)
from src.reinforcement.rl_models import TradingEnvironment, RLTrader
from src.database.data_loader import DataLoader
from src.analysis.eigenportfolio import Eigenportfolio
from src.analysis.arbitrage_detector import ArbitrageDetector
from src.config.settings import TICKERS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_dl_models(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = 50,
    batch_size: int = 32,
    epochs: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[LSTMArbitrageDetector, ArbitrageAutoencoder]:
    """Train deep learning models."""
    # Create sequences for LSTM
    feature_extractor = FeatureExtractor(window_size=window_size)
    sequences = feature_extractor.create_sequences(features)

    # Split data
    train_size = int(0.8 * len(sequences))
    train_sequences = sequences[:train_size]
    train_labels = labels[window_size : train_size + window_size]

    # Initialize models
    lstm_model = LSTMArbitrageDetector(
        input_dim=sequences.shape[2],
        hidden_dim=64,
    ).to(device)

    autoencoder = ArbitrageAutoencoder(
        input_dim=sequences.shape[2],
    ).to(device)

    # Training loop
    optimizer_lstm = torch.optim.Adam(lstm_model.parameters())
    optimizer_ae = torch.optim.Adam(autoencoder.parameters())

    lstm_losses = []
    ae_losses = []

    for epoch in range(epochs):
        # Train LSTM
        lstm_model.train()
        total_lstm_loss = 0

        for i in range(0, len(train_sequences), batch_size):
            batch_sequences = torch.FloatTensor(train_sequences[i : i + batch_size]).to(
                device
            )
            batch_labels = torch.FloatTensor(train_labels[i : i + batch_size]).to(
                device
            )

            optimizer_lstm.zero_grad()
            predictions, _ = lstm_model(batch_sequences)
            loss = F.binary_cross_entropy(predictions.squeeze(), batch_labels)
            loss.backward()
            optimizer_lstm.step()

            total_lstm_loss += loss.item()

        # Train Autoencoder
        autoencoder.train()
        total_ae_loss = 0

        for i in range(0, len(train_sequences), batch_size):
            batch_sequences = torch.FloatTensor(train_sequences[i : i + batch_size]).to(
                device
            )
            # Flatten the input for autoencoder
            flattened_input = batch_sequences.view(batch_sequences.size(0), -1)

            optimizer_ae.zero_grad()
            _, decoded = autoencoder(flattened_input)
            loss = F.mse_loss(decoded, flattened_input)
            loss.backward()
            optimizer_ae.step()

            total_ae_loss += loss.item()

        # Log progress
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"LSTM Loss: {total_lstm_loss / len(train_sequences):.4f} - "
                f"AE Loss: {total_ae_loss / len(train_sequences):.4f}"
            )

        lstm_losses.append(total_lstm_loss / len(train_sequences))
        ae_losses.append(total_ae_loss / len(train_sequences))

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lstm_losses)
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(ae_losses)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig("outputs/plots/dl_training_curves.png")
    plt.close()

    return lstm_model, autoencoder


def train_rl_agent(
    env: TradingEnvironment,
    state_dim: int,
    action_dim: int = 3,  # HOLD, BUY, SELL
    episodes: int = 100,
    max_steps: int = 1000,
    batch_size: int = 64,
    target_update: int = 10,
) -> RLTrader:
    """Train RL agent."""
    try:
        logger.info(
            f"Initializing RL agent with state_dim: {state_dim}, action_dim: {action_dim}"
        )
        agent = RLTrader(env, state_dim, action_dim)

        # Training metrics
        episode_rewards = []
        episode_losses = []

        for episode in range(episodes):
            logger.info(f"Starting episode {episode + 1}/{episodes}")
            state = env.reset()
            total_reward = 0
            total_loss = 0
            steps = 0

            try:
                for step in range(max_steps):
                    # Select and execute action
                    epsilon = max(
                        0.01, 0.1 - 0.09 * episode / episodes
                    )  # Decay epsilon
                    action = agent.select_action(state, epsilon)
                    next_state, reward, done, _ = env.step(action)

                    # Store experience
                    agent.replay_buffer.push(state, action, reward, next_state, done)
                    logger.debug(
                        f"Step {step + 1}: Action={action}, Reward={reward:.4f}"
                    )

                    # Train DQN
                    if len(agent.replay_buffer) > batch_size:
                        loss = agent.train_dqn(batch_size)
                        total_loss += loss
                        if step % 100 == 0:  # Log every 100 steps
                            logger.info(f"Step {step + 1}: Loss={loss:.4f}")

                    # Update state and reward
                    state = next_state
                    total_reward += reward
                    steps += 1

                    if done:
                        logger.info(
                            f"Episode {episode + 1} finished after {steps} steps"
                        )
                        break

                # Update target network
                if episode % target_update == 0:
                    agent.update_target_network()
                    logger.info(f"Updated target network at episode {episode + 1}")

                # Log progress
                episode_rewards.append(total_reward)
                avg_loss = total_loss / steps if steps > 0 else 0
                episode_losses.append(avg_loss)

                logger.info(
                    f"Episode {episode + 1}/{episodes} - "
                    f"Steps: {steps} - "
                    f"Reward: {total_reward:.2f} - "
                    f"Avg Loss: {avg_loss:.4f} - "
                    f"Epsilon: {epsilon:.3f}"
                )

            except Exception as e:
                logger.error(f"Error in episode {episode + 1}: {str(e)}")
                continue

        # Plot training curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")

        plt.subplot(1, 2, 2)
        plt.plot(episode_losses)
        plt.title("Average Loss per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        plt.tight_layout()
        plt.savefig("outputs/plots/rl_training_curves.png")
        plt.close()

        logger.info("RL training completed successfully")
        return agent

    except Exception as e:
        logger.error(f"Error in train_rl_agent: {str(e)}")
        raise


def evaluate_strategy(
    returns: pd.DataFrame,
    predictions: np.ndarray,
    agent: RLTrader,
    window_size: int = 50,
) -> Dict[str, float]:
    """Evaluate the complete strategy."""
    env = TradingEnvironment(
        returns=returns,
        predictions=predictions,
        window_size=window_size,
    )

    state = env.reset()
    done = False
    total_reward = 0
    portfolio_values = [env.balance]

    while not done:
        action = agent.select_action(
            state, epsilon=0
        )  # No exploration during evaluation
        state, reward, done, info = env.step(action)
        total_reward += reward
        portfolio_values.append(info["balance"])

    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    metrics = {
        "total_return": (portfolio_values[-1] / portfolio_values[0] - 1),
        "annual_return": (portfolio_values[-1] / portfolio_values[0])
        ** (252 / len(returns))
        - 1,
        "annual_volatility": np.std(returns) * np.sqrt(252),
        "sharpe_ratio": np.mean(returns) / np.std(returns) * np.sqrt(252),
        "max_drawdown": np.min(
            portfolio_values / np.maximum.accumulate(portfolio_values)
        )
        - 1,
        "total_reward": total_reward,
    }

    # Plot portfolio value
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values)
    plt.title("Portfolio Value")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig("outputs/plots/portfolio_value.png")
    plt.close()

    # Save metrics
    with open("outputs/reports/strategy_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    try:
        # Load and prepare data
        logger.info("Loading and preparing data...")
        data_loader = DataLoader(TICKERS)
        market_data = data_loader.load_data()
        returns = data_loader.prepare_returns_matrix()

        # Set window size for sequence creation
        window_size = 50
        logger.info(f"Using window size: {window_size}")

        # Compute eigenportfolios
        logger.info("Computing eigenportfolios...")
        eigen_analyzer = Eigenportfolio(returns)
        eigenportfolios = eigen_analyzer.compute_eigenportfolios()

        # Generate features and labels
        logger.info("Generating features and labels...")
        signal_detector = ArbitrageDetector(returns, eigenportfolios)
        features, labels = signal_detector.generate_features_and_labels()
        logger.info(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

        # Train deep learning models
        logger.info("Training deep learning models...")
        lstm_model, autoencoder = train_dl_models(
            features, labels, window_size=window_size
        )

        # Generate enhanced predictions
        logger.info("Generating predictions...")
        feature_sequences = FeatureExtractor(window_size=window_size).create_sequences(
            features
        )
        logger.info(f"Feature sequences shape: {feature_sequences.shape}")

        with torch.no_grad():
            # Get LSTM predictions
            lstm_predictions, _ = lstm_model(
                torch.FloatTensor(feature_sequences).to(lstm_model.device)
            )
            lstm_predictions = lstm_predictions.cpu().numpy().squeeze()
            logger.info(f"LSTM predictions shape: {lstm_predictions.shape}")

            # Get autoencoder anomalies
            feature_sequences_tensor = torch.FloatTensor(feature_sequences).to(
                autoencoder.device
            )
            anomalies, _ = autoencoder.detect_anomalies(feature_sequences_tensor)
            anomalies = anomalies.cpu().numpy()
            logger.info(f"Anomalies shape: {anomalies.shape}")

            # Ensure predictions and anomalies have the same length
            min_len = min(len(lstm_predictions), len(anomalies))
            lstm_predictions = lstm_predictions[:min_len]
            anomalies = anomalies[:min_len]
            logger.info(
                f"After alignment - Predictions: {lstm_predictions.shape}, Anomalies: {anomalies.shape}"
            )

        # Combine predictions
        combined_predictions = lstm_predictions.copy()
        combined_predictions[anomalies] *= 0.5  # Reduce confidence for anomalies
        logger.info(f"Combined predictions shape: {combined_predictions.shape}")

        # Create and train RL agent
        logger.info("Creating trading environment...")
        env = TradingEnvironment(returns, combined_predictions, window_size=window_size)
        state_dim = env._get_state().shape[0]
        logger.info(f"State dimension: {state_dim}")

        logger.info("Training RL agent...")
        agent = train_rl_agent(env, state_dim)

        # Evaluate strategy
        logger.info("Evaluating strategy...")
        metrics = evaluate_strategy(
            returns, combined_predictions, agent, window_size=window_size
        )

        logger.info("\nStrategy Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.2%}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise
