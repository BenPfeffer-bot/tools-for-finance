# ml_model_trainer.py
import numpy as np
import pandas as pd
import sys, os
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Dict, Any, Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class MLModelTrainer:
    """Class for training and evaluating machine learning models."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize the ML model trainer.

        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate for gradient boosting
            max_depth: Maximum tree depth
            n_splits: Number of cross-validation splits
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_splits = n_splits
        self.random_state = random_state

        # Initialize model
        self.model = None
        self.feature_importance = None
        self.cv_results = None

        logger.info(
            f"Initialized MLModelTrainer with {n_estimators} estimators, "
            f"learning rate {learning_rate}, max depth {max_depth}"
        )

    def train_model(self, features: pd.DataFrame, labels: pd.Series) -> xgb.Booster:
        """
        Train the XGBoost model on the provided features and labels.

        Args:
            features: DataFrame of input features
            labels: Series of binary labels

        Returns:
            Trained XGBoost classifier
        """
        try:
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )

            # Create DMatrix objects for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            # Set up parameters
            params = {
                "objective": "binary:logistic",
                "eval_metric": ["auc", "error"],
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "gamma": 0,
            }

            # Initialize evaluation results dictionary
            evals_result = {}

            # Train model with early stopping
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
                evals_result=evals_result,
            )

            # Store evaluation results
            self.eval_results = evals_result

            # Log training results
            logger.info(f"Best iteration: {self.model.best_iteration}")
            logger.info(f"Best validation AUC: {self.model.best_score:.4f}")

            return self.model

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def cross_validate(
        self, features: pd.DataFrame, labels: pd.Series, n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation and return performance metrics.

        Args:
            features: DataFrame of input features
            labels: Series of binary labels
            n_splits: Number of cross-validation folds

        Returns:
            Dictionary of performance metrics
        """
        try:
            # Initialize cross-validation
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            # Initialize metrics storage
            metrics = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "auc": [],
            }

            # Perform cross-validation
            for fold, (train_idx, val_idx) in enumerate(cv.split(features, labels)):
                # Split data
                X_train = features.iloc[train_idx]
                X_val = features.iloc[val_idx]
                y_train = labels.iloc[train_idx]
                y_val = labels.iloc[val_idx]

                # Train model
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)

                # Set up parameters
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "n_estimators": self.n_estimators,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 1,
                    "gamma": 0,
                }

                # Train model
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    evals=[(dtrain, "train"), (dval, "val")],
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )

                # Make predictions
                y_pred = model.predict(dval)
                y_pred_binary = (y_pred > 0.5).astype(int)

                # Calculate metrics
                metrics["accuracy"].append(accuracy_score(y_val, y_pred_binary))
                metrics["precision"].append(precision_score(y_val, y_pred_binary))
                metrics["recall"].append(recall_score(y_val, y_pred_binary))
                metrics["f1"].append(f1_score(y_val, y_pred_binary))
                metrics["auc"].append(roc_auc_score(y_val, y_pred))

                logger.info(
                    f"Fold {fold + 1}: AUC = {metrics['auc'][-1]:.4f}, "
                    f"Accuracy = {metrics['accuracy'][-1]:.4f}"
                )

            # Calculate mean and std for each metric
            results = {}
            for metric in metrics:
                results[f"{metric}_mean"] = np.mean(metrics[metric])
                results[f"{metric}_std"] = np.std(metrics[metric])

            logger.info("\nCross-validation results:")
            for metric, value in results.items():
                logger.info(f"{metric}: {value:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}")
            raise

    def plot_feature_importance(
        self, feature_names: Optional[List[str]] = None
    ) -> None:
        """
        Plot feature importance from the trained model.

        Args:
            feature_names: Optional list of feature names
        """
        if self.model is None:
            logger.error("Model must be trained first!")
            return

        try:
            # Get feature importance scores
            importance = self.model.get_score(importance_type="gain")

            if not importance:
                logger.warning("No feature importance available")
                return

            # Convert to DataFrame
            importance_df = pd.DataFrame(
                list(importance.items()), columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)

            # Create plot
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df["Feature"], importance_df["Importance"], alpha=0.8)
            plt.title("Feature Importance (Gain)")
            plt.xlabel("Importance Score")
            plt.tight_layout()

            # Save plot
            plt.savefig("feature_importance.png")
            plt.close()

            logger.info("Feature importance plot saved as 'feature_importance.png'")

        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")

    def plot_learning_curves(self) -> None:
        """
        Plot learning curves from the training history.
        """
        if self.model is None:
            logger.error("Model must be trained first!")
            return

        if not hasattr(self, "eval_results"):
            logger.error("No evaluation results available")
            return

        try:
            # Create plot
            plt.figure(figsize=(12, 6))

            # Plot training and validation metrics
            for metric in ["auc", "error"]:
                plt.plot(
                    self.eval_results["train"][metric],
                    label=f"Training {metric.upper()}",
                )
                plt.plot(
                    self.eval_results["val"][metric],
                    label=f"Validation {metric.upper()}",
                )

            plt.title("Learning Curves")
            plt.xlabel("Iteration")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save plot
            plt.savefig("learning_curves.png")
            plt.close()

            logger.info("Learning curves plot saved as 'learning_curves.png'")

        except Exception as e:
            logger.error(f"Error plotting learning curves: {str(e)}")

    def save_model(self, path: str = "model.json") -> None:
        """
        Save the trained model to a file.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            logger.error("Model must be trained first!")
            return

        try:
            self.model.save_model(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, path: str = "model.json") -> None:
        """
        Load a trained model from a file.

        Args:
            path: Path to the saved model
        """
        try:
            self.model = xgb.Booster()
            self.model.load_model(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
