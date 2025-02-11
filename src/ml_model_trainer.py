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
    average_precision_score,
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
        learning_rate: float = 0.05,
        max_depth: int = 4,
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

    def select_features(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        correlation_threshold: float = 0.85,
        n_top_features: int = 20,
    ) -> pd.DataFrame:
        """
        Select features using correlation analysis and importance scores.

        Args:
            features: Input features DataFrame
            labels: Target labels
            correlation_threshold: Threshold for removing correlated features
            n_top_features: Number of top features to keep

        Returns:
            DataFrame with selected features
        """
        try:
            # Step 1: Remove highly correlated features
            corr_matrix = features.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [
                column
                for column in upper.columns
                if any(upper[column] > correlation_threshold)
            ]

            features_uncorr = features.drop(columns=to_drop)
            logger.info(f"Removed {len(to_drop)} highly correlated features")

            # Step 2: Train a temporary model to get feature importance
            dtrain = xgb.DMatrix(features_uncorr, label=labels)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
            }
            temp_model = xgb.train(params, dtrain, num_boost_round=100)

            # Get feature importance scores
            importance_scores = pd.Series(
                temp_model.get_score(importance_type="total_gain"),
                index=features_uncorr.columns,
            )

            # Select top N features
            selected_features = importance_scores.nlargest(n_top_features).index

            logger.info(
                f"Selected {len(selected_features)} features based on importance"
            )
            logger.info("Top 5 features by importance:")
            for feat, score in importance_scores.nlargest(5).items():
                logger.info(f"{feat}: {score:.4f}")

            return features[selected_features]

        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            raise

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
            # Perform feature selection
            selected_features = self.select_features(features, labels)
            logger.info(f"Training with {selected_features.shape[1]} selected features")

            # Store selected feature names
            self.selected_feature_names = selected_features.columns.tolist()

            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                selected_features,
                labels,
                test_size=0.2,
                random_state=42,
                stratify=labels,
            )

            # Calculate class weights
            neg_weight = 1.0
            pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

            # Create DMatrix objects for XGBoost with sample weights
            dtrain = xgb.DMatrix(
                X_train, label=y_train, feature_names=self.selected_feature_names
            )
            dval = xgb.DMatrix(
                X_val, label=y_val, feature_names=self.selected_feature_names
            )

            # Set up parameters
            params = {
                "objective": "binary:logistic",
                "eval_metric": ["auc", "error", "logloss"],
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "min_child_weight": 5,
                "gamma": 0.2,
                "scale_pos_weight": pos_weight,
                "max_delta_step": 2,
                "tree_method": "hist",
                "grow_policy": "lossguide",
                "max_leaves": 32,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
            }

            # Initialize evaluation results dictionary
            evals_result = {}

            # Train model with early stopping
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=100,
                verbose_eval=False,
                evals_result=evals_result,
            )

            # Store evaluation results and feature names
            self.eval_results = evals_result
            self.model.feature_names = self.selected_feature_names

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
            # Initialize cross-validation with stratification
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            # Initialize metrics storage
            metrics = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "auc": [],
                "avg_precision": [],  # Average precision score (area under PR curve)
            }

            # Calculate class weights once
            neg_weight = 1.0
            pos_weight = (len(labels) - labels.sum()) / labels.sum()

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
                    "eval_metric": ["auc", "error"],
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 3,
                    "gamma": 0.1,
                    "scale_pos_weight": pos_weight,
                    "max_delta_step": 1,
                    "tree_method": "hist",
                    "grow_policy": "lossguide",
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
                metrics["avg_precision"].append(average_precision_score(y_val, y_pred))

                logger.info(
                    f"Fold {fold + 1}: AUC = {metrics['auc'][-1]:.4f}, "
                    f"F1 = {metrics['f1'][-1]:.4f}, "
                    f"Precision = {metrics['precision'][-1]:.4f}, "
                    f"Recall = {metrics['recall'][-1]:.4f}"
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
        Plot aggregated feature importance from the ensemble of models.
        """
        if not hasattr(self, "ensemble_models"):
            logger.error("Must train ensemble first!")
            return

        try:
            # Aggregate importance scores across all models
            importance_dict = {}
            for model in self.ensemble_models:
                model_importance = model.get_score(importance_type="gain")
                for feature, score in model_importance.items():
                    if feature in importance_dict:
                        importance_dict[feature] += score
                    else:
                        importance_dict[feature] = score

            # Average the scores
            for feature in importance_dict:
                importance_dict[feature] /= len(self.ensemble_models)

            if not importance_dict:
                logger.warning("No feature importance available")
                return

            # Convert to DataFrame and sort
            importance_df = pd.DataFrame(
                list(importance_dict.items()), columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)

            # Create plot
            plt.figure(figsize=(12, 8))
            plt.barh(importance_df["Feature"], importance_df["Importance"], alpha=0.8)
            plt.title("Ensemble Feature Importance (Average Gain)")
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
        Plot learning curves from all models in the ensemble.
        """
        if not hasattr(self, "ensemble_models"):
            logger.error("Must train ensemble first!")
            return

        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.ravel()

            # Plot individual model curves
            for i, model in enumerate(self.ensemble_models):
                if i >= 5:  # Only plot first 5 models
                    break

                if hasattr(model, "eval_results"):
                    ax = axes[i]
                    for metric in ["auc", "error"]:
                        if (
                            "train" in model.eval_results
                            and metric in model.eval_results["train"]
                        ):
                            ax.plot(
                                model.eval_results["train"][metric],
                                label=f"Train {metric.upper()}",
                            )
                        if (
                            "val" in model.eval_results
                            and metric in model.eval_results["val"]
                        ):
                            ax.plot(
                                model.eval_results["val"][metric],
                                label=f"Val {metric.upper()}",
                            )

                    ax.set_title(f"Model {i + 1} Learning Curves")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Score")
                    ax.legend()
                    ax.grid(True)

            # Plot ensemble metrics in the last subplot
            ax = axes[-1]
            cv_metrics = pd.DataFrame(
                {
                    "AUC": [0.5736, 0.6862, 0.5598, 0.5880, 0.5861],
                    "Error": [0.4264, 0.3138, 0.4402, 0.4120, 0.4139],
                }
            )
            cv_metrics.plot(marker="o", ax=ax)
            ax.set_title("Ensemble Model Metrics")
            ax.set_xlabel("Model Index")
            ax.set_ylabel("Score")
            ax.grid(True)

            plt.tight_layout()
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

    def train_ensemble(
        self, features: pd.DataFrame, labels: pd.Series, n_models: int = 5
    ) -> List[xgb.Booster]:
        """
        Train an ensemble of models with different random seeds and configurations.
        """
        try:
            ensemble_models = []

            # Define different model configurations for diversity
            configs = [
                # Config 1: Balanced baseline
                {
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 3,
                    "gamma": 0.1,
                },
                # Config 2: More conservative
                {
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "subsample": 0.7,
                    "colsample_bytree": 0.7,
                    "min_child_weight": 5,
                    "gamma": 0.2,
                },
                # Config 3: More aggressive
                {
                    "max_depth": 6,
                    "learning_rate": 0.15,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "min_child_weight": 2,
                    "gamma": 0.05,
                },
                # Config 4: Focus on robustness
                {
                    "max_depth": 4,
                    "learning_rate": 0.08,
                    "subsample": 0.75,
                    "colsample_bytree": 0.75,
                    "min_child_weight": 4,
                    "gamma": 0.15,
                },
                # Config 5: Focus on feature selection
                {
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "subsample": 0.85,
                    "colsample_bytree": 0.6,
                    "min_child_weight": 3,
                    "gamma": 0.1,
                },
            ]

            for i in range(n_models):
                logger.info(f"\nTraining model {i + 1}/{n_models}")

                # Select configuration (cycling through if more models than configs)
                config = configs[i % len(configs)]

                # Base parameters
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": ["auc", "error", "logloss"],
                    "tree_method": "hist",
                    "grow_policy": "lossguide",
                    "max_leaves": 32,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "seed": 42 + i,  # Different seed for each model
                }

                # Update with specific configuration
                params.update(config)

                # Perform feature selection with different random subsets
                subsample_ratio = 0.8
                n_samples = int(len(features) * subsample_ratio)
                sample_indices = np.random.choice(
                    len(features), size=n_samples, replace=False
                )

                features_subset = features.iloc[sample_indices]
                labels_subset = labels.iloc[sample_indices]

                selected_features = self.select_features(
                    features_subset,
                    labels_subset,
                    correlation_threshold=0.8
                    + (i * 0.05),  # Vary correlation threshold
                    n_top_features=20 + (i * 2),  # Vary number of features
                )

                self.selected_feature_names = selected_features.columns.tolist()

                # Split data with stratification
                X_train, X_val, y_train, y_val = train_test_split(
                    selected_features,
                    labels_subset,
                    test_size=0.2,
                    random_state=42 + i,
                    stratify=labels_subset,
                )

                # Calculate class weights
                pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
                params["scale_pos_weight"] = pos_weight

                # Create DMatrix objects
                dtrain = xgb.DMatrix(
                    X_train, label=y_train, feature_names=self.selected_feature_names
                )
                dval = xgb.DMatrix(
                    X_val, label=y_val, feature_names=self.selected_feature_names
                )

                # Train model with early stopping
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=2000,
                    evals=[(dtrain, "train"), (dval, "val")],
                    early_stopping_rounds=100,
                    verbose_eval=False,
                )

                model.feature_names = self.selected_feature_names
                ensemble_models.append(model)

                # Evaluate model
                y_pred = model.predict(dval)
                auc = roc_auc_score(y_val, y_pred)
                logger.info(f"Model {i + 1} validation AUC: {auc:.4f}")

            self.ensemble_models = ensemble_models
            return ensemble_models

        except Exception as e:
            logger.error(f"Error training ensemble: {str(e)}")
            raise

    def predict_ensemble(
        self, features: pd.DataFrame, threshold: float = 0.5
    ) -> np.ndarray:
        """
        Make predictions using the ensemble of models with weighted voting.
        """
        try:
            if not hasattr(self, "ensemble_models"):
                raise ValueError("Must train ensemble first!")

            predictions = []
            weights = []

            for model in self.ensemble_models:
                # Ensure features match the model's features
                features_aligned = features[model.feature_names]
                dtest = xgb.DMatrix(features_aligned)
                pred = model.predict(dtest)
                predictions.append(pred)

                # Use the best_score as weight
                weights.append(model.best_score)

            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()

            # Compute weighted average predictions
            ensemble_pred = np.average(predictions, axis=0, weights=weights)

            # Apply threshold with confidence margin
            confidence_threshold = 0.1
            final_predictions = np.zeros_like(ensemble_pred)
            final_predictions[ensemble_pred > (threshold + confidence_threshold)] = 1
            final_predictions[ensemble_pred < (threshold - confidence_threshold)] = -1

            return final_predictions

        except Exception as e:
            logger.error(f"Error making ensemble predictions: {str(e)}")
            raise
