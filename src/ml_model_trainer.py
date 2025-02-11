# ml_model_trainer.py
import numpy as np
import pandas as pd
import sys, os
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class MLModelTrainer:
    """
    Class for training and managing ensemble of ML models.
    """

    def __init__(
        self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 5
    ):
        """
        Initialize MLModelTrainer.

        Args:
            n_estimators: Number of trees in XGBoost
            learning_rate: Learning rate for XGBoost
            max_depth: Maximum tree depth
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.feature_names = None
        self.scaler = StandardScaler()

        # Model configurations for diversity
        self.model_configs = [
            {"max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "gamma": 0},
            {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.9, "gamma": 0.1},
            {"max_depth": 4, "learning_rate": 0.08, "subsample": 0.7, "gamma": 0.2},
            {"max_depth": 6, "learning_rate": 0.03, "subsample": 0.85, "gamma": 0.15},
            {"max_depth": 4, "learning_rate": 0.12, "subsample": 0.75, "gamma": 0.05},
        ]

    def train_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train ensemble of models with different configurations.

        Args:
            X: Feature matrix
            y: Target labels
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train models with different configurations
        for i, config in enumerate(self.model_configs):
            logger.info(f"\nTraining model {i + 1}/5")

            # Remove highly correlated features
            selector = SelectFromModel(
                XGBClassifier(random_state=42),
                threshold="mean",
                max_features=min(X.shape[1] - i * 2, X.shape[1] - 5),
            )
            X_selected = selector.fit_transform(X_scaled, y)

            # Log feature selection
            n_removed = X.shape[1] - X_selected.shape[1]
            logger.info(f"Removed {n_removed} highly correlated features")

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_selected, y, test_size=0.2, random_state=42
            )

            # Train model
            model = XGBClassifier(
                n_estimators=self.n_estimators, random_state=42 + i, **config
            )
            model.fit(X_train, y_train)

            # Get feature importance
            importance = model.feature_importances_
            top_features = np.argsort(importance)[-5:][::-1]
            logger.info("Top 5 features by importance:")
            for idx in top_features:
                logger.info(f"{idx}: {importance[idx]:.4f}")

            # Evaluate on validation set
            y_pred = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_pred)
            logger.info(f"Model {i + 1} validation AUC: {val_auc:.4f}")

            self.models.append(
                {"model": model, "selector": selector, "val_score": val_auc}
            )

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self.models:
            raise ValueError("Models not trained yet")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get predictions from each model
        predictions = []
        weights = []

        for model_dict in self.models:
            model = model_dict["model"]
            selector = model_dict["selector"]
            val_score = model_dict["val_score"]

            # Select features and predict
            X_selected = selector.transform(X_scaled)
            pred = model.predict_proba(X_selected)[:, 1]

            predictions.append(pred)
            weights.append(val_score)

        # Compute weighted average predictions
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights

        ensemble_pred = np.zeros(len(X))
        for pred, weight in zip(predictions, weights):
            ensemble_pred += pred * weight

        return ensemble_pred

    def plot_feature_importance(self) -> None:
        """
        Plot aggregated feature importance across models.
        """
        if not self.models:
            raise ValueError("Models not trained yet")

        plt.figure(figsize=(12, 6))

        # Aggregate importance across models
        importance_dict = {}
        for model_dict in self.models:
            model = model_dict["model"]
            importance = model_dict["model"].feature_importances_
            for i, imp in enumerate(importance):
                if i not in importance_dict:
                    importance_dict[i] = []
                importance_dict[i].append(imp)

        # Compute mean importance
        mean_importance = {k: np.mean(v) for k, v in importance_dict.items()}

        # Plot top 20 features
        sorted_features = sorted(
            mean_importance.items(), key=lambda x: x[1], reverse=True
        )[:20]
        feature_ids = [str(f[0]) for f in sorted_features]
        importance_values = [f[1] for f in sorted_features]

        plt.bar(feature_ids, importance_values)
        plt.title("Top 20 Features by Importance")
        plt.xlabel("Feature ID")
        plt.ylabel("Mean Importance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        logger.info("Feature importance plot saved as 'feature_importance.png'")

    def plot_learning_curves(self) -> None:
        """
        Plot learning curves for the models.
        """
        if not self.models:
            raise ValueError("Models not trained yet")

        plt.figure(figsize=(10, 6))

        for i, model_dict in enumerate(self.models):
            model = model_dict["model"]
            results = model.evals_result()

            if results:
                epochs = len(results["validation_0"]["auc"])
                x_axis = range(0, epochs)
                plt.plot(x_axis, results["validation_0"]["auc"], label=f"Model {i + 1}")

        plt.title("Learning Curves")
        plt.xlabel("Boosting Round")
        plt.ylabel("AUC Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("learning_curves.png")
        logger.info("Learning curves plot saved as 'learning_curves.png'")

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
