# ml_model_trainer.py
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MLModelTrainer:
    @staticmethod
    def train_model(features, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        model = XGBClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("ML Model Accuracy:", accuracy_score(y_test, predictions))
        return model
