from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
from base_classes import BaseClassifier, ModelConfig
import joblib
import os
import json

class LogisticClassifier(BaseClassifier):
    """
    Logistic Regression classifier with cross-validation.

    Ensures that labels are consistently numeric (0 = non-suicide, 1 = suicide).
    """

    def __init__(self, config: ModelConfig, clf_dirs: dict):
        """
        Args:
            config (ModelConfig): training config
            clf_dirs (dict): subdirectories from pipeline
        """
        super().__init__(config, clf_dirs)
        self.vectorizer = None  # TF-IDF vectorizer
        self.model = None  # Logistic Regression Model

    # ---------------------------------------------------------------------
    # Utility methods (save_model, save_metrics, etc.)
    # ---------------------------------------------------------------------
    def save_model(self, model, prefix="final_model"):
        """
        Save a .joblib model in the 'final' directory.
        """
        model_path = os.path.join(
            self.clf_dirs["final"],
            f"{prefix}_{self.timestamp}.joblib"
        )
        joblib.dump(model, model_path)
        self.logger.info(f"Model saved to {model_path}")

    def save_metrics(self, metrics, prefix="metrics"):
        """
        Save evaluation metrics as JSON in 'cv' directory.
        """
        metrics_path = os.path.join(
            self.clf_dirs["cv"],
            f"{prefix}_{self.timestamp}.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {metrics_path}")

    def save_fold_results(self, fold_results: List[Dict], prefix: str) -> str:
        """
        Save cross-validation fold results as CSV in 'cv' directory.
        """
        out_file = os.path.join(
            self.clf_dirs["cv"],
            f"{prefix}_fold_results_{self.timestamp}.csv"
        )
        pd.DataFrame(fold_results).to_csv(out_file, index=False)
        self.logger.info(f"Saved fold results to {out_file}")
        return out_file

    # ---------------------------------------------------------------------
    # Preprocessing text (TF-IDF)
    # ---------------------------------------------------------------------
    def preprocess_data(self, texts: List[str], is_training: bool = False) -> np.ndarray:
        """
        Convert raw texts → TF-IDF features.
        If is_training=True and vectorizer is None, we fit it here.
        Otherwise, we just transform.
        """
        self.logger.info(f"Preprocessing {'training' if is_training else 'evaluation'} data...")
        self.logger.info(f"Number of texts: {len(texts)}")

        if is_training and self.vectorizer is None:
            self.logger.info("Initializing TF-IDF vectorizer (fitting)...")
            self.vectorizer = TfidfVectorizer(
                max_features=3000,
                min_df=2,
                max_df=0.95
            )
            X = self.vectorizer.fit_transform(texts)
            self.logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            return X

        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted yet. Provide training data first.")

        return self.vectorizer.transform(texts)

    # ---------------------------------------------------------------------
    # Single fold training
    # ---------------------------------------------------------------------
    def train_fold(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
        fold: int
    ) -> Tuple[LogisticRegression, Dict[str, float], Dict]:
        """
        Train a single cross-validation fold.
        """
        self.logger.info(f"===== Training fold {fold+1} =====")
        self.logger.info(f"Fold data sizes: train={len(X_train)}, val={len(X_val)}")

        # Ensure X_train are strings
        if not isinstance(X_train[0], str):
            raise ValueError("LogisticClassifier expects raw text strings for training.")

        # Ensure labels are numeric
        y_train = np.array([1 if lbl == "suicide" else 0 for lbl in y_train], dtype=int)
        y_val = np.array([1 if lbl == "suicide" else 0 for lbl in y_val], dtype=int)

        # TF-IDF
        X_train_vec = self.preprocess_data(X_train, is_training=True)
        X_val_vec = self.preprocess_data(X_val, is_training=False)

        # Initialize logistic regression
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=self.config.random_state + fold
        )
        self.logger.info("Fitting logistic regression on fold data...")
        model.fit(X_train_vec, y_train)

        # Predict and convert predictions to numeric (0/1)
        y_pred = model.predict(X_val_vec)
        y_prob = model.predict_proba(X_val_vec)[:, 1]

        # Evaluate
        metrics_dict = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_prob)
        }
        self.logger.info(f"Fold {fold+1} metrics: {metrics_dict}")

        # Save fold model
        fold_model_path = os.path.join(
            self.clf_dirs["cv"],
            f"logistic_model_fold_{fold}_{self.timestamp}.joblib"
        )
        joblib.dump(model, fold_model_path)
        self.logger.info(f"Saved fold model to {fold_model_path}")

        return model, metrics_dict, {}

    # ---------------------------------------------------------------------
    # Final training after best fold
    # ---------------------------------------------------------------------
    def train_final_model(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        config: Dict
    ) -> LogisticRegression:
        """
        Train final logistic model on entire training set using the best config.
        """
        self.logger.info("===== Training final logistic model on full training set =====")

        # Ensure labels are numeric
        y_train = np.array([1 if lbl == "suicide" else 0 for lbl in y_train], dtype=int)

        # TF-IDF
        X_train_vec = self.preprocess_data(X_train, is_training=True)

        # Train final model
        model = LogisticRegression(
            max_iter=config.get('max_iter', 1000),
            class_weight=config.get('class_weight', 'balanced'),
            random_state=self.config.random_state
        )
        model.fit(X_train_vec, y_train)

        # Save final model
        self.save_model(model, prefix="logistic_model_final")
        self.model = model
        return model

    # ---------------------------------------------------------------------
    # Evaluate on Test Set
    # ---------------------------------------------------------------------
    def evaluate_model(
        self,
        model: LogisticRegression,
        X_test: List[str],
        y_test: np.ndarray
    ) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Evaluate final model on the test set.
        """
        self.logger.info("===== Evaluating logistic model on test set =====")

        # Ensure labels are numeric
        y_test = np.array([1 if lbl == "suicide" else 0 for lbl in y_test], dtype=int)

        # Vectorize X_test
        X_test_vec = self.preprocess_data(X_test, is_training=False)

        # Predictions
        y_pred = model.predict(X_test_vec)
        y_prob = model.predict_proba(X_test_vec)[:, 1]

        # Metrics
        metrics_dict = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        self.logger.info(f"Test set metrics: {metrics_dict}")

        return metrics_dict, y_test, y_prob
