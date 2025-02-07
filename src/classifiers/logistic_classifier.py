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
    Logistic Regression classifier implementation with cross-validation
    
    Uses TF-IDF vectorization for text preprocessing
    Implements proper CV strategy with holdout test set
    Includes model persistence and comprehensive logging
    """

    def __init__(self, config):
        super().__init__(config)
        self.vectorizer = None 
        

    
    def save_model(self, model, prefix="final_model"):
        model_path = os.path.join(self.final_dir, f"{prefix}_{self.timestamp}.joblib")
        joblib.dump(model, model_path)
        self.logger.info(f"Model saved to {model_path}")

    def save_metrics(self, metrics, prefix="metrics"):
        """
        Save evaluation metrics to file.

        Args:
            metrics (dict): Dictionary containing evaluation metrics.
            prefix (str): Prefix for the output file name.
        """
        metrics_file = os.path.join(self.cv_dir, f"{prefix}_{self.timestamp}.json")

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {metrics_file}")

    def save_fold_results(self, fold_results: List[Dict], prefix: str) -> str:
        """
        Save results from all folds
        
        Args:
            fold_results (List[Dict]): List of dictionaries containing fold results
            prefix (str): Prefix for the output file name
            
        Returns:
            str: Path to saved results file
        """
        try:
            output_file = os.path.join(self.cv_dir, f'{prefix}_fold_results_{self.timestamp}.csv')
            pd.DataFrame(fold_results).to_csv(output_file, index=False)
            self.logger.info(f"Saved fold results to {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Error saving fold results: {str(e)}")
            raise

    def preprocess_data(self, texts: List[str], is_training: bool = False) -> np.ndarray:
        """
        Preprocess text data using TF-IDF vectorization
        
        Args:
            texts (List[str]): List of text samples
            is_training (bool): Whether this is training data
            
        Returns:
            np.ndarray: Vectorized text features
        """
        self.logger.info(f"Preprocessing {'training' if is_training else 'evaluation'} data...")
        self.logger.info(f"Number of texts: {len(texts)}")
        
        if is_training and self.vectorizer is None:
            self.logger.info("Initializing and fitting TF-IDF vectorizer...")
            self.vectorizer = TfidfVectorizer(
                max_features=3000, #reduce the number of features to 3000 NEW
                min_df=2,
                max_df=0.95
            )
            X = self.vectorizer.fit_transform(texts)
            self.logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            return X
            
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call with training data first.")
            
        return self.vectorizer.transform(texts)
        
    def train_fold(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
        fold: int
    ) -> Tuple[LogisticRegression, Dict[str, float], Dict]:
        """
        Train model on a single fold with statistical validation.

        Args:
            X_train (List[str]): Training texts.
            y_train (np.ndarray): Training labels.
            X_val (List[str]): Validation texts.
            y_val (np.ndarray): Validation labels.
            fold (int): Current fold number.

        Returns:
            Tuple[LogisticRegression, Dict[str, float], Dict]:
                - Trained Logistic Regression model.
                - Performance metrics (accuracy, precision, recall, F1-score, etc.).
                - Model configuration.
        """

        self.logger.info(f"Training fold {fold+1}")
        self.logger.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
    
        # Validate that raw data is being used
        if not isinstance(X_train[0], str):
            raise ValueError("LogisticClassifier requires raw text data for training.")

        # Debug label information
        self.logger.debug(f"Original label types - Training: {type(y_train[0])}, Validation: {type(y_val[0])}")
        self.logger.debug(f"Original unique labels - Training: {np.unique(y_train)}, Validation: {np.unique(y_val)}")
    
        # Convert labels to numeric values (if they are still strings)
        if isinstance(y_train[0], str):
            y_train = np.array([1 if label == "suicide" else 0 for label in y_train], dtype=int)
            y_val = np.array([1 if label == "suicide" else 0 for label in y_val], dtype=int)
            self.logger.info("Converted string labels to numeric (1 = suicide, 0 = non-suicide)")

        # Log label types
        self.logger.debug(f"Final label types - Training: {type(y_train[0])}, Validation: {type(y_val[0])}")
        self.logger.debug(f"Final unique labels - Training: {np.unique(y_train)}, Validation: {np.unique(y_val)}")

        # Preprocess data
        X_train_vec = self.preprocess_data(X_train, is_training=True)
        X_val_vec = self.preprocess_data(X_val, is_training=False)
    
        # Initialize and train model
        self.logger.info("Training Logistic Regression model...")
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=self.config.random_state + fold
        )
        model.fit(X_train_vec, y_train)
    
        # Get predictions
        y_pred = model.predict(X_val_vec)
        y_prob = model.predict_proba(X_val_vec)[:, 1]
    
        try:
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, pos_label=1),
                'recall': recall_score(y_val, y_pred, pos_label=1),
                'f1': f1_score(y_val, y_pred, pos_label=1),
                'roc_auc': roc_auc_score(np.array([1 if label == "suicide" else 0 for label in y_val]), y_prob)
            }
        
            self.logger.info(f"Fold {fold+1} metrics: {metrics}")
        
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            self.logger.error(f"y_val unique values: {np.unique(y_val)}")
            self.logger.error(f"y_pred unique values: {np.unique(y_pred)}")
            raise
        finally:
            # Cleanup resources
            del X_train_vec, X_val_vec
        
    
        # Save fold model
        model_path = os.path.join(
            self.cv_dir,
            f"logistic_model_fold_{fold}_{self.timestamp}.joblib"
        )

        joblib.dump(model, model_path)
        self.logger.info(f"Saved fold model to {model_path}")
    
        # Track configuration
        config = {
            'max_iter': 1000,
            'class_weight': 'balanced',
            'vocab_size': len(self.vectorizer.vocabulary_),
            'fold': fold
        }
    
        return model, metrics, config


    def train_final_model(self,
                        X_train: List[str],
                        y_train: np.ndarray,
                        config: Dict) -> LogisticRegression:
        """
        Train final model on entire training set.
        
        Args:
            X_train: Training texts.
            y_train: Training labels.
            config: Best configuration from CV.
            
        Returns:
            LogisticRegression: Trained final model.
        """
        self.logger.info("Training final model on complete training set...")
        self.logger.info(f"Training set size: {len(X_train)}")

        # Preprocess data
        X_train_vec = self.preprocess_data(X_train, is_training=True)

        # Initialize and train model
        model = LogisticRegression(
            max_iter=config['max_iter'],
            class_weight=config['class_weight'],
            random_state=self.config.random_state
        )

        self.logger.info("Fitting final model...")
        model.fit(X_train_vec, y_train)

        # ✅ Store the trained model inside the class
        self.model = model  

        # Save model and vectorizer
        model_path = os.path.join(
            self.final_dir,
            f"logistic_model_final_{self.timestamp}.joblib"
        )

        vectorizer_path = os.path.join(
            self.config.output_dir,
            f"tfidf_vectorizer_{self.timestamp}.joblib"
        )

        joblib.dump(model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

        self.logger.info(f"Saved final model to {model_path}")
        self.logger.info(f"Saved vectorizer to {vectorizer_path}")

        return model

        
    def evaluate_model(self,
                    model: LogisticRegression,
                    X_test: List[str],
                    y_test: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained Logistic Regression model.
            X_test: List of test texts.
            y_test: Test labels.
            
        Returns:
            Tuple[Dict, np.ndarray, np.ndarray]:
                - A dictionary of evaluation metrics.
                - An array of true labels.
                - An array of predicted probabilities for the positive class.
        """
        self.logger.info("Evaluating on test set...")
        self.logger.info(f"Test set size: {len(X_test)}")
        
        # Preprocess test data
        X_test_vec = self.preprocess_data(X_test, is_training=False)
        
        # Get predictions
        y_pred = np.array([1 if label == "suicide" else 0 for label in model.predict(X_test_vec)])
        y_prob = model.predict_proba(X_test_vec)[:, 1]
        
        # Ensure numeric labels for evaluation
        y_test = np.array([1 if label == "suicide" else 0 for label in y_test])
        

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label=1),
            'recall': recall_score(y_test, y_pred, pos_label=1),
            'f1': f1_score(y_test, y_pred, pos_label=1),
            'roc_auc': roc_auc_score(y_test, y_prob)  # No need for explicit casting
        }
        
        self.logger.info(f"Test set metrics: {metrics}")
        
        # Save predictions to CSV
        pd.DataFrame({
            'text': X_test,
            'prediction': y_pred,
            'actual': y_test,
            'probability': y_prob
        }).to_csv(
            f"{self.config.output_dir}/logistic_test_predictions_{self.timestamp}.csv",
            index=False
        )
        
        return metrics, y_test.tolist(), y_prob.tolist()



        
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions on new texts
        
        Args:
            texts (List[str]): List of text samples
            
        Returns:
            np.ndarray: Predicted labels
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model or vectorizer not initialized. Train the model first.")
            
        self.logger.info(f"Making predictions on {len(texts)} new texts")
        X = self.preprocess_data(texts, is_training=False)
        
        predictions = self.model.predict(X)
        self.logger.info("Predictions complete")
        
        return predictions
        
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities for new texts
        
        Args:
            texts (List[str]): List of text samples
            
        Returns:
            np.ndarray: Predicted probabilities for positive class
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model or vectorizer not initialized. Train the model first.")
            
        self.logger.info(f"Getting prediction probabilities for {len(texts)} texts")
        X = self.preprocess_data(texts, is_training=False)
        
        probabilities = self.model.predict_proba(X)[:, 1]
        self.logger.info("Probability calculations complete")
        
        return probabilities