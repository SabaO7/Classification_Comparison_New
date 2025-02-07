from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
import json
import sys

"""
Cross-Validation and Final Model Training Process:

1. **Train-Test Split (80-20)**
   - The dataset is first split into **80% training** and **20% testing**.
   - The **test set (20%) remains untouched** until the final evaluation.

2. **5-Fold Cross-Validation on the 80% Training Data**
   - The **80% training data** is further split into **5 equal parts (folds)**.
   - The model is trained **5 times**, each time:
     - **4 folds (64% of total data) are used for training.**
     - **1 fold (16% of total data) is used for validation.**
   - This ensures that every data point is used for **validation once** 
     and **training multiple times**.

3. **Selecting the Best Model from Cross-Validation**
   - After 5-fold CV, the **best-performing fold** is selected based on
     metrics like **F1-score, accuracy, precision, recall, and ROC-AUC**.
   - The **best model configuration** (hyperparameters, training setup) is recorded.

4. **Final Training on the Entire 80% Training Data**
   - Instead of training on only 4 folds, the **final model is trained using
     the full 80% training data**.
   - This **maximizes the available training data** to improve generalization.

5. **Final Model Evaluation on the 20% Test Set**
   - The **final trained model is tested ONCE on the 20% test set**.
   - This provides an **unbiased** estimate of real-world performance.
   - The test set **was never used in training or validation**,
     ensuring a **fair evaluation**.
"""

class ModelType(Enum):
    """Enumeration of supported model types"""
    LOGISTIC = "logistic"
    BERT = "bert"
    FEW_SHOT = "few_shot"

@dataclass
class ModelConfig:
    """
    Configuration for model training
    
    Attributes:
        batch_size (int): Size of batches for training
        learning_rate (float): Learning rate for model optimization
        epochs (int): Number of training epochs
        max_length (int): Maximum sequence length
        train_size (float): Proportion of data for training (80-20 split)
        random_state (int): Random seed for reproducibility
        num_iterations (int): Number of cross-validation folds
        output_dir (str): Directory for saving outputs
    """
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 3
    max_length: int = 128
    train_size: float = 0.8
    random_state: int = 42
    num_iterations: int = 5  # Number of CV folds
    output_dir: str = "outputs"

    def to_dict(self) -> Dict:
        """Convert config to dictionary for saving"""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'max_length': self.max_length,
            'train_size': self.train_size,
            'random_state': self.random_state,
            'num_iterations': self.num_iterations,
            'output_dir': self.output_dir
        }

class BaseClassifier:
    """
    Base classifier class with common functionality and proper cross-validation.

    Implements best practices for model evaluation:
      1. Initial train/test split (80-20)
      2. K-fold cross-validation on training data
      3. Final model training on full training set
      4. Evaluation on held-out test set
    """
    
    def __init__(self, config: ModelConfig, clf_dirs: dict):
        """
        Args:
            config (ModelConfig): Configuration for model training.
            clf_dirs (dict): Dictionary of directories for this classifier
                             (keys = "cv", "final", "visualizations", etc.),
                             passed from the pipeline.
        """
        self.config = config
        self.clf_dirs = clf_dirs
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.logger.info(f"Initialized {self.__class__.__name__} with directories: {self.clf_dirs}")

    def save_config(self):
        """Save model configuration as JSON in the 'final' directory."""
        config_path = os.path.join(self.clf_dirs["final"], f'config_{self.timestamp}.json')
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=4)
            self.logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")

    def convert_numpy(self, obj):
        """
        Convert NumPy and Pandas objects to JSON-serializable formats.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, dict):
            return {k: self.convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy(v) for v in obj]
        return obj

    def save_metrics(self, metrics: Dict, prefix: str) -> str:
        """
        Save evaluation metrics (Dict) to a JSON file in the 'cv' directory.
        Ensures NumPy arrays are converted first.
        """
        try:
            metrics_serializable = self.convert_numpy(metrics)
            output_file = os.path.join(self.clf_dirs["cv"], f'{prefix}_metrics_{self.timestamp}.json')
            with open(output_file, 'w') as f:
                json.dump(metrics_serializable, f, indent=4)
            self.logger.info(f"Saved metrics to {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise

    def calculate_aggregate_metrics(self, fold_results: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Calculate mean and std of metrics across folds.
        """
        try:
            metrics_df = pd.DataFrame(fold_results)
            mean_metrics = self.convert_numpy(metrics_df.mean().to_dict())
            std_metrics = self.convert_numpy(metrics_df.std().to_dict())

            self.logger.info("Calculated aggregate metrics (mean ± std):")
            for metric, value in mean_metrics.items():
                self.logger.info(f"{metric}: {value:.4f} ± {std_metrics[metric]:.4f}")

            return mean_metrics, std_metrics
        except Exception as e:
            self.logger.error(f"Error calculating aggregate metrics: {str(e)}")
            raise

    def save_comparison_results(self, results: dict) -> dict:
        """
        Save model comparison results (JSON+CSV) in the 'final' directory.
        """
        self.logger.info("Saving comparison results...")

        best_model = None
        best_f1_score = 0.0
        comparison_data = []

        for model_name, (cv_metrics, final_metrics, best_config) in results.items():
            final_metrics = self.convert_numpy(final_metrics)
            best_config = self.convert_numpy(best_config)

            comparison_data.append({
                "Model": model_name,
                "F1 Score": final_metrics["f1"],
                "Accuracy": final_metrics["accuracy"],
                "Precision": final_metrics["precision"],
                "Recall": final_metrics["recall"],
                "ROC AUC": final_metrics["roc_auc"]
            })

            if final_metrics["f1"] > best_f1_score:
                best_f1_score = final_metrics["f1"]
                best_model = model_name

        summary = {"best_model": best_model, "best_f1_score": best_f1_score}

        csv_path = os.path.join(self.clf_dirs["final"], f"comparison_summary_{self.timestamp}.csv")
        json_path = os.path.join(self.clf_dirs["final"], f"comparison_summary_{self.timestamp}.json")

        pd.DataFrame(comparison_data).to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)

        self.logger.info(f"Comparison summary saved to {csv_path} and {json_path}")
        return summary

    def save_fold_results(self, fold_metrics: List[Dict], prefix: str) -> None:
        """
        Save fold results in JSON + CSV inside the 'cv' directory.
        """
        output_json = os.path.join(self.clf_dirs["cv"], f"{prefix}_{self.timestamp}.json")
        output_csv = os.path.join(self.clf_dirs["cv"], f"{prefix}_{self.timestamp}.csv")

        try:
            fold_metrics = self.convert_numpy(fold_metrics)
            with open(output_json, "w") as f:
                json.dump(fold_metrics, f, indent=4)
            self.logger.info(f"Saved fold results to {output_json}")

            df = pd.DataFrame(fold_metrics)
            df.to_csv(output_csv, index=False)
            self.logger.info(f"Saved fold results to {output_csv}")
        except Exception as e:
            self.logger.error(f"Error saving fold results: {str(e)}")
            raise

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize a single text string.
        """
        try:
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = ' '.join(text.split())
            return text
        except Exception as e:
            logging.error(f"Error cleaning text: {str(e)}")
            return ""

    def log_data_split(self, X_train, X_test, y_train, y_test):
        """Log info about data splits."""
        train_dist = pd.Series(y_train).value_counts(normalize=True)
        test_dist = pd.Series(y_test).value_counts(normalize=True)
        self.logger.info("Data Split Info:")
        self.logger.info(f"Training set size: {len(X_train)}")
        self.logger.info(f"Test set size: {len(X_test)}")
        self.logger.info(f"Training class distribution:\n{train_dist}")
        self.logger.info(f"Test class distribution:\n{test_dist}")

    def train(self, texts: List[str], labels: np.ndarray) -> Tuple[List[Dict], Dict, Dict, np.ndarray, np.ndarray]:
        """
        Train and evaluate model with cross-validation + final hold-out test.

        Returns: 
            (cv_metrics_list, final_test_metrics, best_config, final_labels, final_probs)
        """
        try:
            self.logger.info("Starting model training pipeline...")

            # 1. train/test split
            self.logger.info("Performing initial train-test split...")
            X_train, X_test, y_train, y_test = train_test_split(
                texts,
                labels,
                test_size=1 - self.config.train_size,
                stratify=labels,
                random_state=self.config.random_state
            )
            self.log_data_split(X_train, X_test, y_train, y_test)

            # 2. K-fold cross-validation
            cv_metrics = []
            cv_configs = []
            skf = StratifiedKFold(
                n_splits=self.config.num_iterations,
                shuffle=True,
                random_state=self.config.random_state
            )

            self.logger.info("Starting cross-validation...")
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                self.logger.info(f"\nTraining fold {fold+1}/{self.config.num_iterations}")

                # Grab fold data
                if isinstance(X_train, (pd.DataFrame, pd.Series)):
                    fold_X_train = [X_train.iloc[i] for i in train_idx]
                    fold_y_train = y_train.iloc[train_idx]
                    fold_X_val = [X_train.iloc[i] for i in val_idx]
                    fold_y_val = y_train.iloc[val_idx]
                elif isinstance(X_train, list):
                    fold_X_train = [X_train[i] for i in train_idx]
                    fold_y_train = y_train[train_idx]
                    fold_X_val = [X_train[i] for i in val_idx]
                    fold_y_val = y_train[val_idx]
                else:
                    raise ValueError("Unsupported data type for X_train/y_train")

                # Train fold
                fold_model, fold_metrics_dict, fold_config = self.train_fold(
                    fold_X_train, fold_y_train,
                    fold_X_val, fold_y_val,
                    fold
                )

                fold_metrics_dict["fold"] = fold
                cv_metrics.append(fold_metrics_dict)
                cv_configs.append(fold_config)
                self.logger.info(f"Fold {fold} metrics: {fold_metrics_dict}")

            # 3. Decide best fold by F1
            best_fold_idx = np.argmax([m["f1"] for m in cv_metrics])
            best_config = cv_configs[best_fold_idx]
            self.logger.info(f"Best configuration from fold {best_fold_idx+1}: {best_config}")

            # 4. Train final model on entire training set
            self.logger.info("Training final model on full training set...")
            final_model = self.train_final_model(X_train, y_train, best_config)

            # 5. Evaluate on 20% test
            self.logger.info("Evaluating on held-out test set...")
            test_metrics, final_labels, final_probs = self.evaluate_model(final_model, X_test, y_test)
            self.logger.info(f"Final test metrics: {test_metrics}")

            # 6. Save metrics
            mean_cv_metrics, std_cv_metrics = self.calculate_aggregate_metrics(cv_metrics)
            self.save_fold_results(cv_metrics, f"{self.__class__.__name__}_cv")
            self.save_metrics(test_metrics, f"{self.__class__.__name__}_test")
            self.save_metrics(mean_cv_metrics, f"{self.__class__.__name__}_cv_mean")
            self.save_metrics(std_cv_metrics, f"{self.__class__.__name__}_cv_std")

            self.logger.info("Training pipeline completed successfully!")
            return cv_metrics, test_metrics, best_config, final_labels, final_probs

        except Exception as e:
            self.logger.error(f"Error in training pipeline: {str(e)}")
            raise

    def train_fold(
        self, X_train, y_train, X_val, y_val, fold: int
    ) -> Tuple[Any, Dict, Dict]:
        """
        Train on a single fold. Must be overridden by each classifier.
        """
        raise NotImplementedError

    def train_final_model(
        self, X_train, y_train, config: Dict
    ) -> Any:
        """
        Train final model on entire training set. Must be overridden.
        """
        raise NotImplementedError

    def evaluate_model(
        self, model: Any, X_test: List[str], y_test: np.ndarray
    ) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Evaluate on the test set. Must be overridden.
        Returns: (test_metrics_dict, final_labels, final_probs)
        """
        raise NotImplementedError
