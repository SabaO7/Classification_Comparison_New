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
   - This ensures that every data point is used for **validation once** and **training multiple times**.

3. **Selecting the Best Model from Cross-Validation**
   - After 5-fold CV, the **best-performing fold** is selected based on metrics like **F1-score, accuracy, precision, recall, and ROC-AUC**.
   - The **best model configuration** (hyperparameters, training setup) is recorded.

4. **Final Training on the Entire 80% Training Data**
   - Instead of training on only 4 folds, the **final model is trained using the full 80% training data**.
   - This **maximizes the available training data** to improve generalization.

5. **Final Model Evaluation on the 20% Test Set**
   - The **final trained model is tested ONCE on the 20% test set**.
   - This provides an **unbiased** estimate of real-world performance.
   - The test set **was never used in training or validation**, ensuring a **fair evaluation**.

### **Why This Process?**
✅ Prevents **data leakage** by keeping the test set separate.  
✅ Ensures **comparability** between different models.  
✅ Maximizes the **amount of training data** before final evaluation.  
✅ Provides a **robust estimate** of real-world performance.  
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
    Base classifier class with common functionality and proper cross-validation
    
    Implements best practices for model evaluation:
    1. Initial train/test split
    2. K-fold cross-validation on training data
    3. Final model training on full training set
    4. Evaluation on held-out test set
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = os.path.join(config.output_dir, self.__class__.__name__.replace("Classifier", "").lower())
        
        # New Subfolders: logs, cv, final, visualizations
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.cv_dir = os.path.join(self.base_dir, "cv")
        self.final_dir = os.path.join(self.base_dir, "final")
        self.visualizations_dir = os.path.join(self.base_dir, "visualizations")

        for directory in [
            self.logs_dir,
            self.cv_dir,
            self.final_dir,
            self.visualizations_dir
        ]:
            os.makedirs(directory, exist_ok=True)

        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the classifier."""
        log_file = os.path.join(self.logs_dir, f"{self.__class__.__name__}_{self.timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler(sys.stdout)
        
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Logging initialized. Log file: {log_file}")
        
    
    def save_config(self):
        """Save model configuration"""
        config_file = os.path.join(self.final_dir, f'config_{self.timestamp}.json')

        try:
            with open(config_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=4)
            self.logger.info(f"Saved configuration to {config_file}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
        

    def convert_numpy(self, obj):
        """
        Convert NumPy and Pandas objects to JSON-serializable formats.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to list
        elif isinstance(obj, np.generic): 
            return obj.item()  # Convert NumPy scalar to Python int/float
        elif isinstance(obj, pd.Series): 
            return obj.tolist()  # Convert Pandas Series to list
        elif isinstance(obj, pd.DataFrame): 
            return obj.to_dict(orient="records")  # Convert DataFrame to list of dicts
        elif isinstance(obj, dict):  # Ensure nested dicts are also converted
            return {k: self.convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):  # Ensure lists with NumPy objects are converted
            return [self.convert_numpy(v) for v in obj]
        return obj  # Return as-is if not a NumPy/Pandas type



    def save_metrics(self, metrics: Dict, prefix: str) -> str:
        """
        Save evaluation metrics to file, ensuring NumPy arrays are converted.
        """
        try:
            # Convert everything before saving
            metrics_serializable = self.convert_numpy(metrics)

            # Debugging step to check if conversion worked
            print(f"Debug (save_metrics): {metrics_serializable}")

            output_file = os.path.join(self.cv_dir, f'{prefix}_metrics_{self.timestamp}.json')
            
            with open(output_file, 'w') as f:
                json.dump(metrics_serializable, f, indent=4)

            self.logger.info(f"Saved metrics to {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise


    def calculate_aggregate_metrics(self, fold_results: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Calculate mean and standard deviation of metrics across folds,
        ensuring all values are JSON serializable.
        """
        try:
            metrics_df = pd.DataFrame(fold_results)

            mean_metrics = self.convert_numpy(metrics_df.mean().to_dict())
            std_metrics = self.convert_numpy(metrics_df.std().to_dict())

            # Debugging: Print converted values
            print(f"Debug (mean_metrics): {mean_metrics}")
            print(f"Debug (std_metrics): {std_metrics}")

            self.logger.info("Calculated aggregate metrics:")
            for metric, value in mean_metrics.items():
                self.logger.info(f"{metric}: {value:.4f} ± {std_metrics[metric]:.4f}")
            
            return mean_metrics, std_metrics
        except Exception as e:
            self.logger.error(f"Error calculating aggregate metrics: {str(e)}")
            raise

        
    def save_comparison_results(self, results: dict) -> dict:
        """
        Save model comparison results in both JSON and CSV.

        Args:
            results (dict): Dictionary containing results for each model.

        Returns:
            dict: Summary of best-performing model.
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

        csv_path = os.path.join(self.final_dir, f"comparison_summary_{self.timestamp}.csv")
        json_path = os.path.join(self.final_dir, f"comparison_summary_{self.timestamp}.json")


        pd.DataFrame(comparison_data).to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)

        self.logger.info(f"Comparison summary saved to {csv_path} and {json_path}")
        return summary

    
    def save_fold_results(self, fold_metrics: List[Dict], prefix: str) -> None:
        """
        Save fold results in both JSON and CSV format.

        Args:
            fold_metrics (List[Dict]): List of fold evaluation metrics.
            prefix (str): Prefix for filenames.
        """
        output_json = os.path.join(self.cv_dir, f"{prefix}_{self.timestamp}.json")
        output_csv = os.path.join(self.cv_dir, f"{prefix}_{self.timestamp}.csv")

        try:
            # Convert NumPy to native types
            fold_metrics = self.convert_numpy(fold_metrics)

            # Save as JSON
            with open(output_json, "w") as f:
                json.dump(fold_metrics, f, indent=4)
            self.logger.info(f"Saved fold results to {output_json}")

            # Save as CSV
            df = pd.DataFrame(fold_metrics)
            df.to_csv(output_csv, index=False)
            self.logger.info(f"Saved fold results to {output_csv}")

        except Exception as e:
            self.logger.error(f"Error saving fold results: {str(e)}")
            raise

 
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize input text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        try:
            if pd.isna(text):
                return ""
            # Convert to string and lowercase
            text = str(text).lower()
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text
        except Exception as e:
            logging.error(f"Error cleaning text: {str(e)}")
            return ""
        
    def log_data_split(self, X_train, X_test, y_train, y_test):
        """Log information about data splits"""
        train_dist = pd.Series(y_train).value_counts(normalize=True)
        test_dist = pd.Series(y_test).value_counts(normalize=True)
        
        self.logger.info(f"\nData Split Information:")
        self.logger.info(f"Training set size: {len(X_train)}")
        self.logger.info(f"Test set size: {len(X_test)}")
        self.logger.info(f"Training class distribution:\n{train_dist}")
        self.logger.info(f"Test class distribution:\n{test_dist}")
        
    def train(self, texts: List[str], labels: np.ndarray) -> Tuple[List[Dict], Dict, Dict]:
        """
        Train and evaluate model using proper cross-validation strategy
        
        Implementation steps:
        1. Split data into train (80%) and test (20%) sets
        2. Perform k-fold cross-validation on training data
        3. Use CV results to determine best model configuration
        4. Train final model on entire training set
        5. Evaluate on held-out test set
        
        Args:
            texts (List[str]): List of text samples
            labels (np.ndarray): Labels for text samples
            
        Returns:
            Tuple[List[Dict], Dict, Dict]: 
                - List of CV fold metrics
                - Final test metrics
                - Best model configuration
        """
        try:
            self.logger.info("Starting model training pipeline...")
            
            # 1. Initial train-test split
            self.logger.info("Performing initial train-test split...")
            X_train, X_test, y_train, y_test = train_test_split(
                texts, 
                labels,
                test_size=1-self.config.train_size,
                stratify=labels,
                random_state=self.config.random_state
            )
            
            self.log_data_split(X_train, X_test, y_train, y_test)
            
            # 2. Initialize k-fold cross validation
            cv_metrics = []
            cv_configs = []
            skf = StratifiedKFold(
                n_splits=self.config.num_iterations,
                shuffle=True,
                random_state=self.config.random_state
            )
            
            # 3. Perform cross-validation
            self.logger.info("Starting cross-validation...")
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)): 
                self.logger.info(f"\nTraining fold {fold+1}/{self.config.num_iterations}")
                
                # Get fold data
                if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
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
                    raise ValueError("Unsupported data type for X_train and y_train")
                    
                # Log fold distribution
                self.logger.debug(f"Fold {fold} sizes - Train: {len(fold_X_train)}, Val: {len(fold_X_val)}")
                
                # Train and evaluate on this fold
                fold_model, fold_metrics, fold_config = self.train_fold(
                    fold_X_train, fold_y_train,
                    fold_X_val, fold_y_val,
                    fold
                )
                
                # Track metrics and configs
                fold_metrics['fold'] = fold
                cv_metrics.append(fold_metrics)
                cv_configs.append(fold_config)
                
                self.logger.info(f"Fold {fold} metrics: {fold_metrics}")
            
            # 4. Find best configuration
            best_fold_idx = np.argmax([m['f1'] for m in cv_metrics])
            best_config = cv_configs[best_fold_idx]
            self.logger.info(f"Best configuration from fold {best_fold_idx + 1}: {best_config}")
            
            # 5. Train final model on entire training set
            self.logger.info("Training final model on full training set...")
            final_model = self.train_final_model(X_train, y_train, best_config)
            
            # 6. Evaluate on held-out test set
            self.logger.info("Evaluating on held-out test set...")
            test_metrics, final_labels, final_probs = self.evaluate_model(final_model, X_test, y_test)
            self.logger.info(f"Final test metrics: {test_metrics}")

            
            # Calculate and save metrics
            mean_cv_metrics, std_cv_metrics = self.calculate_aggregate_metrics(cv_metrics)
            
            # Save all results
            self.save_fold_results(cv_metrics, f"{self.__class__.__name__}_cv")
            self.save_metrics(test_metrics, f"{self.__class__.__name__}_test")
            self.save_metrics(mean_cv_metrics, f"{self.__class__.__name__}_cv_mean")
            self.save_metrics(std_cv_metrics, f"{self.__class__.__name__}_cv_std")
            
            self.logger.info("Training pipeline completed successfully!")
            return cv_metrics, test_metrics, best_config, final_labels, final_probs
        
            
            
        except Exception as e:
            self.logger.error(f"Error in training pipeline: {str(e)}")
            raise
    
    def train_fold(self, X_train, y_train, X_val, y_val, fold: int) -> Tuple[Any, Dict, Dict]:
        """To be implemented by specific classifier classes"""
        raise NotImplementedError
        
    def train_final_model(self, X_train, y_train, config: Dict) -> Any:
        """To be implemented by specific classifier classes"""
        raise NotImplementedError
        
    def evaluate_model(self, model: Any, X_test: List[str], y_test: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """To be implemented by specific classifier classes.
        
        Returns:
            A tuple containing:
                - A dictionary of evaluation metrics.
                - An array of true labels.
                - An array of predicted probabilities.
        """
        raise NotImplementedError
