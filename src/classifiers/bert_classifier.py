from logging import Logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
from base_classes import BaseClassifier, ModelConfig
import shutil
import json
import pandas as pd
import gc
from transformers import DataCollatorWithPadding
from sklearn.model_selection import StratifiedKFold  
from scipy.special import softmax


# Disable tokenizer parallelism to avoid deadlock warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------------------------------------------
# 1) Check existing folds that have completed
# ----------------------------------------------------------------
metrics_dir = "outputs/bert/logs"
completed_folds = []

for fold_idx in range(5):
    metrics_file = os.path.join(metrics_dir, f"metrics_fold_{fold_idx}.json")
    if os.path.exists(metrics_file):
        completed_folds.append(fold_idx)

print("Completed folds:", completed_folds)

# ----------------------------------------------------------------
# 2) Single load of the data BEFORE the folds
# ----------------------------------------------------------------
print("Loading tokenized DataFrame once...")
full_dataset = pd.read_pickle("../data/tokenized/tokenized_data.pkl")
print(f"Loaded dataset with {len(full_dataset)} rows.")


class TextDataset(Dataset):
    """
    Dataset class for BERT model with tokenized data.
    
    We accept an existing DataFrame to avoid re-reading
    from disk for each fold. The DataFrame must have:
      - "tokenized": a dict of {input_ids, attention_mask, ...}
      - "class": either 'suicide' or 'non-suicide'
    """
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize dataset from a pre-loaded DataFrame.
        """
        self.data = dataframe.copy()

        # ✅ Ensure tokenized column exists and is not empty
        if "tokenized" not in self.data.columns:
            raise ValueError("Dataset is missing 'tokenized' column.")

        # ✅ Drop rows with missing tokenized data
        self.data = self.data.dropna(subset=["tokenized"]).reset_index(drop=True)

        print(f"✅ Loaded dataset with {len(self.data)} valid rows.")

    def __getitem__(self, idx: int) -> Dict:
        """Retrieve pre-tokenized item by index."""
        record = self.data.iloc[idx]

        try:
            # ✅ Ensure the tokenized data is valid
            if not isinstance(record["tokenized"], dict):
                raise ValueError(f"Invalid tokenized format at index {idx}: {record['tokenized']}")

            label = 1 if record["class"] == "suicide" else 0
            item = {key: torch.tensor(val) for key, val in record["tokenized"].items()}
            item["labels"] = torch.tensor(label, dtype=torch.long)

            return item
        except Exception as e:
            print(f"❌ Error processing record {idx}: {e}")
            raise

    def __len__(self):
        return len(self.data)


class BERTClassifier(BaseClassifier):
    """
    BERT-based classifier implementation with cross-validation.
    """

    def __init__(self, config: ModelConfig, dataset: TextDataset):
        super().__init__(config)
        self.logger.info("Initializing BERT classifier...")

        # ✅ Store the dataset once to avoid reloading
        self.dataset = dataset

        # ✅ Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # ✅ Initialize tokenizer
        self.logger.info("Loading BERT tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

        # ✅ DataCollator handles dynamic padding
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = None

    def train(self, texts: List[str], labels: np.ndarray) -> Tuple[List[Dict], Dict, Dict, Any, Any]:
        """
        Override the generic train method.
        
        For the BERTClassifier we already have a custom training routine that
        loads the tokenized data once and uses index-based cross-validation.
        This override ensures that when the pipeline calls train(), it uses the
        BERT-specific training routine rather than the generic one (which expects
        raw texts and labels).
        
        Returns:
            Tuple containing:
            - A list of cross-validation metrics (dummy placeholder here),
            - A dictionary of final test metrics (dummy placeholder),
            - A dictionary representing the best configuration (dummy placeholder),
            - Final test labels (dummy placeholder),
            - Final test probabilities (dummy placeholder).
        """
        # Call the BERT-specific training routine.
        self.train_model()
        
        # In your current BERT code, you are not returning CV metrics or final evaluation results.
        # You can either extend your train_model() and train_final_model() methods to compute these
        # or, if you do not need them, return placeholder values.
        cv_metrics = []      # Replace with actual CV metrics if available.
        best_config = {}     # Replace with the best configuration if computed.
        test_metrics = {}    # Replace with final test metrics if final evaluation is performed.
        final_labels = None  # Replace with the final test labels if computed.
        final_probs = None   # Replace with the final test probabilities if computed.
        
        return cv_metrics, test_metrics, best_config, final_labels, final_probs


    def train_model(self):
        """
        Runs cross-validation for training the BERT model.

        This method:
        - Loads the dataset ONCE
        - Uses StratifiedKFold to create train/val splits
        - Calls `train_fold()` with index-based subsets (instead of reloading the dataset)
        """
        self.logger.info("Starting cross-validation...")

        # Use the dataset that was passed from main.py
        dataset = self.dataset  # ✅ Correct: Use preloaded dataset


        # ✅ Stratified K-Fold split (preserves class balance)
        skf = StratifiedKFold(n_splits=self.config.num_iterations, shuffle=True, random_state=self.config.random_state)

        # ✅ Convert labels to numpy array
        labels = np.array(full_dataset["class"].map({"suicide": 1, "non-suicide": 0}))

        cv_metrics = []
        cv_configs = []

        # ✅ Loop through each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(full_dataset["text"], labels)):
            self.logger.info(f"\nTraining fold {fold + 1}/{self.config.num_iterations}")

            # ✅ Call train_fold with index-based subsets
            model, fold_metrics, fold_config = self.train_fold(
                dataset, train_idx, val_idx, fold
            )

            # ✅ Store fold results
            cv_metrics.append(fold_metrics)
            cv_configs.append(fold_config)

        self.logger.info("✅ Cross-validation complete. Saving results.")

    def train_fold(self, dataset, train_idx, val_idx, fold: int) -> Tuple[Any, Dict, Dict]:
        """
        Train model on a single fold using the provided train/val sets.

        Returns:
            (model, metrics, training_args)
        """
        self.logger.info(f"Training fold {fold + 1}")
        self.logger.info("Using distinct train/val subsets for real cross-validation.")

        try:
            # ✅ Subset dataset using provided indices
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # -------------------------------------------------
            # ✅ Load Pretrained BERT Model for Classification
            # -------------------------------------------------
            model = AutoModelForSequenceClassification.from_pretrained(
                "prajjwal1/bert-tiny",
                num_labels=2  # Ensure model initializes classification head properly
            ).to(self.device)

            # -------------------------------------------------
            # ✅ Define Training Output Directory
            # -------------------------------------------------
            output_dir = os.path.join(self.models_dir, f"fold_{fold}_{self.timestamp}")
            os.makedirs(output_dir, exist_ok=True)

            # -------------------------------------------------
            # ✅ Define TrainingArguments for Hugging Face Trainer
            # -------------------------------------------------
            training_args = TrainingArguments(
                output_dir=output_dir,        # Save model checkpoints here
                learning_rate=3e-5,           # Learning rate for optimizer
                per_device_train_batch_size=16,  # Training batch size
                per_device_eval_batch_size=16,   # Evaluation batch size
                num_train_epochs=self.config.epochs,  # Number of training epochs
                weight_decay=0.01,             # Regularization to prevent overfitting
                evaluation_strategy="epoch",   # Evaluate at the end of each epoch
                save_strategy="no",            # No intermediate model saving
                logging_dir=f"{self.config.output_dir}/logs",  # Logging directory
                logging_steps=1000,            # Log every 1000 steps
                gradient_accumulation_steps=2, # Accumulate gradients to avoid out-of-memory issues
                fp16=False,                    # Disable mixed precision for stability
                max_grad_norm=1.0,              # Gradient clipping
            )

            # -------------------------------------------------
            # ✅ Initialize Trainer
            # -------------------------------------------------
            trainer = Trainer(
                model=model,         # Model instance
                args=training_args,  # Training configuration
                train_dataset=train_subset,  # ✅ Use defined subset
                eval_dataset=val_subset,    # ✅ Use defined subset
                data_collator=self.data_collator, # Handles batch padding dynamically
                compute_metrics=self.compute_metrics # Function to compute performance metrics
            )


            # -------------------------------------------------
            # ✅ Train and Evaluate Model
            # -------------------------------------------------
            trainer.train()  # Train model
            eval_metrics = trainer.evaluate()  # Evaluate on validation set

            # ✅ Process evaluation metrics
            # ✅ Convert NumPy arrays to lists before saving JSON
            cleaned_metrics = {k.replace("eval_", ""): float(v) for k, v in eval_metrics.items()}
            for key, value in cleaned_metrics.items():
                if isinstance(value, np.ndarray):
                    cleaned_metrics[key] = value.tolist()  # ✅ Convert to list before saving

            # ✅ Save fold metrics to JSON file
            metrics_file = os.path.join(self.logs_dir, f"metrics_fold_{fold}.json")
            with open(metrics_file, "w") as f:
                json.dump(cleaned_metrics, f, indent=4)

            self.logger.info(f"Metrics for fold {fold + 1} saved to {metrics_file}")

            # -------------------------------------------------
            # ✅ Return trained model, metrics, and arguments
            # -------------------------------------------------
            return model, cleaned_metrics, training_args.to_dict()

        except Exception as e:
            # ❌ Log any errors that occur during training
            self.logger.error(f"Error in fold {fold} training: {str(e)}")
            raise

        finally:
            # -------------------------------------------------
            # ✅ Cleanup: Free GPU Memory After Each Fold
            # -------------------------------------------------
            del model, trainer  # Delete model and trainer instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Free GPU memory
            gc.collect()  # Run garbage collection to free CPU memory
            self.logger.info("✅ Cleared GPU memory after fold.")


    def save_model(self, model, prefix="final_model"):
        """Save the trained model"""
        model_dir = os.path.join(self.models_dir, f"{prefix}_{self.timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        self.logger.info(f"Model saved to {model_dir}")


    def save_model(self, model, prefix="final_model"):
        """
        Save the trained model
        """
        model_dir = os.path.join(self.models_dir, f"{prefix}_{self.timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        self.logger.info(f"Model saved to {model_dir}")
        
    def train_final_model(
        self, 
        X_train: List[str],
        y_train: np.ndarray,
        config: Dict
    ) -> Any:
        """
        Train final model on entire training set (optional).
        If your data is already fully tokenized in full_df, we can just reuse that.
        """
        self.logger.info("Training final model on complete training set.")
        self.logger.info(f"Training set size: {len(X_train)}")
        self.logger.info(f"Using config: {config}")

        try:
            # Just reuse the single loaded DataFrame if you want all data
            final_dataset = TextDataset(full_df)

            model = AutoModelForSequenceClassification.from_pretrained(
                "prajjwal1/bert-tiny", num_labels=2
            ).to(self.device)

            training_args = TrainingArguments(
                output_dir=f"{self.config.output_dir}/bert_final_{self.timestamp}",
                learning_rate=config["learning_rate"],
                per_device_train_batch_size=config["per_device_train_batch_size"],
                num_train_epochs=config["num_train_epochs"],
                weight_decay=config["weight_decay"],
                save_strategy="epoch",
                logging_dir=f"{self.config.output_dir}/logs_final_{self.timestamp}",
                logging_steps=10
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=final_dataset,
                data_collator=self.data_collator
            )

            trainer.train()

        except Exception as e:
            self.logger.error(f"Error in final model training: {str(e)}")
            raise

        finally:
            del model, trainer, final_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return None  # or return model if you prefer
    
    def evaluate_model(
        self, 
        model: AutoModelForSequenceClassification,
        X_test: List[Any],
        y_test: Any,
        file_path: str = None
    ) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Evaluate model on a test set.
        
        Returns:
            Tuple[Dict, np.ndarray, np.ndarray]: 
                - A dictionary of evaluation metrics.
                - An array of ground-truth labels.
                - An array of probabilities for the positive class.
        """
        self.logger.info("Evaluating model on test set...")

        if file_path is None:
            file_path = "../data/tokenized/tokenized_data.pkl"

        try:
            test_df = pd.read_pickle(file_path)
            test_dataset = TextDataset(test_df)

            trainer = Trainer(
                model=model,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics
            )

            predictions = trainer.predict(test_dataset)
            logits = predictions.predictions    # shape (N, 2)
            label_ids = predictions.label_ids   # ground-truth labels as 0/1

            # Convert logits -> probabilities for the positive class (index=1)
            probs = softmax(logits, axis=1)[:, 1]

            metrics = self.compute_metrics(predictions)
            cleaned = {k: float(v) for k, v in metrics.items()}
            self.logger.info(f"Evaluation metrics: {cleaned}")

            # Save predictions to CSV
            pred_labels = np.argmax(logits, axis=1)
            df_out = pd.DataFrame({
                'prediction': pred_labels,
                'actual': label_ids,
                'probability': probs
            })
            df_out.to_csv(
                f"{self.config.output_dir}/bert_test_predictions_{self.timestamp}.csv",
                index=False
            )
            return cleaned, label_ids, probs

        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise

    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """
        Compute evaluation metrics from the model predictions.
        
        Args:
            eval_pred (Tuple[np.ndarray, np.ndarray]): A tuple containing:
                - logits: Model output logits.
                - labels: Ground truth labels.
                
        Returns:
            Dict: A dictionary with evaluation metrics.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Compute metrics using sklearn
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, pos_label=1)
        recall = recall_score(labels, predictions, pos_label=1)
        f1 = f1_score(labels, predictions, pos_label=1)
        # Compute ROC AUC (ensure labels are binary integers 0 and 1)
        # Use softmax to convert logits to probabilities for the positive class
        probs = softmax(logits, axis=1)[:, 1]
        roc_auc = roc_auc_score(labels, probs)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }

