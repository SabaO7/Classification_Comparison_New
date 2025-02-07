from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import gc
import json
from scipy.special import softmax
from sklearn.model_selection import train_test_split, StratifiedKFold

from base_classes import BaseClassifier, ModelConfig


class TokenizedSubset(Dataset):
    """
    A subset of a pre-tokenized DataFrame (with columns: 'tokenized', 'class'),
    given a list of row indices.
    """
    def __init__(self, df: pd.DataFrame, indices: List[int]):
        self.df = df
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]
        token_dict = row["tokenized"] 
        if not isinstance(token_dict, dict):
            raise ValueError(f"Row {real_idx} has invalid 'tokenized' data: {token_dict}")
        
        # Convert "suicide" -> 1, otherwise 0
        label = 1 if row["class"] == "suicide" else 0

        item = {
            "input_ids": torch.tensor(token_dict["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(token_dict["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        # Some tokenizers also have 'token_type_ids'
        if "token_type_ids" in token_dict and token_dict["token_type_ids"] is not None:
            item["token_type_ids"] = torch.tensor(token_dict["token_type_ids"], dtype=torch.long)

        return item


class BERTClassifier(BaseClassifier):
    """
    A BERT classifier that relies on:
      - BaseClassifier for cross-validation and final training logic.
      - A pre-tokenized DataFrame with columns ["tokenized", "class"].
    """

    def __init__(self, config: ModelConfig, clf_dirs: dict, df: pd.DataFrame):
        """
        Args:
            config (ModelConfig): Training configuration.
            clf_dirs (dict): Dictionary of subdirectories (cv, final, etc.) from pipeline.
            df (pd.DataFrame): Pre-tokenized DataFrame with columns ["tokenized", "class"].
        """
        super().__init__(config, clf_dirs)
        self.logger.info("Initializing BERT classifier with pre-tokenized DataFrame...")

        self.df = df.reset_index(drop=True)
        if "tokenized" not in self.df.columns:
            raise ValueError("DataFrame is missing 'tokenized' column.")
        if "class" not in self.df.columns:
            raise ValueError("DataFrame is missing 'class' column.")
        self.df.dropna(subset=["tokenized"], inplace=True)
        self.df = self.df.reset_index(drop=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Tokenizer
        self.logger.info("Loading BERT tokenizer (prajjwal1/bert-tiny)...")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

        # DataCollator
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train(
        self, 
        texts: List[str], 
        labels: np.ndarray
    ) -> Tuple[List[Dict], Dict, Dict, Any, Any]:
        """
        1) train_test_split(80-20)
        2) StratifiedKFold on that 80%
        3) best config
        4) final training on 80%
        5) evaluate on 20%
        """
        try:
            self.logger.info("Starting model training pipeline...")

            # 1. train_test_split
            self.logger.info("Performing initial train-test split...")
            X_train, X_test, y_train, y_test = train_test_split(
                texts, 
                labels,
                test_size=1 - self.config.train_size,
                stratify=labels,
                random_state=self.config.random_state
            )
            self.log_data_split(X_train, X_test, y_train, y_test)

            # 2. K-fold CV
            cv_metrics = []
            cv_configs = []
            skf = StratifiedKFold(
                n_splits=self.config.num_iterations,
                shuffle=True,
                random_state=self.config.random_state
            )

            # 3. cross-validation
            self.logger.info("Starting cross-validation...")
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                self.logger.info(f"\nTraining fold {fold+1}/{self.config.num_iterations}")

                # Convert to workable lists
                if isinstance(X_train, (pd.DataFrame, pd.Series)):
                    fold_X_train = [X_train.iloc[i] for i in train_idx]
                    fold_y_train = y_train.iloc[train_idx]
                    fold_X_val   = [X_train.iloc[i] for i in val_idx]
                    fold_y_val   = y_train.iloc[val_idx]
                elif isinstance(X_train, (list, np.ndarray)):
                    X_train_list = X_train.tolist() if isinstance(X_train, np.ndarray) else X_train
                    fold_X_train = [X_train_list[i] for i in train_idx]
                    fold_y_train = y_train[train_idx]
                    fold_X_val   = [X_train_list[i] for i in val_idx]
                    fold_y_val   = y_train[val_idx]
                else:
                    raise ValueError("Unsupported data type for X_train and y_train")

                self.logger.debug(f"Fold {fold} sizes - Train: {len(fold_X_train)}, Val: {len(fold_X_val)}")

                # train fold
                fold_model, fold_metrics_dict, fold_config = self.train_fold(
                    fold_X_train, 
                    fold_y_train,
                    fold_X_val, 
                    fold_y_val,
                    fold
                )
                fold_metrics_dict["fold"] = fold
                cv_metrics.append(fold_metrics_dict)
                cv_configs.append(fold_config)

                self.logger.info(f"Fold {fold} metrics: {fold_metrics_dict}")

            # 4. pick best fold
            best_fold_idx = np.argmax([m["f1"] for m in cv_metrics])
            best_config = cv_configs[best_fold_idx]
            self.logger.info(f"Best configuration from fold {best_fold_idx + 1}: {best_config}")

            # 5. final train on entire 80%
            self.logger.info("Training final model on full training set...")
            final_model = self.train_final_model(X_train, y_train, best_config)

            # 6. evaluate on the 20% test
            self.logger.info("Evaluating on held-out test set...")
            test_metrics, final_labels, final_probs = self.evaluate_model(final_model, X_test, y_test)
            self.logger.info(f"Final test metrics: {test_metrics}")

            # aggregate
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
        self,
        X_train_list: List[Any],
        y_train_arr: np.ndarray,
        X_val_list: List[Any],
        y_val_arr: np.ndarray,
        fold: int
    ) -> Tuple[Any, Dict, Dict]:
        """
        Actually train on a single fold. Create TokenizedSubset from our stored self.df
        using row indices from X_train_list, X_val_list (numeric indices).
        """
        self.logger.info(f"train_fold: Creating TokenizedSubset for fold {fold}...")

        # Build dataset subsets
        train_subset = TokenizedSubset(self.df, X_train_list)
        val_subset   = TokenizedSubset(self.df, X_val_list)

        # Initialize BERT model
        model = AutoModelForSequenceClassification.from_pretrained(
            "prajjwal1/bert-tiny", num_labels=2
        ).to(self.device)

        # Output dir for this fold inside the "cv" folder
        fold_dir = os.path.join(self.clf_dirs["cv"], f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=fold_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_dir=os.path.join(fold_dir, "logs"),
            logging_steps=50,
            gradient_accumulation_steps=1,
            fp16=False,
            disable_tqdm=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        eval_metrics = trainer.evaluate()

        # parse metrics
        fold_metrics = {}
        for k, v in eval_metrics.items():
            if k.startswith("eval_"):
                fold_metrics[k.replace("eval_", "")] = float(v)
            else:
                fold_metrics[k] = float(v)

        self.logger.info(f"Fold {fold} -> metrics: {fold_metrics}")

        # example fold config
        fold_config = {
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs
        }

        return model, fold_metrics, fold_config

    def train_final_model(
        self,
        X_train_idx: List[int],
        y_train_idx: np.ndarray,
        config: Dict
    ) -> Any:
        """
        Train final on entire 80% subset (indices in X_train_idx).
        """
        self.logger.info("Training final BERT model on full training subset.")

        # Create final subfolder for the model
        final_dir = os.path.join(self.clf_dirs["final"], f"model_{self.timestamp}")
        os.makedirs(final_dir, exist_ok=True)

        train_subset = TokenizedSubset(self.df, X_train_idx)

        model = AutoModelForSequenceClassification.from_pretrained(
            "prajjwal1/bert-tiny", num_labels=2
        ).to(self.device)

        training_args = TrainingArguments(
            output_dir=final_dir,
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            num_train_epochs=config["epochs"],
            evaluation_strategy="no",
            save_strategy="epoch",
            logging_dir=os.path.join(final_dir, "logs_final"),
            logging_steps=50
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            data_collator=self.data_collator
        )

        trainer.train()

        # Save final checkpoint
        checkpoint_path = os.path.join(final_dir, "checkpoint")
        model.save_pretrained(checkpoint_path)
        self.logger.info(f"Final model saved in {checkpoint_path}")
        return model

    def evaluate_model(
        self,
        model: Any,
        X_test_idx: List[int],
        y_test_idx: np.ndarray
    ) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Evaluate on the parent's 20% test set. Indices X_test_idx, numeric labels y_test_idx.
        We'll build a TokenizedSubset with those row indices and run HF Trainer.predict.
        """
        self.logger.info("Evaluating final BERT model on test set...")
        test_subset = TokenizedSubset(self.df, X_test_idx)

        trainer = Trainer(
            model=model,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        preds_output = trainer.predict(test_subset)
        logits = preds_output.predictions  
        label_ids = preds_output.label_ids  

        probs = softmax(logits, axis=1)[:, 1]

        raw_metrics = self.compute_metrics((logits, label_ids))
        test_metrics = {k: float(v) for k, v in raw_metrics.items()}

        self.logger.info(f"Final test metrics: {test_metrics}")
        return test_metrics, label_ids, probs

    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Standard classification metrics: accuracy, precision, recall, F1, roc_auc.
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, pos_label=1)
        rec = recall_score(labels, preds, pos_label=1)
        f1_ = f1_score(labels, preds, pos_label=1)
        probs = softmax(logits, axis=1)[:, 1]
        roc = roc_auc_score(labels, probs)

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1_,
            "roc_auc": roc
        }
