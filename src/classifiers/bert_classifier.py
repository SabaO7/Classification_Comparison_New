from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
from base_classes import BaseClassifier, ModelConfig
import shutil 
import json
import pandas as pd
from torch.utils.data import DataLoader #adding this based on the comment from Helen to handle dynamic batch loading!! which hopefully reduces the load


# Disable tokenizer parallelism to avoid deadlock warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TextDataset(Dataset): #updated this to use the tokenized data
    """Dataset class for BERT model with tokenized data"""

    def __init__(self, file_path: str):
        """
        Load pre-tokenized dataset from disk.
        
        Args:
            file_path (str): Path to the pre-tokenized dataset.
        """
        self.data = pd.read_pickle(file_path)

    def __getitem__(self, idx: int) -> Dict:
        """Retrieve pre-tokenized item by index."""
        record = self.data.iloc[idx]
        try:
            # Convert string labels to integers
            label = 1 if record['class'] == 'suicide' else 0
            
            # Convert tokenized data to tensors
            item = {key: torch.tensor(val) for key, val in record['tokenized'].items()}
            item['labels'] = torch.tensor(label, dtype=torch.long)

            # Debugging: Print types and contents
            print(f"Item {idx}:")
            print(f"Input IDs Type: {type(item['input_ids'])}, Shape: {item['input_ids'].shape}")
            print(f"Attention Mask Type: {type(item['attention_mask'])}, Shape: {item['attention_mask'].shape}")
            print(f"Label Type: {type(item['labels'])}, Value: {item['labels']}")

            return item
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
            raise




    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)

class BERTClassifier(BaseClassifier):
    """
    BERT-based classifier implementation with cross-validation
    
    Uses tiny-BERT model for efficient training while maintaining performance
    Implements proper CV strategy with holdout test set
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize BERT classifier
        
        Args:
            config (ModelConfig): Configuration object
        """
        super().__init__(config)
        self.logger.info("Initializing BERT classifier...")
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.logger.info("Loading BERT tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        
        self.model = None

    def compute_metrics(self, pred) -> Dict:
        """
        Compute comprehensive evaluation metrics
        
        Args:
            pred: Prediction outputs
            
        Returns:
            Dict: Dictionary of computed metrics
        """
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        probs = pred.predictions[:, 1]  # Probability of positive class
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds),
            'recall': recall_score(labels, preds),
            'f1': f1_score(labels, preds),
            'roc_auc': roc_auc_score(labels, probs)
        }
        
        self.logger.debug(f"Computed metrics: {metrics}")
        return metrics

    #updated this part as well to use the DataLoader for dynamic batch loading - also cleaned the memory after each fold 
    def train_fold(self, fold: int) -> Tuple[Any, Dict, Dict]:
        """
        Train model on a single fold using dynamic data loading with DataLoader.
        
        Args:
            fold (int): Fold number
            
        Returns:
            Tuple[Any, Dict, Dict]: 
                - Trained model
                - Evaluation metrics
                - Training arguments
        """
        self.logger.info(f"Training fold {fold + 1}")  # Log the fold number
        self.logger.info(f"Using pre-tokenized data for training and validation.")

        try:
            # Initialize train and validation datasets from pre-tokenized file
            train_dataset = TextDataset(file_path="../../data/tokenized/tokenized_data.pkl")
            print(train_dataset[0])
            val_dataset = TextDataset(file_path="../../data/tokenized/tokenized_data.pkl")

            # Use DataLoader for dynamic data loading in batches
            train_loader = DataLoader(
                train_dataset, batch_size=32, shuffle=True, num_workers=4
            )
            val_loader = DataLoader(
                val_dataset, batch_size=32, shuffle=False, num_workers=4
            )

### STILL GETTING THIS ERROR Error in bert classifier: np.int64(7), something is wrong with the new method
            # Debugging: Inspect the first batch from train_loader
            for batch in train_loader:
                print("Inspecting first batch:")
                print(f"Batch Input IDs: {batch['input_ids'].shape}, Type: {batch['input_ids'].dtype}")
                print(f"Batch Attention Mask: {batch['attention_mask'].shape}, Type: {batch['attention_mask'].dtype}")
                print(f"Batch Labels: {batch['labels'].shape}, Type: {batch['labels'].dtype}")
                print(f"First Label in Batch: {batch['labels'][0]}")
                break  # Only inspect the first batch

            # Load the BERT model for sequence classification
            model = AutoModelForSequenceClassification.from_pretrained(
                'prajjwal1/bert-tiny', num_labels=2
            ).to(self.device)  # Move model to the appropriate device (CPU/GPU)

            # Define output directory for fold-specific training artifacts
            output_dir = os.path.join(self.models_dir, f"fold_{fold}_{self.timestamp}")
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

            # Define training arguments for the Trainer API
            training_args = TrainingArguments(
                output_dir=output_dir,  # Directory to save model checkpoints and outputs
                learning_rate=3e-5,  # Fixed learning rate for fine-tuning
                per_device_train_batch_size=32,  # Batch size managed by DataLoader
                per_device_eval_batch_size=32,  # Batch size managed by DataLoader
                num_train_epochs=self.config.epochs,  # Number of training epochs
                weight_decay=0.01,  # Regularization to avoid overfitting
                evaluation_strategy="steps",  # Evaluate after a specific number of steps
                eval_steps=500,  # Evaluate every 500 steps
                save_strategy="steps",  # Save checkpoints frequently
                save_steps=500,  # Save checkpoints every 500 steps
                save_total_limit=1,  # Keep only the most recent checkpoint
                load_best_model_at_end=False,  # Disable best model loading to save memory
                metric_for_best_model="f1",  # Use F1 score to evaluate the best model
                logging_dir=f"{self.config.output_dir}/logs",  # Logging directory
                logging_steps=1000,  # Log every 1000 steps
                gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batches
                fp16=False,  # Disable mixed precision for compatibility
                max_grad_norm=1.0,  # Clip gradients to avoid explosion
            )

            # Define Trainer object for managing training and evaluation
            trainer = Trainer(
                model=model,  # Model instance
                args=training_args,  # Training arguments
                train_dataset=train_loader,  # Pass DataLoader for training
                eval_dataset=val_loader,  # Pass DataLoader for validation
                compute_metrics=self.compute_metrics  # Custom metrics computation
            )

            # Train the model and evaluate
            trainer.train()
            metrics = trainer.evaluate()  # Evaluate the model after training
            cleaned_metrics = {k.replace('eval_', ''): v for k, v in metrics.items()}  # Clean metric names

            # Save fold metrics to JSON
            metrics_file = os.path.join(self.logs_dir, f"metrics_fold_{fold}.json")
            with open(metrics_file, 'w') as f:
                json.dump(cleaned_metrics, f, indent=4)
            self.logger.info(f"Metrics for fold {fold + 1} saved to {metrics_file}")

        except Exception as e:
            # Log errors during training
            self.logger.error(f"Error in fold training: {str(e)}")
            raise

        finally:
            # Free up GPU memory by explicitly deleting unused objects
            del model, trainer, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear the CUDA cache
                torch.cuda.synchronize()  # Synchronize CUDA operations to ensure all memory is cleared
            self.logger.info("Cleared GPU memory after fold.")

            # Clean up fold-specific output directory if training failed
            if os.path.exists(output_dir) and len(os.listdir(output_dir)) == 0:
                shutil.rmtree(output_dir)

        # Return the trained model, metrics, and training arguments
        return None, cleaned_metrics, training_args.to_dict()


            
        
        

    def save_model(self, model, prefix="final_model"):
        """
        Save the trained model
        
        Args:
            model: Trained model
            prefix (str): Prefix for the model file name
        """
        model_dir = os.path.join(self.models_dir, f"{prefix}_{self.timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        self.logger.info(f"Model saved to {model_dir}")
        
        
    def train_final_model(self, 
                         X_train: List[str],
                         y_train: np.ndarray,
                         config: Dict) -> Any:
        """
        Train final model on entire training set
        
        Args:
            X_train: Full training texts
            y_train: Full training labels
            config: Best configuration from CV
            
        Returns:
            Any: Trained final model
        """
        self.logger.info("Training final model on complete training set...")
        self.logger.info(f"Training set size: {len(X_train)}")
        self.logger.info(f"Using configuration: {config}")
        
        try:
            train_dataset = TextDataset(file_path= "../../data/tokenized/tokenized_data.pkl")
            model = AutoModelForSequenceClassification.from_pretrained(
                'prajjwal1/bert-tiny', num_labels=2
            ).to(self.device)

            training_args = TrainingArguments(
                output_dir=f"{self.config.output_dir}/bert_final_{self.timestamp}",
                learning_rate=config['learning_rate'],
                per_device_train_batch_size=config['batch_size'],
                num_train_epochs=config['epochs'],
                weight_decay=config['weight_decay'],
                save_strategy="epoch",
                logging_dir=f"{self.config.output_dir}/logs_final_{self.timestamp}",
                logging_steps=10
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset
            )

            trainer.train()

        except Exception as e:
            self.logger.error(f"Error in final model training: {str(e)}")
            raise

        finally:
            del model, trainer, train_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return model
    
    #updated this part as well to use dynamic batch loading
    def evaluate_model(self, 
                   model: AutoModelForSequenceClassification,
                   file_path: str) -> Dict:
        """
        Evaluate model on test set using dynamic data loading.
        
        Args:
            model: Trained model
            file_path: Path to pre-tokenized dataset.
            
        Returns:
            Dict: Evaluation metrics
        """
        self.logger.info("Evaluating model on test set...")

        try:
            # Initialize test dataset and DataLoader
            test_dataset = TextDataset(file_path=file_path)
            test_loader = DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=4
            )

            # Define Trainer for evaluation
            trainer = Trainer(
                model=model,
                compute_metrics=self.compute_metrics
            )

            # Evaluate
            metrics = trainer.evaluate(eval_dataset=test_loader)
            cleaned_metrics = {k.replace('eval_', ''): v for k, v in metrics.items()}

            self.logger.info(f"Evaluation metrics: {cleaned_metrics}")
            return cleaned_metrics

        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise

        
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions on new texts
        
        Args:
            texts (List[str]): List of text samples
            
        Returns:
            np.ndarray: Predicted labels
        """
        if self.model is None:
            raise ValueError("Model hasn't been trained yet")
            
        # Create dataset
        dataset = TextDataset(texts, np.zeros(len(texts)), self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(model=self.model)
        
        # Get predictions
        predictions = trainer.predict(dataset)
        return np.argmax(predictions.predictions, axis=1)
        
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities for new texts
        
        Args:
            texts (List[str]): List of text samples
            
        Returns:
            np.ndarray: Predicted probabilities for positive class
        """
        if self.model is None:
            raise ValueError("Model hasn't been trained yet")
            
        # Create dataset
        dataset = TextDataset(texts, np.zeros(len(texts)), self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(model=self.model)
        
        # Get predictions
        predictions = trainer.predict(dataset)
        return predictions.predictions[:, 1]  # Return probability of positive class