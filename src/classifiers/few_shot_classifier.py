import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from base_classes import BaseClassifier, ModelConfig
import openai
from dotenv import load_dotenv
import os
import time
import json
from functools import lru_cache
import concurrent.futures
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from openai import OpenAIError

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)


class FewShotClassifier(BaseClassifier):
    """
    Few-shot learning classifier implementation using OpenAI GPT.
    
    Implements a zero-shot or few-shot approach using GPT.
    Uses parallel processing, caching for efficiency,
    and a retry mechanism for API rate limits.

    """
    
    def __init__(self, config: ModelConfig, clf_dirs: dict):
        """
        Args:
            config (ModelConfig): Training configuration
            clf_dirs (dict): Dictionary of subdirectories (cv, final, etc.) from pipeline
        """
        super().__init__(config, clf_dirs)
        self.logger.info("Initializing Few-Shot classifier...")
        
        # Check for API key
        self.logger.info("Loading environment variables...")
        if not api_key:
            self.logger.error("OpenAI API key not found in environment variables")
            raise ValueError("OpenAI API key not found")
        
        # Define few-shot examples
        self.logger.info("Setting up few-shot examples...")
        self.examples = [
            {"text": "I feel like cutting myself.", "label": "suicide"},
            {"text": "I am going to the gym.",      "label": "non-suicide"},
            {"text": "I want to hurt myself.",      "label": "suicide"},
            {"text": "I am feeling happy today.",   "label": "non-suicide"},
        ]
        
        # Setup prompt templates
        self.setup_prompt_templates()

    def prepare_data(
        self, 
        X: List[str], 
        y: np.ndarray, 
        random_state: int = 42
    ) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
        """
        Split 10% dataset into 80% reference (train) and 20% validation (inference).
        """
        self.logger.info("Splitting dataset into 80% training (reference) and 20% validation (inference).")
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=0.8, random_state=random_state
        )

        self.logger.info(f"Data split: {len(X_train)} training (reference), {len(X_val)} validation.")
        return X_train, y_train, X_val, y_val

    def save_results(self, results, prefix="results"):
        """
        Save final results in a JSON file inside 'final' subdir.
        """
        results_file = os.path.join(
            self.clf_dirs["final"], 
            f"{prefix}_{self.timestamp}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Results saved to {results_file}")
        
    def setup_prompt_templates(self):
        """Set up the few-shot prompt templates using langchain."""
        self.logger.info("Setting up prompt templates...")
        
        example_template = """
        Text: {text}
        Label: {label}
        """
        example_prompt = PromptTemplate(
            input_variables=["text", "label"],
            template=example_template
        )
        
        self.few_shot_prompt = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=example_prompt,
            prefix=(
                "You are an expert in mental health and suicide prevention. "
                "Carefully analyze the following text and classify it "
                "as either 'suicide' or 'non-suicide'. Here are some examples:"
            ),
            suffix="Now, classify the following text:\nText: {input}\nLabel:",
            input_variables=["input"],
            example_separator="\n\n"
        )
        self.logger.info("Prompt templates setup complete")
    
    @lru_cache(maxsize=50000)
    def classify_text_cached(self, text: str) -> str:
        """
        Classify a single text using GPT with caching (to avoid repeated API calls).
        """
        prompt = self.few_shot_prompt.format(input=text)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=50,
                temperature=0
            )
            label = response.choices[0].message.content.strip().lower()
            return "non-suicide" if "non-suicide" in label else "suicide"
        except OpenAIError as e:
            self.logger.error(f"Rate limit error: {str(e)}")
            return "unclassified"
        except openai.error.AuthenticationError as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return "unclassified"
        except Exception as e:
            self.logger.error(f"Error classifying text: {str(e)}")
            return "unclassified"

    def classify_text_with_retries(self, text: str, max_retries: int = 5) -> str:
        """
        Classify text with a retry mechanism (exponential backoff) for rate limits / errors.
        """
        retries = 0
        base_wait_time = 8
        max_wait_time = 64

        while retries < max_retries:
            try:
                return self.classify_text_cached(text)
            except Exception as e:
                error_message = str(e)
                retries += 1
                
                if "insufficient_quota" in error_message:
                    self.logger.error("API quota exceeded")
                    return "unclassified"

                if "rate_limit" in error_message:
                    wait_time = base_wait_time * (2 ** retries)
                    # If error message has "try again in Xs." we can parse that
                    if "try again in" in error_message.lower():
                        try:
                            wait_str = error_message.split("try again in")[1].split("s.")[0]
                            wait_time = float(wait_str.strip()) + 1
                        except:
                            pass
                    wait_time = min(wait_time, max_wait_time)
                    
                    self.logger.warning(
                        f"Rate limit reached. Retry {retries}/{max_retries}. Waiting {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Unexpected error: {str(e)}")
                    if retries == max_retries:
                        return "unclassified"
                    time.sleep(base_wait_time)
                    
        return "unclassified"

    def classify_batch(self, texts: List[str]) -> List[str]:
        """
        Classify a batch of texts in parallel threads, chunked to avoid rate limits.
        """
        self.logger.info(f"Processing batch of {len(texts)} texts")
        results = []
        num_threads = min(10, os.cpu_count())  
        batch_size = 50
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            self.logger.info(f"Processing batch {i // batch_size + 1}/{(len(texts) // batch_size) + 1}")
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_index = {
                    executor.submit(self.classify_text_with_retries, text): idx + i
                    for idx, text in enumerate(batch)
                }
                
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        label = future.result()
                        results.append((index, label))
                    except Exception as e:
                        self.logger.error(f"Failed to classify text at index {index}: {str(e)}")
                        results.append((index, "unclassified"))
                    
                    # Periodic logging
                    if len(results) % 10 == 0:
                        self.logger.info(f"Processed {len(results)}/{len(texts)} texts")
            
            # Small delay between batches
            if i + batch_size < len(texts):
                time.sleep(5)
        
        # Sort by original index so outputs line up with inputs
        return [label for _, label in sorted(results, key=lambda x: x[0])]

    def train_fold(
        self, 
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
        fold: int
    ) -> Tuple[Any, Dict, Dict]:
        """
        Evaluate one cross-validation fold in few-shot mode.
        
        Because it's few-shot, we basically do no training:
        we only classify the validation set using GPT.
        """
        self.logger.info(f"Evaluating fold {fold+1}")
        self.logger.info(
            "Few-shot: 80% of the (10%) dataset is reference, inference on 20% validation."
        )
        self.logger.info(f"Validation set size: {len(X_val)}")
        
        # Classify the validation texts
        val_preds = self.classify_batch(X_val)
        val_preds = np.array(val_preds)

        # Clean up unexpected labels
        valid_labels = {"suicide", "non-suicide"}
        val_preds = np.array([
            label if label in valid_labels else "non-suicide"
            for label in val_preds
        ])

        # Convert predictions to probabilities (1 for suicide, 0 otherwise)
        val_probs = (val_preds == 'suicide').astype(float)

        # If y_val is numeric, convert it to text
        if isinstance(y_val[0], (int, np.integer)):
            y_val = np.array([
                'suicide' if label == 1 else 'non-suicide'
                for label in y_val
            ])

        # Compute metrics
        metrics_dict = {
            'accuracy':  accuracy_score(y_val, val_preds),
            'precision': precision_score(y_val, val_preds, pos_label='suicide'),
            'recall':    recall_score(y_val, val_preds, pos_label='suicide'),
            'f1':        f1_score(y_val, val_preds, pos_label='suicide'),
            'roc_auc':   roc_auc_score((y_val == 'suicide').astype(int), val_probs)
        }
        self.logger.info(f"Fold {fold+1} metrics: {metrics_dict}")
        
        # "model" is None, as there's no actual training
        config = {
            'examples': self.examples,
            'prompt_template': str(self.few_shot_prompt),
            'fold': fold
        }
        return None, metrics_dict, config
        
    def train_final_model(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        config: Dict
    ) -> Any:
        """
        No actual final training is needed for few-shot. 
        We remain in inference mode.
        """
        self.logger.info("Few-shot classifier does not require final training.")
        return None
        
    def evaluate_model(
        self,
        model: Any,
        X_test: List[str],
        y_test: np.ndarray
    ) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Evaluate on a test set for binary classification in few-shot mode.
        """
        self.logger.info("Evaluating in few-shot mode on the test set...")
        self.logger.info(f"Test set size: {len(X_test)}")
        
        # Classify test
        test_preds = self.classify_batch(X_test)
        test_preds = np.array(test_preds)

        # Clean up labels
        valid_labels = {"suicide", "non-suicide"}
        test_preds = np.array([
            label if label in valid_labels else "non-suicide"
            for label in test_preds
        ])

        # If y_test is numeric, convert to text
        if isinstance(y_test[0], (int, np.integer)):
            y_test = np.array(['suicide' if lbl == 1 else 'non-suicide' for lbl in y_test])

        # Convert to probabilities
        test_probs = (test_preds == 'suicide').astype(float)

        try:
            metrics_dict = {
                'accuracy':  accuracy_score(y_test, test_preds),
                'precision': precision_score(y_test, test_preds, pos_label='suicide'),
                'recall':    recall_score(y_test, test_preds, pos_label='suicide'),
                'f1':        f1_score(y_test, test_preds, pos_label='suicide'),
                'roc_auc':   roc_auc_score((y_test == 'suicide').astype(int), test_probs)
            }
        except ValueError as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            self.logger.error(f"Unique test labels: {np.unique(y_test)}")
            self.logger.error(f"Unique predictions: {np.unique(test_preds)}")
            raise
        
        self.logger.info(f"Test set metrics: {metrics_dict}")

        # Save predictions to CSV in the 'final' folder
        output_path = os.path.join(
            self.clf_dirs["final"], 
            f"few_shot_test_predictions_{self.timestamp}.csv"
        )
        pd.DataFrame({
            'text':        X_test,
            'prediction':  test_preds,
            'actual':      y_test,
            'probability': test_probs
        }).to_csv(output_path, index=False)
        
        self.logger.info(f"Predictions saved to {output_path}")

        # Convert y_test to binary for returning
        test_labels_binary = (y_test == "suicide").astype(int)
        return metrics_dict, test_labels_binary, test_probs

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict on new texts in few-shot mode (GPT).
        """
        self.logger.info(f"Making predictions on {len(texts)} new texts (few-shot inference).")
        return self.classify_batch(texts)
        
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get probabilities for 'suicide' class in few-shot mode.
        """
        self.logger.info(f"Getting probabilities for {len(texts)} texts (few-shot inference).")
        preds = self.predict(texts)
        return np.array([1.0 if p == 'suicide' else 0.0 for p in preds])
