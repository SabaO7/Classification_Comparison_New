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

class FewShotClassifier(BaseClassifier):
    """
    Few-shot learning classifier implementation using OpenAI GPT
    
    Implements zero-shot learning approach using GPT-4o
    Uses parallel processing and caching for efficiency
    Includes retry mechanism for API rate limits
    """
    
    def __init__(self, config):
        """
        Initialize Few-shot classifier
        
        Args:
            config (ModelConfig): Configuration object
        """
        super().__init__(config)
        self.logger.info("Initializing Few-Shot classifier...")
        
        # Load API key
        self.logger.info("Loading environment variables...")
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            self.logger.error("OpenAI API key not found in environment variables")
            raise ValueError("OpenAI API key not found")
        
        # Define few-shot examples
        self.logger.info("Setting up few-shot examples...")
        self.examples = [
            {"text": "I feel like cutting myself.", "label": "suicide"},
            {"text": "I am going to the gym.", "label": "non-suicide"},
            {"text": "I want to hurt myself.", "label": "suicide"},
            {"text": "I am feeling happy today.", "label": "non-suicide"},
        ]
        
        # Setup prompt templates
        self.setup_prompt_templates()

    def save_results(self, results, prefix="results"):
        results_file = os.path.join(self.results_dir, f"{prefix}_{self.timestamp}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Results saved to {results_file}")
        
    def setup_prompt_templates(self):
        """Setup the few-shot prompt templates using langchain"""
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
            prefix="You are an expert in mental health and suicide prevention. Carefully analyze the following text and classify it as either 'suicide' or 'non-suicide'. Here are some examples:",
            suffix="Now, classify the following text:\nText: {input}\nLabel:",
            input_variables=["input"],
            example_separator="\n\n"
        )
        self.logger.info("Prompt templates setup complete")
    
    @lru_cache(maxsize=10000)
    def classify_text_cached(self, text: str) -> str:
        """Classify a single text using GPT with caching"""
        prompt = self.few_shot_prompt.format(input=text)
        try:
            response = openai.ChatCompletion.create( 
                model="gpt-3.5-turbo",  # Using 3.5-turbo as it has better quota limits
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=50,
                temperature=0
            )
            # Use old OpenAI format (0.28)
            label = response['choices'][0]['message']['content'].strip().lower()
            return "non-suicide" if "non-suicide" in label else "suicide"
        except openai.error.RateLimitError as e:
            self.logger.error(f"Rate limit error: {str(e)}")
            return "unclassified"
        except openai.error.AuthenticationError as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return "unclassified"
        except Exception as e:
            self.logger.error(f"Error classifying text: {str(e)}")
            return "unclassified"

    def classify_text_with_retries(self, text: str, max_retries: int = 5) -> str:
        """Classify text with retry mechanism"""
        retries = 0
        base_wait_time = 8
        max_wait_time = 64

        while retries < max_retries:
            try:
                return self.classify_text_cached(text)
            except Exception as e:
                error_message = str(e)
                retries += 1
                
                # Check for different error types
                if "insufficient_quota" in error_message:
                    self.logger.error("API quota exceeded")
                    return "unclassified"  # Return unclassified instead of defaulting
                    
                if "rate_limit" in error_message:
                    wait_time = base_wait_time * (2 ** retries)
                    if "try again in" in error_message.lower():
                        try:
                            wait_str = error_message.split("try again in")[1].split("s.")[0]
                            wait_time = float(wait_str.strip()) + 1  # Add 1 second buffer
                        except:
                            pass
                    wait_time = min(wait_time, max_wait_time)
                    
                    self.logger.warning(
                        f"Rate limit reached. Retry {retries}/{max_retries}. "
                        f"Waiting {wait_time:.2f} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Unexpected error: {str(e)}")
                    if retries == max_retries:
                        return "unclassified"
                    time.sleep(base_wait_time)
                    
        return "unclassified"  # Return unclassified after all retries exhausted
        
    
    def classify_batch(self, texts: List[str]) -> List[str]:
        """Classify a batch of texts"""
        self.logger.info(f"Processing batch of {len(texts)} texts")
        results = []
        num_threads = 2  # Reduced from 3 to 2
        batch_size = 20  # Reduced from 25 to 20
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{len(texts)//batch_size + 1}")
            
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
                    
                    if len(results) % 10 == 0:
                        self.logger.info(f"Processed {len(results)}/{len(texts)} texts")
            
            # Add larger delay between batches
            if i + batch_size < len(texts):
                time.sleep(5)  # Increased from 2 to 5 seconds
        
        return [label for _, label in sorted(results, key=lambda x: x[0])]

    def train_fold(self, 
                  X_train: List[str],
                  y_train: np.ndarray,
                  X_val: List[str],
                  y_val: np.ndarray,
                  fold: int) -> Tuple[Any, Dict, Dict]:
        """
        Evaluate one fold using the LLM
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            fold: Current fold number
            
        Returns:
            Tuple[Any, Dict, Dict]: 
                - None (no model to return for few-shot)
                - Metrics for this fold
                - Configuration used
        """
        self.logger.info(f"Evaluating fold {fold+1}")
        self.logger.info(f"Validation set size: {len(X_val)}")
        
        # Classify validation texts
        val_preds = self.classify_batch(X_val)
        
        # Convert predictions and labels to proper format
        val_preds = np.array(val_preds)
        val_probs = (val_preds == 'suicide').astype(float)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, val_preds),
            'precision': precision_score(y_val, val_preds, pos_label='suicide'),
            'recall': recall_score(y_val, val_preds, pos_label='suicide'),
            'f1': f1_score(y_val, val_preds, pos_label='suicide'),
            'roc_auc': roc_auc_score(
                (y_val == 'suicide').astype(int),
                val_probs
            )
        }
        
        self.logger.info(f"Fold {fold+1} metrics: {metrics}")
        
        # Configuration for few-shot is just the examples and prompts
        config = {
            'examples': self.examples,
            'prompt_template': str(self.few_shot_prompt),
            'fold': fold
        }
        
        return None, metrics, config
        
    def train_final_model(self,
                         X_train: List[str],
                         y_train: np.ndarray,
                         config: Dict) -> Any:
        """
        No actual training needed for few-shot model
        
        Args:
            X_train: Training texts
            y_train: Training labels
            config: Best configuration from CV
            
        Returns:
            Any: None (no model to return)
        """
        self.logger.info("Few-shot classifier doesn't require final training")
        return None
        
    def evaluate_model(self,
                      model: Any,
                      X_test: List[str],
                      y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            model: Not used for few-shot
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            Dict: Evaluation metrics
        """
        self.logger.info("Evaluating on test set...")
        self.logger.info(f"Test set size: {len(X_test)}")
        
        # Get predictions
        test_preds = self.classify_batch(X_test)
        test_preds = np.array(test_preds)
        test_probs = (test_preds == 'suicide').astype(float)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, test_preds),
            'precision': precision_score(y_test, test_preds, pos_label='suicide'),
            'recall': recall_score(y_test, test_preds, pos_label='suicide'),
            'f1': f1_score(y_test, test_preds, pos_label='suicide'),
            'roc_auc': roc_auc_score(
                (y_test == 'suicide').astype(int),
                test_probs
            )
        }
        
        self.logger.info(f"Test set metrics: {metrics}")
        
        # Save predictions
        pd.DataFrame({
            'text': X_test,
            'prediction': test_preds,
            'actual': y_test,
            'probability': test_probs
        }).to_csv(f"{self.config.output_dir}/few_shot_test_predictions_{self.timestamp}.csv",
                 index=False)
        
        return metrics

    def predict(self, texts: List[str]) -> List[str]:
        """
        Make predictions on new texts
        
        Args:
            texts (List[str]): List of text samples
            
        Returns:
            List[str]: Predicted labels
        """
        self.logger.info(f"Making predictions on {len(texts)} new texts")
        return self.classify_batch(texts)
        
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities for new texts
        
        Args:
            texts (List[str]): List of text samples
            
        Returns:
            np.ndarray: Predicted probabilities for suicide class
        """
        self.logger.info(f"Getting prediction probabilities for {len(texts)} texts")
        predictions = self.predict(texts)
        # Convert to probabilities (1 for suicide, 0 for non-suicide)
        return np.array([(1.0 if pred == 'suicide' else 0.0) for pred in predictions])