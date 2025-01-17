import pandas as pd
import os
from typing import List, Dict, Tuple
import logging
from datetime import datetime
import json
import sys
import torch
from utils.performance_metrics import PerformanceMetrics
from base_classes import ModelConfig, BaseClassifier
from classifiers.logistic_classifier import LogisticClassifier
from classifiers.bert_classifier import BERTClassifier
from classifiers.few_shot_classifier import FewShotClassifier
from utils.visualization import VisualizationManager


class ClassificationPipeline:
    """
    Main pipeline for running all classifiers.
    
    Orchestrates execution of multiple classification models, manages data loading, preprocessing,
    and results comparison. Implements comprehensive logging and visualization.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize classification pipeline."""
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup directories and logging
        self.setup_directories()
        self.setup_logging()
        
        self.logger.info("Initialized ClassificationPipeline")
    
    def setup_directories(self):
        """Create organized directory structure."""
        self.comparison_dir = os.path.join(self.config.output_dir, "comparison")
        self.logs_dir = os.path.join(self.comparison_dir, "logs")
        self.metrics_dir = os.path.join(self.comparison_dir, "metrics")
        self.visualizations_dir = os.path.join(self.comparison_dir, "visualizations")
        self.results_dir = os.path.join(self.comparison_dir, "results")

        for directory in [self.logs_dir, self.metrics_dir, self.visualizations_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)

    def setup_logging(self):
        """Configure comprehensive logging system."""
        log_file = os.path.join(self.logs_dir, f'pipeline_{self.timestamp}.log')
        
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler with detailed logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler with info-level logging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging setup complete. Log file: {log_file}")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess raw data with validation.
        
        Args:
            file_path (str): Path to the raw dataset file.
            
        Returns:
            pd.DataFrame: Loaded and preprocessed DataFrame.
        """
        self.logger.info(f"Loading raw data from {file_path}")
        
        if not os.path.exists(file_path):
            self.logger.error(f"Input file not found: {file_path}")
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        try:
            # Attempt to load with multiple encodings
            encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.logger.debug(f"Attempting to read with {encoding} encoding")
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not read file with any encoding")
            
            # Validate required columns
            required_columns = ['text', 'class']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Log and preprocess
            self.logger.info(f"Initial data statistics: Total samples: {len(df)}")
            df = df.dropna(subset=required_columns)
            initial_size = len(df)
            df['text'] = df['text'].apply(BaseClassifier.clean_text)
            df = df[df['text'].str.strip() != ""]
            final_size = len(df)
            
            # Log preprocessing results
            self.logger.info(f"Removed {initial_size - final_size} empty/invalid samples.")
            self.logger.info(f"Final sample count: {final_size}")
            self.logger.info(f"Class distribution:\n{df['class'].value_counts(normalize=True)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading raw data: {str(e)}")
            raise

    def run_pipeline(self, raw_file: str, tokenized_file: str):
        """
        Run the complete classification pipeline.
        
        Args:
            raw_file (str): Path to the raw dataset file.
            tokenized_file (str): Path to the pre-tokenized dataset file.
            
        Returns:
            Dict: Summary of the best-performing classifier, or None if no results.
        """
        try:
            self.logger.info(f"Starting classification pipeline at {self.timestamp}")

            # Load raw data
            raw_df = self.load_data(raw_file)
            raw_texts = raw_df['text'].tolist()
            raw_labels = raw_df['class'].values

            # Load tokenized data for BERTClassifier
            self.logger.info(f"Loading tokenized data from {tokenized_file}")
            tokenized_df = pd.read_pickle(tokenized_file)

            # Initialize classifiers
            classifier_configs = {
                'logistic': LogisticClassifier,
                'bert': BERTClassifier,
                'few_shot': FewShotClassifier
            }
            performance_tracker = PerformanceMetrics(self.metrics_dir)

            results = {}
            for name, classifier_class in classifier_configs.items():
                try:
                    self.logger.info(f"Running {name} classifier...")
                    classifier = classifier_class(self.config)

                    if name == "bert":
                        # Tokenized data for BERT
                        iteration_metrics, mean_metrics, std_metrics = performance_tracker.track_performance(
                            name, classifier.train, tokenized_df['tokenized'], tokenized_df['class']
                        )
                    else:
                        # Raw data for Logistic and Few-Shot
                        iteration_metrics, mean_metrics, std_metrics = performance_tracker.track_performance(
                            name, classifier.train, raw_texts, raw_labels
                        )

                    results[name] = (iteration_metrics, mean_metrics, std_metrics)

                except Exception as e:
                    self.logger.error(f"Error in {name} classifier: {str(e)}")
                    continue

            # Save and visualize results
            if results:
                summary = self.save_comparison_results(results)
                self.logger.info("Pipeline completed successfully.")
                return summary
            else:
                self.logger.warning("No valid results to compare.")
                return None

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the pipeline."""
    try:
        config = ModelConfig(
            batch_size=32,
            learning_rate=2e-5,
            epochs=3,
            output_dir="outputs",
            num_iterations=5
        )
        raw_file = "../data/raw/Suicide_Detection.csv"
        tokenized_file = "../data/tokenized/tokenized_data.pkl"

        pipeline = ClassificationPipeline(config)
        summary = pipeline.run_pipeline(raw_file, tokenized_file)
        if summary:
            print("\nPipeline completed successfully!")
            print(f"Best model: {summary['best_model']}")
            print(f"Best F1 score: {summary['best_f1_score']:.4f}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
