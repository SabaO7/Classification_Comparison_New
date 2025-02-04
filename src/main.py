import pandas as pd
import os
from typing import List, Dict, Tuple
import logging
from datetime import datetime
import json
import sys
import torch
import multiprocessing
from utils.performance_metrics import PerformanceMetrics
from base_classes import ModelConfig
from classifiers.logistic_classifier import LogisticClassifier
from classifiers.bert_classifier import BERTClassifier, TextDataset
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

        # Keep a reference to PerformanceMetrics so we can save/plot time usage
        self.performance_tracker = None
        
        # VisualizationManager for plotting
        self.visual_manager = VisualizationManager(
            os.path.join(self.config.output_dir, "comparison", "visualizations")
        )
    
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
        
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging setup complete. Log file: {log_file}")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load raw data (CSV) with fallback encodings. Simple validation for 'text' and 'class' columns.
        """
        self.logger.info(f"Loading raw data from {file_path}")
        
        if not os.path.exists(file_path):
            self.logger.error(f"Input file not found: {file_path}")
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        try:
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
            
            required_columns = ['text', 'class']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            self.logger.info(f"Initial data statistics: {len(df)} samples")
            df = df.dropna(subset=required_columns)
            
            # Minimal cleaning
            df['text'] = df['text'].str.lower().str.strip()
            df = df[df['text'] != ""]
            self.logger.info(f"Final sample count after cleaning: {len(df)}")
            self.logger.info(f"Class distribution:\n{df['class'].value_counts(normalize=True)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading raw data: {str(e)}")
            raise

    def validate_tokenized_data(self, tokenized_file: str):
        """Loads tokenized data from PyTorch format and validates it."""
        try:
            # ✅ Load tokenized data correctly
            data_dict = torch.load(tokenized_file)
            df = pd.DataFrame.from_dict(data_dict)

            # ✅ Check for valid tokenized format
            invalid_rows = df[df["tokenized"].apply(lambda x: not isinstance(x, dict) or "input_ids" not in x)]
            if not invalid_rows.empty:
                self.logger.warning(f"⚠️ Found {len(invalid_rows)} invalid rows in tokenized dataset!")
                print(invalid_rows.head(5))
                raise ValueError("Tokenized dataset contains improperly formatted rows!")

            self.logger.info(f"✅ Tokenized dataset is correctly formatted with {len(df)} samples.")

            return df

        except Exception as e:
            self.logger.error(f"❌ Error loading tokenized data: {str(e)}")
            raise


    def run_pipeline(
        self, 
        raw_file: str, 
        tokenized_file: str, 
        run_logistic: bool = True, 
        run_bert: bool = True,
        run_few_shot: bool = True,
    ) -> Dict:
        """
        Run the complete classification pipeline with optional classifier selection.
        
        This method loads the raw data (if needed), loads tokenized data (if needed),
        and runs each classifier’s training pipeline.

        Returns:
            dict: Summary of the best-performing classifier or None if no valid results.
        """
        try:
            self.logger.info(f"Starting classification pipeline at {self.timestamp}")

            # PerformanceMetrics tracker
            self.performance_tracker = PerformanceMetrics(self.metrics_dir)

            # ---------------------------------------------------------
            # 1) Conditionally load raw data for Logistic / Few-Shot
            # ---------------------------------------------------------
            raw_df = None
            raw_texts = None
            raw_labels = None

            if run_logistic or run_few_shot:
                self.logger.info(f"Loading raw data from {raw_file}")
                raw_df = self.load_data(raw_file)
                raw_texts = raw_df['text'].tolist()
                raw_labels = raw_df['class'].values

            # ---------------------------------------------------------
            # 2) Conditionally load tokenized data for BERT
            # ---------------------------------------------------------
            tokenized_df = None
            if run_bert:
                self.logger.info(f"Loading tokenized data from {tokenized_file}")
                tokenized_df = pd.read_pickle(tokenized_file)  # ✅ Load the tokenized data first
                dataset = TextDataset(tokenized_df)  # ✅ Pass the DataFrame, not file_path


            # ---------------------------------------------------------
            # 3) Build dictionary of classifiers to run
            # ---------------------------------------------------------
            classifier_configs = [
                ("logistic", LogisticClassifier) if run_logistic else None,
                ("bert", BERTClassifier) if run_bert else None,
                ("few_shot", FewShotClassifier) if run_few_shot else None
            ]

            # Remove None values (for cases where a classifier is disabled)
            classifier_configs = [entry for entry in classifier_configs if entry is not None]

            results = {}

            # ---------------------------------------------------------
            # 4) For each classifier:
            #    - do cross-validation
            #    - final train
            #    - final test eval
            #    - get test metrics + (y_test, y_prob) for ROC
            # ---------------------------------------------------------
            for name, classifier_class in classifier_configs:
                self.logger.info(f"Running {name} classifier...")

                # ✅ Pass dataset only to BERT
                if name == "bert":
                    classifier = classifier_class(self.config, dataset=dataset)
                else:
                    classifier = classifier_class(self.config)

                try:
                    output = self.performance_tracker.track_performance(
                        classifier_name=name,
                        func=classifier.train,
                        texts=(
                            dataset.data['tokenized'] if name == "bert" 
                            else raw_texts
                        ),
                        labels=(
                            dataset.data['class'] if name == "bert"
                            else raw_labels
                        )
                    )

                    (cv_metrics, final_metrics, best_config, 
                    final_labels, final_probs) = output

                    results[name] = (cv_metrics, final_metrics, best_config)

                    if final_labels is not None and final_probs is not None:
                        self.visual_manager.plot_roc_curves(
                            y_true=final_labels,
                            y_prob=final_probs,
                            model_name=name,
                            fold=None
                        )

                except Exception as e:
                    self.logger.error(f"Error in {name} classifier: {str(e)}")
                    continue  # Skip to next classifier if error

            # ---------------------------------------------------------
            # 5) If we got results, save a comparison summary
            # ---------------------------------------------------------
            if results:
                summary = self.save_comparison_results(results)
                self.logger.info("Pipeline completed successfully.")
            else:
                self.logger.warning("No valid results to compare.")
                summary = None

            # ---------------------------------------------------------
            # 6) Free memory if needed
            # ---------------------------------------------------------
            del raw_df, tokenized_df
            import gc
            gc.collect()

            return summary

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def save_comparison_results(self, results: dict) -> dict:
        """
        Example method to compare final results from each classifier.
        
        'results' is a dict like:
            {
              "logistic": ([cv_metrics], final_metrics, best_config),
              "few_shot":  ...,
              "bert": ...
            }
        """
        self.logger.info("Saving comparison results...")
        
        best_model = None
        best_f1_score = 0.0
        
        # 'results[name]' -> (cv_metrics_list, final_test_metrics_dict, best_config_dict)
        for model_name, (fold_metrics, final_metrics, best_config) in results.items():
            # final_metrics is a dictionary with e.g. {'accuracy':..., 'f1':..., ...}
            if final_metrics["f1"] > best_f1_score:
                best_f1_score = final_metrics["f1"]
                best_model = model_name
        
        summary = {
            "best_model": best_model,
            "best_f1_score": best_f1_score
        }
        
        # Save summary as JSON
        out_file = os.path.join(self.metrics_dir, f"comparison_summary_{self.timestamp}.json")
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=4)
        
        self.logger.info(f"Comparison summary saved to {out_file}")
        
        # A) Save & plot performance metrics (time, GPU) from PerformanceTracker
        df_metrics = self.performance_tracker.save_results()
        self.performance_tracker.plot_results(df_metrics)
        
        # B) Optionally, we can also do bar plots for each fold metric
        for model_name, (fold_metrics, final_m, config_m) in results.items():
            self.visual_manager.plot_metrics_across_iterations(
                metrics_list=fold_metrics,    
                mean_metrics=final_m,
                std_metrics={},   # if you want std, store it in final_m or another dict
                model_name=model_name
            )
        
        return summary


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

import multiprocessing

if __name__ == "__main__":
   main()

