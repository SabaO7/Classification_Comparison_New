import pandas as pd
import os
from typing import List, Dict, Tuple
import logging
from datetime import datetime
import json
import sys
import torch
import multiprocessing
import numpy as np

from utils.performance_metrics import PerformanceMetrics
from base_classes import ModelConfig
from classifiers.logistic_classifier import LogisticClassifier
from classifiers.bert_classifier import BERTClassifier
from classifiers.few_shot_classifier import FewShotClassifier
from utils.visualization import VisualizationManager


class ClassificationPipeline:
    """
    Main pipeline for running all classifiers.
    
    Orchestrates execution of multiple classification models, manages data loading, 
    cross-validation, final training, and results comparison. Implements comprehensive
    logging and visualization.
    """

    def __init__(self, config: ModelConfig):
        """Initialize classification pipeline."""
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup directories and logging
        self.setup_directories()
        self.setup_logging()

        self.logger.info("Initialized ClassificationPipeline")

        # Performance tracker for timing, GPU usage, etc.
        self.performance_tracker = PerformanceMetrics(self.metrics_dir)

        # Visualization manager for global comparisons
        self.visual_manager = VisualizationManager(
            os.path.join(self.config.output_dir, "comparison", "visualizations")
        )

    def setup_directories(self):
        """Create top-level 'comparison' directories, plus a run folder with classifier subfolders."""
        # Overall "comparison" folder
        self.comparison_dir = os.path.join(self.config.output_dir, "comparison")
        self.logs_dir = os.path.join(self.comparison_dir, "logs")
        self.metrics_dir = os.path.join(self.comparison_dir, "metrics")
        self.visualizations_dir = os.path.join(self.comparison_dir, "visualizations")
        self.results_dir = os.path.join(self.comparison_dir, "results")

        for directory in [self.logs_dir, self.metrics_dir, self.visualizations_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)

        # A "run" folder for classifier subdirectories
        self.run_dir = os.path.join(self.config.output_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.classifiers = ["logistic", "bert", "few_shot"]
        self.classifiers_dir = os.path.join(self.run_dir, "classifiers")
        for clf in self.classifiers:
            for subfolder in ["cv", "final", "visualizations"]:
                os.makedirs(os.path.join(self.classifiers_dir, clf, subfolder), exist_ok=True)

        # "overall" results in the run folder
        self.overall_results_dir = os.path.join(self.run_dir, "overall")
        os.makedirs(self.overall_results_dir, exist_ok=True)

    def setup_logging(self):
        """Configure logging system for the pipeline."""
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
        Load raw CSV data with fallback encodings. Check for 'text' & 'class' columns, minimal cleaning.
        """
        self.logger.info(f"Loading raw data from {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        try:
            encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not read file with any encoding")

            if 'text' not in df.columns or 'class' not in df.columns:
                raise ValueError("Missing required columns: 'text' or 'class'")

            self.logger.info(f"Initial data statistics: {len(df)} samples")
            df = df.dropna(subset=['text','class'])
            df['text'] = df['text'].astype(str).str.lower().str.strip()
            df = df[df['text'] != ""]
            self.logger.info(f"Final sample count after cleaning: {len(df)}")

            class_dist = df['class'].value_counts(normalize=True)
            self.logger.info(f"Class distribution:\n{class_dist}")
            return df

        except Exception as e:
            self.logger.error(f"Error loading raw data: {str(e)}")
            raise

    def run_pipeline(
        self, 
        raw_file: str, 
        tokenized_file: str, 
        run_logistic: bool = True, 
        run_bert: bool = False,
        run_few_shot: bool = False,
    ) -> Dict:
        """
        Run classification pipeline for logistic, BERT, and/or few-shot. 
        Each classifier sees only 10% of the data. 
        We do cross-validation + final training for each, then compare results.
        """
        try:
            self.logger.info(f"Starting classification pipeline at {self.timestamp}")

            # 1) Possibly load raw data (for logistic/few-shot)
            raw_df = None
            X_log, y_log = None, None
            if run_logistic or run_few_shot:
                self.logger.info(f"Loading raw data from {raw_file}")
                raw_df = self.load_data(raw_file)

                # Take 10% sample
                raw_df = raw_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
                X_log = raw_df["text"].tolist()
                y_log = raw_df["class"].values  # still strings "suicide"/"non-suicide" is fine

            # 2) Possibly load pre-tokenized data for BERT
            bert_df = None
            X_bert, y_bert = None, None
            if run_bert:
                self.logger.info(f"Loading tokenized data from {tokenized_file}")
                bert_df = pd.read_pickle(tokenized_file)

                # Sample 10%
                bert_df = bert_df.sample(frac=0.1, random_state=42).reset_index(drop=True)

                # We'll create numeric index array for the BERT classifier
                X_bert = np.arange(len(bert_df))
                y_bert = np.where(bert_df["class"]=="suicide", 1, 0)

            # 3) Build a list of (name, classifier_class) for each to run
            classifier_configs = []
            if run_logistic:
                classifier_configs.append(("logistic", LogisticClassifier))
            if run_bert:
                classifier_configs.append(("bert", BERTClassifier))
            if run_few_shot:
                classifier_configs.append(("few_shot", FewShotClassifier))

            results = {}

            # 4) Train each classifier
            for name, ClfClass in classifier_configs:
                self.logger.info(f"Running {name} classifier...")

                # Create classifier instance
                if name == "logistic":
                    classifier = ClfClass(self.config)
                    X_input, y_input = X_log, y_log

                elif name == "bert":
                    classifier = ClfClass(self.config, df=bert_df)
                    X_input, y_input = X_bert, y_bert

                elif name == "few_shot":
                    classifier = ClfClass(self.config)
                    X_input, y_input = X_log, y_log

                # 5) Track performance (time, GPU usage)
                try:
                    out = self.performance_tracker.track_performance(
                        classifier_name=name,
                        func=classifier.train,
                        *[X_input, y_input]
                    )
                    (cv_metrics, final_metrics, best_config) = out[:3]
                    final_labels, final_probs = out[3], out[4]

                    results[name] = (cv_metrics, final_metrics, best_config)

                    # Possibly plot ROC
                    if final_labels is not None and final_probs is not None:
                        clf_vis_dir = os.path.join(self.classifiers_dir, name, "visualizations")
                        VisualizationManager(clf_vis_dir).plot_roc_curves(
                            y_true=final_labels,
                            y_prob=final_probs,
                            model_name=name,
                            fold=None
                        )

                except Exception as e:
                    self.logger.error(f"Error in {name} classifier: {str(e)}")
                    continue

            # 6) Compare results
            if results:
                summary = self.save_comparison_results(results)
                self.logger.info("Pipeline completed successfully.")
            else:
                self.logger.warning("No valid results to compare.")
                summary = None

            # Cleanup
            del raw_df, bert_df
            import gc
            gc.collect()

            return summary

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def save_comparison_results(self, results: dict) -> dict:
        """
        Compare final results from each classifier, pick best model by final F1, 
        and save summary in overall folder.
        """
        self.logger.info("Saving comparison results...")
        best_model = None
        best_f1_score = 0.0

        for model_name, (fold_metrics, final_metrics, best_config) in results.items():
            # Check final F1
            if final_metrics and "f1" in final_metrics:
                if final_metrics["f1"] > best_f1_score:
                    best_f1_score = final_metrics["f1"]
                    best_model = model_name

            # Save cross-validation results inside the classifier-specific "cv" folder
            clf_cv_dir = os.path.join(self.classifiers_dir, model_name, "cv")
            cv_file = os.path.join(clf_cv_dir, "cross_validation_results.json")
            with open(cv_file, "w") as f:
                json.dump({
                    "fold_metrics": fold_metrics,
                    "final_metrics": final_metrics,
                    "best_config": best_config
                }, f, indent=4)
            self.logger.info(f"Saved CV results for {model_name} to {cv_file}")

            # Optionally plot metrics across folds
            if fold_metrics:
                clf_vis_dir = os.path.join(self.classifiers_dir, model_name, "visualizations")
                VisualizationManager(clf_vis_dir).plot_metrics_across_iterations(
                    metrics_list=fold_metrics,
                    mean_metrics=final_metrics,
                    std_metrics={},  # or real std if you have it
                    model_name=model_name
                )

        summary = {
            "best_model": best_model,
            "best_f1_score": best_f1_score
        }
        overall_summary_file = os.path.join(self.overall_results_dir, "comparison_summary.json")
        with open(overall_summary_file, "w") as f:
            json.dump(summary, f, indent=4)

        self.logger.info(f"Overall comparison summary saved to {overall_summary_file}")
        return summary


    # --------------------------------------------------------------------
    # Optionally, a "run_final_training_and_evaluation" approach if needed
    # --------------------------------------------------------------------
    def run_final_training_and_evaluation(self, raw_file, tokenized_file):
        """
        Example approach if you want to re-run final training after picking best model from
        stored cross-validation results. 
        But this is optional.
        """
        self.logger.info("🚀 Starting final training/evaluation pipeline... (optional)")

        # 1. Load cross-validation results from a previous run
        # 2. Decide best model
        # 3. Load raw or tokenized data
        # 4. Train final model on entire 80%
        # 5. Evaluate on 20%

        pass


def main():
    """Main function to run the pipeline."""
    try:
        # 1) Build config
        config = ModelConfig(
            batch_size=32,
            learning_rate=2e-5,
            epochs=3,
            output_dir="outputs",
            num_iterations=5
        )

        # 2) Create pipeline
        pipeline = ClassificationPipeline(config)

        # 3) Decide file paths
        raw_file = "../data/raw/Suicide_Detection.csv"
        tokenized_file = "../data/tokenized/tokenized_data.pkl"

        # 4) If "final" in arguments, do final training, else do cross-validation pipeline
        if len(sys.argv) > 1 and sys.argv[1] == "final":
            pipeline.run_final_training_and_evaluation(raw_file, tokenized_file)
        else:
            # By default, run logistic, BERT, and few-shot
            summary = pipeline.run_pipeline(
                raw_file, 
                tokenized_file,
                run_logistic=True,
                run_bert=True,
                run_few_shot=True
            )
            if summary:
                print("\nPipeline (cross-validation) completed successfully!")
                print(f"Best model: {summary['best_model']}")
                print(f"Best F1 Score: {summary['best_f1_score']:.4f}")

    except Exception as e:
        logging.error(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
