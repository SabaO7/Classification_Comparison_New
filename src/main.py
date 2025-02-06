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
    cross-validation, final training, and results comparison. Implements comprehensive logging and visualization.
    """

    def __init__(self, config: ModelConfig):
        """Initialize classification pipeline."""
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup directories and logging
        self.setup_directories()
        self.setup_logging()

        self.logger.info("Initialized ClassificationPipeline")

        # Performance tracking and visualization
        # (For performance metrics, we use the overall metrics directory as before.)
        self.performance_tracker = PerformanceMetrics(self.metrics_dir)
        # Retain the original visualization manager for overall comparisons.
        self.visual_manager = VisualizationManager(
            os.path.join(self.config.output_dir, "comparison", "visualizations")
        )

    def setup_directories(self):
        """Create organized directory structure."""
        # Original overall structure
        self.comparison_dir = os.path.join(self.config.output_dir, "comparison")
        self.logs_dir = os.path.join(self.comparison_dir, "logs")
        self.metrics_dir = os.path.join(self.comparison_dir, "metrics")
        self.visualizations_dir = os.path.join(self.comparison_dir, "visualizations")
        self.results_dir = os.path.join(self.comparison_dir, "results")

        for directory in [self.logs_dir, self.metrics_dir, self.visualizations_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)

        # NEW: Build classifier-specific directories under a new folder within outputs.
        # Here we create a run folder (based on timestamp) and within it a classifiers folder,
        # with subfolders for each classifier and further subfolders for CV, final, and visualizations.
        self.run_dir = os.path.join(self.config.output_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.classifiers = ["logistic", "bert", "few_shot"]
        self.classifiers_dir = os.path.join(self.run_dir, "classifiers")
        for clf in self.classifiers:
            for subfolder in ["cv", "final", "visualizations"]:
                os.makedirs(os.path.join(self.classifiers_dir, clf, subfolder), exist_ok=True)
        # Also create an overall results folder inside the run folder.
        self.overall_results_dir = os.path.join(self.run_dir, "overall")
        os.makedirs(self.overall_results_dir, exist_ok=True)

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
        Load raw data (CSV) with fallback encodings and minimal cleaning.
        Validates for required 'text' and 'class' columns.
        """
        self.logger.info(f"Loading raw data from {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"Input file not found: {file_path}")
            raise FileNotFoundError(f"Input file not found: {file_path}")

        try:
            # Try several encodings for robustness
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
            df['text'] = df['text'].str.lower().str.strip()
            df = df[df['text'] != ""]
            self.logger.info(f"Final sample count after cleaning: {len(df)}")
            self.logger.info(f"Class distribution:\n{df['class'].value_counts(normalize=True)}")
            return df

        except Exception as e:
            self.logger.error(f"Error loading raw data: {str(e)}")
            raise

    def validate_tokenized_data(self, tokenized_file: str) -> pd.DataFrame:
        """Loads tokenized data from PyTorch format and validates it."""
        try:
            data_dict = torch.load(tokenized_file)
            df = pd.DataFrame.from_dict(data_dict)

            # Check for valid tokenized format (must be a dict with key "input_ids")
            invalid_rows = df[df["tokenized"].apply(lambda x: not isinstance(x, dict) or "input_ids" not in x)]
            if not invalid_rows.empty:
                self.logger.warning(f"⚠️ Found {len(invalid_rows)} invalid rows in tokenized dataset!")
                self.logger.debug(invalid_rows.head(5))
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
        run_bert: bool = False,
        run_few_shot: bool = False,
    ) -> Dict:
        """
        Run the complete classification pipeline with optional classifier selection.
        Performs cross-validation training and evaluation for each classifier.
        
        Returns:
            dict: Summary of the best-performing classifier.
        """
        try:
            self.logger.info(f"Starting classification pipeline at {self.timestamp}")

            # 1) Conditionally load raw data for Logistic / Few-Shot classifiers
            raw_df = None
            raw_texts = None
            raw_labels = None

            if run_logistic or run_few_shot:
                self.logger.info(f"Loading raw data from {raw_file}")
                raw_df = self.load_data(raw_file)
                if run_few_shot:
                    self.logger.info("Subsampling data to 10% for few-shot learning.")
                    raw_df = raw_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
                raw_texts = raw_df['text'].tolist()
                raw_labels = raw_df['class'].values


            # 2) Conditionally load tokenized data for BERT
            tokenized_df = None
            if run_bert:
                self.logger.info(f"Loading tokenized data from {tokenized_file}")
                tokenized_df = pd.read_pickle(tokenized_file)  # Alternatively, call validate_tokenized_data()
                dataset = TextDataset(tokenized_df)

            # 3) Build dictionary of classifiers to run
            classifier_configs = [
                ("logistic", LogisticClassifier) if run_logistic else None,
                ("bert", BERTClassifier) if run_bert else None,
                ("few_shot", FewShotClassifier) if run_few_shot else None
            ]
            # Remove None values for classifiers that are not selected
            classifier_configs = [entry for entry in classifier_configs if entry is not None]

            results = {}

            # 4) For each classifier, perform cross-validation training and final evaluation
            for name, classifier_class in classifier_configs:
                self.logger.info(f"Running {name} classifier...")

                # For BERT, pass the tokenized dataset; otherwise use raw texts/labels
                if name == "bert":
                    classifier = classifier_class(self.config, dataset=dataset)
                else:
                    classifier = classifier_class(self.config)

                try:
                    output = self.performance_tracker.track_performance(
                        classifier_name=name,
                        func=classifier.train,
                        texts=(dataset.data['tokenized'] if name == "bert" else raw_texts),
                        labels=(dataset.data['class'] if name == "bert" else raw_labels)
                    )
                    # Expected output: (cv_metrics, final_metrics, best_config, final_labels, final_probs)
                    (cv_metrics, final_metrics, best_config, final_labels, final_probs) = output

                    results[name] = (cv_metrics, final_metrics, best_config)

                    # Instead of passing an unexpected keyword argument to plot_roc_curves,
                    # create a new VisualizationManager with the classifier-specific directory.
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
                    continue  # Skip to next classifier if error occurs

            # 5) Save comparison results if any classifier produced results
            if results:
                summary = self.save_comparison_results(results)
                self.logger.info("Pipeline completed successfully.")
            else:
                self.logger.warning("No valid results to compare.")
                summary = None

            # 6) Clean up memory
            del raw_df, tokenized_df
            import gc
            gc.collect()

            return summary

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def save_comparison_results(self, results: dict) -> dict:
        """
        Compare final results from each classifier and save the summary.
        Also saves the full cross-validation results to a JSON file for later use.
        
        'results' is a dict like:
            {
              "logistic": (cv_metrics_list, final_test_metrics_dict, best_config_dict),
              "few_shot":  ...,
              "bert": ...
            }
        """
        self.logger.info("Saving comparison results...")
        best_model = None
        best_f1_score = 0.0

        for model_name, (fold_metrics, final_metrics, best_config) in results.items():
            # Compare based on the final F1-score
            if final_metrics["f1"] > best_f1_score:
                best_f1_score = final_metrics["f1"]
                best_model = model_name

            # Save CV results in the classifier-specific folder.
            clf_cv_dir = os.path.join(self.classifiers_dir, model_name, "cv")
            cv_file = os.path.join(clf_cv_dir, "cross_validation_results.json")
            with open(cv_file, "w") as f:
                json.dump({
                    "fold_metrics": fold_metrics,
                    "final_metrics": final_metrics,
                    "best_config": best_config
                }, f, indent=4)
            self.logger.info(f"Saved CV results for {model_name} to {cv_file}")

            # Create a new VisualizationManager for the classifier-specific visualization folder
            clf_vis_dir = os.path.join(self.classifiers_dir, model_name, "visualizations")
            VisualizationManager(clf_vis_dir).plot_metrics_across_iterations(
                metrics_list=fold_metrics,
                mean_metrics=final_metrics,
                std_metrics={},   # Optionally, include std if available
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

    def load_cross_validation_results(self):
        """Load saved cross-validation results from previous runs."""
        results_file = os.path.join(self.metrics_dir, "cross_validation_results.json")
        if not os.path.exists(results_file):
            self.logger.error("❌ Cross-validation results not found. Run cross-validation first.")
            raise FileNotFoundError("Cross-validation results not found. Run full training first.")

        self.logger.info("✅ Loading saved cross-validation results...")
        with open(results_file, "r") as f:
            results = json.load(f)
        return results

    def select_best_model(self, results):
        """
        Select the best model based on the highest average F1-score computed over the cross-validation folds.
        """
        best_model = None
        best_score = float("-inf")

        for model_name, (fold_metrics, final_metrics, best_config) in results.items():
            if fold_metrics:
                avg_f1 = sum(fold["f1"] for fold in fold_metrics) / len(fold_metrics)
            else:
                avg_f1 = 0.0
            if avg_f1 > best_score:
                best_score = avg_f1
                best_model = model_name

        self.logger.info(f"🏆 Best model selected: {best_model} (Avg F1-score: {best_score:.4f})")
        return best_model

    def train_final_model(self, model_name, X_train, y_train, tokenized_data=None):
        """Train the selected model on the full training data (80% of the dataset)."""
        self.logger.info(f"🚀 Retraining {model_name} on the full 80% training data...")

        model_classes = {
            "logistic": LogisticClassifier,
            "bert": BERTClassifier,
            "few_shot": FewShotClassifier
        }

        # For BERT, pass the tokenized data wrapped in a TextDataset; for others, no dataset is needed.
        if model_name == "bert":
            model = model_classes[model_name](self.config, dataset=TextDataset(tokenized_data))
        else:
            model = model_classes[model_name](self.config)

        # Some classifier classes may have a specialized final training method.
        if hasattr(model, "train_final_model"):
            final_model = model.train_final_model(X_train, y_train)
        else:
            final_model = model.train(X_train, y_train)
        return final_model

    def evaluate_final_model(self, final_model, X_test, y_test, model_name):
        """
        Evaluate the final trained model on the test set and plot the ROC curve.
        Save final test metrics in the classifier-specific folder.
        """
        self.logger.info("📊 Evaluating final model on the 20% test set...")
        y_pred, y_prob = final_model.predict(X_test)
        test_metrics = final_model.evaluate(y_test, y_pred)

        # Save final test metrics to a file (saved in the overall metrics folder as before)
        metrics_file = os.path.join(self.metrics_dir, f"final_test_metrics_{self.timestamp}.json")
        with open(metrics_file, "w") as f:
            json.dump(test_metrics, f, indent=4)
        self.logger.info(f"📁 Saved final test metrics to {metrics_file}")

        # Instead of using the global visualization manager, create one for the classifier-specific directory.
        clf_vis_dir = os.path.join(self.classifiers_dir, model_name, "visualizations")
        VisualizationManager(clf_vis_dir).plot_roc_curves(y_test, y_prob, model_name="final_model", fold=None)

        return test_metrics

    def run_final_training_and_evaluation(self, raw_file, tokenized_file):
        """
        Runs the final model training and evaluation:
          1. Loads previous cross-validation results.
          2. Selects the best model.
          3. Splits the data into 80% training and 20% testing.
          4. Trains the best model on the training data.
          5. Evaluates the model on the test set.
        """
        self.logger.info("🚀 Starting final training and evaluation pipeline...")

        # Load previously saved cross-validation results
        results = self.load_cross_validation_results()

        # Select the best model based on CV results
        best_model = self.select_best_model(results)

        # Load raw dataset and extract texts and labels
        raw_df = self.load_data(raw_file)
        raw_texts = raw_df['text'].tolist()
        raw_labels = raw_df['class'].values

        # Split dataset: 80% train, 20% test
        train_size = int(len(raw_texts) * 0.8)
        X_train, y_train = raw_texts[:train_size], raw_labels[:train_size]
        X_test, y_test = raw_texts[train_size:], raw_labels[train_size:]

        # For BERT, load tokenized data if needed
        tokenized_data = pd.read_pickle(tokenized_file) if best_model == "bert" else None

        # Train the best model on the full training data
        final_model = self.train_final_model(best_model, X_train, y_train, tokenized_data)

        # Evaluate on the test set
        final_metrics = self.evaluate_final_model(final_model, X_test, y_test, best_model)

        self.logger.info(f"🎉 Pipeline complete. Best model: {best_model}, Final F1-score: {final_metrics['f1']:.4f}")

        return {
            "best_model": best_model,
            "final_f1_score": final_metrics["f1"]
        }


def main():
    """Main function to run the pipeline.
    
    By default, the cross-validation pipeline is executed.
    To run final training and evaluation, pass 'final' as a command-line argument.
    """
    try:
        # Set up model configuration (adjust parameters as needed)
        config = ModelConfig(
            batch_size=32,
            learning_rate=2e-5,
            epochs=3,
            output_dir="outputs",
            num_iterations=5
        )
        # Update the file paths as appropriate
        raw_file = "../data/raw/Suicide_Detection.csv"  
        tokenized_file = "../data/tokenized/tokenized_data.pkl"

        pipeline = ClassificationPipeline(config)

        # Choose pipeline based on command-line argument:
        #   "final" -> final training & evaluation; otherwise, run cross-validation.
        if len(sys.argv) > 1 and sys.argv[1] == "final":
            summary = pipeline.run_final_training_and_evaluation(raw_file, tokenized_file)
            if summary:
                print("\n🎉 Final training and evaluation pipeline completed successfully!")
                print(f"🏆 Best Model: {summary['best_model']}")
                print(f"📊 Final F1 Score: {summary['final_f1_score']:.4f}")
        else:
            summary = pipeline.run_pipeline(raw_file, tokenized_file,
                                            run_logistic=False,
                                            run_bert=False,
                                            run_few_shot=True)
            if summary:
                print("\nPipeline (cross-validation) completed successfully!")
                print(f"Best model: {summary['best_model']}")
                print(f"Best F1 Score: {summary['best_f1_score']:.4f}")

    except Exception as e:
        logging.error(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
