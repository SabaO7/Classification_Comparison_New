import time
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
from typing import List
import json
from datetime import datetime

class PerformanceMetrics:
    """
    Tracks and records performance metrics for classifiers, including
    execution time and GPU memory usage.
    
    Attributes:
        results (list): List to store performance metrics for each classifier.
        output_dir (str): Directory to save metrics and plots.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the PerformanceMetrics class.

        Args:
            output_dir (str): Directory to save performance metrics and plots.
        """
        self.results = []
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        os.makedirs(output_dir, exist_ok=True)

    def track_performance(self, *args, classifier_name, func, **kwargs):
        """
        Track the time and GPU memory usage of a given function.

        Args:
            *args:            Positional arguments to pass along to `func`.
            classifier_name:  Name of the classifier being tracked (keyword-only).
            func:             The function to execute (keyword-only).
            **kwargs:         Keyword arguments to pass along to `func`.

        Returns:
            Any: Result of the function execution.
        """
        start_time = time.time()
        gpu_usage_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Execute the function with the given args/kwargs
        result = func(*args, **kwargs)

        # Capture time taken and GPU memory usage after execution
        end_time = time.time()
        gpu_usage_end = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        time_taken = end_time - start_time
        gpu_usage = gpu_usage_end - gpu_usage_start

        # result[1] should be the final_metrics dict, so we can safely do:
        final_metrics = result[1] if len(result) > 1 else {}

        self.results.append({
            "Classifier": classifier_name,
            "Time (s)": time_taken,
            "GPU Usage (bytes)": gpu_usage,
            "Accuracy": final_metrics.get("accuracy", None),
            "Precision": final_metrics.get("precision", None),
            "Recall": final_metrics.get("recall", None),
            "F1": final_metrics.get("f1", None),
            "ROC AUC": final_metrics.get("roc_auc", None),
        })

        return result
    
    @staticmethod
    def compare_models(baseline_scores: List[float], new_scores: List[float]) -> bool:
        """
        Perform a t-test to compare two models' performance scores.
        """
        t_stat, p_value = stats.ttest_ind(baseline_scores, new_scores)
        return p_value < 0.05
    
    def save_results(self) -> pd.DataFrame:
        """
        Save performance metrics to both CSV and JSON.

        Returns:
            pd.DataFrame: DataFrame containing the recorded metrics.
        """
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.output_dir, f"performance_metrics_{self.timestamp}.csv")
        json_path = os.path.join(self.output_dir, f"performance_metrics_{self.timestamp}.json")

        # Save CSV
        df.to_csv(csv_path, index=False)
        print(f"Performance metrics saved to {csv_path}")

        # Save JSON
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"Performance metrics saved to {json_path}")

        return df

    def plot_results(self, df: pd.DataFrame):
        """
        Plot the recorded performance metrics as a bar chart.
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot execution time
        ax1.set_xlabel("Classifier")
        ax1.set_ylabel("Time (s)", color="tab:blue")
        ax1.bar(df["Classifier"], df["Time (s)"], color="tab:blue", alpha=0.7, label="Time (s)")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Plot GPU usage on a secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("GPU Usage (bytes)", color="tab:orange")
        ax2.plot(df["Classifier"], df["GPU Usage (bytes)"], color="tab:orange", marker="o", label="GPU Usage")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        fig.tight_layout()
        plot_path = os.path.join(self.output_dir, "performance_metrics.png")
        plt.title("Performance Metrics")
        plt.savefig(plot_path)
        plt.close()
        print(f"Performance metrics plot saved to {plot_path}")
