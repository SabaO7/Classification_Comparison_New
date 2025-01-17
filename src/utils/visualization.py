import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import pandas as pd
import os
import logging
from sklearn.metrics import roc_curve, confusion_matrix, auc
import numpy as np
from datetime import datetime

class VisualizationManager:
    """
    Manager class for creating and saving visualizations
    
    Implements comprehensive visualization creation and management
    Includes error handling and logging
    Saves high-quality plots with consistent styling
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize visualization manager
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.setup_logging()
        
        # Set style configurations
        plt.style.use('seaborn-v0_8')
        self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        self.colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6']
        
        self.logger.info(f"Initialized VisualizationManager with output dir: {output_dir}")
        
    def setup_logging(self):
        """Configure logging for visualization manager"""
        log_file = os.path.join(self.output_dir, f'visualization_{self.timestamp}.log')
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Setup stream handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Logging setup complete")
        
    def save_plot(self, 
                 filename: str,
                 fig: Optional[plt.Figure] = None,
                 dpi: int = 300) -> str:
        """
        Save plot with error handling and logging
        
        Args:
            filename (str): Name for the plot file
            fig (Optional[plt.Figure]): Figure to save
            dpi (int): DPI for saved image
            
        Returns:
            str: Path to saved plot
        """
        try:
            filepath = os.path.join(self.output_dir, filename)
            if fig is None:
                fig = plt.gcf()
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"Saved plot to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving plot {filename}: {str(e)}")
            raise
        
    # def plot_metrics_across_iterations(self,
    #                                 metrics_list: List[Dict],
    #                                 mean_metrics: Dict,
    #                                 std_metrics: Dict,
    #                                 model_name: str) -> None:
    #     """Plot metrics across iterations"""
    #     self.logger.info(f"Plotting metrics across iterations for {model_name}")
        
    #     try:
    #         # Convert metrics to regular Python types
    #         metrics_df = pd.DataFrame([
    #             {k: float(v) if isinstance(v, (np.float64, np.float32, np.integer)) else v 
    #             for k, v in m.items()}
    #             for m in metrics_list
    #         ])
            
    #         # Convert mean and std metrics to regular Python floats
    #         mean_metrics = {k: float(v) if isinstance(v, (np.float64, np.float32, np.integer)) else v 
    #                       for k, v in mean_metrics.items()}
    #         std_metrics = {k: float(v) if isinstance(v, (np.float64, np.float32, np.integer)) else v 
    #                       for k, v in std_metrics.items()}
            
    #         fig, ax = plt.subplots(figsize=(12, 8))
            
    #         for metric, color in zip(self.metrics, self.colors):
    #             if metric in metrics_df.columns:
    #                 # Plot metric line
    #                 ax.plot(metrics_df.index, metrics_df[metric],
    #                       label=metric.capitalize(), color=color, linewidth=2)
                    
    #                 # Plot mean line if metric exists in mean_metrics
    #                 if metric in mean_metrics and metric in std_metrics:
    #                     ax.axhline(y=mean_metrics[metric], color=color,
    #                             linestyle='--', label=f'Mean {metric}')
                        
    #                     # Add confidence interval
    #                     ax.fill_between(metrics_df.index,
    #                                   mean_metrics[metric] - std_metrics[metric],
    #                                   mean_metrics[metric] + std_metrics[metric],
    #                                   color=color, alpha=0.1)
            
    #         ax.set_xlabel('Iteration', fontsize=12)
    #         ax.set_ylabel('Score', fontsize=12)
    #         ax.set_title(f'{model_name} Performance Metrics Across Iterations',
    #                     fontsize=14, pad=20)
    #         ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #         ax.grid(True, alpha=0.3)
    #         plt.tight_layout()
            
    #         self.save_plot(f'metrics_across_iterations_{model_name}_{self.timestamp}.png',
    #                       fig)
            
    #     except Exception as e:
    #         self.logger.error(f"Error plotting metrics: {str(e)}")
    #         raise
    
    def plot_metrics_across_iterations(
        self,
        metrics_list: List[Dict[str, float]],
        mean_metrics: Dict[str, float],
        std_metrics: Dict[str, float],
        model_name: str
    ) -> None:
        """
        Plot metrics across iterations with confidence intervals.

        Args:
            metrics_list (List[Dict[str, float]]): List of metrics for each iteration.
            mean_metrics (Dict[str, float]): Mean metrics.
            std_metrics (Dict[str, float]): Standard deviation metrics.
            model_name (str): Name of the model.
        """
        self.logger.info(f"Plotting metrics across iterations for {model_name}")

        try:
            metrics_df = pd.DataFrame(metrics_list)
            fig, ax = plt.subplots(figsize=(12, 8))

            for metric, color in zip(self.metrics, self.colors):
                if metric in metrics_df.columns:
                    # Plot metric line
                    ax.plot(metrics_df.index, metrics_df[metric],
                            label=metric.capitalize(), color=color, linewidth=2)

                    # Plot mean and confidence intervals
                    if metric in mean_metrics and metric in std_metrics:
                        ax.axhline(y=mean_metrics[metric], color=color, linestyle='--',
                                label=f'{metric.capitalize()} Mean')
                        ax.fill_between(
                            metrics_df.index,
                            mean_metrics[metric] - std_metrics[metric],
                            mean_metrics[metric] + std_metrics[metric],
                            color=color, alpha=0.2, label=f'{metric.capitalize()} CI'
                        )

            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'{model_name} Performance Metrics Across Iterations',
                        fontsize=14, pad=20)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            self.save_plot(f'metrics_across_iterations_{model_name}_{self.timestamp}.png', fig)

        except Exception as e:
            self.logger.error(f"Error plotting metrics: {str(e)}")
            raise

        
    def plot_roc_curves(self,
                       y_true: np.ndarray,
                       y_prob: np.ndarray,
                       model_name: str,
                       fold: Optional[int] = None) -> None:
        """
        Plot ROC curve with AUC score
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
            fold (Optional[int]): Fold number if applicable
        """
        fold_str = f"Fold {fold}" if fold is not None else "Final"
        self.logger.info(f"Plotting ROC curve for {model_name} - {fold_str}")
        
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curve - {model_name} {fold_str}', fontsize=14, pad=20)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
            self.save_plot(
                f'roc_curve_{model_name}_{fold_str.lower()}_{self.timestamp}.png',
                fig
            )
            
        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {str(e)}")
            raise
        
    def plot_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            model_name: str,
                            fold: Optional[int] = None) -> None:
        """
        Plot confusion matrix with normalized values
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
            fold (Optional[int]): Fold number if applicable
        """
        fold_str = f"Fold {fold}" if fold is not None else "Final"
        self.logger.info(f"Plotting confusion matrix for {model_name} - {fold_str}")
        
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create subplots for both raw and normalized matrices
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot raw counts
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('True')
            ax1.set_title('Raw Counts')
            
            # Plot normalized values
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2)
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('True')
            ax2.set_title('Normalized')
            
            plt.suptitle(f'Confusion Matrix - {model_name} {fold_str}',
                        fontsize=14, y=1.05)
            
            self.save_plot(
                f'confusion_matrix_{model_name}_{fold_str.lower()}_{self.timestamp}.png',
                fig
            )
            
            # Save raw numbers to CSV
            cm_df = pd.DataFrame(
                cm,
                index=['Actual Negative', 'Actual Positive'],
                columns=['Predicted Negative', 'Predicted Positive']
            )
            cm_df.to_csv(os.path.join(
                self.output_dir,
                f'confusion_matrix_{model_name}_{fold_str.lower()}_{self.timestamp}.csv'
            ))
            
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise
        
    def plot_model_comparison(self,
                            comparison_results: Dict,
                            plot_type: str = 'bar') -> None:
        """
        Create comprehensive model comparison visualizations
        
        Args:
            comparison_results (Dict): Dictionary containing results for each model
            plot_type (str): Type of plot ('bar' or 'heatmap')
        """
        self.logger.info("Creating model comparison visualizations")
        
        try:
            model_names = list(comparison_results.keys())
            
            if plot_type == 'bar':
                # Create individual bar plots for each metric
                for metric in self.metrics:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    x = np.arange(len(model_names))
                    width = 0.35
                    
                    # Get means and standard deviations
                    means = [comparison_results[model]['mean_metrics'][metric]
                            for model in model_names]
                    stds = [comparison_results[model]['std_metrics'][metric]
                           for model in model_names]
                    
                    # Create bar plot
                    ax.bar(x, means, width, yerr=stds, capsize=5)
                    ax.set_xlabel('Model Type', fontsize=12)
                    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12)
                    ax.set_title(f'Model Comparison - {metric.upper()}',
                               fontsize=14, pad=20)
                    ax.set_xticks(x)
                    ax.set_xticklabels(model_names)
                    ax.grid(True, alpha=0.3)
                    
                    self.save_plot(
                        f'model_comparison_{metric}_{self.timestamp}.png',
                        fig
                    )
            
            # Create summary heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare data for heatmap
            data = []
            for model in model_names:
                model_data = []
                for metric in self.metrics:
                    mean = comparison_results[model]['mean_metrics'][metric]
                    std = comparison_results[model]['std_metrics'][metric]
                    model_data.append(f"{mean:.3f}±{std:.3f}")
                data.append(model_data)
            
            # Create heatmap
            sns.heatmap(
                [[float(x.split('±')[0]) for x in row] for row in data],
                annot=np.array(data),
                fmt='',
                xticklabels=self.metrics,
                yticklabels=model_names,
                cmap='YlOrRd',
                ax=ax
            )
            
            ax.set_title('Model Comparison Summary', fontsize=14, pad=20)
            plt.tight_layout()
            
            self.save_plot(f'model_comparison_summary_{self.timestamp}.png', fig)
            
            # Save comparison results to CSV
            comparison_df = pd.DataFrame(data, columns=self.metrics, index=model_names)
            comparison_df.to_csv(os.path.join(
                self.output_dir,
                f'model_comparison_{self.timestamp}.csv'
            ))
            
        except Exception as e:
            self.logger.error(f"Error creating model comparison: {str(e)}")
            raise
        
    def plot_class_distribution(self,
                              class_distributions: List[Dict],
                              model_name: str) -> None:
        """
        Plot class distribution across iterations
        
        Args:
            class_distributions (List[Dict]): List of class distributions
            model_name (str): Name of the model
        """
        self.logger.info(f"Plotting class distribution for {model_name}")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            df = pd.DataFrame(class_distributions)
            df.plot(kind='bar', stacked=True, ax=ax)
            
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Proportion', fontsize=12)
            ax.set_title(f'Class Distribution Across Iterations - {model_name}',
                        fontsize=14, pad=20)
            ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            self.save_plot(
                f'class_distribution_{model_name}_{self.timestamp}.png',
                fig
            )
            
            # Save distribution data to CSV
            df.to_csv(os.path.join(
                self.output_dir,
                f'class_distribution_{model_name}_{self.timestamp}.csv'
            ))
            
        except Exception as e:
            self.logger.error(f"Error plotting class distribution: {str(e)}")
            raise