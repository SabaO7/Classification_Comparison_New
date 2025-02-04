import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
from datetime import datetime
import json
import sys
from typing import Tuple, Dict
import argparse
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import gc  # For garbage collection
import torch


class DataPreprocessor:
    """
    Class for handling data preprocessing and sampling with memory optimizations.
    
    Implements comprehensive data cleaning, balancing, and validation.
    Includes detailed logging and error tracking.
    Saves processing statistics and data quality metrics.
    Memory usage is optimized by early dataset sampling and in-place operations.
    """
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        Initialize DataPreprocessor.

        Args:
            raw_data_path (str): Path to raw data directory.
            processed_data_path (str): Path to save processed data.
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Validate that raw data path exists.
        if not os.path.exists(raw_data_path):
            raise ValueError(f"Raw data path does not exist: {raw_data_path}")
        
        # Create the directory for processed data if it doesn't exist.
        os.makedirs(processed_data_path, exist_ok=True)
        
        # Setup logging for debugging and processing information.
        self.setup_logging()
        
        self.logger.info("Initialized DataPreprocessor")
        self.logger.info(f"Raw data path: {raw_data_path}")
        self.logger.info(f"Processed data path: {processed_data_path}")
    
    def setup_logging(self):
        """
        Configure comprehensive logging system.
        
        Creates both file and console handlers for detailed debug and info level logging.
        """
        log_file = os.path.join(self.processed_data_path, f'preprocessing_{self.timestamp}.log')
        
        # Define a logging format.
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler: logs debug messages to a file.
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler: logs info messages to stdout.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup the logger.
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers if logger already has them.
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging setup complete. Log file: {log_file}")
        
    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """
        Load raw data with robust encoding handling.

        Args:
            filename (str): Name of the raw data file.
            
        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        file_path = os.path.join(self.raw_data_path, filename)
        self.logger.info(f"Attempting to load file: {file_path}")
        
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try a list of encodings to successfully load the file.
        encodings = ['utf-8', 'ISO-8859-1', 'cp1252', 'latin1']
        
        for encoding in encodings:
            try:
                self.logger.debug(f"Trying encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding)
                self.logger.info(f"Successfully loaded data with {encoding} encoding")
                
                # Validate required columns.
                required_columns = ['text', 'class']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                return df
                
            except UnicodeDecodeError:
                self.logger.debug(f"Failed to read with {encoding} encoding")
                continue
            except Exception as e:
                self.logger.error(f"Error reading file with {encoding} encoding: {str(e)}")
                continue
            
        raise ValueError(f"Could not read file {filename} with any encoding")
    
    def log_dataframe_stats(self, df: pd.DataFrame, stage: str) -> Dict:
        """
        Log comprehensive dataframe statistics.

        Args:
            df (pd.DataFrame): DataFrame to analyze.
            stage (str): Current processing stage.
            
        Returns:
            Dict: A dictionary of DataFrame statistics.
        """
        stats = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'class_distribution': df['class'].value_counts().to_dict(),
            'text_length_stats': df['text'].str.len().describe().to_dict(),
            'unique_texts': len(df['text'].unique()),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2  # in MB
        }
        
        self.logger.info(f"\nDataFrame Statistics ({stage}):")
        self.logger.info(f"Total rows: {stats['total_rows']}")
        self.logger.info(f"Class distribution:\n{pd.Series(stats['class_distribution'])}")
        self.logger.info(f"Missing values:\n{pd.Series(stats['missing_values'])}")
        self.logger.info(f"Text length statistics:\n{pd.Series(stats['text_length_stats'])}")
        self.logger.info(f"Unique texts: {stats['unique_texts']}")
        self.logger.info(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")
        
        return stats
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.

        Args:
            text (str): Input text.
            
        Returns:
            str: Cleaned text.
        """
        try:
            # Return empty string if text is NaN.
            if pd.isna(text):
                return ""
                
            # Convert text to string and lower case.
            text = str(text).lower()
            
            # Remove extra whitespace.
            text = ' '.join(text.split())
            
            # Warn if text is very short after cleaning.
            if len(text) < 5:
                self.logger.warning(f"Very short text after cleaning: '{text}'")
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error cleaning text: '{text[:100]}...' - {str(e)}")
            return ""
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data with comprehensive validation.

        Args:
            df (pd.DataFrame): Raw dataframe.
            
        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        self.logger.info("Starting data preprocessing...")
        
        # Log initial statistics of the raw dataframe.
        initial_stats = self.log_dataframe_stats(df, "initial")
        
        # Clean text data in-place to reduce memory overhead.
        self.logger.info("Cleaning text data...")
        df['text'] = df['text'].apply(self.clean_text)
        
        # Remove rows with empty or null text values.
        initial_rows = len(df)
        df.dropna(subset=['text'], inplace=True)
        df = df[df['text'].str.strip() != ""].copy() #copied it to improve performance/memory
        removed_rows = initial_rows - len(df)
        self.logger.info(f"Removed {removed_rows} rows with empty or null text")
        
        # Remove duplicate text entries.
        initial_rows = len(df)
        df.drop_duplicates(subset=['text'], inplace=True)
        removed_duplicates = initial_rows - len(df)
        self.logger.info(f"Removed {removed_duplicates} duplicate texts")
        
        # Standardize class labels
        df['class'] = df['class'].astype(str).str.lower().str.strip()

        # Map text labels to numeric values
        class_mapping = {'suicide': 1, 'non-suicide': 0}
        df['class'] = df['class'].map(class_mapping)

        # Remove rows with invalid labels
        df.dropna(subset=['class'], inplace=True)

        # Ensure class column is integer type
        df['class'] = df['class'].astype(int)

        self.logger.info(f"Updated class labels: {df['class'].unique()}")

        
        # Log final statistics after preprocessing.
        final_stats = self.log_dataframe_stats(df, "after_preprocessing")
        
        
        del initial_stats  # Free memory of statistics dictionary
        del final_stats    # Free memory of statistics dictionary
        gc.collect() 
        return df
        
    def sample_balanced_data(self, df: pd.DataFrame, sample_fraction: float = 1.0) -> pd.DataFrame:
        """
        Sample data while maintaining class distribution.

        Args:
            df (pd.DataFrame): Input dataframe.
            sample_fraction (float): Fraction of data to sample from each class.
            
        Returns:
            pd.DataFrame: Balanced and sampled dataframe.
        """
        self.logger.info(f"Sampling {sample_fraction*100}% of data from each class for balancing...")
        
        # Validate input fraction.
        if not 0 < sample_fraction <= 1:
            raise ValueError(f"Invalid sample_fraction: {sample_fraction}")
        
        # Get class distribution counts.
        class_counts = df['class'].value_counts()
        # Determine the number of samples to extract per class based on the minimum count.
        min_class_count = int(class_counts.min() * sample_fraction)
        
        self.logger.info("Original class distribution:")
        self.logger.info(class_counts)
        
        sampled_dfs = []
        # Iterate over each class to sample data equally.
        for class_label in class_counts.index:
            class_df = df[df['class'] == class_label]
            if len(class_df) < min_class_count:
                self.logger.warning(
                    f"Insufficient samples for class {class_label}. "
                    f"Required: {min_class_count}, Available: {len(class_df)}"
                )
            sampled_class = class_df.sample(n=min_class_count, random_state=42)
            sampled_dfs.append(sampled_class)
            self.logger.info(f"Sampled {min_class_count} records from class {class_label}")
        
        # Combine the balanced samples from all classes.
        sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        # Shuffle the final dataset to mix classes.
        sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Log final class distribution after sampling.
        final_dist = sampled_df['class'].value_counts()
        self.logger.info("Final class distribution after balanced sampling:")
        self.logger.info(final_dist)
        
        return sampled_df
        
    def pre_tokenize_and_save(self, tokenizer, output_path: str, sample_fraction: float = 1.0):
        """Pre-tokenize the dataset for BERT and save it in PyTorch-compatible format."""
        self.logger.info("Starting pre-tokenization for BERT...")

        try:
            raw_data_file = "Suicide_Detection.csv"
            data = self.load_raw_data(raw_data_file)

            # Sample data for efficiency
            if sample_fraction < 1.0:
                data = data.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

            if 'text' not in data.columns:
                raise ValueError("❌ Error: The dataset must have a 'text' column.")

            def tokenize_text(text):
                encoding = tokenizer(text, truncation=True, padding='max_length', max_length=64, return_tensors="pt")
                return {
                    "input_ids": encoding["input_ids"].squeeze(0).tolist(),
                    "attention_mask": encoding["attention_mask"].squeeze(0).tolist(),
                    "token_type_ids": encoding.get("token_type_ids", None)  # ✅ Ensure `token_type_ids` is handled properly
                }

            def validate_tokenized_entry(entry):
                required_keys = ["input_ids", "attention_mask"]
                if not isinstance(entry, dict) or not all(key in entry for key in required_keys):
                    raise ValueError(f"Invalid tokenized format: {entry}")

            # ✅ Apply tokenization and validate each entry **before saving**
            data["tokenized"] = data["text"].apply(lambda x: tokenize_text(x))
            data["tokenized"].apply(validate_tokenized_entry)

            # ✅ Save dataset correctly
            data.to_pickle(output_path)
            self.logger.info(f"✅ Pre-tokenized dataset saved to {output_path}")

        except Exception as e:
            self.logger.error(f"❌ Error during pre-tokenization: {str(e)}")
            raise




    def process_and_save_data(self, input_filename: str, sample_fraction: float = 1.0) -> Tuple[str, Dict]:
        """
        Main method to process raw data and save cleaned version with comprehensive logging.
        
        Memory optimization is enforced by early sampling of the dataset.

        Args:
            input_filename (str): Name of input file.
            sample_fraction (float): Fraction of data to sample early to reduce memory usage.
            
        Returns:
            Tuple[str, Dict]: Path to processed data and processing statistics.
        """
        try:
            self.logger.info(f"Starting data processing pipeline for {input_filename}")
            
            # Load raw data from the specified file.
            raw_df = self.load_raw_data(input_filename)
            
            # Log statistics for the raw data.
            initial_stats = self.log_dataframe_stats(raw_df, "raw_data")
            
            # Early sampling to enforce processing only a subset of the data to save memory.
            if sample_fraction < 1.0:
                self.logger.info(f"Early sampling of raw data to {sample_fraction*100}% to reduce memory usage...")
                raw_df = raw_df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
                self.logger.info("Early sampling completed.")
            
            # Preprocess data: cleaning text, removing duplicates, and standardizing class labels.
            self.logger.info("Preprocessing data...")
            cleaned_df = self.preprocess_data(raw_df)
            
            # Perform balanced sampling to ensure equal representation of classes.
            sampled_df = self.sample_balanced_data(cleaned_df, sample_fraction)
            
            # Log final statistics after processing.
            final_stats = self.log_dataframe_stats(sampled_df, "final")
            
            # Define output filename and path for the processed data.
            output_filename = f"processed_data_{self.timestamp}.csv"
            output_path = os.path.join(self.processed_data_path, output_filename)
            
            # Save the processed and sampled data to CSV.
            sampled_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved processed data to {output_path}")
            
            # Compile processing statistics.
            stats = {
                "timestamp": self.timestamp,
                "input_file": input_filename,
                "output_file": output_filename,
                "initial_stats": initial_stats,
                "final_stats": final_stats,
                "early_sampling_fraction": sample_fraction,
                "processing_summary": {
                    "initial_rows": len(raw_df),
                    "after_cleaning": len(cleaned_df),
                    "final_rows": len(sampled_df),
                    "removed_rows": len(raw_df) - len(cleaned_df),
                    "sampling_reduction": len(cleaned_df) - len(sampled_df)
                }
            }
            
            # Save processing statistics as a JSON file.
            stats_file = os.path.join(
                self.processed_data_path,
                f"processing_stats_{self.timestamp}.json"
            )
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
                
            self.logger.info(f"Saved processing stats to {stats_file}")
            self.logger.info("Data processing pipeline completed successfully")
            
            # Free up memory.
            del raw_df, cleaned_df, sampled_df
            gc.collect()
            
            return output_path, stats
            
        except Exception as e:
            self.logger.error(f"Error in processing pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    # Parse command-line arguments for data preprocessing.
    parser = argparse.ArgumentParser(description="Data Preprocessor")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to the raw data directory")
    parser.add_argument("--processed_data_path", type=str, required=True, help="Path to save processed data")
    parser.add_argument("--input_filename", type=str, required=True, help="Name of the input file (e.g., data.csv)")
    parser.add_argument("--sample_fraction", type=float, default=0.1, help="Fraction of data to sample early (e.g., 0.1 for 10%)")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased", help="Tokenizer name for pre-tokenization")
    parser.add_argument("--pre_tokenize", action="store_true", help="Whether to perform pre-tokenization")
    
    args = parser.parse_args()

    # Initialize the DataPreprocessor with specified raw and processed data paths.
    preprocessor = DataPreprocessor(args.raw_data_path, args.processed_data_path)

    try:
        if args.pre_tokenize:
            # If pre-tokenization is requested, load the tokenizer and perform pre-tokenization.
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
            output_tokenized_path = os.path.join(args.processed_data_path, "tokenized_data.pkl")
            preprocessor.pre_tokenize_and_save(tokenizer, output_tokenized_path, sample_fraction=args.sample_fraction)
        else:
            # Otherwise, run the full data processing pipeline.
            processed_file, stats = preprocessor.process_and_save_data(
                input_filename=args.input_filename,
                sample_fraction=args.sample_fraction
            )
            print(f"Processed data saved at: {processed_file}")
            print(f"Processing statistics saved in JSON format at: {os.path.join(args.processed_data_path, f'processing_stats_{preprocessor.timestamp}.json')}")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
