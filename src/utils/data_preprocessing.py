import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
from datetime import datetime
import json
import sys
from typing import Tuple, Dict

class DataPreprocessor:
    """
    Class for handling data preprocessing and sampling
    
    Implements comprehensive data cleaning, balancing, and validation
    Includes detailed logging and error tracking
    Saves processing statistics and data quality metrics
    """
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        Initialize data preprocessor
        
        Args:
            raw_data_path (str): Path to raw data directory
            processed_data_path (str): Path to save processed data
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Validate paths
        if not os.path.exists(raw_data_path):
            raise ValueError(f"Raw data path does not exist: {raw_data_path}")
        
        # Create processed data directory
        os.makedirs(processed_data_path, exist_ok=True)
        self.setup_logging()
        
        self.logger.info("Initialized DataPreprocessor")
        self.logger.info(f"Raw data path: {raw_data_path}")
        self.logger.info(f"Processed data path: {processed_data_path}")
    

    def pre_tokenize_and_save(self, tokenizer, output_path: str):
        """
        Pre-tokenize the dataset for BERT and save it to disk. 
        This is because I can use the tokenized data later on with other modesl
        
        Args:
            tokenizer: Tokenizer for BERT (e.g., AutoTokenizer).
            output_path (str): Path to save the pre-tokenized dataset.
        """
        self.logger.info("Starting pre-tokenization for BERT...")
        try:
            # Load the raw dataset
            raw_data_file = "Suicide_Detection.csv" 
            data = self.load_raw_data(raw_data_file)
            
            # Ensure text column exists
            if 'text' not in data.columns:
                raise ValueError("The dataset must have a 'text' column.")
            
            # Pre-tokenize text
            data['tokenized'] = data['text'].apply(lambda x: tokenizer(x, truncation=True, padding=True, max_length=128))
            
            # Save the tokenized dataset
            data.to_pickle(output_path)
            self.logger.info(f"Pre-tokenized dataset saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error during pre-tokenization: {str(e)}")
            raise


    def setup_logging(self):
        """Configure comprehensive logging system"""
        log_file = os.path.join(self.processed_data_path, f'preprocessing_{self.timestamp}.log')
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler with detailed logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler with info level
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging setup complete. Log file: {log_file}")
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        try:
            if pd.isna(text):
                return ""
                
            # Convert to string and lowercase
            text = str(text).lower()
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Log potentially problematic texts
            if len(text) < 5:
                self.logger.warning(f"Very short text after cleaning: '{text}'")
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error cleaning text: '{text[:100]}...' - {str(e)}")
            return ""
        
    def log_dataframe_stats(self, df: pd.DataFrame, stage: str):
        """
        Log comprehensive dataframe statistics
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            stage (str): Current processing stage
        """
        stats = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'class_distribution': df['class'].value_counts().to_dict(),
            'text_length_stats': df['text'].str.len().describe().to_dict(),
            'unique_texts': len(df['text'].unique()),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        self.logger.info(f"\nDataFrame Statistics ({stage}):")
        self.logger.info(f"Total rows: {stats['total_rows']}")
        self.logger.info(f"Class distribution:\n{pd.Series(stats['class_distribution'])}")
        self.logger.info(f"Missing values:\n{pd.Series(stats['missing_values'])}")
        self.logger.info(f"Text length statistics:\n{pd.Series(stats['text_length_stats'])}")
        self.logger.info(f"Unique texts: {stats['unique_texts']}")
        self.logger.info(f"Memory usage: {stats['memory_usage']:.2f} MB")
        
        return stats
        
    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """
        Load raw data with robust encoding handling
        
        Args:
            filename (str): Name of the raw data file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        file_path = os.path.join(self.raw_data_path, filename)
        self.logger.info(f"Attempting to load file: {file_path}")
        
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try different encodings
        encodings = ['utf-8', 'ISO-8859-1', 'cp1252', 'latin1']
        
        for encoding in encodings:
            try:
                self.logger.debug(f"Trying encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding)
                self.logger.info(f"Successfully loaded data with {encoding} encoding")
                
                # Validate required columns
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
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data with comprehensive validation
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        self.logger.info("Starting data preprocessing...")
        
        # Log initial stats
        initial_stats = self.log_dataframe_stats(df, "initial")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        try:
            # Clean text
            self.logger.info("Cleaning text data...")
            df['text'] = df['text'].apply(self.clean_text)
            
            # Remove empty texts
            initial_rows = len(df)
            df = df.dropna(subset=['text'])
            df = df[df['text'].str.strip() != ""]
            removed_rows = initial_rows - len(df)
            self.logger.info(f"Removed {removed_rows} rows with empty/null text")
            
            # Remove duplicates
            initial_rows = len(df)
            df = df.drop_duplicates(subset=['text'])
            removed_duplicates = initial_rows - len(df)
            self.logger.info(f"Removed {removed_duplicates} duplicate texts")
            
            # Standardize class labels
            df['class'] = df['class'].str.lower()
            unique_classes = df['class'].unique()
            self.logger.info(f"Unique class labels: {unique_classes}")
            
            # Log final stats
            final_stats = self.log_dataframe_stats(df, "after_preprocessing")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}")
            raise
        
    def sample_balanced_data(self, df: pd.DataFrame, sample_fraction: float = 0.1) -> pd.DataFrame:
        """
        Sample data while maintaining class distribution with validation
        
        Args:
            df (pd.DataFrame): Input dataframe
            sample_fraction (float): Fraction of data to sample
            
        Returns:
            pd.DataFrame: Sampled dataframe
        """
        self.logger.info(f"Sampling {sample_fraction*100}% of data...")
        
        # Validate input
        if not 0 < sample_fraction <= 1:
            raise ValueError(f"Invalid sample_fraction: {sample_fraction}")
        
        # Get class distribution
        class_counts = df['class'].value_counts()
        min_class_count = int(class_counts.min() * sample_fraction)
        
        self.logger.info("Original class distribution:")
        self.logger.info(class_counts)
        
        # Sample equally from each class
        sampled_dfs = []
        for class_label in class_counts.index:
            class_df = df[df['class'] == class_label]
            if len(class_df) < min_class_count:
                self.logger.warning(
                    f"Insufficient samples for class {class_label}. "
                    f"Required: {min_class_count}, Available: {len(class_df)}"
                )
            
            sampled_class = class_df.sample(
                n=min_class_count,
                random_state=42
            )
            sampled_dfs.append(sampled_class)
            
            self.logger.info(f"Sampled {min_class_count} records from class {class_label}")
        
        # Combine sampled data
        sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Shuffle the final dataset
        sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Log final distribution
        final_dist = sampled_df['class'].value_counts()
        self.logger.info("Final class distribution:")
        self.logger.info(final_dist)
        
        return sampled_df
        
    def process_and_save_data(self, input_filename: str, sample_fraction: float = 0.1) -> Tuple[str, Dict]:
        """
        Main method to process raw data and save cleaned version with comprehensive logging
        
        Args:
            input_filename (str): Name of input file
            sample_fraction (float): Fraction of data to sample
            
        Returns:
            Tuple[str, Dict]: Path to processed data and processing statistics
        """
        try:
            self.logger.info(f"Starting data processing pipeline for {input_filename}")
            
            # Load raw data
            raw_df = self.load_raw_data(input_filename)
            
            # Log initial data stats
            initial_stats = self.log_dataframe_stats(raw_df, "raw_data")
            
            # Preprocess data
            self.logger.info("Preprocessing data...")
            cleaned_df = self.preprocess_data(raw_df)
            
            # Sample data
            sampled_df = self.sample_balanced_data(cleaned_df, sample_fraction)
            
            # Log final stats
            final_stats = self.log_dataframe_stats(sampled_df, "final")
            
            # Save processed data
            output_filename = f"processed_data_{self.timestamp}.csv"
            output_path = os.path.join(self.processed_data_path, output_filename)
            
            sampled_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved processed data to {output_path}")
            
            # Compile comprehensive statistics
            stats = {
                "timestamp": self.timestamp,
                "input_file": input_filename,
                "output_file": output_filename,
                "initial_stats": initial_stats,
                "final_stats": final_stats,
                "sampling_fraction": sample_fraction,
                "processing_summary": {
                    "initial_rows": len(raw_df),
                    "after_cleaning": len(cleaned_df),
                    "final_rows": len(sampled_df),
                    "removed_rows": len(raw_df) - len(cleaned_df),
                    "sampling_reduction": len(cleaned_df) - len(sampled_df)
                }
            }
            
            # Save processing stats
            stats_file = os.path.join(
                self.processed_data_path,
                f"processing_stats_{self.timestamp}.json"
            )
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
                
            self.logger.info(f"Saved processing stats to {stats_file}")
            self.logger.info("Data processing pipeline completed successfully")
            
            return output_path, stats
            
        except Exception as e:
            self.logger.error(f"Error in processing pipeline: {str(e)}")
            raise