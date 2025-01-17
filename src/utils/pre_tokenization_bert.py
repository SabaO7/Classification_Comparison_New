from transformers import AutoTokenizer
from data_preprocessing import DataPreprocessor

def preprocess():
    # Define paths
    raw_data_path = "../../data/raw"
    processed_data_path = "../../data/tokenized"
    pre_tokenized_path = f"{processed_data_path}/tokenized_data.pkl"
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    
    # Initialize the data preprocessor
    preprocessor = DataPreprocessor(raw_data_path, processed_data_path)
    
    # Pre-tokenize and save the dataset
    preprocessor.pre_tokenize_and_save(tokenizer, pre_tokenized_path)
    print(f"tokenized dataset saved at: {pre_tokenized_path}")

if __name__ == "__main__":
    preprocess()
