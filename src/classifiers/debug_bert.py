import pandas as pd

# Load pre-tokenized dataset
tokenized_data = pd.read_pickle("../../data/tokenized/tokenized_data.pkl")

# Inspect the first few rows
print(tokenized_data.head())

# Check for null or unexpected values
print(tokenized_data.isnull().sum())

# Validate structure of 'tokenized' column
print("Sample tokenized entry:", tokenized_data['tokenized'].iloc[0])
print("Unique classes:", tokenized_data['class'].unique())
