import pandas as pd
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

# Load raw dataset and remove the unnecessary column
df = pd.read_csv("../../data/raw/Suicide_Detection.csv")
df = df[["text", "class"]]  # Keep only necessary columns

# Apply tokenizer correctly
df["tokenized"] = df["text"].apply(lambda x: tokenizer(x, truncation=True, padding="max_length", max_length=128))

# Convert tokenized objects to JSON-compatible format
df["tokenized"] = df["tokenized"].apply(lambda x: {k: v for k, v in x.items()})

# Save the fixed dataset
df.to_pickle("../../data/tokenized/tokenized_data.pkl")
print("Fixed tokenization! Saved to tokenized_data.pkl")
