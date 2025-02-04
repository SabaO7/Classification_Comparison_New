import pandas as pd

# Load the tokenized dataset
df = pd.read_pickle("../../data/tokenized/tokenized_data.pkl")

# Print a few random records to check tokenized format
print(df.sample(5)["tokenized"])

# Check if any rows have missing or malformed tokenized entries
invalid_rows = df[df["tokenized"].apply(lambda x: not isinstance(x, dict) or "input_ids" not in x)]
if not invalid_rows.empty:
    print(f"❌ Warning: Found {len(invalid_rows)} invalid rows in tokenized dataset!")
    print(invalid_rows.head(5))
else:
    print("✅ All tokenized records are correctly formatted!")
