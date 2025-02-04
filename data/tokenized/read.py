import pandas as pd

# Load the .pkl file
file_path = "tokenized_data.pkl"  # Update the path if needed
df = pd.read_pickle(file_path)

# Display the first few rows
print(df.head())

# Print all column names
print("\nColumns in the DataFrame:")
print(df.columns)
