import pandas as pd

# Dataset path
file_path = "data/raw/NF-UQ-NIDS-v2.csv"

# Load dataset
df = pd.read_csv(file_path)

# Display first 5 rows
print("\nFirst 5 Rows:\n")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())