import pandas as pd

# File paths
benign_path = "data/raw/Benign-Monday-no-metadata.parquet"
portscan_path = "data/raw/Portscan-Friday-no-metadata.parquet"
webattack_path = "data/raw/WebAttacks-Thursday-no-metadata.parquet"

# Load datasets
benign_df = pd.read_parquet(benign_path)
portscan_df = pd.read_parquet(portscan_path)
webattack_df = pd.read_parquet(webattack_path)

# Display information
print("Benign Dataset:", benign_df.shape)
print("PortScan Dataset:", portscan_df.shape)
print("WebAttack Dataset:", webattack_df.shape)

print("\nBenign Columns:")
print(benign_df.columns.tolist())

datasets = {
    "Benign": benign_df,
    "PortScan": portscan_df,
    "WebAttack": webattack_df
}

for name, df in datasets.items():
    print(f"\n{'='*50}")
    print(f"{name} Dataset")
    print(f"{'='*50}")

    print("Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nFirst 5 Rows:")
    print(df.head())

    # Merge the datasets
combined_df = pd.concat(
    [benign_df, portscan_df, webattack_df],
    ignore_index=True
)

print("\nMerged Dataset Shape:", combined_df.shape)
# Save merged dataset
combined_df.to_parquet(
    "data/processed/merged_dataset.parquet",
    index=False
)

print("Merged dataset saved successfully!")
print("\nMerged Dataset Shape:", combined_df.shape)