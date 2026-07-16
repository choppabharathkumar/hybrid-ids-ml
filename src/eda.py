import pandas as pd

# Load merged dataset
df = pd.read_parquet("data/processed/merged_dataset.parquet")

# Basic information
print("Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print(df.duplicated().sum())

print(df["Label"].value_counts())

# Save cleaned dataset
df.to_parquet(
    "data/processed/cleaned_dataset.parquet",
    index=False
)

print("\nCleaned dataset saved successfully!")