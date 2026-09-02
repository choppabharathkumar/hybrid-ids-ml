import pandas as pd
import os

# =============================
# Canonical Label Taxonomy
# =============================

LABEL_MAP = {
    "Benign": "Benign",
    "BENIGN": "Benign",
    "PortScan": "PortScan",
    "Web Attack \ufffd Brute Force": "Web Attack - Brute Force",
    "Web Attack \u00ef\u00bf\u00bd Brute Force": "Web Attack - Brute Force",
    "Web Attack \u2013 Brute Force": "Web Attack - Brute Force",
    "Web Attack - Brute Force": "Web Attack - Brute Force",
    "Web Attack \ufffd XSS": "Web Attack - XSS",
    "Web Attack \u00ef\u00bf\u00bd XSS": "Web Attack - XSS",
    "Web Attack \u2013 XSS": "Web Attack - XSS",
    "Web Attack - XSS": "Web Attack - XSS",
    "Web Attack \ufffd Sql Injection": "Web Attack - SQL Injection",
    "Web Attack \u00ef\u00bf\u00bd Sql Injection": "Web Attack - SQL Injection",
    "Web Attack \u2013 Sql Injection": "Web Attack - SQL Injection",
    "Web Attack - Sql Injection": "Web Attack - SQL Injection",
    "Web Attack - SQL Injection": "Web Attack - SQL Injection",
}

CANONICAL_LABELS = [
    "Benign",
    "PortScan",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - SQL Injection",
]

RAW_FILES = {
    "Benign": "data/raw/Benign-Monday-no-metadata.parquet",
    "PortScan": "data/raw/Portscan-Friday-no-metadata.parquet",
    "WebAttack": "data/raw/WebAttacks-Thursday-no-metadata.parquet",
}


# =============================
# 1. Load Individual Datasets
# =============================

def load_raw_datasets(base_dir="."):
    """Load all raw parquet files and return as a dict of DataFrames."""
    datasets = {}
    for name, rel_path in RAW_FILES.items():
        path = os.path.join(base_dir, rel_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw data file not found: {path}")
        df = pd.read_parquet(path)
        datasets[name] = df
        print(f"Loaded {name}: {df.shape}")
    return datasets


# =============================
# 2. Normalize Labels
# =============================

def normalize_labels(df, label_column="Label"):
    """Map all label variants to canonical names.

    Handles Unicode corruption artifacts from CICIDS2017
    (replacement char U+FFFD, en-dash U+2013, etc.).
    """
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in DataFrame")

    # Convert categorical to string for replacement
    if hasattr(df[label_column], "cat"):
        df[label_column] = df[label_column].astype(str)

    df[label_column] = df[label_column].map(LABEL_MAP)

    unmapped = df[label_column].isna()
    if unmapped.any():
        bad_labels = df.loc[unmapped, label_column].unique()
        raise ValueError(f"Unmapped labels found: {bad_labels}")

    return df


# =============================
# 3. Merge Datasets
# =============================

def merge_datasets(datasets):
    """Concatenate all DataFrames into a single merged DataFrame."""
    combined = pd.concat(datasets.values(), ignore_index=True)
    print(f"Merged dataset: {combined.shape}")
    return combined


# =============================
# 4. Full Load Pipeline
# =============================

def load_and_merge(base_dir=".", save_path=None):
    """Load raw files, normalize labels, merge, and optionally save."""
    datasets = load_raw_datasets(base_dir)

    for name, df in datasets.items():
        datasets[name] = normalize_labels(df)
        print(f"  {name} labels: {datasets[name]['Label'].value_counts().to_dict()}")

    merged = merge_datasets(datasets)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        merged.to_parquet(save_path, index=False)
        print(f"Saved merged dataset to: {save_path}")

    return merged


# =============================
# Main
# =============================

if __name__ == "__main__":
    merged = load_and_merge(
        base_dir=".",
        save_path="data/processed/merged_dataset.parquet"
    )

    print(f"\nFinal shape: {merged.shape}")
    print(f"\nLabel distribution:")
    print(merged["Label"].value_counts())