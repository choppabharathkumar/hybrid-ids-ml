import pandas as pd
import numpy as np
import json
import os

# =============================
# 1. Dataset Summary
# =============================

def dataset_summary(df):
    """Print concise dataset overview."""
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nShape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nLabel Distribution:")
    label_counts = df["Label"].value_counts()
    total = len(df)
    for label, count in label_counts.items():
        pct = count / total * 100
        print(f"  {label}: {count:,} ({pct:.2f}%)")

    return label_counts


# =============================
# 2. Detect Data Quality Issues
# =============================

def detect_quality_issues(df):
    """Identify data quality problems and return a report dict."""
    report = {}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Negative Flow Duration
    if "Flow Duration" in df.columns:
        neg_duration = (df["Flow Duration"] < 0).sum()
        report["negative_flow_duration"] = int(neg_duration)
        if neg_duration > 0:
            print(f"WARNING: {neg_duration} rows with negative Flow Duration")

    # Infinite values
    inf_cols = []
    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            inf_cols.append({"column": col, "count": int(n_inf)})
    report["infinite_value_columns"] = inf_cols
    if inf_cols:
        print(f"WARNING: {len(inf_cols)} columns with infinite values")

    # Zero-variance columns
    zv_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
    report["zero_variance_columns"] = zv_cols
    if zv_cols:
        print(f"INFO: {len(zv_cols)} zero-variance columns: {zv_cols}")

    # Missing values per column
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].to_dict()
    report["missing_value_columns"] = {k: int(v) for k, v in missing_cols.items()}

    return report


# =============================
# 3. Remove Duplicates
# =============================

def remove_duplicates(df):
    """Remove duplicate rows and report how many were removed."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"Duplicates removed: {removed:,} ({removed / before * 100:.2f}%)")
    return df


# =============================
# 4. Remove Invalid Rows
# =============================

def remove_invalid_rows(df):
    """Remove rows with physically impossible values."""
    before = len(df)

    # Remove negative Flow Duration
    if "Flow Duration" in df.columns:
        mask = df["Flow Duration"] >= 0
        df = df[mask].reset_index(drop=True)

    # Remove rows with infinite values
    numeric_cols = df.select_dtypes(include="number").columns
    inf_mask = np.isinf(df[numeric_cols]).any(axis=1)
    df = df[~inf_mask].reset_index(drop=True)

    removed = before - len(df)
    print(f"Invalid rows removed: {removed:,}")
    return df


# =============================
# 5. Remove Zero-Variance Features
# =============================

def remove_zero_variance_features(df, exclude_cols=None):
    """Drop numeric columns with zero variance (carry no information)."""
    if exclude_cols is None:
        exclude_cols = ["Label"]

    numeric_cols = df.select_dtypes(include="number").columns
    candidates = [c for c in numeric_cols if c not in exclude_cols]

    zv_cols = [c for c in candidates if df[c].nunique() <= 1]

    if zv_cols:
        df = df.drop(columns=zv_cols)
        print(f"Removed {len(zv_cols)} zero-variance columns: {zv_cols}")
    else:
        print("No zero-variance columns found")

    return df, zv_cols


# =============================
# 6. Full EDA Pipeline
# =============================

def run_eda(df, output_dir="outputs"):
    """Run the complete EDA pipeline and save a report."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Summary
    label_counts = dataset_summary(df)

    # Quality issues
    quality_report = detect_quality_issues(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Remove invalid rows
    df = remove_invalid_rows(df)

    # Remove zero-variance features
    df, zv_cols = remove_zero_variance_features(df)

    # Final summary
    print(f"\nFinal shape after cleaning: {df.shape}")
    print(f"Label distribution after cleaning:")
    print(df["Label"].value_counts())

    # Save EDA report
    eda_report = {
        "original_shape": [int(x) for x in label_counts.index.shape] + [int(label_counts.sum())],
        "final_shape": list(df.shape),
        "label_distribution": df["Label"].value_counts().to_dict(),
        "quality_issues": quality_report,
        "zero_variance_removed": zv_cols,
    }

    report_path = os.path.join(output_dir, "eda_report.json")
    with open(report_path, "w") as f:
        json.dump(eda_report, f, indent=2)
    print(f"\nEDA report saved to: {report_path}")

    return df


# =============================
# Main
# =============================

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/merged_dataset.parquet")
    df = run_eda(df, output_dir="outputs")
    df.to_parquet("data/processed/cleaned_dataset.parquet", index=False)
    print("\nCleaned dataset saved successfully!")