import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# =============================
# 1. Feature Selection
# =============================

# Columns to exclude from model features
EXCLUDE_COLUMNS = ["Label"]

# Features known to be redundant or zero-variance (detected in EDA)
KNOWN_ZERO_VARIANCE = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]


def get_feature_columns(df):
    """Return list of numeric feature columns, excluding label and zero-variance."""
    all_cols = df.columns.tolist()
    feature_cols = [
        c for c in all_cols
        if c not in EXCLUDE_COLUMNS
        and c not in KNOWN_ZERO_VARIANCE
    ]
    return feature_cols


# =============================
# 2. Train/Validation/Test Split
# =============================

def split_dataset(df, test_size=0.2, val_size=0.1, random_state=42):
    """Split into train/validation/test with stratification on Label.

    Default: 70% train, 10% validation, 20% test.
    Stratification ensures label proportions are preserved in each split.
    """
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Label"],
    )

    # Second split: separate validation from training
    val_fraction = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        random_state=random_state,
        stratify=train_val_df["Label"],
    )

    print(f"Train: {train_df.shape}  ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {val_df.shape}  ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test:  {test_df.shape}  ({len(test_df)/len(df)*100:.1f}%)")

    # Verify label proportions
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split_df["Label"].value_counts(normalize=True)
        print(f"  {name} label proportions: { {k: f'{v:.4f}' for k, v in dist.items()} }")

    return train_df, val_df, test_df


# =============================
# 3. Fit Preprocessing (Training Only)
# =============================

def fit_preprocessing(train_df, feature_columns):
    """Fit scaler and label encoder on TRAINING DATA ONLY.

    This prevents data leakage — validation and test sets
    are transformed using parameters learned from training data.
    """
    X_train = train_df[feature_columns]

    # Handle any remaining categorical columns via one-hot encoding
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        print(f"One-hot encoded categorical columns: {cat_cols}")

    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Fit label encoder on training labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["Label"])

    print(f"\nScaler fitted on {X_train.shape[1]} features (training data only)")
    print(f"Label classes: {label_encoder.classes_.tolist()}")

    # Store the final feature names after one-hot encoding
    final_feature_names = X_train.columns.tolist()

    return scaler, label_encoder, final_feature_names, cat_cols


# =============================
# 4. Transform Data
# =============================

def transform_features(df, feature_columns, scaler, cat_cols):
    """Apply preprocessing transformations using pre-fitted scaler."""
    X = df[feature_columns].copy()

    # Apply same one-hot encoding
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X_scaled = scaler.transform(X)

    return X_scaled, X.columns.tolist()


def encode_labels(df, label_encoder):
    """Encode labels using pre-fitted encoder."""
    return label_encoder.transform(df["Label"])


# =============================
# 5. Save/Load Preprocessing Artifacts
# =============================

def save_preprocessing_artifacts(
    scaler, label_encoder, feature_names, cat_cols,
    split_info, output_dir="models"
):
    """Save all preprocessing artifacts for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))

    config = {
        "feature_names": feature_names,
        "categorical_columns": cat_cols,
        "n_features": len(feature_names),
        "label_classes": label_encoder.classes_.tolist(),
        "split_info": split_info,
    }
    config_path = os.path.join(output_dir, "preprocessing_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nPreprocessing artifacts saved to: {output_dir}/")
    print(f"  - scaler.joblib")
    print(f"  - label_encoder.joblib")
    print(f"  - preprocessing_config.json")


def load_preprocessing_artifacts(model_dir="models"):
    """Load saved preprocessing artifacts."""
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

    config_path = os.path.join(model_dir, "preprocessing_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    return scaler, label_encoder, config


# =============================
# 6. Full Preprocessing Pipeline
# =============================

def run_preprocessing(df, output_dir="models", data_output_dir="data/processed"):
    """Execute the complete preprocessing pipeline with proper data splitting."""

    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Get feature columns
    feature_columns = get_feature_columns(df)
    print(f"\nFeature columns: {len(feature_columns)}")

    # Split dataset (BEFORE fitting any transformations)
    train_df, val_df, test_df = split_dataset(df)

    # Fit preprocessing on TRAINING DATA ONLY
    scaler, label_encoder, final_features, cat_cols = fit_preprocessing(
        train_df, feature_columns
    )

    # Transform all splits using training-fitted parameters
    X_train, _ = transform_features(train_df, feature_columns, scaler, cat_cols)
    X_val, _ = transform_features(val_df, feature_columns, scaler, cat_cols)
    X_test, _ = transform_features(test_df, feature_columns, scaler, cat_cols)

    y_train = encode_labels(train_df, label_encoder)
    y_val = encode_labels(val_df, label_encoder)
    y_test = encode_labels(test_df, label_encoder)

    print(f"\nTransformed shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # Save split DataFrames (with original features for rule-based detection)
    os.makedirs(data_output_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(data_output_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(data_output_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(data_output_dir, "test.parquet"), index=False)
    print(f"\nSplit datasets saved to: {data_output_dir}/")

    # Save preprocessing artifacts
    split_info = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "total_size": len(df),
        "random_state": 42,
        "test_fraction": 0.2,
        "val_fraction": 0.1,
    }

    save_preprocessing_artifacts(
        scaler, label_encoder, final_features, cat_cols,
        split_info, output_dir
    )

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "scaler": scaler, "label_encoder": label_encoder,
        "feature_names": final_features,
    }


# =============================
# Main
# =============================

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/cleaned_dataset.parquet")
    results = run_preprocessing(df)
    print("\nPreprocessing complete!")