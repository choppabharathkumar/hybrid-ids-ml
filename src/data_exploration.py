"""Data Exploration Module.

Provides detailed statistical analysis of features per class,
correlation analysis, and distribution profiling to support
rule derivation and model feature selection.
"""
import pandas as pd
import numpy as np
import json
import os


# =============================
# 1. Feature Distribution Analysis
# =============================

def analyze_feature_distributions(df, output_dir="outputs"):
    """Compute per-class statistics for all numeric features.

    Returns a dict mapping each label to feature-level
    percentile and distribution statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    labels = sorted(df["Label"].unique())

    print("=" * 60)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 60)

    analysis = {}
    for label in labels:
        subset = df[df["Label"] == label][numeric_cols]
        stats = {}
        for col in numeric_cols:
            series = subset[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(series) == 0:
                continue
            stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "p75": float(series.quantile(0.75)),
                "p90": float(series.quantile(0.90)),
                "p95": float(series.quantile(0.95)),
                "p99": float(series.quantile(0.99)),
                "max": float(series.max()),
                "n_unique": int(series.nunique()),
                "n_zeros": int((series == 0).sum()),
                "zero_pct": float((series == 0).sum() / len(series) * 100),
            }
        analysis[label] = stats
        print(f"  {label}: {len(numeric_cols)} features analyzed ({len(subset):,} samples)")

    # Save full analysis
    output_path = os.path.join(output_dir, "feature_distribution_analysis.json")
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved to: {output_path}")

    return analysis


# =============================
# 2. Correlation Analysis
# =============================

def analyze_correlations(df, threshold=0.95, output_dir="outputs"):
    """Identify highly correlated feature pairs.

    Flags pairs with |correlation| >= threshold for potential removal.
    """
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    corr_matrix = df[numeric_cols].corr()

    # Find pairs above threshold
    high_corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    "feature_1": numeric_cols[i],
                    "feature_2": numeric_cols[j],
                    "correlation": round(float(corr_val), 4),
                })

    print(f"\nHighly correlated pairs (|r| >= {threshold}): {len(high_corr_pairs)}")
    for pair in high_corr_pairs[:10]:
        print(f"  {pair['feature_1']} ↔ {pair['feature_2']}: r={pair['correlation']}")

    output_path = os.path.join(output_dir, "correlation_analysis.json")
    with open(output_path, "w") as f:
        json.dump({
            "threshold": threshold,
            "n_features": len(numeric_cols),
            "n_high_corr_pairs": len(high_corr_pairs),
            "pairs": high_corr_pairs,
        }, f, indent=2)

    return high_corr_pairs


# =============================
# 3. Class Separation Analysis
# =============================

def analyze_class_separation(df, output_dir="outputs"):
    """Identify features that best separate attack classes from benign.

    Uses the ratio of inter-class to intra-class variance (Fisher criterion)
    to rank discriminative features.
    """
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    benign = df[df["Label"] == "Benign"]
    attacks = df[df["Label"] != "Benign"]

    rankings = []
    for col in numeric_cols:
        benign_vals = benign[col].replace([np.inf, -np.inf], np.nan).dropna()
        attack_vals = attacks[col].replace([np.inf, -np.inf], np.nan).dropna()

        if len(benign_vals) == 0 or len(attack_vals) == 0:
            continue

        # Fisher criterion: (mean_diff)^2 / (var_benign + var_attack)
        mean_diff = abs(benign_vals.mean() - attack_vals.mean())
        pooled_var = benign_vals.var() + attack_vals.var()

        if pooled_var > 0:
            fisher_score = (mean_diff ** 2) / pooled_var
        else:
            fisher_score = 0.0

        rankings.append({
            "feature": col,
            "fisher_score": round(float(fisher_score), 6),
            "benign_mean": round(float(benign_vals.mean()), 4),
            "attack_mean": round(float(attack_vals.mean()), 4),
            "mean_diff": round(float(mean_diff), 4),
        })

    rankings.sort(key=lambda x: x["fisher_score"], reverse=True)

    print(f"\nTop discriminative features (Fisher criterion):")
    for i, r in enumerate(rankings[:15]):
        print(f"  {i+1}. {r['feature']}: Fisher={r['fisher_score']:.4f}")

    output_path = os.path.join(output_dir, "class_separation_analysis.json")
    with open(output_path, "w") as f:
        json.dump(rankings, f, indent=2)

    return rankings


# =============================
# 4. Full Exploration Pipeline
# =============================

def run_exploration(df, output_dir="outputs"):
    """Execute the complete data exploration pipeline."""
    print("\n" + "=" * 60)
    print("DATA EXPLORATION PIPELINE")
    print("=" * 60)

    analysis = analyze_feature_distributions(df, output_dir)
    corr_pairs = analyze_correlations(df, threshold=0.95, output_dir=output_dir)
    rankings = analyze_class_separation(df, output_dir)

    print(f"\nExploration complete. Results in: {output_dir}/")
    return analysis, corr_pairs, rankings


# =============================
# Main
# =============================

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/cleaned_dataset.parquet")
    run_exploration(df)
