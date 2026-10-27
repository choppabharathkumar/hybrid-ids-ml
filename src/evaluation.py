"""Hybrid IDS Evaluation Module.

Evaluates the complete hybrid detection pipeline on the test set.
Computes overall and per-class metrics, prints a detailed report,
and saves results to outputs/evaluation/.
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_hybrid_predictions(predictions_path="outputs/hybrid/hybrid_predictions.parquet",
                                output_dir="outputs/evaluation"):
    """Evaluate hybrid predictions against ground truth."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("HYBRID IDS EVALUATION")
    print("=" * 70)

    df = pd.read_parquet(predictions_path)
    print(f"Total flows evaluated: {len(df):,}")

    # Determine final prediction column
    if "Hybrid_Prediction" in df.columns:
        pred_col = "Hybrid_Prediction"
    elif "Final_Prediction" in df.columns:
        pred_col = "Final_Prediction"
    else:
        # Build from Signature + Anomaly
        pred_col = "_eval_prediction"
        df[pred_col] = df["Signature_Prediction"].replace("Unmatched", "Benign")
        if "Is_Anomaly" in df.columns:
            anomaly_mask = (
                (df["Signature_Prediction"] == "Unmatched")
                & (df["Is_Anomaly"] == True)
            )
            df.loc[anomaly_mask, pred_col] = "Potential Unknown Attack"
        elif "Anomaly_Prediction" in df.columns:
            anomaly_mask = (
                (df["Signature_Prediction"] == "Unmatched")
                & (df["Anomaly_Prediction"] == -1)
            )
            df.loc[anomaly_mask, pred_col] = "Potential Unknown Attack"

    print(f"Using prediction column: {pred_col}")
    print(f"\nPrediction distribution:")
    print(df[pred_col].value_counts())
    print(f"\nGround truth distribution:")
    print(df["Label"].value_counts())

    # ---- Binary: Attack vs Benign ----
    y_true_binary = (df["Label"] != "Benign").astype(int)
    y_pred_binary = (df[pred_col] != "Benign").astype(int)

    tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
    tn = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
    fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
    fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()

    binary_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    binary_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    binary_f1 = 2 * binary_precision * binary_recall / (binary_precision + binary_recall) \
        if (binary_precision + binary_recall) > 0 else 0
    binary_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print("\n" + "-" * 50)
    print("BINARY METRICS (Attack vs Benign)")
    print("-" * 50)
    print(f"  TP: {tp:,}  |  FP: {fp:,}")
    print(f"  FN: {fn:,}  |  TN: {tn:,}")
    print(f"  Precision: {binary_precision:.4f}")
    print(f"  Recall:    {binary_recall:.4f}")
    print(f"  F1 Score:  {binary_f1:.4f}")
    print(f"  FPR:       {binary_fpr:.6f}")

    # ---- Detection by type ----
    print("\n" + "-" * 50)
    print("DETECTION BY MECHANISM")
    print("-" * 50)

    sig_detected = df["Signature_Matched"].sum() if "Signature_Matched" in df.columns else 0
    anom_detected = 0
    if "Is_Anomaly" in df.columns:
        anom_mask = (~df["Signature_Matched"]) & (df["Is_Anomaly"] == True)
        anom_detected = anom_mask.sum()
    elif "Anomaly_Prediction" in df.columns:
        anom_mask = (~df["Signature_Matched"]) & (df["Anomaly_Prediction"] == -1)
        anom_detected = anom_mask.sum()
    normal = len(df) - sig_detected - anom_detected

    print(f"  Signature detections: {sig_detected:,}")
    print(f"  Anomaly detections:   {anom_detected:,}")
    print(f"  Normal (benign):      {normal:,}")

    # ---- Per-attack recall (most important metric) ----
    print("\n" + "-" * 50)
    print("PER-ATTACK DETECTION RATE (RECALL)")
    print("-" * 50)

    attack_labels = [l for l in sorted(df["Label"].unique()) if l != "Benign"]
    per_attack = {}

    for label in attack_labels:
        total = (df["Label"] == label).sum()
        detected = ((df["Label"] == label) & (y_pred_binary == 1)).sum()
        recall = detected / total if total > 0 else 0
        per_attack[label] = {
            "total": int(total),
            "detected": int(detected),
            "missed": int(total - detected),
            "recall": float(recall),
        }
        status = "✓" if recall >= 0.80 else "✗"
        print(f"  {status} {label}: {detected}/{total} = {recall:.1%}")

    # ---- Aggregate metrics ----
    overall_accuracy = accuracy_score(y_true_binary, y_pred_binary)

    print(f"\n  Overall Binary Accuracy: {overall_accuracy:.4f}")
    print(f"  Overall Attack Recall:  {binary_recall:.4f}")

    # ---- Save results ----
    results = {
        "binary_metrics": {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "precision": float(binary_precision),
            "recall": float(binary_recall),
            "f1_score": float(binary_f1),
            "fpr": float(binary_fpr),
            "accuracy": float(overall_accuracy),
        },
        "detection_by_mechanism": {
            "signature_detections": int(sig_detected),
            "anomaly_detections": int(anom_detected),
            "normal_flows": int(normal),
        },
        "per_attack_recall": per_attack,
        "total_flows": int(len(df)),
    }

    results_path = os.path.join(output_dir, "hybrid_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    evaluate_hybrid_predictions()
