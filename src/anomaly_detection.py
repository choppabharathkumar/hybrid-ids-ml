import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.data_preprocessing import load_preprocessing_artifacts, transform_features


# =============================
# 1. Prepare Benign-Only Training Data
# =============================

def prepare_benign_training_data(train_df, scaler, config):
    """Extract and scale benign-only flows for Isolation Forest training.

    The model learns what 'normal' network traffic looks like.
    Attack labels are NOT used for training — only for evaluation.
    """
    benign_df = train_df[train_df["Label"] == "Benign"].copy()
    print(f"Benign training samples: {len(benign_df):,}")
    print(f"  ({len(benign_df)/len(train_df)*100:.1f}% of training data)")

    feature_cols = config["feature_names"]
    cat_cols = config.get("categorical_columns", [])
    X_benign, _ = transform_features(benign_df, feature_cols, scaler, cat_cols)

    return X_benign, benign_df


# =============================
# 2. Train Isolation Forest
# =============================

def train_isolation_forest(
    X_train,
    contamination=0.01,
    n_estimators=200,
    max_samples="auto",
    random_state=42,
):
    """Train Isolation Forest on benign-only data.

    Parameters
    ----------
    X_train : scaled feature matrix (benign flows only)
    contamination : expected proportion of anomalies in training data.
        Since we train on benign-only data, this should be very low.
        A small value (0.005-0.02) allows the model to treat rare
        benign behaviors as borderline normal rather than anomalous.
    n_estimators : number of isolation trees
    """
    print(f"\nTraining Isolation Forest:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  contamination: {contamination}")
    print(f"  max_samples: {max_samples}")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Features: {X_train.shape[1]}")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    print("  Training complete.")

    return model


# =============================
# 3. Predict Anomalies
# =============================

def predict_anomalies(model, X, threshold=None):
    """Generate anomaly predictions and scores.

    Returns
    -------
    predictions : array of 1 (normal) or -1 (anomaly)
    scores : raw anomaly scores (lower = more anomalous)
    """
    predictions = model.predict(X)
    scores = model.decision_function(X)

    if threshold is not None:
        predictions = np.where(scores < threshold, -1, 1)

    n_anomalies = (predictions == -1).sum()
    n_normal = (predictions == 1).sum()
    print(f"\nPredictions: {n_normal:,} normal, {n_anomalies:,} anomalies")
    print(f"  Anomaly rate: {n_anomalies / len(predictions) * 100:.2f}%")

    return predictions, scores


# =============================
# 4. Calibrate Threshold
# =============================

def calibrate_threshold(model, X_val, y_val_labels, target_fpr=0.05):
    """Calibrate anomaly threshold using validation data.

    Searches for the threshold that maximizes attack recall while
    keeping the false positive rate at or below target_fpr.

    Parameters
    ----------
    target_fpr : maximum acceptable false positive rate on benign traffic
    """
    scores = model.decision_function(X_val)

    benign_scores = scores[y_val_labels == "Benign"]
    attack_scores = scores[y_val_labels != "Benign"]

    print(f"\nThreshold Calibration (target FPR <= {target_fpr:.2%}):")
    print(f"  Benign validation samples: {len(benign_scores):,}")
    print(f"  Attack validation samples: {len(attack_scores):,}")
    print(f"  Benign score range: [{benign_scores.min():.4f}, {benign_scores.max():.4f}]")
    print(f"  Attack score range: [{attack_scores.min():.4f}, {attack_scores.max():.4f}]")

    # Search across a wide range of benign score percentiles
    candidate_percentiles = np.arange(0, 50, 0.25)
    thresholds_from_benign = np.percentile(benign_scores, candidate_percentiles)

    # Also consider thresholds at attack score percentiles
    if len(attack_scores) > 0:
        attack_percentiles = np.percentile(attack_scores, np.arange(10, 100, 5))
        all_thresholds = np.unique(np.concatenate([thresholds_from_benign, attack_percentiles]))
    else:
        all_thresholds = np.unique(thresholds_from_benign)

    best_threshold = None
    best_recall = 0

    for t in sorted(all_thresholds):
        fp = (benign_scores < t).sum()
        fpr = fp / len(benign_scores)
        tp = (attack_scores < t).sum()
        recall = tp / len(attack_scores) if len(attack_scores) > 0 else 0

        if fpr <= target_fpr and recall > best_recall:
            best_threshold = t
            best_recall = recall

    if best_threshold is None:
        best_threshold = np.percentile(benign_scores, target_fpr * 100)
        print(f"  WARNING: Could not find threshold with FPR <= {target_fpr:.2%}")

    # Report final metrics at chosen threshold
    final_fp = (benign_scores < best_threshold).sum()
    final_fpr = final_fp / len(benign_scores)
    final_tp = (attack_scores < best_threshold).sum()
    final_recall = final_tp / len(attack_scores) if len(attack_scores) > 0 else 0

    print(f"\n  Calibrated threshold: {best_threshold:.6f}")
    print(f"  FPR at threshold: {final_fpr:.4f} ({final_fp:,} false positives)")
    print(f"  Attack detection rate: {final_recall:.4f} ({final_tp:,} / {len(attack_scores):,})")

    return best_threshold


# =============================
# 5. Evaluate Anomaly Detector
# =============================

def evaluate_anomaly_detector(
    predictions, scores, y_true_labels, output_dir="outputs/anomaly"
):
    """Evaluate anomaly detection in binary terms (normal vs anomaly)."""
    os.makedirs(output_dir, exist_ok=True)

    # Binary: Benign=Normal(1), Attack=Anomaly(-1)
    y_binary = np.where(y_true_labels == "Benign", 1, -1)

    print("\n" + "=" * 70)
    print("ANOMALY DETECTION EVALUATION")
    print("=" * 70)

    # Confusion matrix (binary)
    cm = confusion_matrix(y_binary, predictions, labels=[1, -1])
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n  Binary Classification (Normal vs Anomaly):")
    print(f"  TP (attacks detected):     {tp:,}")
    print(f"  TN (benign correct):       {tn:,}")
    print(f"  FP (benign flagged):       {fp:,}")
    print(f"  FN (attacks missed):       {fn:,}")
    print(f"\n  Precision:         {precision:.4f}")
    print(f"  Recall (Det Rate): {recall:.4f}")
    print(f"  F1 Score:          {f1:.4f}")
    print(f"  FPR:               {fpr:.6f}")
    print(f"  FNR:               {fnr:.4f}")

    # Per-attack-class detection rates
    print(f"\n  Per-Attack-Class Detection Rates:")
    unique_labels = sorted(set(y_true_labels))
    per_class = {}
    for label in unique_labels:
        if label == "Benign":
            continue
        mask = y_true_labels == label
        detected = (predictions[mask] == -1).sum()
        total = mask.sum()
        rate = detected / total if total > 0 else 0
        print(f"    {label}: {detected}/{total} ({rate:.2%})")
        per_class[label] = {"detected": int(detected), "total": int(total), "rate": float(rate)}

    # Score statistics
    print(f"\n  Anomaly Score Statistics:")
    for label in unique_labels:
        mask = y_true_labels == label
        label_scores = scores[mask]
        print(f"    {label}: mean={label_scores.mean():.4f}, median={np.median(label_scores):.4f}, "
              f"min={label_scores.min():.4f}, max={label_scores.max():.4f}")

    # Save evaluation results
    eval_results = {
        "binary_metrics": {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "fpr": float(fpr),
            "fnr": float(fnr),
        },
        "per_class_detection": per_class,
        "confusion_matrix": cm.tolist(),
    }

    eval_path = os.path.join(output_dir, "anomaly_evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n  Evaluation saved to: {eval_path}")

    return eval_results


# =============================
# 6. Save/Load Model
# =============================

def save_anomaly_model(model, threshold, model_params, output_dir="models"):
    """Save trained model and configuration."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "isolation_forest.joblib")
    joblib.dump(model, model_path)

    config = {
        "model_type": "IsolationForest",
        "calibrated_threshold": float(threshold),
        "model_params": model_params,
    }
    config_path = os.path.join(output_dir, "anomaly_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"Config saved to: {config_path}")


def load_anomaly_model(model_dir="models"):
    """Load trained model and configuration."""
    model = joblib.load(os.path.join(model_dir, "isolation_forest.joblib"))
    with open(os.path.join(model_dir, "anomaly_config.json"), "r") as f:
        config = json.load(f)
    return model, config


# =============================
# 7. Full Pipeline
# =============================

def run_anomaly_detection_pipeline(
    contamination=0.05,
    n_estimators=300,
    target_fpr=0.05,
):
    """Execute the complete anomaly detection pipeline."""

    print("=" * 70)
    print("ANOMALY DETECTION PIPELINE")
    print("=" * 70)

    # Load preprocessing artifacts
    scaler, label_encoder, config = load_preprocessing_artifacts("models")
    feature_cols = config["feature_names"]
    cat_cols = config.get("categorical_columns", [])

    # Load data splits
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    # Prepare benign-only training data
    X_benign, _ = prepare_benign_training_data(train_df, scaler, config)

    # Train Isolation Forest
    model_params = {
        "contamination": contamination,
        "n_estimators": n_estimators,
        "max_samples": "auto",
        "random_state": 42,
    }
    model = train_isolation_forest(
        X_benign,
        contamination=contamination,
        n_estimators=n_estimators,
    )

    # Calibrate threshold on validation data
    X_val, _ = transform_features(val_df, feature_cols, scaler, cat_cols)
    threshold = calibrate_threshold(
        model, X_val, val_df["Label"].values, target_fpr=target_fpr
    )

    # Evaluate on test data
    X_test, _ = transform_features(test_df, feature_cols, scaler, cat_cols)
    predictions, scores = predict_anomalies(model, X_test, threshold=threshold)
    eval_results = evaluate_anomaly_detector(
        predictions, scores, test_df["Label"].values,
        output_dir="outputs/anomaly_test"
    )

    # Also evaluate on validation for comparison
    val_predictions, val_scores = predict_anomalies(model, X_val, threshold=threshold)
    evaluate_anomaly_detector(
        val_predictions, val_scores, val_df["Label"].values,
        output_dir="outputs/anomaly_val"
    )

    # Save model and artifacts
    model_params["calibrated_threshold"] = float(threshold)
    save_anomaly_model(model, threshold, model_params)

    return model, threshold, eval_results


# =============================
# Main
# =============================

if __name__ == "__main__":
    model, threshold, results = run_anomaly_detection_pipeline()
    print("\nAnomaly detection pipeline complete!")
