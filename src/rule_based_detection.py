import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# =============================
# Rule Documentation
# =============================

RULE_DOCUMENTATION = {
    "PORTSCAN_001": {
        "rule_id": "PORTSCAN_001",
        "attack_category": "PortScan",
        "description": "Detects port scanning behavior based on small packet counts, "
                       "low packet payload, and single-direction push flag pattern.",
        "features_used": [
            "PSH Flag Count",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Fwd Act Data Packets",
            "Packet Length Mean",
        ],
        "thresholds": {
            "PSH Flag Count": "== 1",
            "Total Fwd Packets": "<= 5",
            "Total Backward Packets": "<= 3",
            "Fwd Act Data Packets": "<= 1",
            "Packet Length Mean": "< 100",
        },
        "justification": (
            "PortScan flows in CICIDS2017 characteristically have PSH=1 (96% of flows), "
            "very few forward/backward packets (median TotalFwd=2, TotalBwd=1), "
            "minimal actual data transmission (FwdActData <= 1 for 99.8%), "
            "and low Packet Length Mean (median ~2.4). These thresholds were derived "
            "from the p95 boundaries of the training set's PortScan distribution."
        ),
        "limitations": (
            "Captures ~77-92% of PortScan flows depending on strictness. "
            "Some short-lived benign TCP connections may match (FPR ~0.2-1.2%). "
            "Does not detect SYN-based port scans (SYN Flag Count is always 0 in this dataset)."
        ),
        "training_metrics": {},
    },
    "WEBATTACK_001": {
        "rule_id": "WEBATTACK_001",
        "attack_category": "Web Attack (Brute Force / XSS)",
        "description": "Detects web application attacks (Brute Force and XSS) "
                       "based on TCP push-without-acknowledge pattern, zero data transmission, "
                       "long flow duration, and specific TCP window sizes from the attack tool.",
        "features_used": [
            "PSH Flag Count",
            "ACK Flag Count",
            "Fwd Act Data Packets",
            "Flow Duration",
            "Init Fwd Win Bytes",
            "Init Bwd Win Bytes",
            "Down/Up Ratio",
            "Flow Bytes/s",
        ],
        "thresholds": {
            "PSH Flag Count": "== 1",
            "ACK Flag Count": "== 0",
            "Fwd Act Data Packets": "== 0",
            "Flow Duration": "> 1,000,000 microseconds (~1 second)",
            "Init Fwd Win Bytes": "== 29200",
            "Init Bwd Win Bytes": "== 28960",
            "Down/Up Ratio": "== 0",
            "Flow Bytes/s": "== 0",
        },
        "justification": (
            "Web attacks in CICIDS2017 show a distinctive pattern: PSH flag set without ACK "
            "(PSH=1, ACK=0), zero actual data packets forwarded, long flow durations "
            "(median ~5.5M microseconds), and specific TCP window sizes (InitFwdWin=29200, "
            "InitBwdWin=28960) characteristic of the attack tool's TCP stack configuration. "
            "The zero Flow Bytes/s indicates the attack tool maintains connections without "
            "significant data transfer. These values were derived from training data distributions."
        ),
        "sub_classification": (
            "Brute Force and XSS are indistinguishable at the network-flow level. Both have "
            "identical median feature values (Packet Length Mean=0, same PSH/ACK/InitWin pattern). "
            "Distinguishing them requires HTTP payload inspection which is unavailable in flow data. "
            "Both are classified as 'Web Attack - Brute Force' by this rule when Packet Length Mean <= 50. "
            "Flows with Packet Length Mean > 50 matching this pattern are classified as XSS."
        ),
        "limitations": (
            "Precision is limited (~30-48%) because some benign long-lived TCP connections "
            "share similar flow characteristics at the network level. Flow-level features "
            "do not contain HTTP payload content. Brute Force vs XSS distinction is unreliable "
            "at the flow level. Flows not matching this exact pattern are deferred to anomaly detection."
        ),
        "training_metrics": {},
    },
    "SQLI_001": {
        "rule_id": "SQLI_001",
        "attack_category": "Web Attack - SQL Injection",
        "description": "Detects SQL Injection attacks based on distinctive TCP window sizes "
                       "and actual data transmission indicating SQL payload delivery.",
        "features_used": [
            "Init Fwd Win Bytes",
            "Init Bwd Win Bytes",
            "Fwd Act Data Packets",
            "Packet Length Mean",
            "Flow Duration",
        ],
        "thresholds": {
            "Init Fwd Win Bytes": "== 29200",
            "Init Bwd Win Bytes": "in [235, 236]",
            "Fwd Act Data Packets": ">= 1",
            "Packet Length Mean": "> 50",
            "Flow Duration": "> 1,000,000 microseconds",
        },
        "justification": (
            "SQL Injection flows in CICIDS2017 show a unique pattern distinct from Brute Force/XSS: "
            "the forward TCP window is 29200 (same attack tool), but the backward window is "
            "235-236 (NOT 28960 as in BF/XSS). This smaller backward window likely reflects "
            "different server behavior when processing SQL payloads. The flows have "
            "Fwd Act Data Packets >= 1 (actual data sent) and higher Packet Length Mean (median ~117) "
            "indicating larger payloads containing SQL statements. These values come from the "
            "15 SQL Injection training samples."
        ),
        "limitations": (
            "Only 15 training samples and 4 test samples for SQL Injection. "
            "Metrics are statistically fragile and should not be treated as reliable. "
            "The Init Bwd Win Bytes threshold of 235-236 may be specific to the lab environment. "
            "Only matches the subset of SQL Injection with InitFwdWin=29200."
        ),
        "training_metrics": {},
    },
}


# =============================
# 1. Signature Rules
# =============================

def classify_portscan(row):
    """PORTSCAN_001: Detect port scanning behavior."""
    return (
        row["PSH Flag Count"] == 1
        and row["Total Fwd Packets"] <= 5
        and row["Total Backward Packets"] <= 3
        and row["Fwd Act Data Packets"] <= 1
        and row["Packet Length Mean"] < 100
    )


def classify_web_attack(row):
    """WEBATTACK_001: Detect web application attacks (Brute Force / XSS)."""
    return (
        row["PSH Flag Count"] == 1
        and row["ACK Flag Count"] == 0
        and row["Fwd Act Data Packets"] == 0
        and row["Flow Duration"] > 1000000
        and row["Init Fwd Win Bytes"] == 29200
        and row["Init Bwd Win Bytes"] == 28960
        and row["Down/Up Ratio"] == 0
        and row["Flow Bytes/s"] == 0
    )


def classify_sql_injection(row):
    """SQLI_001: Detect SQL Injection attacks."""
    return (
        row["Init Fwd Win Bytes"] == 29200
        and row["Init Bwd Win Bytes"] in (235, 236)
        and row["Fwd Act Data Packets"] >= 1
        and row["Packet Length Mean"] > 50
        and row["Flow Duration"] > 1000000
    )


def classify_web_attack_subtype(row):
    """Sub-classify Brute Force vs XSS based on Packet Length Mean.

    Note: this distinction is unreliable at the flow level since
    both attacks have identical median feature values. Flows with
    higher Packet Length Mean are more likely XSS (script payload).
    """
    if row["Packet Length Mean"] > 50:
        return "Web Attack - XSS"
    else:
        return "Web Attack - Brute Force"


# =============================
# 2. Signature Detection Engine
# =============================

def apply_signature_detection(df):
    """Apply all signature rules to classify each flow.

    Returns DataFrame with added columns:
      - Signature_Prediction: predicted attack label or 'Unmatched'
      - Signature_Rule_ID: which rule matched (or 'NONE')
      - Signature_Matched: boolean flag
    """
    predictions = []
    rule_ids = []

    for _, row in df.iterrows():
        if classify_sql_injection(row):
            predictions.append("Web Attack - SQL Injection")
            rule_ids.append("SQLI_001")
        elif classify_web_attack(row):
            subtype = classify_web_attack_subtype(row)
            predictions.append(subtype)
            rule_ids.append("WEBATTACK_001")
        elif classify_portscan(row):
            predictions.append("PortScan")
            rule_ids.append("PORTSCAN_001")
        else:
            predictions.append("Unmatched")
            rule_ids.append("NONE")

    df = df.copy()
    df["Signature_Prediction"] = predictions
    df["Signature_Rule_ID"] = rule_ids
    df["Signature_Matched"] = df["Signature_Prediction"] != "Unmatched"

    return df


def apply_signature_detection_vectorized(df):
    """Vectorized version for large datasets — much faster than row-by-row."""
    df = df.copy()

    df["Signature_Prediction"] = "Unmatched"
    df["Signature_Rule_ID"] = "NONE"

    # SQLI_001 (checked first — most specific rule)
    sqli_mask = (
        (df["Init Fwd Win Bytes"] == 29200)
        & (df["Init Bwd Win Bytes"].isin([235, 236]))
        & (df["Fwd Act Data Packets"] >= 1)
        & (df["Packet Length Mean"] > 50)
        & (df["Flow Duration"] > 1000000)
    )
    df.loc[sqli_mask, "Signature_Prediction"] = "Web Attack - SQL Injection"
    df.loc[sqli_mask, "Signature_Rule_ID"] = "SQLI_001"

    # WEBATTACK_001 (only for flows NOT already matched)
    web_mask = (
        (df["Signature_Prediction"] == "Unmatched")
        & (df["PSH Flag Count"] == 1)
        & (df["ACK Flag Count"] == 0)
        & (df["Fwd Act Data Packets"] == 0)
        & (df["Flow Duration"] > 1000000)
        & (df["Init Fwd Win Bytes"] == 29200)
        & (df["Init Bwd Win Bytes"] == 28960)
        & (df["Down/Up Ratio"] == 0)
        & (df["Flow Bytes/s"] == 0)
    )

    # Sub-classify: XSS when Packet Length Mean > 50, else Brute Force
    xss_mask = web_mask & (df["Packet Length Mean"] > 50)
    bf_mask = web_mask & (df["Packet Length Mean"] <= 50)

    df.loc[xss_mask, "Signature_Prediction"] = "Web Attack - XSS"
    df.loc[xss_mask, "Signature_Rule_ID"] = "WEBATTACK_001"
    df.loc[bf_mask, "Signature_Prediction"] = "Web Attack - Brute Force"
    df.loc[bf_mask, "Signature_Rule_ID"] = "WEBATTACK_001"

    # PORTSCAN_001 (only for flows NOT already matched)
    ps_mask = (
        (df["Signature_Prediction"] == "Unmatched")
        & (df["PSH Flag Count"] == 1)
        & (df["Total Fwd Packets"] <= 5)
        & (df["Total Backward Packets"] <= 3)
        & (df["Fwd Act Data Packets"] <= 1)
        & (df["Packet Length Mean"] < 100)
    )
    df.loc[ps_mask, "Signature_Prediction"] = "PortScan"
    df.loc[ps_mask, "Signature_Rule_ID"] = "PORTSCAN_001"

    df["Signature_Matched"] = df["Signature_Prediction"] != "Unmatched"

    return df


# =============================
# 3. Evaluation
# =============================

def evaluate_signature_engine(df, output_dir="outputs"):
    """Evaluate signature detection against ground truth labels."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("SIGNATURE ENGINE EVALUATION")
    print("=" * 70)

    print("\nPrediction Distribution:")
    print(df["Signature_Prediction"].value_counts())

    print("\nGround Truth Distribution:")
    print(df["Label"].value_counts())

    # For evaluation, treat Unmatched as "Benign" prediction
    df_eval = df.copy()
    sig_pred_binary = df_eval["Signature_Prediction"].replace("Unmatched", "Benign")

    accuracy = accuracy_score(df_eval["Label"], sig_pred_binary)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    labels = sorted(df_eval["Label"].unique())
    print("\nClassification Report:")
    report_str = classification_report(
        df_eval["Label"], sig_pred_binary,
        labels=labels, zero_division=0
    )
    print(report_str)

    cm = confusion_matrix(df_eval["Label"], sig_pred_binary, labels=labels)
    print("Confusion Matrix:")
    print(f"Labels: {labels}")
    print(cm)

    report_dict = classification_report(
        df_eval["Label"], sig_pred_binary,
        labels=labels, output_dict=True, zero_division=0
    )

    macro_f1 = f1_score(df_eval["Label"], sig_pred_binary, average="macro", zero_division=0)
    weighted_f1 = f1_score(df_eval["Label"], sig_pred_binary, average="weighted", zero_division=0)

    print(f"\nMacro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\n" + "=" * 70)
    print("DETECTION-SPECIFIC METRICS")
    print("=" * 70)

    for attack_label in labels:
        if attack_label == "Benign":
            continue
        tp = ((df_eval["Label"] == attack_label) & (sig_pred_binary == attack_label)).sum()
        fp = ((df_eval["Label"] != attack_label) & (sig_pred_binary == attack_label)).sum()
        fn = ((df_eval["Label"] == attack_label) & (sig_pred_binary != attack_label)).sum()
        tn = ((df_eval["Label"] != attack_label) & (sig_pred_binary != attack_label)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        print(f"\n  {attack_label}:")
        print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall (Detection Rate): {recall:.4f}")
        print(f"    FPR: {fpr:.6f}")
        print(f"    FNR: {fnr:.4f}")

    eval_results = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "prediction_distribution": df["Signature_Prediction"].value_counts().to_dict(),
        "label_distribution": df["Label"].value_counts().to_dict(),
    }

    eval_path = os.path.join(output_dir, "signature_evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEvaluation results saved to: {eval_path}")

    doc_path = os.path.join(output_dir, "rule_documentation.json")
    with open(doc_path, "w") as f:
        json.dump(RULE_DOCUMENTATION, f, indent=2)
    print(f"Rule documentation saved to: {doc_path}")

    return eval_results


# =============================
# 4. Feature Analysis
# =============================

def analyze_features_by_class(df, output_path=None):
    """Generate per-class feature statistics for rule derivation."""
    important_features = [
        "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
        "Total Fwd Packets", "Total Backward Packets",
        "Packet Length Mean", "Packet Length Max",
        "SYN Flag Count", "ACK Flag Count", "PSH Flag Count",
        "Init Fwd Win Bytes", "Init Bwd Win Bytes",
        "Fwd Act Data Packets", "Down/Up Ratio", "Avg Packet Size",
    ]

    existing_features = [f for f in important_features if f in df.columns]

    print("=" * 70)
    print("FEATURE ANALYSIS BY CLASS")
    print("=" * 70)

    stats = df.groupby("Label")[existing_features].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )

    if output_path:
        stats.to_csv(output_path)
        print(f"Feature analysis saved to: {output_path}")

    return stats


# =============================
# Main
# =============================

def main():
    print("=" * 70)
    print("SIGNATURE-BASED DETECTION ENGINE")
    print("=" * 70)

    train_df = pd.read_parquet("data/processed/train.parquet")
    print(f"\nTraining data: {train_df.shape}")

    analyze_features_by_class(train_df, "outputs/feature_analysis.csv")

    print("\n--- Training Set Detection ---")
    train_detected = apply_signature_detection_vectorized(train_df)
    evaluate_signature_engine(train_detected, output_dir="outputs/signature_train")

    val_df = pd.read_parquet("data/processed/val.parquet")
    print(f"\n\n--- Validation Set Detection ---")
    val_detected = apply_signature_detection_vectorized(val_df)
    evaluate_signature_engine(val_detected, output_dir="outputs/signature_val")

    test_df = pd.read_parquet("data/processed/test.parquet")
    print(f"\n\n--- Test Set Detection ---")
    test_detected = apply_signature_detection_vectorized(test_df)
    evaluate_signature_engine(test_detected, output_dir="outputs/signature_test")

    test_detected.to_parquet("outputs/rule_based_predictions.parquet", index=False)
    print("\nTest predictions saved to: outputs/rule_based_predictions.parquet")


if __name__ == "__main__":
    main()