import pandas as pd
import numpy as np
import json
import os
import joblib
from src.rule_based_detection import apply_signature_detection_vectorized
from src.anomaly_detection import load_anomaly_model, predict_anomalies
from src.explainability import create_explainer, explain_anomalies, save_explanations
from src.risk_engine import generate_risk_report
from src.remediation import generate_remediation, simulate_response
from src.data_preprocessing import load_preprocessing_artifacts, transform_features


# =============================
# 1. Hybrid Detection Pipeline
# =============================

def run_hybrid_detection(df, models_dir="models", output_dir="outputs/hybrid"):
    """Execute the complete hybrid detection pipeline.

    Stage 1: Signature Detection — check against known attack signatures
    Stage 2: Anomaly Detection — send unmatched flows to Isolation Forest
    Stage 3: SHAP Explainability — explain anomaly decisions
    Stage 4: Risk Assessment — score and classify severity
    Stage 5: Remediation — generate response recommendations
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("HYBRID DETECTION PIPELINE")
    print("=" * 70)

    # Load preprocessing artifacts
    scaler, label_encoder, config = load_preprocessing_artifacts(models_dir)
    feature_names = config["feature_names"]
    cat_cols = config.get("categorical_columns", [])

    # Load anomaly model
    anomaly_model, anomaly_config = load_anomaly_model(models_dir)
    threshold = anomaly_config["calibrated_threshold"]

    # =============================
    # Stage 1: Signature Detection
    # =============================
    print("\n--- Stage 1: Signature Detection ---")
    df = apply_signature_detection_vectorized(df)

    matched = df[df["Signature_Matched"]].copy()
    unmatched = df[~df["Signature_Matched"]].copy()

    print(f"  Signature matched: {len(matched):,}")
    print(f"  Unmatched (sent to anomaly detection): {len(unmatched):,}")

    if len(matched) > 0:
        print(f"  Signature predictions: {matched['Signature_Prediction'].value_counts().to_dict()}")

    # =============================
    # Stage 2: Anomaly Detection
    # =============================
    print("\n--- Stage 2: Anomaly Detection ---")

    if len(unmatched) > 0:
        X_unmatched, _ = transform_features(unmatched, feature_names, scaler, cat_cols)
        anomaly_preds, anomaly_scores = predict_anomalies(
            anomaly_model, X_unmatched, threshold=threshold
        )

        unmatched["Anomaly_Prediction"] = anomaly_preds
        unmatched["Anomaly_Score"] = anomaly_scores
        unmatched["Is_Anomaly"] = anomaly_preds == -1

        n_anomalies = (anomaly_preds == -1).sum()
        print(f"  Anomalies detected: {n_anomalies:,}")
    else:
        unmatched["Anomaly_Prediction"] = []
        unmatched["Anomaly_Score"] = []
        unmatched["Is_Anomaly"] = []

    # =============================
    # Stage 3: SHAP Explainability
    # =============================
    print("\n--- Stage 3: SHAP Explainability ---")

    anomalous_flows = unmatched[unmatched.get("Is_Anomaly", False) == True]
    explanations = []

    if len(anomalous_flows) > 0:
        # Prepare background data for SHAP
        X_bg, _ = transform_features(
            unmatched.head(min(500, len(unmatched))), feature_names, scaler, cat_cols
        )
        explainer = create_explainer(anomaly_model, X_bg)

        X_anomalous, _ = transform_features(anomalous_flows, feature_names, scaler, cat_cols)
        anomalous_scores = anomalous_flows["Anomaly_Score"].values

        # Limit SHAP to first 200 anomalies for performance
        max_explain = min(200, len(X_anomalous))
        explanations = explain_anomalies(
            explainer, X_anomalous[:max_explain], feature_names,
            anomaly_scores=anomalous_scores[:max_explain], top_k=5
        )
    else:
        print("  No anomalies to explain.")

    # Save SHAP explanations
    if explanations:
        save_explanations(explanations, output_dir="outputs/shap")

    # =============================
    # Stage 4 & 5: Risk & Remediation
    # =============================
    print("\n--- Stage 4 & 5: Risk Assessment & Remediation ---")

    hybrid_results = []

    # Process signature-matched flows
    for idx, row in matched.iterrows():
        det_result = {
            "detection_type": "Signature",
            "prediction": row["Signature_Prediction"],
            "rule_id": row["Signature_Rule_ID"],
            "flow_index": idx,
        }
        risk = generate_risk_report(det_result)
        remediation = generate_remediation(det_result, risk)

        hybrid_results.append({
            **det_result,
            "risk_score": risk["risk_score"],
            "severity": risk["severity"],
            "evidence": risk["evidence"],
            "remediation": remediation["description"],
            "recommended_actions": remediation["recommended_actions"][:3],
        })

    # Process anomaly-detected flows
    explanation_idx = 0
    for idx, row in anomalous_flows.iterrows():
        exp = explanations[explanation_idx] if explanation_idx < len(explanations) else None
        explanation_idx += 1

        det_result = {
            "detection_type": "Anomaly",
            "prediction": "Potential Unknown Attack",
            "anomaly_score": float(row["Anomaly_Score"]),
            "flow_index": idx,
            "shap_features": exp["top_contributing_features"] if exp else [],
        }
        risk = generate_risk_report(det_result)
        remediation = generate_remediation(det_result, risk)

        hybrid_results.append({
            **det_result,
            "risk_score": risk["risk_score"],
            "severity": risk["severity"],
            "evidence": risk["evidence"],
            "remediation": remediation["description"],
            "recommended_actions": remediation["recommended_actions"][:3],
            "explanation": exp["human_readable"] if exp else None,
        })

    # =============================
    # Final Output
    # =============================
    print("\n--- Detection Summary ---")
    print(f"  Total flows analyzed: {len(df):,}")
    print(f"  Signature detections: {len(matched):,}")
    print(f"  Anomaly detections: {len(anomalous_flows):,}")
    print(f"  Normal (no threat): {len(unmatched) - len(anomalous_flows):,}")

    # Build final DataFrame with all columns
    # Set defaults for matched flows (no anomaly detection needed)
    matched["Anomaly_Prediction"] = 1
    matched["Anomaly_Score"] = np.nan
    matched["Is_Anomaly"] = False

    # Set defaults for unmatched flows (no signature match)
    # (already set from signature detection)

    # Combine
    final_df = pd.concat([matched, unmatched], ignore_index=False).sort_index()

    # Add final classification column
    final_df["Hybrid_Prediction"] = "Benign"
    final_df.loc[final_df["Signature_Matched"], "Hybrid_Prediction"] = final_df.loc[
        final_df["Signature_Matched"], "Signature_Prediction"
    ]
    final_df.loc[
        final_df.get("Is_Anomaly", False) == True, "Hybrid_Prediction"
    ] = "Potential Unknown Attack"

    # Save results
    results_path = os.path.join(output_dir, "hybrid_results.json")
    with open(results_path, "w") as f:
        json.dump(hybrid_results[:500], f, indent=2)  # Save first 500 for file size

    final_df.to_parquet(os.path.join(output_dir, "hybrid_predictions.parquet"), index=False)

    summary = {
        "total_flows": len(df),
        "signature_detections": len(matched),
        "anomaly_detections": int(len(anomalous_flows)),
        "normal_flows": int(len(unmatched) - len(anomalous_flows)),
        "signature_distribution": matched["Signature_Prediction"].value_counts().to_dict() if len(matched) > 0 else {},
        "total_explanations": len(explanations),
    }

    summary_path = os.path.join(output_dir, "hybrid_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to: {output_dir}/")
    return final_df, hybrid_results, explanations


# =============================
# Main
# =============================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    test_df = pd.read_parquet("data/processed/test.parquet")
    final_df, results, explanations = run_hybrid_detection(test_df)
    print("\nHybrid detection pipeline complete!")
