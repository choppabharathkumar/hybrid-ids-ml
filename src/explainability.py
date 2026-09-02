import numpy as np
import json
import os
import shap


# =============================
# 1. Create SHAP Explainer
# =============================

def create_explainer(model, X_background, max_background=500):
    """Create a SHAP TreeExplainer for Isolation Forest.

    Uses a background sample for efficiency with large datasets.
    """
    if len(X_background) > max_background:
        indices = np.random.RandomState(42).choice(
            len(X_background), max_background, replace=False
        )
        X_bg = X_background[indices]
    else:
        X_bg = X_background

    explainer = shap.TreeExplainer(model, data=X_bg)
    print(f"SHAP TreeExplainer created (background samples: {len(X_bg)})")
    return explainer


# =============================
# 2. Explain Anomalies
# =============================

def explain_anomalies(
    explainer, X_anomalies, feature_names, anomaly_scores=None, top_k=5
):
    """Generate SHAP explanations for detected anomalies.

    Returns a list of explanation dicts for each anomalous flow.
    """
    if len(X_anomalies) == 0:
        print("No anomalies to explain.")
        return []

    print(f"Generating SHAP explanations for {len(X_anomalies)} anomalies...")
    shap_values = explainer.shap_values(X_anomalies)

    explanations = []
    for i in range(len(X_anomalies)):
        sv = shap_values[i]
        feature_vals = X_anomalies[i]

        # Sort by absolute SHAP value (most influential first)
        top_indices = np.argsort(np.abs(sv))[::-1][:top_k]

        top_features = []
        for idx in top_indices:
            top_features.append({
                "feature": feature_names[idx],
                "shap_value": float(sv[idx]),
                "feature_value": float(feature_vals[idx]),
                "direction": "increases anomaly" if sv[idx] < 0 else "decreases anomaly",
            })

        explanation = {
            "anomaly_index": i,
            "anomaly_score": float(anomaly_scores[i]) if anomaly_scores is not None else None,
            "top_contributing_features": top_features,
            "human_readable": generate_human_explanation(top_features),
        }
        explanations.append(explanation)

    print(f"Generated {len(explanations)} explanations.")
    return explanations


# =============================
# 3. Human-Readable Explanations
# =============================

# Feature-to-cybersecurity language mapping
FEATURE_DESCRIPTIONS = {
    "Flow Duration": "flow session duration",
    "Total Fwd Packets": "number of forward packets",
    "Total Backward Packets": "number of backward packets",
    "Flow Bytes/s": "data transfer rate",
    "Flow Packets/s": "packet transmission rate",
    "Packet Length Mean": "average packet size",
    "Packet Length Max": "maximum packet size",
    "Packet Length Min": "minimum packet size",
    "Fwd Packet Length Mean": "average forward packet size",
    "Bwd Packet Length Mean": "average backward packet size",
    "SYN Flag Count": "TCP SYN flag activity",
    "ACK Flag Count": "TCP ACK flag activity",
    "PSH Flag Count": "TCP PSH flag activity",
    "FIN Flag Count": "TCP FIN flag activity",
    "RST Flag Count": "TCP RST flag activity",
    "Flow IAT Mean": "average inter-arrival time",
    "Flow IAT Max": "maximum inter-arrival time",
    "Flow IAT Min": "minimum inter-arrival time",
    "Fwd Packets/s": "forward packet rate",
    "Bwd Packets/s": "backward packet rate",
    "Init Fwd Win Bytes": "initial TCP forward window size",
    "Init Bwd Win Bytes": "initial TCP backward window size",
    "Fwd Act Data Packets": "forward data packet count",
    "Down/Up Ratio": "download-to-upload ratio",
    "Avg Packet Size": "average packet size",
    "Fwd Header Length": "forward header size",
    "Bwd Header Length": "backward header size",
}


def generate_human_explanation(top_features):
    """Translate SHAP feature contributions into cybersecurity language.

    Note: SHAP shows model contribution, not proof of causation.
    """
    parts = []
    for feat in top_features[:3]:
        name = feat["feature"]
        direction = feat["direction"]
        desc = FEATURE_DESCRIPTIONS.get(name, name.lower())

        if direction == "increases anomaly":
            parts.append(f"unusual {desc}")
        else:
            parts.append(f"atypical {desc}")

    if parts:
        explanation = (
            "This flow was flagged as anomalous primarily due to: "
            + ", ".join(parts)
            + ". These features deviate from the model's learned profile of normal traffic."
        )
    else:
        explanation = "Anomaly detected based on overall feature deviation from normal traffic patterns."

    return explanation


# =============================
# 4. Save Explanations
# =============================

def save_explanations(explanations, output_dir="outputs/shap"):
    """Save SHAP explanations to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Save individual explanations (limit to first 100 for file size)
    sample_explanations = explanations[:100]
    path = os.path.join(output_dir, "anomaly_explanations.json")
    with open(path, "w") as f:
        json.dump(sample_explanations, f, indent=2)
    print(f"Explanations saved to: {path} ({len(sample_explanations)} samples)")

    # Save summary statistics
    if explanations:
        all_features = {}
        for exp in explanations:
            for feat in exp["top_contributing_features"]:
                name = feat["feature"]
                if name not in all_features:
                    all_features[name] = {"count": 0, "total_abs_shap": 0}
                all_features[name]["count"] += 1
                all_features[name]["total_abs_shap"] += abs(feat["shap_value"])

        # Rank by frequency of appearance in top features
        feature_importance = sorted(
            all_features.items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        )

        summary = {
            "total_anomalies_explained": len(explanations),
            "top_contributing_features": [
                {
                    "feature": name,
                    "times_in_top_k": data["count"],
                    "mean_abs_shap": data["total_abs_shap"] / data["count"],
                }
                for name, data in feature_importance[:15]
            ],
        }

        summary_path = os.path.join(output_dir, "shap_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")

    return


# =============================
# Main
# =============================

if __name__ == "__main__":
    print("Explainability module loaded. Use via hybrid_detection.py or import directly.")
