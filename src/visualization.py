import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os


# =============================
# 1. Class Distribution
# =============================

def plot_class_distribution(df, output_dir="outputs/plots"):
    """Bar chart of label distribution in the dataset."""
    os.makedirs(output_dir, exist_ok=True)

    counts = df["Label"].value_counts()
    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(counts)), counts.values, color=colors[:len(counts)])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.set_ylabel("Number of Flows")
    ax.set_title("Class Distribution")

    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{count:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150)
    plt.close()
    print("  Saved: class_distribution.png")


# =============================
# 2. Feature Distributions
# =============================

def plot_feature_distributions(df, output_dir="outputs/plots"):
    """Box plots of important features grouped by class."""
    os.makedirs(output_dir, exist_ok=True)

    features = [
        "Flow Duration", "Packet Length Mean", "Total Fwd Packets",
        "Total Backward Packets", "Flow Bytes/s", "PSH Flag Count",
    ]
    existing = [f for f in features if f in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    labels = sorted(df["Label"].unique())
    for i, feat in enumerate(existing):
        if i >= len(axes):
            break
        ax = axes[i]
        data = []
        tick_labels = []
        for label in labels:
            subset = df[df["Label"] == label][feat].dropna()
            clipped = subset.clip(upper=subset.quantile(0.99))
            data.append(clipped.values)
            tick_labels.append(label[:15])

        bp = ax.boxplot(data, labels=tick_labels, patch_artist=True)
        colors = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db", "#9b59b6"]
        for patch, color in zip(bp["boxes"], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(feat, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    for j in range(len(existing), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Class", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"), dpi=150)
    plt.close()
    print("  Saved: feature_distributions.png")


# =============================
# 3. Confusion Matrix
# =============================

def plot_confusion_matrix(cm, labels, title, filename, output_dir="outputs/plots"):
    """Heatmap-style confusion matrix using Matplotlib."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# =============================
# 4. Precision / Recall / F1
# =============================

def plot_per_class_metrics(report_dict, title, filename, output_dir="outputs/plots"):
    """Grouped bar chart comparing precision, recall, F1 per class."""
    os.makedirs(output_dir, exist_ok=True)

    classes = []
    precisions = []
    recalls = []
    f1s = []

    for cls, metrics in report_dict.items():
        if isinstance(metrics, dict) and "precision" in metrics and cls not in ("macro avg", "weighted avg", "accuracy"):
            classes.append(cls[:20])
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1s.append(metrics["f1-score"])

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precisions, width, label="Precision", color="#3498db")
    ax.bar(x, recalls, width, label="Recall", color="#e74c3c")
    ax.bar(x + width, f1s, width, label="F1-Score", color="#2ecc71")

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# =============================
# 5. Anomaly Score Distribution
# =============================

def plot_anomaly_scores(df, output_dir="outputs/plots"):
    """Histogram of anomaly scores by label."""
    os.makedirs(output_dir, exist_ok=True)

    if "Anomaly_Score" not in df.columns:
        print("  Skipped: anomaly_score_distribution.png (no scores)")
        return

    score_df = df[df["Anomaly_Score"].notna()].copy()
    if len(score_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = sorted(score_df["Label"].unique())
    colors = {"Benign": "#2ecc71", "PortScan": "#e74c3c",
              "Web Attack - Brute Force": "#e67e22", "Web Attack - XSS": "#3498db",
              "Web Attack - SQL Injection": "#9b59b6"}

    for label in labels:
        subset = score_df[score_df["Label"] == label]["Anomaly_Score"]
        if len(subset) > 5000:
            subset = subset.sample(5000, random_state=42)
        ax.hist(subset, bins=50, alpha=0.6, label=label,
                color=colors.get(label, "#95a5a6"))

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Anomaly Score Distribution by Label")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anomaly_score_distribution.png"), dpi=150)
    plt.close()
    print("  Saved: anomaly_score_distribution.png")


# =============================
# 6. Threshold Analysis
# =============================

def plot_threshold_analysis(scores, y_labels, output_dir="outputs/plots"):
    """Plot FPR and detection rate vs threshold."""
    os.makedirs(output_dir, exist_ok=True)

    benign_scores = scores[y_labels == "Benign"]
    attack_scores = scores[y_labels != "Benign"]

    thresholds = np.linspace(scores.min(), np.percentile(benign_scores, 20), 200)
    fprs = []
    recalls = []

    for t in thresholds:
        fp = (benign_scores < t).sum()
        fpr = fp / len(benign_scores)
        tp = (attack_scores < t).sum()
        recall = tp / len(attack_scores) if len(attack_scores) > 0 else 0
        fprs.append(fpr)
        recalls.append(recall)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thresholds, recalls, "b-", label="Attack Detection Rate", linewidth=2)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Detection Rate", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, fprs, "r--", label="False Positive Rate", linewidth=2)
    ax2.set_ylabel("False Positive Rate", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    ax1.set_title("Anomaly Detection: Threshold vs Detection/FPR Trade-off")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "threshold_analysis.png"), dpi=150)
    plt.close()
    print("  Saved: threshold_analysis.png")


# =============================
# 7. False Positive Analysis
# =============================

def plot_false_positive_analysis(df, output_dir="outputs/plots"):
    """Analyze false positives from signature detection."""
    os.makedirs(output_dir, exist_ok=True)

    if "Signature_Prediction" not in df.columns:
        return

    sig_pred = df["Signature_Prediction"].replace("Unmatched", "Benign")
    fp_mask = (df["Label"] == "Benign") & (sig_pred != "Benign")
    fp_df = df[fp_mask]

    if len(fp_df) == 0:
        print("  Skipped: false_positive_analysis.png (no FPs)")
        return

    fp_types = sig_pred[fp_mask].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(fp_types.index, fp_types.values, color="#e74c3c", alpha=0.8)
    ax.set_ylabel("Count")
    ax.set_title(f"False Positives by Predicted Category (Total: {len(fp_df):,})")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "false_positive_analysis.png"), dpi=150)
    plt.close()
    print("  Saved: false_positive_analysis.png")


# =============================
# 8. False Negative Analysis
# =============================

def plot_false_negative_analysis(df, output_dir="outputs/plots"):
    """Analyze false negatives from signature detection."""
    os.makedirs(output_dir, exist_ok=True)

    if "Signature_Prediction" not in df.columns:
        return

    sig_pred = df["Signature_Prediction"].replace("Unmatched", "Benign")

    fn_data = {}
    attack_labels = [l for l in df["Label"].unique() if l != "Benign"]
    for label in attack_labels:
        total = (df["Label"] == label).sum()
        missed = ((df["Label"] == label) & (sig_pred != label)).sum()
        fn_data[label] = {"missed": missed, "total": total,
                          "fnr": missed / total if total > 0 else 0}

    classes = list(fn_data.keys())
    fnrs = [fn_data[c]["fnr"] for c in classes]
    totals = [fn_data[c]["total"] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(classes)), fnrs, color="#e67e22", alpha=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c[:20] for c in classes], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("False Negative Rate")
    ax.set_title("Signature Engine: False Negative Rate by Attack Class")
    ax.set_ylim(0, 1.1)

    for bar, fnr, total in zip(bars, fnrs, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{fnr:.1%}\n(n={total})", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "false_negative_analysis.png"), dpi=150)
    plt.close()
    print("  Saved: false_negative_analysis.png")


# =============================
# 9. SHAP Feature Importance
# =============================

def plot_shap_importance(shap_summary_path, output_dir="outputs/plots"):
    """Horizontal bar chart of top SHAP features."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(shap_summary_path):
        print("  Skipped: shap_feature_importance.png (no summary)")
        return

    with open(shap_summary_path, "r") as f:
        summary = json.load(f)

    top_feats = summary.get("top_contributing_features", [])
    if not top_feats:
        return

    features = [f["feature"][:25] for f in top_feats[:10]]
    values = [f["mean_abs_shap"] for f in top_feats[:10]]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(features))
    ax.barh(y_pos, values, color="#8e44ad", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Top Features by SHAP Importance (Anomaly Explanations)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"), dpi=150)
    plt.close()
    print("  Saved: shap_feature_importance.png")


# =============================
# 10. Detection Summary
# =============================

def plot_detection_summary(hybrid_summary, output_dir="outputs/plots"):
    """Pie chart of detection type distribution."""
    os.makedirs(output_dir, exist_ok=True)

    if hybrid_summary is None:
        return

    sizes = [
        hybrid_summary.get("signature_detections", 0),
        hybrid_summary.get("anomaly_detections", 0),
        hybrid_summary.get("normal_flows", 0),
    ]
    labels_text = ["Signature\nDetections", "Anomaly\nDetections", "Normal\nFlows"]
    colors = ["#e74c3c", "#e67e22", "#2ecc71"]

    # Remove zero entries
    filtered = [(s, l, c) for s, l, c in zip(sizes, labels_text, colors) if s > 0]
    if not filtered:
        return
    sizes, labels_text, colors = zip(*filtered)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels_text, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.85
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("Hybrid Detection Summary")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detection_summary.png"), dpi=150)
    plt.close()
    print("  Saved: detection_summary.png")


# =============================
# Main: Generate All Plots
# =============================

def generate_all_plots(output_dir="outputs/plots"):
    """Generate the complete set of visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 70)

    # Load data
    test_df = pd.read_parquet("data/processed/test.parquet")

    # 1. Class distribution
    plot_class_distribution(test_df, output_dir)

    # 2. Feature distributions
    plot_feature_distributions(test_df, output_dir)

    # 3-4. Signature evaluation
    sig_eval_path = "outputs/signature_test/signature_evaluation.json"
    if os.path.exists(sig_eval_path):
        with open(sig_eval_path) as f:
            sig_eval = json.load(f)
        cm = np.array(sig_eval["confusion_matrix"])
        labels = sig_eval["confusion_matrix_labels"]
        plot_confusion_matrix(cm, labels, "Signature Detection Confusion Matrix",
                              "signature_confusion_matrix.png", output_dir)
        plot_per_class_metrics(sig_eval["classification_report"],
                               "Signature Engine: Per-Class Metrics",
                               "signature_per_class_metrics.png", output_dir)

    # 5. Anomaly score distribution
    hybrid_path = "outputs/hybrid/hybrid_predictions.parquet"
    if os.path.exists(hybrid_path):
        hybrid_df = pd.read_parquet(hybrid_path)
        plot_anomaly_scores(hybrid_df, output_dir)
        plot_false_positive_analysis(hybrid_df, output_dir)
        plot_false_negative_analysis(hybrid_df, output_dir)

    # 6. Threshold analysis — requires re-computing scores
    anomaly_eval_path = "outputs/anomaly_test/anomaly_evaluation.json"
    if os.path.exists(anomaly_eval_path):
        with open(anomaly_eval_path) as f:
            anom_eval = json.load(f)
        cm_a = np.array(anom_eval["confusion_matrix"])
        plot_confusion_matrix(cm_a, ["Normal", "Anomaly"],
                              "Anomaly Detection Confusion Matrix",
                              "anomaly_confusion_matrix.png", output_dir)

    # 9. SHAP importance
    plot_shap_importance("outputs/shap/shap_summary.json", output_dir)

    # 10. Detection summary
    summary_path = "outputs/hybrid/hybrid_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            hybrid_summary = json.load(f)
        plot_detection_summary(hybrid_summary, output_dir)

    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    generate_all_plots()
