import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os


# =============================
# Page Configuration
# =============================

st.set_page_config(
    page_title="Hybrid IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================
# Custom CSS
# =============================

st.markdown("""
<style>
    .stMetric > div { background: #1e1e2e; border-radius: 10px; padding: 15px; }
    .severity-critical { color: #ff4444; font-weight: bold; }
    .severity-high { color: #ff8800; font-weight: bold; }
    .severity-medium { color: #ffcc00; font-weight: bold; }
    .severity-low { color: #44bb44; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# =============================
# Data Loading
# =============================

@st.cache_data
def load_hybrid_results():
    """Load hybrid detection results."""
    results_path = "outputs/hybrid/hybrid_predictions.parquet"
    if os.path.exists(results_path):
        return pd.read_parquet(results_path)
    return None


@st.cache_data
def load_hybrid_summary():
    """Load hybrid detection summary."""
    path = "outputs/hybrid/hybrid_summary.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_detailed_results():
    """Load detailed detection results with explanations."""
    path = "outputs/hybrid/hybrid_results.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


@st.cache_data
def load_signature_eval():
    """Load signature evaluation metrics."""
    path = "outputs/signature_test/signature_evaluation.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_anomaly_eval():
    """Load anomaly detection evaluation metrics."""
    path = "outputs/anomaly_test/anomaly_evaluation.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_shap_summary():
    """Load SHAP explanation summary."""
    path = "outputs/shap/shap_summary.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_rule_docs():
    """Load rule documentation."""
    path = "outputs/signature_test/rule_documentation.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# =============================
# Sidebar Navigation
# =============================

st.sidebar.title("🛡️ Hybrid IDS")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview",
        "🔍 Detection Results",
        "📝 Signature Detection",
        "🔮 Anomaly Detection",
        "💡 SHAP Explainability",
        "⚠️ Risk & Severity",
        "🔧 Remediation",
        "📈 Model Performance",
        "ℹ️ System Information",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**CICIDS2017 Dataset**")
st.sidebar.markdown("Offline Analysis Mode")


# =============================
# Page: Overview
# =============================

if page == "📊 Overview":
    st.title("🛡️ Hybrid Network Intrusion Detection System")
    st.markdown("Combining **Signature-Based** and **Anomaly-Based** detection for comprehensive network security monitoring.")

    summary = load_hybrid_summary()
    df = load_hybrid_results()

    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Flows", f"{summary['total_flows']:,}")
        with col2:
            st.metric("Signature Detections", f"{summary['signature_detections']:,}")
        with col3:
            st.metric("Anomaly Detections", f"{summary['anomaly_detections']:,}")
        with col4:
            st.metric("Normal Flows", f"{summary['normal_flows']:,}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Detection Distribution")
            if df is not None and "Hybrid_Prediction" in df.columns:
                pred_counts = df["Hybrid_Prediction"].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db", "#9b59b6"]
                wedges, texts, autotexts = ax.pie(
                    pred_counts.values, labels=pred_counts.index,
                    autopct="%1.1f%%", colors=colors[:len(pred_counts)],
                    startangle=90, pctdistance=0.8
                )
                for t in texts:
                    t.set_fontsize(7)
                for t in autotexts:
                    t.set_fontsize(7)
                ax.set_title("Hybrid Detection Results", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        with col2:
            st.subheader("Ground Truth Distribution")
            if df is not None:
                label_counts = df["Label"].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ["#27ae60", "#c0392b", "#d35400", "#2980b9", "#8e44ad"]
                wedges, texts, autotexts = ax.pie(
                    label_counts.values, labels=label_counts.index,
                    autopct="%1.1f%%", colors=colors[:len(label_counts)],
                    startangle=90, pctdistance=0.8
                )
                for t in texts:
                    t.set_fontsize(7)
                for t in autotexts:
                    t.set_fontsize(7)
                ax.set_title("Actual Label Distribution", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.markdown("---")
        st.subheader("Architecture")
        st.markdown("""
        ```
        Network Flow → Signature Detection → Known Attack?
                                    ↓ YES                ↓ NO
                            Attack Classification    Isolation Forest
                            + Evidence               ↓
                            + Risk Score         Normal / Anomaly
                            + Remediation            ↓
                                                SHAP Explanation
                                                + Risk Score
                                                + Remediation
        ```
        """)
    else:
        st.warning("No hybrid detection results found. Run `hybrid_detection.py` first.")


# =============================
# Page: Detection Results
# =============================

elif page == "🔍 Detection Results":
    st.title("🔍 Detection Results")

    df = load_hybrid_results()
    detailed = load_detailed_results()

    if df is not None:
        # Determine prediction column
        pred_col = "Hybrid_Prediction" if "Hybrid_Prediction" in df.columns else "Signature_Prediction"

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            pred_filter = st.multiselect(
                "Filter by Prediction",
                df[pred_col].unique().tolist(),
                default=df[pred_col].unique().tolist(),
            )
        with col2:
            det_type_filter = st.multiselect(
                "Filter by Detection Type",
                ["Signature", "Anomaly", "Normal"],
                default=["Signature", "Anomaly"],
            )

        filtered = df[df[pred_col].isin(pred_filter)]

        # Filter by detection type
        type_mask = pd.Series(False, index=filtered.index)
        if "Signature" in det_type_filter:
            type_mask |= filtered["Signature_Matched"]
        if "Anomaly" in det_type_filter:
            type_mask |= (filtered.get("Is_Anomaly", False) == True)
        if "Normal" in det_type_filter:
            type_mask |= (~filtered["Signature_Matched"]) & (filtered.get("Is_Anomaly", False) != True)
        filtered = filtered[type_mask]

        st.metric("Filtered Results", f"{len(filtered):,}")

        # Show detection table
        display_cols = ["Label", pred_col, "Signature_Rule_ID", "Anomaly_Score"]
        available_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[available_cols].head(100), use_container_width=True)

        # Detailed inspection
        if detailed:
            st.markdown("---")
            st.subheader("Detailed Detection Inspection")

            det_idx = st.slider("Select detection", 0, min(len(detailed)-1, 99), 0)
            det = detailed[det_idx]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Detection Type:** {det.get('detection_type', 'N/A')}")
                st.markdown(f"**Prediction:** {det.get('prediction', 'N/A')}")
                st.markdown(f"**Risk Score:** {det.get('risk_score', 'N/A')}/100")
                st.markdown(f"**Severity:** {det.get('severity', 'N/A')}")
            with col2:
                if det.get("rule_id"):
                    st.markdown(f"**Rule ID:** {det['rule_id']}")
                if det.get("anomaly_score"):
                    st.markdown(f"**Anomaly Score:** {det['anomaly_score']:.4f}")
                if det.get("explanation"):
                    st.markdown(f"**Explanation:** {det['explanation']}")

            if det.get("evidence"):
                st.markdown("**Evidence:**")
                for e in det["evidence"]:
                    st.markdown(f"- {e}")

            if det.get("recommended_actions"):
                st.markdown("**Recommended Actions:**")
                for a in det["recommended_actions"]:
                    st.markdown(f"- {a}")
    else:
        st.warning("No detection results found.")


# =============================
# Page: Signature Detection
# =============================

elif page == "📝 Signature Detection":
    st.title("📝 Signature-Based Detection")

    rule_docs = load_rule_docs()
    sig_eval = load_signature_eval()

    if rule_docs:
        for rule_id, doc in rule_docs.items():
            with st.expander(f"Rule: {rule_id} — {doc['attack_category']}", expanded=True):
                st.markdown(f"**Description:** {doc['description']}")
                st.markdown("**Features & Thresholds:**")
                for feat, thresh in doc["thresholds"].items():
                    st.markdown(f"- `{feat}` {thresh}")
                st.markdown(f"**Justification:** {doc['justification']}")
                st.markdown(f"**Limitations:** {doc['limitations']}")

    if sig_eval:
        st.markdown("---")
        st.subheader("Signature Engine Performance (Test Set)")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{sig_eval['accuracy']:.4f}")
        with col2:
            st.metric("Macro F1", f"{sig_eval['macro_f1']:.4f}")
        with col3:
            st.metric("Weighted F1", f"{sig_eval['weighted_f1']:.4f}")

        # Confusion matrix heatmap using Matplotlib
        cm = np.array(sig_eval["confusion_matrix"])
        labels = sig_eval["confusion_matrix_labels"]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("Signature Detection Confusion Matrix")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        thresh = cm.max() / 2
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color=color, fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# =============================
# Page: Anomaly Detection
# =============================

elif page == "🔮 Anomaly Detection":
    st.title("🔮 Anomaly-Based Detection")

    anomaly_eval = load_anomaly_eval()
    df = load_hybrid_results()

    st.markdown("""
    **Isolation Forest** learns the profile of normal (benign) network traffic.
    Flows that deviate significantly from this profile are flagged as anomalous.
    """)

    if anomaly_eval:
        metrics = anomaly_eval["binary_metrics"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col2:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col3:
            st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
        with col4:
            st.metric("FPR", f"{metrics['fpr']:.4f}")

        st.markdown("---")
        st.subheader("Per-Class Detection Rates")
        per_class = anomaly_eval.get("per_class_detection", {})
        if per_class:
            class_data = []
            for label, data in per_class.items():
                class_data.append({
                    "Attack Class": label,
                    "Detected": data["detected"],
                    "Total": data["total"],
                    "Detection Rate": f"{data['rate']:.2%}",
                })
            st.table(pd.DataFrame(class_data))

        st.info(
            "**Note:** Isolation Forest detection rates for this CICIDS2017 subset are modest because "
            "the attack flows are similar to benign traffic at the flow-feature level. "
            "The signature engine handles the majority of detectable attacks; "
            "the anomaly detector provides a safety net for novel/unknown patterns."
        )

    if df is not None and "Anomaly_Score" in df.columns:
        st.markdown("---")
        st.subheader("Anomaly Score Distribution")

        score_data = df[df["Anomaly_Score"].notna()].copy()
        if len(score_data) > 0:
            sample = score_data.sample(min(10000, len(score_data)), random_state=42)
            fig, ax = plt.subplots(figsize=(10, 5))
            colors_map = {"Benign": "#2ecc71", "PortScan": "#e74c3c",
                          "Web Attack - Brute Force": "#e67e22",
                          "Web Attack - XSS": "#3498db",
                          "Web Attack - SQL Injection": "#9b59b6"}
            for label in sorted(sample["Label"].unique()):
                subset = sample[sample["Label"] == label]["Anomaly_Score"]
                ax.hist(subset, bins=50, alpha=0.6, label=label,
                        color=colors_map.get(label, "#95a5a6"))
            ax.set_xlabel("Anomaly Score")
            ax.set_ylabel("Frequency")
            ax.set_title("Anomaly Score Distribution by Label")
            ax.legend(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# =============================
# Page: SHAP Explainability
# =============================

elif page == "💡 SHAP Explainability":
    st.title("💡 SHAP Explainability")

    st.markdown("""
    **SHAP** (SHapley Additive exPlanations) explains *why* the Isolation Forest
    flagged a specific flow as anomalous by showing which features contributed most
    to the anomaly decision.

    > SHAP shows model contribution, not proof of causation.
    """)

    shap_summary = load_shap_summary()
    detailed = load_detailed_results()

    if shap_summary:
        st.subheader("Most Influential Features (Across All Anomalies)")
        top_feats = shap_summary.get("top_contributing_features", [])
        if top_feats:
            features = [f["feature"][:25] for f in top_feats[:10]]
            values = [f["mean_abs_shap"] for f in top_feats[:10]]

            fig, ax = plt.subplots(figsize=(10, 5))
            y_pos = range(len(features))
            ax.barh(y_pos, values, color="#8e44ad", alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title("Top Features by Mean |SHAP Value|")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Show individual explanations
    anomaly_dets = [d for d in detailed if d.get("detection_type") == "Anomaly"]
    if anomaly_dets:
        st.markdown("---")
        st.subheader("Individual Anomaly Explanations")
        idx = st.slider("Select anomaly", 0, min(len(anomaly_dets)-1, 49), 0)
        det = anomaly_dets[idx]

        st.markdown(f"**Anomaly Score:** {det.get('anomaly_score', 'N/A'):.4f}")
        st.markdown(f"**Severity:** {det.get('severity', 'N/A')}")

        if det.get("explanation"):
            st.info(det["explanation"])

        if det.get("shap_features"):
            st.markdown("**Top Contributing Features:**")
            for feat in det["shap_features"][:5]:
                direction = "🔴" if feat["direction"] == "increases anomaly" else "🟢"
                st.markdown(
                    f"{direction} **{feat['feature']}**: "
                    f"SHAP={feat['shap_value']:.4f}, Value={feat['feature_value']:.4f}"
                )
    elif not shap_summary:
        st.warning("No SHAP explanations available. Run hybrid detection first.")


# =============================
# Page: Risk & Severity
# =============================

elif page == "⚠️ Risk & Severity":
    st.title("⚠️ Risk & Severity Assessment")

    detailed = load_detailed_results()

    if detailed:
        # Severity distribution
        severities = [d.get("severity", "INFO") for d in detailed]
        sev_counts = pd.Series(severities).value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Severity Distribution")
            sev_colors = {"CRITICAL": "#ff4444", "HIGH": "#ff8800", "MEDIUM": "#ffcc00",
                          "LOW": "#44bb44", "INFO": "#888888"}
            fig, ax = plt.subplots(figsize=(6, 4))
            bar_colors = [sev_colors.get(s, "#888888") for s in sev_counts.index]
            ax.bar(sev_counts.index, sev_counts.values, color=bar_colors)
            ax.set_ylabel("Count")
            ax.set_title("Detection Severity Distribution")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Risk Score Distribution")
            risk_scores = [d.get("risk_score", 0) for d in detailed]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(risk_scores, bins=20, color="#3498db", alpha=0.8, edgecolor="white")
            ax.set_xlabel("Risk Score")
            ax.set_ylabel("Count")
            ax.set_title("Risk Score Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.subheader("Risk Methodology")
        st.markdown("""
        | Component | Weight | Description |
        |---|---|---|
        | Detection Source | High | Signature = higher confidence than anomaly |
        | Attack Category | Medium | SQL Injection > XSS > Brute Force > PortScan |
        | Anomaly Score | Medium | Lower score = more anomalous = higher risk |
        | Feature Evidence | Low | Number of unusual features detected |
        """)
    else:
        st.warning("No risk assessment data available.")


# =============================
# Page: Remediation
# =============================

elif page == "🔧 Remediation":
    st.title("🔧 Remediation Guidance")

    detailed = load_detailed_results()

    if detailed:
        # Filter to detections with remediation
        detections_with_remed = [d for d in detailed if d.get("recommended_actions")]

        if detections_with_remed:
            st.metric("Detections with Remediation", len(detections_with_remed))

            idx = st.slider("Select detection", 0, min(len(detections_with_remed)-1, 49), 0)
            det = detections_with_remed[idx]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Detection Type:** {det.get('detection_type')}")
                st.markdown(f"**Prediction:** {det.get('prediction')}")
                st.markdown(f"**Severity:** {det.get('severity')}")
            with col2:
                st.markdown(f"**Risk Score:** {det.get('risk_score')}/100")
                st.markdown(f"**Remediation:** {det.get('remediation', 'N/A')}")

            st.markdown("---")
            st.subheader("Recommended Actions")
            for action in det.get("recommended_actions", []):
                st.markdown(f"✅ {action}")

            st.markdown("---")
            st.subheader("Response Mode")
            st.info(
                "⚡ **Current Mode: RECOMMENDATION** — All responses are advisory only. "
                "No automated actions are taken. This system operates in offline analysis mode "
                "using historical CICIDS2017 data."
            )
    else:
        st.warning("No remediation data available.")


# =============================
# Page: Model Performance
# =============================

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    sig_eval = load_signature_eval()
    anomaly_eval = load_anomaly_eval()

    tab1, tab2 = st.tabs(["Signature Engine", "Anomaly Detector"])

    with tab1:
        if sig_eval:
            st.subheader("Signature Engine — Test Set Metrics")

            report = sig_eval.get("classification_report", {})
            metrics_data = []
            for label, metrics in report.items():
                if isinstance(metrics, dict) and "precision" in metrics and label not in ("macro avg", "weighted avg", "accuracy"):
                    metrics_data.append({
                        "Class": label,
                        "Precision": f"{metrics['precision']:.4f}",
                        "Recall": f"{metrics['recall']:.4f}",
                        "F1-Score": f"{metrics['f1-score']:.4f}",
                        "Support": metrics.get("support", "N/A"),
                    })
            if metrics_data:
                st.table(pd.DataFrame(metrics_data))

            # Confusion matrix using Matplotlib
            cm = np.array(sig_eval["confusion_matrix"])
            labels = sig_eval["confusion_matrix_labels"]
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title("Confusion Matrix")
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            thresh = cm.max() / 2
            for i in range(len(labels)):
                for j in range(len(labels)):
                    color = "white" if cm[i, j] > thresh else "black"
                    ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                            color=color, fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("No signature evaluation data.")

    with tab2:
        if anomaly_eval:
            st.subheader("Anomaly Detector — Test Set Metrics")
            metrics = anomaly_eval["binary_metrics"]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Binary Classification (Normal vs Anomaly)**")
                cm_data = {
                    "": ["Predicted Normal", "Predicted Anomaly"],
                    "Actual Normal": [metrics["tn"], metrics["fp"]],
                    "Actual Anomaly": [metrics["fn"], metrics["tp"]],
                }
                st.table(pd.DataFrame(cm_data).set_index(""))

            with col2:
                st.markdown("**Key Metrics**")
                st.markdown(f"- Precision: {metrics['precision']:.4f}")
                st.markdown(f"- Recall: {metrics['recall']:.4f}")
                st.markdown(f"- F1 Score: {metrics['f1_score']:.4f}")
                st.markdown(f"- FPR: {metrics['fpr']:.4f}")
                st.markdown(f"- FNR: {metrics['fnr']:.4f}")

            per_class = anomaly_eval.get("per_class_detection", {})
            if per_class:
                st.markdown("---")
                st.subheader("Per-Attack Detection Rates")
                for label, data in per_class.items():
                    st.markdown(f"- **{label}**: {data['detected']}/{data['total']} ({data['rate']:.2%})")
        else:
            st.warning("No anomaly evaluation data.")


# =============================
# Page: System Information
# =============================

elif page == "ℹ️ System Information":
    st.title("ℹ️ System Information")

    st.subheader("Project")
    st.markdown("""
    - **Project**: Hybrid Network Intrusion Detection System (HIDS)
    - **Dataset**: CICIDS2017 (Benign, PortScan, Web Attacks)
    - **Mode**: Offline Analysis
    - **Author**: Choppa Bharath Kumar
    """)

    st.subheader("Architecture")
    st.markdown("""
    1. **Stage 1 — Signature Detection**: Rule-based matching against known attack patterns
    2. **Stage 2 — Anomaly Detection**: Isolation Forest trained on benign-only traffic
    3. **Stage 3 — SHAP Explainability**: Feature contribution analysis for anomalies
    4. **Stage 4 — Risk Assessment**: Severity scoring (0-100) with evidence
    5. **Stage 5 — Remediation**: Attack-specific response recommendations
    """)

    st.subheader("Model Configuration")
    config_path = "models/anomaly_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        st.json(config)

    preproc_path = "models/preprocessing_config.json"
    if os.path.exists(preproc_path):
        with open(preproc_path, "r") as f:
            preproc = json.load(f)
        st.subheader("Preprocessing Configuration")
        st.markdown(f"- Features: {preproc['n_features']}")
        st.markdown(f"- Label classes: {preproc['label_classes']}")
        st.markdown(f"- Train/Val/Test split: {preproc['split_info']}")

    st.subheader("Limitations")
    st.markdown("""
    - Operates on historical CICIDS2017 data, not live network traffic
    - Isolation Forest detection rates are modest due to attack/benign feature overlap
    - SQL Injection has only 21 samples — metrics are unreliable for this class
    - Signature rules are specific to CICIDS2017 flow characteristics
    - No HTTP payload inspection (flow-level features only)
    - Automated responses are simulated, not connected to live infrastructure
    """)
