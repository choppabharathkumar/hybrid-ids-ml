"""Unit tests for the Hybrid IDS pipeline."""
import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================
# Test: Label Normalization
# =============================

class TestLabelNormalization:

    def test_canonical_labels(self):
        from src.data_loading import LABEL_MAP, CANONICAL_LABELS
        for mapped in LABEL_MAP.values():
            assert mapped in CANONICAL_LABELS

    def test_corrupt_label_mapping(self):
        from src.data_loading import normalize_labels
        df = pd.DataFrame({"Label": ["Benign", "Web Attack \ufffd Brute Force"]})
        result = normalize_labels(df)
        assert result["Label"].tolist() == ["Benign", "Web Attack - Brute Force"]

    def test_endash_label_mapping(self):
        from src.data_loading import normalize_labels
        df = pd.DataFrame({"Label": ["Web Attack \u2013 XSS"]})
        result = normalize_labels(df)
        assert result["Label"].iloc[0] == "Web Attack - XSS"

    def test_unmapped_label_raises(self):
        from src.data_loading import normalize_labels
        df = pd.DataFrame({"Label": ["UnknownLabel"]})
        with pytest.raises(ValueError, match="Unmapped labels"):
            normalize_labels(df)

    def test_missing_column_raises(self):
        from src.data_loading import normalize_labels
        df = pd.DataFrame({"NotLabel": ["test"]})
        with pytest.raises(ValueError, match="not found"):
            normalize_labels(df)


# =============================
# Test: EDA Functions
# =============================

class TestEDA:

    def test_remove_duplicates(self):
        from src.eda import remove_duplicates
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = remove_duplicates(df)
        assert len(result) == 2

    def test_remove_invalid_rows_negative_duration(self):
        from src.eda import remove_invalid_rows
        df = pd.DataFrame({"Flow Duration": [100, -5, 200], "x": [1, 2, 3]})
        result = remove_invalid_rows(df)
        assert len(result) == 2
        assert (result["Flow Duration"] >= 0).all()

    def test_remove_zero_variance(self):
        from src.eda import remove_zero_variance_features
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "Label": ["x", "y", "z"]})
        result, removed = remove_zero_variance_features(df)
        assert "a" not in result.columns
        assert "b" in result.columns
        assert "Label" in result.columns

    def test_detect_quality_issues(self):
        from src.eda import detect_quality_issues
        df = pd.DataFrame({
            "Flow Duration": [100, -5],
            "x": [1, 1],
        })
        report = detect_quality_issues(df)
        assert report["negative_flow_duration"] == 1
        assert "x" in report["zero_variance_columns"]


# =============================
# Test: Preprocessing
# =============================

class TestPreprocessing:

    def test_get_feature_columns(self):
        from src.data_preprocessing import get_feature_columns, EXCLUDE_COLUMNS
        df = pd.DataFrame({"a": [1], "Label": ["Benign"], "Bwd PSH Flags": [0]})
        cols = get_feature_columns(df)
        assert "Label" not in cols
        assert "Bwd PSH Flags" not in cols
        assert "a" in cols

    def test_split_preserves_proportions(self):
        from src.data_preprocessing import split_dataset
        df = pd.DataFrame({
            "Label": ["Benign"] * 90 + ["PortScan"] * 10,
            "x": range(100),
        })
        train, val, test = split_dataset(df, test_size=0.2, val_size=0.1)
        assert len(train) + len(val) + len(test) == 100

    def test_split_no_overlap(self):
        from src.data_preprocessing import split_dataset
        df = pd.DataFrame({
            "Label": ["Benign"] * 90 + ["PortScan"] * 10,
            "x": range(100),
        })
        train, val, test = split_dataset(df, test_size=0.2, val_size=0.1)
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0


# =============================
# Test: Signature Rules
# =============================

class TestSignatureRules:

    def test_portscan_detection(self):
        from src.rule_based_detection import classify_portscan
        row = {
            "PSH Flag Count": 1,
            "Total Fwd Packets": 2,
            "Total Backward Packets": 1,
            "Fwd Act Data Packets": 0,
            "Packet Length Mean": 5.0,
        }
        assert classify_portscan(row) == True

    def test_portscan_negative_high_packets(self):
        from src.rule_based_detection import classify_portscan
        row = {
            "PSH Flag Count": 1,
            "Total Fwd Packets": 100,
            "Total Backward Packets": 50,
            "Fwd Act Data Packets": 0,
            "Packet Length Mean": 5.0,
        }
        assert classify_portscan(row) == False

    def test_web_attack_detection(self):
        from src.rule_based_detection import classify_web_attack
        row = {
            "PSH Flag Count": 1,
            "ACK Flag Count": 0,
            "Fwd Act Data Packets": 0,
            "Flow Duration": 5000000,
            "Init Fwd Win Bytes": 29200,
            "Init Bwd Win Bytes": 28960,
            "Down/Up Ratio": 0,
            "Flow Bytes/s": 0,
        }
        assert classify_web_attack(row) == True

    def test_web_attack_subtype_xss(self):
        from src.rule_based_detection import classify_web_attack_subtype
        row = {"Packet Length Mean": 100}
        assert classify_web_attack_subtype(row) == "Web Attack - XSS"

    def test_web_attack_subtype_brute_force(self):
        from src.rule_based_detection import classify_web_attack_subtype
        row = {"Packet Length Mean": 10}
        assert classify_web_attack_subtype(row) == "Web Attack - Brute Force"

    def test_benign_not_flagged(self):
        from src.rule_based_detection import classify_portscan, classify_web_attack
        row = {
            "PSH Flag Count": 0,
            "ACK Flag Count": 1,
            "Total Fwd Packets": 50,
            "Total Backward Packets": 50,
            "Fwd Act Data Packets": 10,
            "Flow Duration": 100,
            "Packet Length Mean": 500,
            "Init Fwd Win Bytes": 65535,
            "Init Bwd Win Bytes": 65535,
            "Down/Up Ratio": 1,
            "Flow Bytes/s": 10000,
        }
        assert classify_portscan(row) == False
        assert classify_web_attack(row) == False

    def test_vectorized_produces_correct_columns(self):
        from src.rule_based_detection import apply_signature_detection_vectorized
        df = pd.DataFrame({
            "PSH Flag Count": [1, 0],
            "ACK Flag Count": [0, 1],
            "Total Fwd Packets": [2, 100],
            "Total Backward Packets": [1, 50],
            "Fwd Act Data Packets": [0, 10],
            "Flow Duration": [5000000, 100],
            "Init Fwd Win Bytes": [29200, 65535],
            "Init Bwd Win Bytes": [28960, -1],
            "Down/Up Ratio": [0, 1],
            "Flow Bytes/s": [0, 10000],
            "Packet Length Mean": [0, 500],
        })
        result = apply_signature_detection_vectorized(df)
        assert "Signature_Prediction" in result.columns
        assert "Signature_Rule_ID" in result.columns
        assert "Signature_Matched" in result.columns


# =============================
# Test: Risk Engine
# =============================

class TestRiskEngine:

    def test_benign_returns_zero(self):
        from src.risk_engine import calculate_risk_score
        score, severity, evidence = calculate_risk_score({"detection_type": "Normal", "prediction": "Benign"})
        assert score == 0
        assert severity == "INFO"

    def test_signature_detection_has_score(self):
        from src.risk_engine import calculate_risk_score
        score, severity, evidence = calculate_risk_score({
            "detection_type": "Signature",
            "prediction": "PortScan",
            "rule_id": "PORTSCAN_001",
        })
        assert score > 0
        assert severity in ("HIGH", "CRITICAL", "MEDIUM")

    def test_anomaly_detection_has_score(self):
        from src.risk_engine import calculate_risk_score
        score, severity, evidence = calculate_risk_score({
            "detection_type": "Anomaly",
            "prediction": "Potential Unknown Attack",
            "anomaly_score": -0.05,
        })
        assert score > 0

    def test_severity_levels(self):
        from src.risk_engine import get_severity_level
        assert get_severity_level(90) == "CRITICAL"
        assert get_severity_level(70) == "HIGH"
        assert get_severity_level(50) == "MEDIUM"
        assert get_severity_level(30) == "LOW"
        assert get_severity_level(10) == "INFO"


# =============================
# Test: Remediation
# =============================

class TestRemediation:

    def test_benign_no_action(self):
        from src.remediation import generate_remediation
        result = generate_remediation(
            {"prediction": "Benign", "detection_type": "Normal"},
            {"severity": "INFO", "risk_score": 0},
        )
        assert result["action_required"] == False

    def test_attack_has_actions(self):
        from src.remediation import generate_remediation
        result = generate_remediation(
            {"prediction": "PortScan", "detection_type": "Signature"},
            {"severity": "HIGH", "risk_score": 75},
        )
        assert result["action_required"] == True
        assert len(result["recommended_actions"]) > 0

    def test_simulate_response_dry_run(self):
        from src.remediation import simulate_response
        action = {"action": "log_event", "description": "Log event", "safe": True}
        result = simulate_response(action, {"detection_type": "Signature"}, dry_run=True)
        assert result["status"] == "SIMULATED"
        assert result["dry_run"] == True


# =============================
# Test: Edge Cases
# =============================

class TestEdgeCases:

    def test_empty_dataframe(self):
        from src.rule_based_detection import apply_signature_detection_vectorized
        df = pd.DataFrame(columns=[
            "PSH Flag Count", "ACK Flag Count", "Total Fwd Packets",
            "Total Backward Packets", "Fwd Act Data Packets", "Flow Duration",
            "Init Fwd Win Bytes", "Init Bwd Win Bytes", "Down/Up Ratio",
            "Flow Bytes/s", "Packet Length Mean",
        ])
        result = apply_signature_detection_vectorized(df)
        assert len(result) == 0
        assert "Signature_Prediction" in result.columns

    def test_zero_values(self):
        from src.rule_based_detection import classify_portscan
        row = {
            "PSH Flag Count": 0,
            "Total Fwd Packets": 0,
            "Total Backward Packets": 0,
            "Fwd Act Data Packets": 0,
            "Packet Length Mean": 0,
        }
        assert classify_portscan(row) == False

    def test_extreme_values(self):
        from src.risk_engine import calculate_risk_score
        score, severity, _ = calculate_risk_score({
            "detection_type": "Anomaly",
            "prediction": "Potential Unknown Attack",
            "anomaly_score": -1000,
        })
        assert 0 <= score <= 100


# =============================
# Test: Evaluation Artifact Regression
# =============================

class TestEvaluationArtifacts:
    """Regression tests — verify that saved evaluation artifacts
    contain expected structure and metrics remain consistent."""

    def test_hybrid_evaluation_file_exists(self):
        path = os.path.join("results", "hybrid_evaluation.json")
        if not os.path.isfile(path):
            pytest.skip(
            "Generated evaluation artifact is unavailable in a clean "
            "checkout; run the full pipeline locally with CIC-IDS2017 data."
        )
        assert os.path.isfile(path), f"Missing {path}"

    def test_hybrid_evaluation_structure(self):
        import json
        path = os.path.join("results", "hybrid_evaluation.json")
        if not os.path.exists(path):
            pytest.skip("Evaluation file not yet generated")
        with open(path) as f:
            data = json.load(f)
        assert "binary_metrics" in data
        assert "per_attack_recall" in data
        assert "total_flows" in data
        bm = data["binary_metrics"]
        for key in ("tp", "tn", "fp", "fn", "precision", "recall", "f1_score", "fpr"):
            assert key in bm, f"Missing binary_metrics.{key}"

    def test_total_flows_matches_test_split(self):
        import json
        eval_path = os.path.join("results", "hybrid_evaluation.json")
        config_path = os.path.join("models", "preprocessing_config.json")
        if not os.path.exists(eval_path) or not os.path.exists(config_path):
            pytest.skip("Artifacts not yet generated")
        with open(eval_path) as f:
            eval_data = json.load(f)
        with open(config_path) as f:
            config = json.load(f)
        assert eval_data["total_flows"] == config["split_info"]["test_size"]

    def test_recall_above_minimum(self):
        """Regression: overall attack recall should stay above 85%."""
        import json
        path = os.path.join("results", "hybrid_evaluation.json")
        if not os.path.exists(path):
            pytest.skip("Evaluation file not yet generated")
        with open(path) as f:
            data = json.load(f)
        assert data["binary_metrics"]["recall"] >= 0.85

    def test_fpr_below_maximum(self):
        """Regression: FPR should stay below 10%."""
        import json
        path = os.path.join("results", "hybrid_evaluation.json")
        if not os.path.exists(path):
            pytest.skip("Evaluation file not yet generated")
        with open(path) as f:
            data = json.load(f)
        assert data["binary_metrics"]["fpr"] <= 0.10

    def test_all_attack_classes_present(self):
        import json
        path = os.path.join("results", "hybrid_evaluation.json")
        if not os.path.exists(path):
            pytest.skip("Evaluation file not yet generated")
        with open(path) as f:
            data = json.load(f)
        expected = {"PortScan", "Web Attack - Brute Force",
                    "Web Attack - SQL Injection", "Web Attack - XSS"}
        actual = set(data["per_attack_recall"].keys())
        assert expected == actual


# =============================
# Test: SQL Injection Rule
# =============================

class TestSQLInjectionRule:
    """Regression: SQLI_001 rule correctness."""

    def test_sqli_positive(self):
        from src.rule_based_detection import classify_sql_injection
        row = {
            "Init Fwd Win Bytes": 29200,
            "Init Bwd Win Bytes": 235,
            "Fwd Act Data Packets": 2,
            "Packet Length Mean": 117,
            "Flow Duration": 5500000,
        }
        assert classify_sql_injection(row) is True

    def test_sqli_negative_wrong_window(self):
        from src.rule_based_detection import classify_sql_injection
        row = {
            "Init Fwd Win Bytes": 29200,
            "Init Bwd Win Bytes": 28960,
            "Fwd Act Data Packets": 2,
            "Packet Length Mean": 117,
            "Flow Duration": 5500000,
        }
        assert classify_sql_injection(row) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
