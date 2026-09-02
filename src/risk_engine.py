import numpy as np


# =============================
# Risk Score Methodology
# =============================
#
# Risk Score (0-100) is calculated from:
#   1. Detection source weight (signature = higher confidence)
#   2. Attack category severity
#   3. Anomaly score magnitude (for anomaly-based detections)
#   4. Feature evidence strength
#
# Severity levels:
#   CRITICAL (80-100): Known high-impact attacks, strong anomaly evidence
#   HIGH     (60-79):  Known attacks, moderate anomaly evidence
#   MEDIUM   (40-59):  Potential threats, weak anomaly evidence
#   LOW      (20-39):  Minor anomalies, likely benign unusual behavior
#   INFO     (0-19):   Normal traffic, no action needed

SEVERITY_LEVELS = {
    "CRITICAL": (80, 100),
    "HIGH": (60, 79),
    "MEDIUM": (40, 59),
    "LOW": (20, 39),
    "INFO": (0, 19),
}

# Base severity by attack category (signature detections)
ATTACK_SEVERITY = {
    "PortScan": 65,
    "Web Attack - Brute Force": 75,
    "Web Attack - XSS": 80,
    "Web Attack - SQL Injection": 90,
    "Benign": 0,
}


# =============================
# 1. Calculate Risk Score
# =============================

def calculate_risk_score(detection_result):
    """Calculate risk score (0-100) for a detection result.

    Parameters
    ----------
    detection_result : dict with keys:
        - detection_type: 'Signature' or 'Anomaly' or 'Normal'
        - prediction: attack label or 'Benign'
        - anomaly_score: float (for anomaly detections)
        - rule_id: str (for signature detections)
        - shap_features: list of top SHAP features (optional)
    """
    det_type = detection_result.get("detection_type", "Normal")
    prediction = detection_result.get("prediction", "Benign")

    if det_type == "Normal" or prediction == "Benign":
        return 0, "INFO", []

    evidence = []

    if det_type == "Signature":
        # Signature-based: higher confidence
        base_score = ATTACK_SEVERITY.get(prediction, 50)
        rule_id = detection_result.get("rule_id", "UNKNOWN")
        evidence.append(f"Matched signature rule: {rule_id}")
        evidence.append(f"Attack category: {prediction}")

        # Adjust based on rule specificity
        if rule_id == "WEBATTACK_001":
            evidence.append("TCP push-without-acknowledge pattern detected")
            evidence.append("Attack tool TCP window signature matched")
        elif rule_id == "PORTSCAN_001":
            evidence.append("Small packet probing behavior detected")
            evidence.append("Low data transmission with PSH flag activity")

        risk_score = min(base_score + 10, 100)  # Boost for confirmed signature

    elif det_type == "Anomaly":
        # Anomaly-based: score depends on anomaly magnitude
        anomaly_score = detection_result.get("anomaly_score", 0)
        shap_features = detection_result.get("shap_features", [])

        # Normalize anomaly score to 0-100 range
        # Lower anomaly_score = more anomalous (Isolation Forest convention)
        # Typical range: -0.15 to 0.3, threshold ~0.01-0.02
        normalized = max(0, min(1, (0.05 - anomaly_score) / 0.2))
        base_score = int(40 + normalized * 45)  # Range: 40-85

        evidence.append(f"Anomaly score: {anomaly_score:.4f}")
        evidence.append("Behavior deviates from learned normal traffic profile")

        # Add SHAP feature evidence
        for feat in shap_features[:3]:
            evidence.append(f"Unusual {feat.get('feature', 'unknown')}")

        risk_score = base_score

    else:
        risk_score = 0

    # Determine severity level
    severity = get_severity_level(risk_score)

    return risk_score, severity, evidence


# =============================
# 2. Get Severity Level
# =============================

def get_severity_level(risk_score):
    """Map risk score to severity level."""
    for level, (low, high) in SEVERITY_LEVELS.items():
        if low <= risk_score <= high:
            return level
    return "INFO"


# =============================
# 3. Risk Assessment Report
# =============================

def generate_risk_report(detection_result):
    """Generate a complete risk assessment for a detection."""
    risk_score, severity, evidence = calculate_risk_score(detection_result)

    report = {
        "detection_type": detection_result.get("detection_type", "Normal"),
        "prediction": detection_result.get("prediction", "Benign"),
        "risk_score": risk_score,
        "severity": severity,
        "evidence": evidence,
        "requires_investigation": severity in ("CRITICAL", "HIGH"),
        "requires_immediate_action": severity == "CRITICAL",
    }

    return report


# =============================
# 4. Batch Risk Assessment
# =============================

def batch_risk_assessment(detection_results):
    """Calculate risk scores for a batch of detections."""
    reports = []
    for result in detection_results:
        report = generate_risk_report(result)
        reports.append(report)
    return reports
