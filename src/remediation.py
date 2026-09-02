import json
import os
from datetime import datetime


# =============================
# Remediation Knowledge Base
# =============================

REMEDIATION_ACTIONS = {
    "PortScan": {
        "description": "Port scanning detected — attacker probing for open services.",
        "recommended_actions": [
            "Investigate source IP for additional suspicious activity",
            "Review firewall logs for connection patterns from this source",
            "Check if the scanned ports correspond to critical services",
            "Consider rate-limiting or blocking the source IP",
            "Verify that unnecessary ports are closed",
            "Review IDS/IPS alerts for correlated activity",
        ],
        "automated_responses": [
            {"action": "block_source_ip", "description": "Block source IP at firewall", "safe": False},
            {"action": "rate_limit", "description": "Rate-limit connections from source", "safe": True},
            {"action": "alert_soc", "description": "Send alert to SOC team", "safe": True},
            {"action": "log_event", "description": "Log detection event", "safe": True},
        ],
    },
    "Web Attack - Brute Force": {
        "description": "Brute force attack detected — repeated authentication attempts.",
        "recommended_actions": [
            "Review authentication logs for failed login attempts",
            "Check if the target account has been compromised",
            "Enable or verify account lockout policies",
            "Investigate source IP reputation",
            "Review web application firewall (WAF) logs",
            "Consider implementing CAPTCHA or multi-factor authentication",
        ],
        "automated_responses": [
            {"action": "block_source_ip", "description": "Block source IP at WAF/firewall", "safe": False},
            {"action": "lock_account", "description": "Temporarily lock targeted account", "safe": False},
            {"action": "alert_soc", "description": "Send alert to SOC team", "safe": True},
            {"action": "log_event", "description": "Log detection event", "safe": True},
        ],
    },
    "Web Attack - XSS": {
        "description": "Cross-Site Scripting (XSS) attack detected — malicious script injection attempt.",
        "recommended_actions": [
            "Review web application logs for injection payloads",
            "Inspect the targeted endpoint for input validation vulnerabilities",
            "Check if the attack was reflected or stored",
            "Review Content Security Policy (CSP) headers",
            "Investigate source IP for additional attack patterns",
            "Update WAF rules to block the specific attack signature",
        ],
        "automated_responses": [
            {"action": "block_source_ip", "description": "Block source IP at WAF", "safe": False},
            {"action": "alert_soc", "description": "Send alert to SOC team", "safe": True},
            {"action": "log_event", "description": "Log detection event", "safe": True},
        ],
    },
    "Web Attack - SQL Injection": {
        "description": "SQL Injection attack detected — database manipulation attempt.",
        "recommended_actions": [
            "IMMEDIATELY review database access logs",
            "Check for data exfiltration indicators",
            "Inspect the targeted endpoint for parameterized query usage",
            "Review database user permissions (principle of least privilege)",
            "Check if the attack succeeded by reviewing query logs",
            "Investigate source IP and block if confirmed malicious",
            "Consider isolating the affected application server",
        ],
        "automated_responses": [
            {"action": "block_source_ip", "description": "Block source IP at WAF/firewall", "safe": False},
            {"action": "isolate_server", "description": "Isolate affected application server", "safe": False},
            {"action": "alert_soc", "description": "Send CRITICAL alert to SOC team", "safe": True},
            {"action": "log_event", "description": "Log detection event", "safe": True},
        ],
    },
    "Potential Unknown Attack": {
        "description": "Anomalous network behavior detected — potentially unknown or novel attack.",
        "recommended_actions": [
            "Investigate the flow's source and destination for context",
            "Review related flows in the same time window",
            "Check if the behavior correlates with known vulnerability exploits",
            "Monitor the source IP for continued anomalous activity",
            "Review the SHAP explanation for feature-level insights",
            "Escalate to security analyst if pattern persists",
        ],
        "automated_responses": [
            {"action": "alert_soc", "description": "Send alert to SOC team for investigation", "safe": True},
            {"action": "log_event", "description": "Log anomaly event with SHAP explanation", "safe": True},
            {"action": "monitor_source", "description": "Add source to enhanced monitoring list", "safe": True},
        ],
    },
}


# =============================
# 1. Generate Remediation
# =============================

def generate_remediation(detection_result, risk_report):
    """Generate remediation guidance for a detection.

    Parameters
    ----------
    detection_result : dict with detection details
    risk_report : dict from risk_engine.generate_risk_report
    """
    prediction = detection_result.get("prediction", "Benign")
    det_type = detection_result.get("detection_type", "Normal")

    if prediction == "Benign" or det_type == "Normal":
        return {
            "action_required": False,
            "description": "No action required — traffic classified as normal.",
            "recommended_actions": [],
            "automated_responses": [],
        }

    # Determine remediation category
    if det_type == "Anomaly":
        category = "Potential Unknown Attack"
    else:
        category = prediction

    knowledge = REMEDIATION_ACTIONS.get(category, REMEDIATION_ACTIONS["Potential Unknown Attack"])

    # Filter automated responses based on severity
    severity = risk_report.get("severity", "LOW")
    available_responses = []
    for resp in knowledge["automated_responses"]:
        if resp["safe"]:
            available_responses.append(resp)
        elif severity in ("CRITICAL", "HIGH"):
            resp_copy = resp.copy()
            resp_copy["requires_confirmation"] = True
            available_responses.append(resp_copy)

    remediation = {
        "action_required": True,
        "description": knowledge["description"],
        "severity": severity,
        "risk_score": risk_report.get("risk_score", 0),
        "recommended_actions": knowledge["recommended_actions"],
        "automated_responses": available_responses,
        "response_mode": "RECOMMENDATION",
    }

    return remediation


# =============================
# 2. Safe Response Framework
# =============================

def simulate_response(action, detection_result, dry_run=True):
    """Simulate or execute an automated response action.

    For the CICIDS2017 offline project, all actions are simulated.
    In production, this would interface with firewalls, SIEM, etc.
    """
    response_log = {
        "timestamp": datetime.now().isoformat(),
        "action": action["action"],
        "description": action["description"],
        "safe": action.get("safe", True),
        "dry_run": dry_run,
        "detection": {
            "type": detection_result.get("detection_type"),
            "prediction": detection_result.get("prediction"),
        },
    }

    if dry_run:
        response_log["status"] = "SIMULATED"
        response_log["message"] = f"[DRY RUN] Would execute: {action['description']}"
    else:
        # In production: interface with security infrastructure
        response_log["status"] = "NOT_CONNECTED"
        response_log["message"] = "No live security infrastructure connected (offline mode)"

    return response_log


# =============================
# 3. Response Audit Log
# =============================

def save_response_log(responses, output_dir="outputs"):
    """Save response actions to audit log."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "response_audit_log.json")

    # Append to existing log
    existing = []
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)

    existing.extend(responses)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

    return path
