# 🛡️ Hybrid Network Intrusion Detection System (HIDS)

A **Hybrid Intrusion Detection System** that combines **Rule-Based Signature Detection** with **Isolation Forest Anomaly Detection** to detect both known and unknown cyber threats. The project integrates **SHAP** for explainable AI, a **Risk Assessment Engine**, **Remediation Guidance**, and a **Streamlit Dashboard** for interactive visualization.

---

## 🚀 Tech Stack

| Category | Technologies |
|---|---|
| **Core** | Python, Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Isolation Forest) |
| **Explainability** | SHAP (TreeExplainer) |
| **Dashboard** | Streamlit |
| **Visualization** | Matplotlib |
| **Data Format** | Parquet (PyArrow) |

---

## 📂 Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Dataset

| Class | Description |
|---|---|
| Benign | Normal network traffic |
| PortScan | Port scanning reconnaissance |
| Web Attack - Brute Force | Brute force authentication attacks |
| Web Attack - XSS | Cross-Site Scripting attacks |
| Web Attack - SQL Injection | SQL Injection attacks |

---

## 🏗️ Architecture

```
Network Flow Data (CICIDS2017)
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: Signature Engine  │  Rule-based pattern matching
│  (3 rules: PORTSCAN_001,   │  against known attack signatures
│   WEBATTACK_001, SQLI_001)  │
└──────────┬──────────────────┘
           │
     ┌─────┴─────┐
     │ Matched?  │
     └─────┬─────┘
       YES │         NO
           │          │
     ┌─────┴──┐  ┌───┴──────────────────────┐
     │ Attack │  │ Stage 2: Isolation Forest │
     │ Label  │  │ (trained on benign-only)  │
     └────┬───┘  └───────────┬──────────────┘
          │            ┌─────┴─────┐
          │            │ Anomaly?  │
          │            └─────┬─────┘
          │              YES │      NO
          │                  │       │
          │     ┌────────────┴┐  ┌──┴──────┐
          │     │ Stage 3:    │  │ Benign  │
          │     │ SHAP Explain│  │ (Normal)│
          │     └──────┬──────┘  └─────────┘
          │            │
          ▼            ▼
┌─────────────────────────────┐
│  Stage 4: Risk Assessment   │  Score (0-100) + Severity
│  Stage 5: Remediation       │  Attack-specific guidance
└─────────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  Streamlit Dashboard        │  Interactive visualization
└─────────────────────────────┘
```

---

## 📁 Project Structure

```
HIDS/
├── main.py                     # Pipeline orchestrator (entry point)
├── requirements.txt            # Python dependencies
├── README.md
│
├── src/                        # Source modules
│   ├── __init__.py
│   ├── data_loading.py         # Dataset loading & label normalization
│   ├── eda.py                  # Exploratory Data Analysis & cleaning
│   ├── data_preprocessing.py   # Feature scaling, encoding, splitting
│   ├── data_exploration.py     # Feature distribution & correlation analysis
│   ├── rule_based_detection.py # Signature detection engine (3 rules)
│   ├── anomaly_detection.py    # Isolation Forest training & prediction
│   ├── hybrid_detection.py     # Combined detection pipeline
│   ├── explainability.py       # SHAP explanations for anomalies
│   ├── risk_engine.py          # Risk scoring & severity assessment
│   ├── remediation.py          # Response recommendations
│   ├── evaluation.py           # Hybrid evaluation metrics
│   └── visualization.py       # Plot generation (10 chart types)
│
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard
│
├── tests/
│   └── test_pipeline.py        # Unit tests (pytest)
│
├── models/                     # Saved model artifacts
│   ├── preprocessing_config.json
│   └── anomaly_config.json
│
├── data/
│   ├── raw/                    # Raw CICIDS2017 parquet files
│   └── processed/              # Cleaned & split datasets
│
├── outputs/                    # Generated outputs
│   ├── plots/                  # Visualization plots
│   ├── signature_*/            # Signature engine evaluation
│   ├── anomaly_*/              # Anomaly detector evaluation
│   ├── hybrid/                 # Hybrid detection results
│   ├── shap/                   # SHAP explanations
│   └── evaluation/             # Final evaluation metrics
│
├── docs/                       # Documentation
└── notebooks/                  # Jupyter notebooks
```

---

## ⚙️ Installation

```bash
git clone https://github.com/choppabharathkumar/hybrid-ids-ml.git
cd hybrid-ids-ml
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

---

## 🎯 Usage

### Run Full Pipeline
```bash
python main.py
```

### Run Specific Stage
```bash
python main.py --stage signature    # Run signature detection only
python main.py --stage anomaly      # Run anomaly detection only
python main.py --stage hybrid       # Run hybrid pipeline
python main.py --stage evaluation   # Evaluate results
python main.py --stage visualization # Generate plots
python main.py --list-stages        # List all stages
```

### Launch Dashboard
```bash
python main.py --dashboard
# or directly:
streamlit run dashboard/app.py
```

### Run Tests
```bash
pytest tests/ -v
```

---

## 📊 Detection Rules

| Rule ID | Attack | Key Features | Justification |
|---|---|---|---|
| `PORTSCAN_001` | Port Scanning | PSH=1, low packets, small payload | 96% of PortScan flows have PSH=1 with minimal data |
| `WEBATTACK_001` | Brute Force / XSS | PSH=1, ACK=0, specific TCP window sizes | Distinctive push-without-acknowledge pattern |
| `SQLI_001` | SQL Injection | InitBwdWin ∈ {235,236}, data packets ≥ 1 | Unique backward window from SQL payload processing |

---

## 📈 Pipeline Stages

| # | Stage | Module | Description |
|---|---|---|---|
| 1 | Data Loading | `data_loading.py` | Load CICIDS2017, normalize labels, merge |
| 2 | EDA | `eda.py` | Quality checks, dedup, clean |
| 3 | Preprocessing | `data_preprocessing.py` | Scale, encode, train/val/test split |
| 4 | Exploration | `data_exploration.py` | Feature analysis, correlations, Fisher scores |
| 5 | Signature Detection | `rule_based_detection.py` | Apply 3 signature rules |
| 6 | Anomaly Detection | `anomaly_detection.py` | Train Isolation Forest on benign-only |
| 7 | Hybrid Detection | `hybrid_detection.py` | Combine Sig + Anomaly + SHAP + Risk |
| 8 | Evaluation | `evaluation.py` | Binary & per-class metrics |
| 9 | Visualization | `visualization.py` | Generate 10 plot types |

---

## 🔍 Key Design Decisions

- **Benign-only training**: Isolation Forest trained exclusively on benign traffic — no attack labels used during training
- **No data leakage**: Scaler fitted on training split only; validation/test transformed with training parameters
- **Threshold calibration**: Anomaly threshold optimized on validation set to maximize recall under FPR constraint
- **Rule priority**: SQL Injection checked first (most specific), then Web Attack, then PortScan
- **SHAP explanations**: TreeExplainer provides per-feature contribution to each anomaly decision
- **Safe remediation**: All automated responses are advisory-only; destructive actions require confirmation

---

## 📅 Development Log

### ✅ Day 1
- Project structure created
- Git & GitHub initialized
- Loaded CICIDS2017 dataset
- Merged datasets
- Performed Exploratory Data Analysis (EDA)
- Removed duplicate records
- Completed preprocessing pipeline
- Saved cleaned dataset

### ✅ Day 2
- Rule-Based Signature Detection Engine (3 rules)
- Isolation Forest training & threshold calibration
- SHAP Explainability integration
- Risk Assessment Engine (0-100 scoring)
- Remediation Guidance system
- Hybrid Detection Pipeline
- Comprehensive Evaluation module
- Visualization suite (10 chart types)
- Streamlit Dashboard (9 pages)
- Unit test suite (30+ tests)
- Main pipeline orchestrator

---

## 👨‍💻 Author

**Choppa Bharath Kumar**

Cybersecurity | Machine Learning | Python

---

## 📄 License

This project is for educational and research purposes.
