#!/usr/bin/env python3
"""Main entry point for the Hybrid IDS pipeline.

Orchestrates all stages of the detection pipeline:
  1. Data Loading & Merging
  2. EDA & Cleaning
  3. Preprocessing
  4. Data Exploration
  5. Signature-Based Detection
  6. Anomaly Detection (Isolation Forest)
  7. Hybrid Detection (Signature + Anomaly + SHAP + Risk + Remediation)
  8. Evaluation
  9. Visualization

Usage:
    python main.py                  # Run full pipeline
    python main.py --stage hybrid   # Run specific stage
    python main.py --dashboard      # Launch Streamlit dashboard
"""

import sys
import argparse
import os


def run_data_loading():
    """Stage 1: Load and merge raw datasets."""
    from src.data_loading import load_and_merge

    print("\n" + "=" * 70)
    print("STAGE 1: DATA LOADING & MERGING")
    print("=" * 70)

    merged = load_and_merge(
        base_dir=".",
        save_path="data/processed/merged_dataset.parquet",
    )
    print(f"\nFinal shape: {merged.shape}")
    print(f"Label distribution:\n{merged['Label'].value_counts()}")
    return merged


def run_eda():
    """Stage 2: Exploratory Data Analysis & Cleaning."""
    import pandas as pd
    from src.eda import run_eda as eda_pipeline

    print("\n" + "=" * 70)
    print("STAGE 2: EDA & CLEANING")
    print("=" * 70)

    df = pd.read_parquet("data/processed/merged_dataset.parquet")
    df = eda_pipeline(df, output_dir="outputs")
    df.to_parquet("data/processed/cleaned_dataset.parquet", index=False)
    print("\nCleaned dataset saved.")
    return df


def run_preprocessing():
    """Stage 3: Feature scaling, encoding, and train/val/test split."""
    import pandas as pd
    from src.data_preprocessing import run_preprocessing as preprocess_pipeline

    print("\n" + "=" * 70)
    print("STAGE 3: PREPROCESSING")
    print("=" * 70)

    df = pd.read_parquet("data/processed/cleaned_dataset.parquet")
    results = preprocess_pipeline(df)
    print("\nPreprocessing complete!")
    return results


def run_exploration():
    """Stage 4: Detailed data exploration and feature analysis."""
    import pandas as pd
    from src.data_exploration import run_exploration

    print("\n" + "=" * 70)
    print("STAGE 4: DATA EXPLORATION")
    print("=" * 70)

    df = pd.read_parquet("data/processed/cleaned_dataset.parquet")
    run_exploration(df)


def run_signature_detection():
    """Stage 5: Rule-based signature detection."""
    from src.rule_based_detection import main as sig_main

    print("\n" + "=" * 70)
    print("STAGE 5: SIGNATURE-BASED DETECTION")
    print("=" * 70)

    sig_main()


def run_anomaly_detection():
    """Stage 6: Isolation Forest anomaly detection."""
    from src.anomaly_detection import run_anomaly_detection_pipeline

    print("\n" + "=" * 70)
    print("STAGE 6: ANOMALY DETECTION")
    print("=" * 70)

    model, threshold, results = run_anomaly_detection_pipeline()
    print("\nAnomaly detection pipeline complete!")


def run_hybrid_detection():
    """Stage 7: Combined hybrid detection pipeline."""
    import pandas as pd
    from src.hybrid_detection import run_hybrid_detection as hybrid_pipeline

    print("\n" + "=" * 70)
    print("STAGE 7: HYBRID DETECTION")
    print("=" * 70)

    test_df = pd.read_parquet("data/processed/test.parquet")
    final_df, results, explanations = hybrid_pipeline(test_df)
    print("\nHybrid detection pipeline complete!")


def run_evaluation():
    """Stage 8: Evaluate hybrid predictions."""
    from src.evaluation import evaluate_hybrid_predictions

    print("\n" + "=" * 70)
    print("STAGE 8: EVALUATION")
    print("=" * 70)

    evaluate_hybrid_predictions()


def run_visualization():
    """Stage 9: Generate all visualization plots."""
    from src.visualization import generate_all_plots

    print("\n" + "=" * 70)
    print("STAGE 9: VISUALIZATION")
    print("=" * 70)

    generate_all_plots()


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("\nLaunching Streamlit dashboard...")
    os.system("streamlit run dashboard/app.py")


# =============================
# Pipeline Orchestrator
# =============================

STAGES = {
    "loading": run_data_loading,
    "eda": run_eda,
    "preprocessing": run_preprocessing,
    "exploration": run_exploration,
    "signature": run_signature_detection,
    "anomaly": run_anomaly_detection,
    "hybrid": run_hybrid_detection,
    "evaluation": run_evaluation,
    "visualization": run_visualization,
}


def run_full_pipeline():
    """Execute all stages sequentially."""
    print("=" * 70)
    print("HYBRID IDS — FULL PIPELINE")
    print("=" * 70)

    for name, stage_fn in STAGES.items():
        try:
            stage_fn()
        except FileNotFoundError as e:
            print(f"\n⚠️  Stage '{name}' skipped: {e}")
        except Exception as e:
            print(f"\n❌ Stage '{name}' failed: {e}")
            raise

    print("\n" + "=" * 70)
    print("✅ FULL PIPELINE COMPLETE")
    print("=" * 70)


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Hybrid Network Intrusion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run the full pipeline
  python main.py --stage hybrid     Run only the hybrid detection stage
  python main.py --stage evaluation Run only evaluation
  python main.py --dashboard        Launch the Streamlit dashboard
  python main.py --list-stages      List available stages
        """,
    )
    parser.add_argument(
        "--stage", type=str, default=None,
        help=f"Run a specific stage: {', '.join(STAGES.keys())}",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Launch the Streamlit dashboard",
    )
    parser.add_argument(
        "--list-stages", action="store_true",
        help="List all available pipeline stages",
    )

    args = parser.parse_args()

    if args.list_stages:
        print("Available pipeline stages:")
        for i, (name, fn) in enumerate(STAGES.items(), 1):
            print(f"  {i}. {name:15s} — {fn.__doc__.strip()}")
        return

    if args.dashboard:
        launch_dashboard()
        return

    if args.stage:
        stage_name = args.stage.lower()
        if stage_name not in STAGES:
            print(f"Unknown stage: {args.stage}")
            print(f"Available: {', '.join(STAGES.keys())}")
            sys.exit(1)
        STAGES[stage_name]()
        return

    run_full_pipeline()


if __name__ == "__main__":
    main()
