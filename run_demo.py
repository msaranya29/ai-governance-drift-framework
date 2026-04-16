"""
run_demo.py — Standalone demo (no Flask required).
Runs the full pipeline on credit_fraud.csv and prints results.

Usage:
    python run_demo.py
"""
import os
import sys
import textwrap

import numpy as np
import pandas as pd

# ── ensure project root is on path ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data.generate_samples import generate_credit_fraud, generate_employee_churn, generate_house_prices
from data.dataset_handler import DatasetHandler
from models.train_models import ModelFactory
from drift.drift_manager import DriftManager


# ── helpers ───────────────────────────────────────────────────────────────────
def hr(char="─", width=72): print(char * width)
def section(title):
    hr()
    print(f"  {title}")
    hr()

def color(text, code): return f"\033[{code}m{text}\033[0m"
def green(t):  return color(t, "32")
def yellow(t): return color(t, "33")
def red(t):    return color(t, "31")
def bold(t):   return color(t, "1")
def cyan(t):   return color(t, "36")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print()
    print(bold(cyan("  ╔══════════════════════════════════════════════════════╗")))
    print(bold(cyan("  ║   AI Governance Framework — Demo Run                ║")))
    print(bold(cyan("  ║   REVA University | Dr. B. MuthuKumar               ║")))
    print(bold(cyan("  ╚══════════════════════════════════════════════════════╝")))
    print()

    # ── 1. Generate samples ───────────────────────────────────────────────────
    section("STEP 1 — Generating Sample Datasets")
    generate_credit_fraud()
    generate_employee_churn()
    generate_house_prices()

    # ── 2. Load dataset ───────────────────────────────────────────────────────
    section("STEP 2 — Loading credit_fraud.csv")
    filepath = os.path.join("data", "samples", "credit_fraud.csv")
    df = pd.read_csv(filepath)
    print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}")

    handler = DatasetHandler(filepath)
    target  = handler.auto_detect_target(df)
    handler.target_column = target
    profile = handler.analyze_dataset(df)

    print(f"  Target column  : {bold(target)}")
    print(f"  Problem type   : {bold(profile.problem_type)}")
    print(f"  Recommended    : {', '.join(profile.recommended_models[:4])}")

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    section("STEP 3 — Preprocessing")
    X, y, feature_names, scaler = handler.preprocess(df, target)
    print(f"  Feature matrix : {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Features       : {', '.join(feature_names)}")

    # ── 4. Split ──────────────────────────────────────────────────────────────
    section("STEP 4 — Train / Stream / Drift Split")
    X_train, X_test, y_train, y_test, X_stream, y_stream, X_drifted, y_drifted = \
        handler.split_with_drift_simulation(X, y)
    print(f"  Train set      : {X_train.shape[0]} rows")
    print(f"  Test set       : {X_test.shape[0]} rows")
    print(f"  Stream set     : {X_stream.shape[0]} rows")
    print(f"  Drifted batch  : {X_drifted.shape[0]} rows (Gaussian noise σ=1.5 applied)")

    # ── 5. Train models ───────────────────────────────────────────────────────
    section("STEP 5 — Training & Comparing Models")
    print("  (This may take 30–90 seconds depending on your machine…)\n")

    factory = ModelFactory()
    models  = factory.get_models_for_problem_type(profile.problem_type, profile)
    print(f"  Models to train: {len(models)}")
    for name in models:
        print(f"    • {name}")
    print()

    results_df, best_model, best_name, explanation = factory.train_and_compare(
        X_train, X_test, y_train, y_test, profile.problem_type, models
    )

    # Print comparison table
    is_reg   = profile.problem_type == 'regression'
    disp_cols = ['model', 'r2', 'mae', 'rmse'] if is_reg else ['model', 'accuracy', 'f1', 'roc_auc', 'train_time_s']
    disp_cols = [c for c in disp_cols if c in results_df.columns]
    print(results_df[disp_cols].to_string(index=False))
    print()
    print(f"  {green('▶ ' + explanation)}")

    # ── 6. Drift detection ────────────────────────────────────────────────────
    section("STEP 6 — Drift Detection")
    baseline_df = pd.DataFrame(X_train, columns=feature_names)
    stream_df   = pd.DataFrame(X_stream, columns=feature_names)
    drifted_df  = pd.DataFrame(X_drifted, columns=feature_names)

    dm = DriftManager(baseline_df=baseline_df, feature_columns=feature_names)

    print("  [Stream batch — real incoming data]")
    stream_result = dm.check_drift(stream_df)
    _print_drift_result(stream_result, "Stream")

    print()
    print("  [Synthetic drift batch — Gaussian noise applied]")
    synth_result = dm.check_drift(drifted_df)
    _print_drift_result(synth_result, "Synthetic")

    # ── 7. Alerts ─────────────────────────────────────────────────────────────
    section("STEP 7 — Alert Summary")
    alerts = []
    if synth_result.alert:
        alerts.append(("CRITICAL", f"Severe drift in synthetic batch. PSI={synth_result.overall_psi:.4f}",
                        "Retrain model immediately."))
    if stream_result.alert:
        alerts.append(("WARNING", f"Drift in stream batch. PSI={stream_result.overall_psi:.4f}",
                        "Monitor closely."))
    alerts.append(("INFO", f"Pipeline complete. Best model: {best_name}", "Review model comparison."))

    for level, msg, action in alerts:
        icon = red("⛔") if level=="CRITICAL" else yellow("⚠ ") if level=="WARNING" else green("ℹ ")
        print(f"  {icon}  [{level}] {msg}")
        print(f"       Action: {action}")
    print()

    # ── 8. Summary ────────────────────────────────────────────────────────────
    section("FINAL SUMMARY")
    score_col = 'r2' if is_reg else 'f1'
    best_score = results_df.iloc[0].get(score_col, results_df.iloc[0].get('accuracy', 0))
    print(f"  Dataset        : credit_fraud.csv  ({len(df):,} rows)")
    print(f"  Problem type   : {profile.problem_type}")
    print(f"  Best model     : {bold(green(best_name))}")
    print(f"  Best {score_col.upper():<10}: {bold(str(round(float(best_score), 4)))}")
    print(f"  Stream PSI     : {stream_result.overall_psi:.4f}  ({stream_result.psi_report.results[0].severity if stream_result.psi_report.results else 'N/A'})")
    print(f"  Synthetic PSI  : {synth_result.overall_psi:.4f}  ({synth_result.psi_report.results[0].severity if synth_result.psi_report.results else 'N/A'})")
    print(f"  Drift alert    : {red('YES') if synth_result.alert else green('NO')}")
    print()
    hr("═")
    print(f"  {bold('Run `python app.py` to launch the full web dashboard.')}")
    hr("═")
    print()


def _print_drift_result(result, label):
    sev = result.psi_report.results[0].severity if result.psi_report.results else "N/A"
    color_fn = red if result.alert else (yellow if sev == "Moderate" else green)
    print(f"  Overall PSI    : {color_fn(f'{result.overall_psi:.4f}')}  ({sev})")
    print(f"  Drifted feats  : {len(result.drifted_features)}")
    if result.drifted_features:
        for feat in result.drifted_features[:5]:
            r = next((x for x in result.psi_report.results if x.feature == feat), None)
            if r:
                print(f"    • {feat}: PSI={r.psi:.4f} — {r.severity}")
    if result.explanations:
        print(f"  Sample explanation:")
        first_key = next(iter(result.explanations))
        wrapped = textwrap.fill(result.explanations[first_key], width=65,
                                initial_indent="    ", subsequent_indent="    ")
        print(wrapped)


if __name__ == '__main__':
    main()
