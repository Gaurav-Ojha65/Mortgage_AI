"""
Post-HPO Clean Rebuild, Validation & Baseline Comparison — Mortgage AI
======================================================================
Executes a completely fresh, independent rebuild and evaluation of the best HPO
candidate discovered by Optuna.

MANDATORY PROTOCOL:
1. NEVER reuse Optuna fold models or trial predictions.
2. Generate fresh 5-fold OOF predictions on real_train.csv with candidate params.
3. Fit fresh Isotonic calibrator strictly on (p_oof, y_real).
4. Train final candidate LightGBM on full training data with candidate params.
5. Score val.csv and optimize 3-tier policy on validation data only. Freeze policy.
6. Evaluate ONCE on untouched test.csv.
7. Generate side-by-side comparison report against frozen v3.0.0 baseline.
8. Determine whether candidate satisfies all acceptance criteria to replace baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import lightgbm as lgb

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.inference.predict import MODEL_FEATURES
from ml.training.eval_utils import compute_calibration_metrics
from ml.training.oof_calibration import (
    load_real_train_data,
    generate_oof_predictions,
)
from ml.training.calibration import fit_oof_calibrators, _compute_sha256, _get_git_commit_sha
from ml.training.calibrated_predictor import CalibratedPredictor
from risk.decision_policy import DecisionPolicy, CostModel, optimize_three_tier_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = _PROJECT_ROOT / "ml" / "models"
DATA_DIR = _PROJECT_ROOT / "data"
REPORTS_DIR = _PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen v3.0.0 Baseline Benchmarks on untouched test.csv (N=21,398)
BASELINE_V3_TEST_BENCHMARK = {
    "version": "v3.0.0-oof-baseline",
    "model_name": "LightGBM (Baseline)",
    "hyperparameters": {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 25,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
    },
    "test_metrics": {
        "roc_auc": 0.8599,
        "pr_auc": 0.3947,
        "brier_score": 0.0494,
        "weighted_ece": 0.0018,
        "macro_ece": 0.0353,
    },
    "frozen_policy": {
        "policy_name": "frozen_oof_3tier_policy",
        "approve_threshold": 0.055,
        "reject_threshold": 0.405,
        "approval_pct": 0.7240,
        "review_pct": 0.2461,
        "rejection_pct": 0.0299,
        "approved_defaults": 302,
        "rejected_non_defaults": 273,
        "rejected_defaults": 366,
        "fn_cost": 3020000.0,
        "fp_cost": 273000.0,
        "review_cost": 789900.0,
        "total_expected_cost": 4082900.0,
        "cost_per_applicant": 190.81,
        "precision": 0.1937,
        "recall": 0.7911,
        "specificity": 0.7614,
        "f1": 0.3113,
        "fpr": 0.2386,
        "fnr": 0.2089,
    },
}


def run_post_hpo_validation(
    hpo_results_path: Optional[Path] = None,
    explicit_params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute clean post-HPO rebuild, validation-based policy optimization, and single test-set evaluation.
    """
    logger.info("=" * 70)
    logger.info("  STARTING POST-HPO CLEAN REBUILD AND VALIDATION")
    logger.info("=" * 70)

    # 1. Determine candidate hyperparameters
    candidate_params = explicit_params
    best_trial_info = {}

    if candidate_params is None:
        hpo_path = hpo_results_path or (METRICS_DIR / "optuna_results.json")
        if not hpo_path.exists():
            raise FileNotFoundError(f"HPO results not found at {hpo_path}. Run lightgbm_hpo.py first.")
        with open(hpo_path, "r", encoding="utf-8") as f:
            hpo_data = json.load(f)
        candidate_params = hpo_data["best_trial"]["params"]
        best_trial_info = hpo_data["best_trial"]

    logger.info(f"Candidate Hyperparameters for Rebuild:")
    for k, v in candidate_params.items():
        logger.info(f"  {k}: {v}")

    # 2. Fresh 5-fold OOF generation on real_train.csv
    logger.info("\n--- Step 1/5: Fresh 5-Fold OOF Generation on real_train.csv ---")
    t0_oof = time.time()
    X_real, y_real = load_real_train_data(DATA_DIR)

    # Ensure subsample_freq is set if subsample < 1.0
    full_params = dict(candidate_params)
    if full_params.get("subsample", 1.0) < 1.0:
        full_params["subsample_freq"] = 1
    else:
        full_params["subsample_freq"] = 0
    full_params["random_state"] = seed
    full_params["verbose"] = -1
    full_params["n_jobs"] = -1

    fresh_oof_result = generate_oof_predictions(
        X_real, y_real, n_splits=5, random_seed=seed, lgb_params=full_params
    )
    t_oof_elapsed = time.time() - t0_oof
    logger.info(f"Fresh OOF generation finished in {t_oof_elapsed:.1f}s.")
    logger.info(f"Fresh OOF ROC-AUC: {fresh_oof_result['raw_metrics']['roc_auc']:.4f}")

    # 3. Fit fresh OOF Isotonic calibrator strictly on (p_oof, y_real)
    logger.info("\n--- Step 2/5: Fresh OOF Isotonic Calibrator Fitting ---")
    fresh_cal_result = fit_oof_calibrators(
        fresh_oof_result["oof_probs"],
        fresh_oof_result["y_true"],
        model_name="lightgbm_hpo_candidate",
    )
    fresh_isotonic = fresh_cal_result["isotonic_calibrator"]

    # 4. Train final candidate LightGBM base model on full train.csv
    logger.info("\n--- Step 3/5: Training Final Base Model on Full train.csv ---")
    t0_train = time.time()
    train_path = DATA_DIR / "train.csv"
    train_df = pd.read_csv(train_path)
    X_train_full = train_df[MODEL_FEATURES]
    y_train_full = train_df["target"]

    candidate_lgb = lgb.LGBMClassifier(**full_params)
    candidate_lgb.fit(X_train_full, y_train_full)
    t_train_elapsed = time.time() - t0_train
    logger.info(f"Final candidate base model trained in {t_train_elapsed:.2f}s.")

    # Save raw candidate model
    cand_model_path = MODELS_DIR / "lightgbm_hpo_candidate.joblib"
    joblib.dump(candidate_lgb, cand_model_path)
    model_size_kb = round(os.path.getsize(cand_model_path) / 1024, 2)
    logger.info(f"Saved raw candidate model to {cand_model_path} ({model_size_kb} KB)")

    # Bundle into CalibratedPredictor wrapper
    candidate_pipeline = CalibratedPredictor(
        base_model=candidate_lgb,
        calibrator=fresh_isotonic,
        calibration_method="isotonic",
        model_name="lightgbm_hpo_candidate",
        version="v3.1-hpo-candidate",
    )

    # 5. Optimize 3-Tier Policy on val.csv ONLY
    logger.info("\n--- Step 4/5: Optimizing 3-Tier Policy on val.csv (Validation Only) ---")
    val_path = DATA_DIR / "val.csv"
    val_df = pd.read_csv(val_path)
    X_val = val_df[MODEL_FEATURES]
    y_val = val_df["target"].values

    p_val_cal = candidate_pipeline.predict_proba(X_val)[:, 1]
    cost_model = CostModel(cost_fn=10000.0, cost_fp=1000.0, cost_manual_review=150.0)

    candidate_policy = optimize_three_tier_policy(
        y_true_val=y_val,
        p_val=p_val_cal,
        cost_model=cost_model,
        target_review_rate_max=0.25,
    )
    logger.info(
        f"Optimized Policy on val.csv: Approve <= {candidate_policy.approve_threshold:.3f}, "
        f"Reject >= {candidate_policy.reject_threshold:.3f}"
    )

    # 6. Final Evaluation ONCE on untouched test.csv
    logger.info("\n--- Step 5/5: Single Final Evaluation on Untouched test.csv ---")
    test_path = DATA_DIR / "test.csv"
    test_df = pd.read_csv(test_path)
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df["target"].values
    n_test = len(y_test)

    # Inference latency benchmark (1,000 single predictions)
    t0_lat = time.time()
    sample_rows = [X_test.iloc[[i]] for i in range(min(1000, len(X_test)))]
    for row in sample_rows:
        _ = candidate_pipeline.predict_proba(row)
    t_lat_total = time.time() - t0_lat
    latency_per_app_ms = round((t_lat_total / len(sample_rows)) * 1000, 3)
    logger.info(f"Measured single-applicant inference latency: {latency_per_app_ms} ms/app")

    # Predict full test set
    y_prob_cal = candidate_pipeline.predict_proba(X_test)[:, 1]

    # Test ML metrics
    test_roc_auc = round(float(roc_auc_score(y_test, y_prob_cal)), 4)
    test_pr_auc = round(float(average_precision_score(y_test, y_prob_cal)), 4)
    test_brier = round(float(brier_score_loss(y_test, y_prob_cal)), 4)
    cal_metrics = compute_calibration_metrics(y_test, y_prob_cal, n_bins=10)
    test_wece = cal_metrics["ece"]
    test_macro_ece = cal_metrics["macro_ece"]

    logger.info(f"Test ML Metrics (HPO Candidate):")
    logger.info(f"  ROC-AUC:      {test_roc_auc:.4f} (Baseline: {BASELINE_V3_TEST_BENCHMARK['test_metrics']['roc_auc']:.4f})")
    logger.info(f"  PR-AUC:       {test_pr_auc:.4f} (Baseline: {BASELINE_V3_TEST_BENCHMARK['test_metrics']['pr_auc']:.4f})")
    logger.info(f"  Brier Score:  {test_brier:.4f} (Baseline: {BASELINE_V3_TEST_BENCHMARK['test_metrics']['brier_score']:.4f})")
    logger.info(f"  Weighted ECE: {test_wece:.4f} (Baseline: {BASELINE_V3_TEST_BENCHMARK['test_metrics']['weighted_ece']:.4f})")
    logger.info(f"  Macro ECE:    {test_macro_ece:.4f} (Baseline: {BASELINE_V3_TEST_BENCHMARK['test_metrics']['macro_ece']:.4f})")

    # Evaluate 3-tier policy on test set
    app_mask = y_prob_cal <= candidate_policy.approve_threshold
    rej_mask = y_prob_cal >= candidate_policy.reject_threshold
    rev_mask = (~app_mask) & (~rej_mask)

    n_app = int(np.sum(app_mask))
    n_rev = int(np.sum(rev_mask))
    n_rej = int(np.sum(rej_mask))

    app_pct = round(n_app / n_test, 4)
    rev_pct = round(n_rev / n_test, 4)
    rej_pct = round(n_rej / n_test, 4)

    app_defaults = int(np.sum(y_test[app_mask]))
    rej_non_defaults = int(np.sum(1 - y_test[rej_mask]))
    rej_defaults = int(np.sum(y_test[rej_mask]))

    fn_cost = app_defaults * cost_model.cost_fn
    fp_cost = rej_non_defaults * cost_model.cost_fp
    rev_cost = n_rev * cost_model.cost_manual_review
    total_cost = fn_cost + fp_cost + rev_cost
    cost_per_app = round(total_cost / n_test, 2)

    # Binary metrics (Flagged = p > approve_thresh)
    flagged = (y_prob_cal > candidate_policy.approve_threshold).astype(int)
    cm = confusion_matrix(y_test, flagged)
    tn, fp, fn, tp = cm.ravel()

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    f1 = round(f1_score(y_test, flagged), 4)
    fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
    fnr = round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0

    logger.info(f"\nTest 3-Tier Policy Economics (HPO Candidate):")
    logger.info(f"  Approve Threshold:    {candidate_policy.approve_threshold:.3f}")
    logger.info(f"  Reject Threshold:     {candidate_policy.reject_threshold:.3f}")
    logger.info(f"  Approval Rate:        {app_pct:.2%}")
    logger.info(f"  Review Rate:          {rev_pct:.2%} (Constraint <= 25%: {rev_pct <= 0.25})")
    logger.info(f"  Rejection Rate:       {rej_pct:.2%}")
    logger.info(f"  Approved Defaults:    {app_defaults} (FN Cost: ${fn_cost:,.0f})")
    logger.info(f"  Rejected Non-Defs:    {rej_non_defaults} (FP Cost: ${fp_cost:,.0f})")
    logger.info(f"  Review Cost:          ${rev_cost:,.0f}")
    logger.info(f"  Total Expected Cost:  ${total_cost:,.0f} (Baseline: ${BASELINE_V3_TEST_BENCHMARK['frozen_policy']['total_expected_cost']:,.0f})")
    logger.info(f"  Cost Per Applicant:   ${cost_per_app:.2f} (Baseline: ${BASELINE_V3_TEST_BENCHMARK['frozen_policy']['cost_per_applicant']:.2f})")

    # 7. Comprehensive Acceptance Evaluation
    base_m = BASELINE_V3_TEST_BENCHMARK["test_metrics"]
    base_p = BASELINE_V3_TEST_BENCHMARK["frozen_policy"]

    delta_auc = round(test_roc_auc - base_m["roc_auc"], 4)
    delta_pr_auc = round(test_pr_auc - base_m["pr_auc"], 4)
    delta_brier = round(test_brier - base_m["brier_score"], 4)
    delta_wece = round(test_wece - base_m["weighted_ece"], 4)
    delta_cost = round(total_cost - base_p["total_expected_cost"], 2)
    delta_cost_pct = round((delta_cost / base_p["total_expected_cost"]) * 100, 2)

    # Formal criteria checks
    c1_auc = delta_auc >= -0.0020  # Meaningfully preserves or improves ROC-AUC
    c2_pr = delta_pr_auc >= -0.0050  # Does not materially degrade PR-AUC
    c3_brier = delta_brier <= 0.0010  # Does not materially degrade Brier
    c4_wece = delta_wece <= 0.0015  # Does not materially degrade wECE
    c5_cost = delta_cost <= 0.0  # Improves or matches economic cost
    c6_review = rev_pct <= 0.2501  # Review rate remains <= 25%

    all_criteria_met = c1_auc and c2_pr and c3_brier and c4_wece and c5_cost and c6_review
    recommendation = (
        "ADOPT_HPO_CANDIDATE" if (all_criteria_met and (delta_cost < 0 or delta_auc > 0.0010))
        else "KEEP_FROZEN_V3_BASELINE"
    )

    acceptance_evaluation = {
        "c1_roc_auc_preserved_or_improved": {
            "passed": c1_auc,
            "baseline": base_m["roc_auc"],
            "candidate": test_roc_auc,
            "delta": delta_auc,
            "rule": "delta >= -0.0020",
        },
        "c2_pr_auc_preserved": {
            "passed": c2_pr,
            "baseline": base_m["pr_auc"],
            "candidate": test_pr_auc,
            "delta": delta_pr_auc,
            "rule": "delta >= -0.0050",
        },
        "c3_brier_preserved": {
            "passed": c3_brier,
            "baseline": base_m["brier_score"],
            "candidate": test_brier,
            "delta": delta_brier,
            "rule": "delta <= 0.0010",
        },
        "c4_wece_preserved": {
            "passed": c4_wece,
            "baseline": base_m["weighted_ece"],
            "candidate": test_wece,
            "delta": delta_wece,
            "rule": "delta <= 0.0015",
        },
        "c5_economic_cost_improved": {
            "passed": c5_cost,
            "baseline_cost": base_p["total_expected_cost"],
            "candidate_cost": total_cost,
            "cost_delta": delta_cost,
            "cost_delta_pct": delta_cost_pct,
            "rule": "candidate_cost <= baseline_cost",
        },
        "c6_review_capacity_respected": {
            "passed": c6_review,
            "baseline_review_pct": base_p["review_pct"],
            "candidate_review_pct": rev_pct,
            "rule": "candidate_review_pct <= 0.25",
        },
        "all_criteria_met": all_criteria_met,
        "final_recommendation": recommendation,
    }

    # Build Side-by-Side Comparison Object
    candidate_summary = {
        "version": "v3.1-hpo-candidate",
        "model_name": "LightGBM (HPO Best Candidate)",
        "hyperparameters": candidate_params,
        "test_metrics": {
            "roc_auc": test_roc_auc,
            "pr_auc": test_pr_auc,
            "brier_score": test_brier,
            "weighted_ece": test_wece,
            "macro_ece": test_macro_ece,
        },
        "frozen_policy": {
            "policy_name": candidate_policy.policy_name,
            "approve_threshold": candidate_policy.approve_threshold,
            "reject_threshold": candidate_policy.reject_threshold,
            "approval_pct": app_pct,
            "review_pct": rev_pct,
            "rejection_pct": rej_pct,
            "approved_defaults": app_defaults,
            "rejected_non_defaults": rej_non_defaults,
            "rejected_defaults": rej_defaults,
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "review_cost": rev_cost,
            "total_expected_cost": total_cost,
            "cost_per_applicant": cost_per_app,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
        },
        "engineering_metrics": {
            "training_time_seconds": round(t_train_elapsed, 2),
            "oof_cv_runtime_seconds": round(t_oof_elapsed, 2),
            "inference_latency_ms_per_app": latency_per_app_ms,
            "model_artifact_size_kb": model_size_kb,
        },
    }

    hpo_vs_baseline_report = {
        "metadata": {
            "eval_dataset": "data/test.csv (N=21,398 untouched)",
            "git_commit_sha": _get_git_commit_sha(),
            "timestamp": datetime.now().isoformat(),
        },
        "baseline_v3": BASELINE_V3_TEST_BENCHMARK,
        "candidate_hpo": candidate_summary,
        "deltas": {
            "roc_auc_delta": delta_auc,
            "pr_auc_delta": delta_pr_auc,
            "brier_score_delta": delta_brier,
            "weighted_ece_delta": delta_wece,
            "total_cost_delta": delta_cost,
            "cost_delta_pct": delta_cost_pct,
            "cost_per_app_delta": round(cost_per_app - base_p["cost_per_applicant"], 2),
            "review_rate_delta": round(rev_pct - base_p["review_pct"], 4),
        },
        "acceptance_evaluation": acceptance_evaluation,
    }

    # Save Comparison Report
    comp_json_path = METRICS_DIR / "hpo_vs_baseline.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(hpo_vs_baseline_report, f, indent=2)
    logger.info(f"Saved side-by-side comparison report to {comp_json_path}")

    # Save Candidate Metadata
    metadata_obj = {
        "model_name": "LightGBM",
        "version": "v3.1-hpo-candidate",
        "hyperparameters": candidate_params,
        "git_commit_sha": _get_git_commit_sha(),
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "lightgbm_version": lgb.__version__,
            "joblib_version": joblib.__version__,
        },
        "provenance": {
            "real_train_csv_sha256": _compute_sha256(DATA_DIR / "real_train.csv"),
            "train_csv_sha256": _compute_sha256(DATA_DIR / "train.csv"),
            "val_csv_sha256": _compute_sha256(DATA_DIR / "val.csv"),
            "test_csv_sha256": _compute_sha256(DATA_DIR / "test.csv"),
        },
        "optuna_best_trial_source": best_trial_info,
        "measured_metrics": candidate_summary,
        "created_at": datetime.now().isoformat(),
    }

    hpo_meta_path = MODELS_DIR / "hpo_metadata.json"
    with open(hpo_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_obj, f, indent=2)
    logger.info(f"Saved candidate metadata to {hpo_meta_path}")

    return hpo_vs_baseline_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-HPO Clean Rebuild and Comparison")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    run_post_hpo_validation(seed=args.seed)
