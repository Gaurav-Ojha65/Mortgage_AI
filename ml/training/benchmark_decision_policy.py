"""
Decision Policy Benchmark — Out-of-Sample Evaluation
===================================================
Compares F1-optimal, Balanced Accuracy, Cost-Sensitive binary,
and 3-tier Economic policies on the held-out test set.

CRITICAL METHODOLOGICAL PROTOCOL:
1. All policy thresholds are optimized EXCLUSIVELY on the validation split (val.csv).
2. All policy parameters are FROZEN after validation optimization.
3. The frozen policies are evaluated ONCE on the untouched test split (test.csv).
4. No threshold, cost parameter, or decision boundary is optimized on test labels.

Cost Model Parameters (Demonstration/Illustrative values):
- False Negative Cost (C_FN): $10,000 (Loss Given Default)
- False Positive Cost (C_FP): $1,000 (Opportunity loss from good borrower rejection)
- Manual Review Cost (C_Review): $150 (Human underwriting review cost)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score,
)

from ml.training.train import load_data
from risk.decision_policy import (
    CostModel,
    DecisionPolicy,
    optimize_f1_threshold,
    optimize_balanced_accuracy_threshold,
    optimize_cost_sensitive_binary_threshold,
    optimize_three_tier_policy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "metrics"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def compute_policy_test_metrics(
    policy_name: str,
    threshold_desc: str,
    y_test: np.ndarray,
    p_test: np.ndarray,
    decision_func,
    cost_model: CostModel,
) -> Dict[str, Any]:
    """
    Evaluate a frozen decision policy on the untouched test set.
    """
    n_test = len(y_test)
    decisions = [decision_func(p) for p in p_test]

    # For binary metrics mapping: APPROVE -> 0 (negative prediction for default),
    # REJECT -> 1 (positive prediction for default).
    # MANUAL_REVIEW -> handled in cost calculations; for binary metrics, treated as rejected from auto-approval.
    binary_preds = [1 if d in ("REJECT", "MANUAL_REVIEW") else 0 for d in decisions]
    
    # Pure binary predictions for strict binary thresholds
    strict_binary_preds = [1 if d == "REJECT" else 0 for d in decisions]

    # Confusion matrix on auto-approval decision boundary
    cm = confusion_matrix(y_test, binary_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Rates
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    rec = float(recall_score(y_test, binary_preds, zero_division=0))
    prec = float(precision_score(y_test, binary_preds, zero_division=0))
    f1 = float(f1_score(y_test, binary_preds, zero_division=0))
    acc = float(accuracy_score(y_test, binary_preds))

    # Decision distribution
    n_approved = sum(1 for d in decisions if d == "APPROVE")
    n_rejected = sum(1 for d in decisions if d == "REJECT")
    n_reviewed = sum(1 for d in decisions if d == "MANUAL_REVIEW")

    # Economic cost computation
    # 1. Defaults approved (FN) -> full default loss
    approved_defaults = sum(1 for y, d in zip(y_test, decisions) if y == 1 and d == "APPROVE")
    cost_defaults = approved_defaults * cost_model.cost_fn

    # 2. Non-defaults rejected (FP) -> opportunity loss
    rejected_good = sum(1 for y, d in zip(y_test, decisions) if y == 0 and d == "REJECT")
    cost_rejected = rejected_good * cost_model.cost_fp

    # 3. Manual reviews -> triage review operational cost
    cost_reviews = n_reviewed * cost_model.cost_manual_review

    total_expected_cost = cost_defaults + cost_rejected + cost_reviews
    cost_per_applicant = total_expected_cost / n_test

    return {
        "policy_name": policy_name,
        "thresholds": threshold_desc,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "accuracy": round(acc, 4),
        "total_expected_cost": round(total_expected_cost, 2),
        "cost_per_applicant": round(cost_per_applicant, 2),
        "distribution": {
            "approved": n_approved,
            "approved_pct": round(n_approved / n_test, 4),
            "reviewed": n_reviewed,
            "reviewed_pct": round(n_reviewed / n_test, 4),
            "rejected": n_rejected,
            "rejected_pct": round(n_rejected / n_test, 4),
        },
        "cost_breakdown": {
            "approved_default_loss": cost_defaults,
            "approved_defaults_count": approved_defaults,
            "rejected_good_borrower_loss": cost_rejected,
            "rejected_good_count": rejected_good,
            "manual_review_cost": cost_reviews,
            "manual_review_count": n_reviewed,
        }
    }


def run_benchmark(seed: int = 42) -> Dict[str, Any]:
    """Execute out-of-sample policy benchmark."""
    logger.info("=" * 80)
    logger.info("  DECISION POLICY BENCHMARK — VALIDATION OPTIMIZED / TEST EVALUATED")
    logger.info("=" * 80)

    # 1. Load canonical data splits
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(seed)
    y_val_arr = y_val.values
    y_test_arr = y_test.values

    # 2. Load frozen Isotonic Calibrated LightGBM model
    cal_model_path = MODELS_DIR / "lightgbm_calibrated_isotonic.joblib"
    if not cal_model_path.exists():
        raise FileNotFoundError(f"Calibrated model artifact not found at {cal_model_path}.")
    cal_model = joblib.load(cal_model_path)
    logger.info(f"Loaded calibrated model from {cal_model_path}")

    # Generate probabilities
    p_val = cal_model.predict_proba(X_val)[:, 1]
    p_test = cal_model.predict_proba(X_test)[:, 1]

    # Standard demonstration cost model
    demo_cost_model = CostModel(cost_fn=10000.0, cost_fp=1000.0, cost_manual_review=150.0)

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 1: OPTIMIZE ALL POLICIES EXCLUSIVELY ON VALIDATION SPLIT
    # ─────────────────────────────────────────────────────────────────────────────
    logger.info("\n[STEP 1] Optimizing policies on VALIDATION data only...")

    # Policy 1: F1-Optimal binary threshold
    f1_t_val, f1_val_score = optimize_f1_threshold(y_val_arr, p_val)
    logger.info(f"  Policy 1 (F1-Optimal): Frozen threshold = {f1_t_val:.3f} (Val F1 = {f1_val_score:.4f})")

    # Policy 2: Balanced Accuracy binary threshold
    bacc_t_val, bacc_val_score = optimize_balanced_accuracy_threshold(y_val_arr, p_val)
    logger.info(f"  Policy 2 (Balanced Accuracy): Frozen threshold = {bacc_t_val:.3f} (Val B-Acc = {bacc_val_score:.4f})")

    # Policy 3: Cost-Sensitive (10:1 Ratio: C_FN=$10,000, C_FP=$1,000)
    cost10_t_val, cost10_val = optimize_cost_sensitive_binary_threshold(
        y_val_arr, p_val, cost_fn=10000.0, cost_fp=1000.0
    )
    logger.info(f"  Policy 3 (Cost-Min 10:1): Frozen threshold = {cost10_t_val:.3f} (Val Cost = ${cost10_val:,.2f})")

    # Policy 4: Cost-Sensitive (5:1 Ratio: C_FN=$5,000, C_FP=$1,000)
    cost5_t_val, cost5_val = optimize_cost_sensitive_binary_threshold(
        y_val_arr, p_val, cost_fn=5000.0, cost_fp=1000.0
    )
    logger.info(f"  Policy 4 (Cost-Min 5:1): Frozen threshold = {cost5_t_val:.3f}")

    # Policy 5: 3-Tier Economic Policy (APPROVE / MANUAL_REVIEW / REJECT)
    policy_3tier = optimize_three_tier_policy(
        y_val_arr, p_val, cost_model=demo_cost_model, target_review_rate_max=0.20
    )
    logger.info(
        f"  Policy 5 (3-Tier Economic): Frozen Approve <= {policy_3tier.approve_threshold:.3f}, "
        f"Reject >= {policy_3tier.reject_threshold:.3f}"
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 2: EVALUATE FROZEN POLICIES ON UNTOUCHED TEST SET
    # ─────────────────────────────────────────────────────────────────────────────
    logger.info("\n[STEP 2] Evaluating FROZEN policies on untouched TEST set...")

    benchmark_results = []

    # Eval Policy 1
    res1 = compute_policy_test_metrics(
        policy_name="1. F1-Optimal Binary",
        threshold_desc=f"theta = {f1_t_val:.3f}",
        y_test=y_test_arr,
        p_test=p_test,
        decision_func=lambda p: "REJECT" if p >= f1_t_val else "APPROVE",
        cost_model=demo_cost_model,
    )
    benchmark_results.append(res1)

    # Eval Policy 2
    res2 = compute_policy_test_metrics(
        policy_name="2. Balanced Accuracy Binary",
        threshold_desc=f"theta = {bacc_t_val:.3f}",
        y_test=y_test_arr,
        p_test=p_test,
        decision_func=lambda p: "REJECT" if p >= bacc_t_val else "APPROVE",
        cost_model=demo_cost_model,
    )
    benchmark_results.append(res2)

    # Eval Policy 3
    res3 = compute_policy_test_metrics(
        policy_name="3. Cost-Sensitive Binary (10:1)",
        threshold_desc=f"theta = {cost10_t_val:.3f}",
        y_test=y_test_arr,
        p_test=p_test,
        decision_func=lambda p: "REJECT" if p >= cost10_t_val else "APPROVE",
        cost_model=demo_cost_model,
    )
    benchmark_results.append(res3)

    # Eval Policy 4
    res4 = compute_policy_test_metrics(
        policy_name="4. Cost-Sensitive Binary (5:1)",
        threshold_desc=f"theta = {cost5_t_val:.3f}",
        y_test=y_test_arr,
        p_test=p_test,
        decision_func=lambda p: "REJECT" if p >= cost5_t_val else "APPROVE",
        cost_model=demo_cost_model,
    )
    benchmark_results.append(res4)

    # Eval Policy 5 (3-Tier)
    res5 = compute_policy_test_metrics(
        policy_name="5. 3-Tier Economic (Approve/Review/Reject)",
        threshold_desc=f"Approve <= {policy_3tier.approve_threshold:.3f}, Reject >= {policy_3tier.reject_threshold:.3f}",
        y_test=y_test_arr,
        p_test=p_test,
        decision_func=lambda p: "APPROVE" if p <= policy_3tier.approve_threshold else ("REJECT" if p >= policy_3tier.reject_threshold else "MANUAL_REVIEW"),
        cost_model=demo_cost_model,
    )
    benchmark_results.append(res5)

    # Format table output
    table_rows = []
    for r in benchmark_results:
        table_rows.append({
            "Policy": r["policy_name"],
            "Thresholds": r["thresholds"],
            "Prec": r["precision"],
            "Rec": r["recall"],
            "Spec": r["specificity"],
            "F1": r["f1"],
            "FPR": r["fpr"],
            "FNR": r["fnr"],
            "Acc": r["accuracy"],
            "Cost/App": f"${r['cost_per_applicant']:,.2f}",
            "Total Cost": f"${r['total_expected_cost']:,.0f}",
        })

    df_table = pd.DataFrame(table_rows)
    print("\n" + "=" * 110)
    print("  DECISION POLICY COMPARATIVE BENCHMARK (UNTOUCHED TEST SET, N=21,398)")
    print("=" * 110)
    print(df_table.to_string(index=False))

    # Save to json report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "decision_policy_benchmark.json"
    with open(report_path, "w") as f:
        json.dump({
            "cost_model": {
                "cost_fn": demo_cost_model.cost_fn,
                "cost_fp": demo_cost_model.cost_fp,
                "cost_manual_review": demo_cost_model.cost_manual_review,
                "is_demonstration": True,
            },
            "policies": benchmark_results,
        }, f, indent=2)
    logger.info(f"\nSaved benchmark report to {report_path}")

    return {"policies": benchmark_results}


if __name__ == "__main__":
    run_benchmark()
