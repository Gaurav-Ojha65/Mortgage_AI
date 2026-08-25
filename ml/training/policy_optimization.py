"""
Validation-Only Policy Optimization & Parameter Freezing — Mortgage AI
========================================================================
Optimizes all decision thresholds and multi-tier economic routing parameters
STRICTLY on the untouched validation split (`data/val.csv`, N=21,398) using
calibrated probabilities produced by the frozen OOF-calibrated model.

Strict Protocols:
- The probability calibrator is ALREADY frozen from OOF training.
- The validation dataset is used ONLY to find optimal operational cutoffs.
- No test data (`data/test.csv`) is accessed or influenced during optimization.
- All optimized policy parameters are serialized to `ml/models/frozen_policy_config.json`.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.inference.predict import MODEL_FEATURES
from risk.decision_policy import CostModel, DecisionPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = _PROJECT_ROOT / "ml" / "models"
DATA_DIR = _PROJECT_ROOT / "data"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_val_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load canonical validation split."""
    d_dir = data_dir or DATA_DIR
    val_path = d_dir / "val.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation dataset not found at {val_path}")

    df = pd.read_csv(val_path)
    X = df[MODEL_FEATURES]
    y = df["target"]
    logger.info(f"Loaded validation split: {len(df):,} samples | Defaults: {int(y.sum()):,} ({y.mean():.2%})")
    return X, y


def optimize_f1(y_true: np.ndarray, p_val: np.ndarray) -> Tuple[float, float]:
    """Find binary threshold maximizing F1 on validation split."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.01, 0.99, 0.005):
        f = f1_score(y_true, p_val >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(float(best_t), 3), round(float(best_f1), 4)


def optimize_balanced_acc(y_true: np.ndarray, p_val: np.ndarray) -> Tuple[float, float]:
    """Find binary threshold maximizing Balanced Accuracy on validation split."""
    best_t, best_bacc = 0.5, 0.0
    for t in np.arange(0.01, 0.99, 0.005):
        bacc = balanced_accuracy_score(y_true, p_val >= t)
        if bacc > best_bacc:
            best_bacc, best_t = bacc, t
    return round(float(best_t), 3), round(float(best_bacc), 4)


def optimize_cost_binary(
    y_true: np.ndarray,
    p_val: np.ndarray,
    cost_fn: float = 10000.0,
    cost_fp: float = 1000.0,
) -> Tuple[float, float]:
    """Find binary threshold minimizing financial cost on validation split."""
    best_t, min_cost = 0.5, float("inf")
    for t in np.arange(0.01, 0.99, 0.005):
        pred = (p_val >= t).astype(int)
        cm = confusion_matrix(y_true, pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        cost = fn * cost_fn + fp * cost_fp
        if cost < min_cost:
            min_cost, best_t = cost, t
    return round(float(best_t), 3), round(float(min_cost), 2)


def optimize_3tier_economic_policy(
    y_true: np.ndarray,
    p_val: np.ndarray,
    cost_model: CostModel,
    max_review_rate: float = 0.25,
) -> Tuple[float, float, float]:
    """
    Find optimal (approve_threshold, reject_threshold) on validation split.

    Cost Model:
    - Approved Default: $10,000 (FN)
    - Rejected Non-Default: $1,000 (FP)
    - Manual Review: $150 (Triage Operational Cost)
    """
    n = len(y_true)
    best_app = 0.04
    best_rej = 0.15
    min_cost = float("inf")

    for t_app in np.arange(0.01, 0.12, 0.005):
        for t_rej in np.arange(t_app + 0.02, 0.45, 0.01):
            is_app = p_val <= t_app
            is_rej = p_val >= t_rej
            is_rev = (~is_app) & (~is_rej)

            rev_rate = np.sum(is_rev) / n
            if rev_rate > max_review_rate:
                continue

            fn_cost = np.sum((y_true == 1) & is_app) * cost_model.cost_fn
            fp_cost = np.sum((y_true == 0) & is_rej) * cost_model.cost_fp
            rev_cost = np.sum(is_rev) * cost_model.cost_manual_review

            total_cost = fn_cost + fp_cost + rev_cost
            if total_cost < min_cost:
                min_cost = total_cost
                best_app = t_app
                best_rej = t_rej

    return round(float(best_app), 3), round(float(best_rej), 3), round(float(min_cost), 2)


def optimize_and_freeze_policies(
    pipeline_artifact_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute validation policy optimization and freeze parameters.
    """
    p_path = pipeline_artifact_path or (MODELS_DIR / "lightgbm_calibrated_pipeline.joblib")
    if not p_path.exists():
        raise FileNotFoundError(f"Calibrated pipeline artifact not found at {p_path}")

    logger.info(f"Loading calibrated pipeline from {p_path}...")
    pipeline = joblib.load(p_path)

    X_val, y_val = load_val_data(data_dir)
    y_true_val = y_val.values

    # Predict calibrated default probabilities on validation split
    logger.info("Computing calibrated probabilities on validation split...")
    p_cal_val = pipeline.predict_proba(X_val)[:, 1]

    # 1. F1 Optimal
    t_f1, f1_score_val = optimize_f1(y_true_val, p_cal_val)
    logger.info(f"  [Policy 1] F1-Optimal: threshold = {t_f1:.3f} (Val F1 = {f1_score_val:.4f})")

    # 2. Balanced Accuracy Optimal
    t_bacc, bacc_val = optimize_balanced_acc(y_true_val, p_cal_val)
    logger.info(f"  [Policy 2] Balanced Accuracy: threshold = {t_bacc:.3f} (Val B-Acc = {bacc_val:.4f})")

    # 3. Cost-Sensitive Binary (10:1 Demonstration Loss)
    cost_model_10_1 = CostModel(cost_fn=10000.0, cost_fp=1000.0, cost_manual_review=150.0)
    t_cost_10, min_cost_10 = optimize_cost_binary(
        y_true_val, p_cal_val, cost_fn=cost_model_10_1.cost_fn, cost_fp=cost_model_10_1.cost_fp
    )
    logger.info(f"  [Policy 3] Cost-Sensitive Binary (10:1): threshold = {t_cost_10:.3f}")

    # 4. Cost-Sensitive Binary (5:1 Demonstration Loss)
    t_cost_5, min_cost_5 = optimize_cost_binary(
        y_true_val, p_cal_val, cost_fn=5000.0, cost_fp=1000.0
    )
    logger.info(f"  [Policy 4] Cost-Sensitive Binary (5:1): threshold = {t_cost_5:.3f}")

    # 5. 3-Tier Economic Policy (Approve / Review / Reject)
    t_app, t_rej, val_3tier_cost = optimize_3tier_economic_policy(
        y_true_val, p_cal_val, cost_model=cost_model_10_1, max_review_rate=0.25
    )
    logger.info(
        f"  [Policy 5] 3-Tier Economic: Approve <= {t_app:.3f}, Reject >= {t_rej:.3f} "
        f"(Val Total Cost: ${val_3tier_cost:,.2f})"
    )

    frozen_config = {
        "metadata": {
            "source": "validation_split_optimization",
            "val_sample_size": len(X_val),
            "val_default_prevalence": round(float(y_true_val.mean()), 4),
            "calibrated_pipeline_version": getattr(pipeline, "version", "v3.0-oof-calibrated"),
            "optimized_at": datetime.now().isoformat(),
        },
        "cost_model": {
            "cost_fn": cost_model_10_1.cost_fn,
            "cost_fp": cost_model_10_1.cost_fp,
            "cost_manual_review": cost_model_10_1.cost_manual_review,
            "is_demonstration": True,
        },
        "frozen_thresholds": {
            "f1_optimal": {
                "threshold": t_f1,
                "val_f1": f1_score_val,
            },
            "balanced_accuracy": {
                "threshold": t_bacc,
                "val_balanced_acc": bacc_val,
            },
            "cost_sensitive_10_1": {
                "threshold": t_cost_10,
                "cost_ratio": "10:1",
            },
            "cost_sensitive_5_1": {
                "threshold": t_cost_5,
                "cost_ratio": "5:1",
            },
            "three_tier_economic": {
                "approve_threshold": t_app,
                "reject_threshold": t_rej,
                "val_expected_cost": val_3tier_cost,
                "max_review_rate_constraint": 0.25,
            },
        },
    }

    config_path = MODELS_DIR / "frozen_policy_config.json"
    with open(config_path, "w") as f:
        json.dump(frozen_config, f, indent=2)

    logger.info(f"Saved frozen policy configuration to {config_path}")
    return frozen_config


if __name__ == "__main__":
    optimize_and_freeze_policies()
