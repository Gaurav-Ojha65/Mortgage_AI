"""
Shared Evaluation Utilities — Mortgage AI
==========================================
Single source of truth for all cross-validation logic.
Imported by BOTH train.py and evaluate.py to guarantee identical evaluation protocols.

Design decision on Ensemble CV
-------------------------------
Fitting MortgageEnsembleModel inside cross_val_score triggers a nested CV:
each outer fold calls ensemble.fit() which internally runs a 5-fold
cross_val_predict on both XGBoost and LightGBM (≈ 10 base-model fits per outer fold).
With 5 outer folds this totals ≈ 50 base-model fits + 5 meta-learner fits ≈ 3-5 minutes.

  train.py    → pass allow_ensemble_cv=False  (skip during training, use test AUC)
  evaluate.py → pass allow_ensemble_cv=True   (always run, post-training report)

Same function. Same code path. Only the flag differs.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── CV-safe estimator factory ────────────────────────────────────────────────


def make_cv_estimator(model_name: str, model):
    """
    Return a cross-validation-safe estimator for the given model.

    XGBoost and LightGBM trained with ``early_stopping_rounds`` cannot be
    passed to ``cross_val_score`` directly — sklearn calls ``.fit()`` without
    an ``eval_set``, which raises ValueError.  This factory creates **fresh**
    instances with:

    * ``n_estimators`` = ``best_iteration`` from the fitted model
      (preserves effective tree count without early-stopping callback).
    * ``early_stopping_rounds`` = None (removed entirely).

    For LogReg pipelines and Ensemble the original object is returned unchanged.

    Args:
        model_name: Lowercase model identifier
                    (``'xgboost'``, ``'lightgbm'``, ``'logisticregression'``, ``'ensemble'``).
        model:      Fitted model or pipeline.

    Returns:
        Estimator safe for use with ``cross_val_score`` without an eval_set.
    """
    name = model_name.lower().replace(" ", "")

    if name == "xgboost":
        import xgboost as xgb

        params = model.get_params()
        n = getattr(model, "best_iteration", None) or params.get("n_estimators", 200)
        # Build a clean parameter set — drop early-stopping and verbosity keys
        drop = {"n_estimators", "early_stopping_rounds", "eval_metric", "verbosity", "n_jobs"}
        safe = {k: v for k, v in params.items() if k not in drop and v is not None}
        return xgb.XGBClassifier(
            n_estimators=n,
            verbosity=0,
            n_jobs=-1,
            **safe,
        )

    elif name == "lightgbm":
        import lightgbm as lgb

        params = model.get_params()
        n = getattr(model, "best_iteration_", None) or params.get("n_estimators", 200)
        drop = {"n_estimators", "early_stopping_rounds", "verbose", "n_jobs"}
        safe = {k: v for k, v in params.items() if k not in drop and v is not None}
        return lgb.LGBMClassifier(
            n_estimators=n,
            verbose=-1,
            n_jobs=-1,
            **safe,
        )

    else:
        # LogReg Pipeline, Ensemble: use the object directly
        return model


# ─── Canonical cross-validation AUC ──────────────────────────────────────────


def compute_cv_auc(
    model,
    model_name: str,
    X,
    y,
    n_splits: int = 5,
    seed: int = 42,
    allow_ensemble_cv: bool = True,
) -> dict:
    """
    5-fold stratified cross-validation AUC — **canonical shared implementation**.

    Both ``train.py`` and ``evaluate.py`` call this function to guarantee
    identical evaluation protocols.  The only behavioural difference between
    the two callers is the ``allow_ensemble_cv`` flag:

    +--------------+----------------------+
    | Caller       | allow_ensemble_cv    |
    +==============+======================+
    | train.py     | False  (fast pass)   |
    +--------------+----------------------+
    | evaluate.py  | True   (always run)  |
    +--------------+----------------------+

    **XGB / LGB**: ``make_cv_estimator`` creates fresh instances without
    ``early_stopping_rounds`` (uses best iteration count as fixed ``n_estimators``).

    **Ensemble**: Full ``cross_val_score`` on the ``MortgageEnsembleModel``
    object.  Each outer fold triggers an inner nested CV inside ``fit()``.
    ``n_jobs=1`` is used to prevent nested parallelism deadlocks on Windows.

    **LogReg**: direct ``cross_val_score`` on the sklearn Pipeline.

    Args:
        model:              Fitted model or pipeline.
        model_name:         Lowercase model identifier.
        X:                  Features — should be **train+val combined**, never test.
        y:                  Target labels.
        n_splits:           CV folds (default 5).
        seed:               Random seed for ``StratifiedKFold``.
        allow_ensemble_cv:  If ``False`` and model_name is ``'ensemble'``,
                            return NaN with an explanatory note.

    Returns:
        Dict with keys ``cv_auc_mean``, ``cv_auc_std``, ``cv_n_splits``,
        and optional ``cv_note`` when skipped or failed.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    name = model_name.lower().replace(" ", "")

    # ── Ensemble fast-pass ────────────────────────────────────────────────────
    if name == "ensemble" and not allow_ensemble_cv:
        logger.info(
            "  [Ensemble] CV-AUC deferred to evaluate.py "
            "(nested 5-fold during training adds ~3-5 min)."
        )
        return {
            "cv_auc_mean": float("nan"),
            "cv_auc_std": float("nan"),
            "cv_n_splits": n_splits,
            "cv_note": (
                "Skipped in training workflow (nested CV, ~3-5 min overhead). "
                "Computed by evaluate.py post-training."
            ),
        }

    # ── Build a CV-safe estimator ─────────────────────────────────────────────
    cv_estimator = make_cv_estimator(model_name, model)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Use n_jobs=1 for Ensemble to avoid nested parallelism (Windows safe)
    n_jobs = 1 if name == "ensemble" else -1

    try:
        scores = cross_val_score(
            cv_estimator, X, y, cv=skf, scoring="roc_auc", n_jobs=n_jobs
        )
        return {
            "cv_auc_mean": round(float(scores.mean()), 4),
            "cv_auc_std": round(float(scores.std()), 4),
            "cv_n_splits": n_splits,
        }
    except Exception as exc:
        logger.warning(f"  [{model_name}] CV-AUC failed: {exc}")
        return {
            "cv_auc_mean": float("nan"),
            "cv_auc_std": float("nan"),
            "cv_n_splits": n_splits,
            "cv_note": f"Failed: {exc}",
        }


# ─── Canonical Calibration & ECE implementation ──────────────────────────────


def compute_calibration_metrics(
    y_true,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute canonical calibration metrics on predicted probabilities.

    Standard Expected Calibration Error (ECE) (Naeini et al., 2015; Guo et al., 2017)
    is computed as the sample-weighted average across all bins:

        ECE = sum_{b=1}^{B} ( |B_b| / N ) * | p_mean(B_b) - o_mean(B_b) |

    Where:
        - B_b is the subset of samples assigned to bin b
        - N is the total number of test samples
        - p_mean(B_b) is the mean predicted confidence in bin b
        - o_mean(B_b) is the empirical default rate in bin b

    Also returns:
        - macro_ece: Unweighted arithmetic mean across non-empty bins (legacy reference)
        - n_non_empty_bins: Number of bins with at least 1 sample
        - bin_details: Full per-bin summary (range, sample count, percentage, p_mean, o_mean, diff)
        - fraction_of_positives: Non-empty empirical positive rates for plotting
        - mean_predicted_value: Non-empty predicted confidence averages for plotting

    Args:
        y_true: True binary target array.
        y_prob: Predicted probability array for class 1.
        n_bins: Number of equal-width calibration bins (default 10).

    Returns:
        Dictionary containing ece, macro_ece, bin_details, fraction_of_positives,
        and mean_predicted_value.
    """
    from sklearn.calibration import calibration_curve

    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    n = len(y_true_arr)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob_arr, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    weighted_ece = 0.0
    bin_details = []
    active_p_means = []
    active_o_means = []

    for b in range(n_bins):
        mask = bin_assignments == b
        count = int(np.sum(mask))
        lo, hi = bin_edges[b], bin_edges[b + 1]
        bin_label = f"[{lo:.1f}, {hi:.1f})" if b < n_bins - 1 else f"[{lo:.1f}, {hi:.1f}]"

        if count > 0:
            p_m = float(np.mean(y_prob_arr[mask]))
            o_m = float(np.mean(y_true_arr[mask]))
            diff = abs(p_m - o_m)
            weighted_ece += (count / n) * diff
            active_p_means.append(round(p_m, 4))
            active_o_means.append(round(o_m, 4))

            bin_details.append({
                "bin": b + 1,
                "range": bin_label,
                "count": count,
                "pct": round(count / n, 4),
                "p_mean": round(p_m, 4),
                "o_mean": round(o_m, 4),
                "abs_diff": round(diff, 4),
            })
        else:
            bin_details.append({
                "bin": b + 1,
                "range": bin_label,
                "count": 0,
                "pct": 0.0,
                "p_mean": None,
                "o_mean": None,
                "abs_diff": None,
            })

    macro_ece = float(np.mean([b["abs_diff"] for b in bin_details if b["abs_diff"] is not None])) if active_p_means else 0.0

    return {
        "ece": round(weighted_ece, 4),
        "macro_ece": round(macro_ece, 4),
        "n_bins": n_bins,
        "n_non_empty_bins": len(active_p_means),
        "fraction_of_positives": active_o_means,
        "mean_predicted_value": active_p_means,
        "bin_details": bin_details,
        "calibration_note": (
            "ECE (Expected Calibration Error) is the standard sample-weighted average across all bins. "
            "Lower is better; 0.0 = perfect calibration."
        ),
    }
