"""
ML Model Evaluation Pipeline — Mortgage Risk Analytics
=======================================================
Evaluates all trained models (LogisticRegression, XGBoost, LightGBM, Ensemble)
on a held-out test set and generates a comprehensive metrics report.

Metrics reported:
  - Accuracy, Precision, Recall, F1, Specificity
  - ROC-AUC, PR-AUC
  - Brier Score (measures probability calibration quality)
  - Cross-validation AUC (mean ± std)
  - Confusion Matrix
  - Calibration curve (reliability diagram)

Usage:
    python -m ml.training.evaluate
    python -m ml.training.evaluate --seed 42
    python -m ml.training.evaluate --plot      # Save calibration/ROC plots
    python -m ml.training.evaluate --verbose

NOTE:
  This evaluator relies on the same data split logic as train.py.
  It re-loads the saved .joblib model artifacts from ml/models/
  and evaluates against a freshly generated test split, using seed=42 by default
  to ensure results are deterministic and reproducible.
"""

import sys
import json
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

# ─── Project root setup ───────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))  # project root

from ml.inference.predict import MODEL_FEATURES, AVAILABLE_MODELS, MODELS_DIR
from ml.training.train import load_data
# Shared canonical CV & Calibration utilities — identical across all evaluation scripts
from ml.training.eval_utils import compute_cv_auc, compute_calibration_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = _ROOT.parent / "reports" / "metrics"
PLOTS_DIR = _ROOT.parent / "reports" / "plots"


# ─── Metric helpers ───────────────────────────────────────────────────────────

def specificity(y_true, y_pred):
    """True Negative Rate = TN / (TN + FP)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_all_metrics(
    model, X_test, y_test, threshold: float = 0.5, model_name: str = "model"
) -> dict:
    """
    Compute the full suite of binary classification metrics.

    Args:
        model: Fitted scikit-learn compatible classifier.
        X_test: Test features (DataFrame or ndarray).
        y_test: Test labels.
        threshold: Decision threshold for binary classification.
        model_name: Name for logging.

    Returns:
        Dictionary of metric name → value.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "precision":   round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":      round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":          round(f1_score(y_test, y_pred, zero_division=0), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
        "roc_auc":     round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc":      round(average_precision_score(y_test, y_prob), 4),
        "brier_score": round(brier_score_loss(y_test, y_prob), 4),
        "threshold":   threshold,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "support": {
            "total": int(len(y_test)),
            "positives": int(y_test.sum()),
            "default_rate": round(float(y_test.mean()), 4),
        },
    }
    return metrics


# compute_cv_auc is imported from ml.training.eval_utils
# (shared with train.py to guarantee identical protocol)


def calibration_metrics(model, X_test, y_test, n_bins: int = 10) -> dict:
    """
    Compute calibration quality metrics via canonical eval_utils.

    Returns:
        fraction_of_positives, mean_predicted_value, Expected Calibration Error (ECE),
        macro_ece, and per-bin summary breakdown.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    return compute_calibration_metrics(y_test, y_prob, n_bins=n_bins)


# ─── Optimal threshold search ─────────────────────────────────────────────────

def find_optimal_threshold(model, X_val, y_val) -> float:
    """Find threshold maximizing F1 on validation set."""
    y_prob_val = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.80, 0.01):
        f = f1_score(y_val, y_prob_val >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    return round(float(best_thresh), 3)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def save_calibration_plot(results: dict, save_dir: Path):
    """Save calibration (reliability) diagram for all models."""
    try:
        import matplotlib.pyplot as plt

        save_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

        for model_name, data in results.items():
            cal = data.get("calibration", {})
            if "mean_predicted_value" not in cal:
                continue
            ax.plot(
                cal["mean_predicted_value"],
                cal["fraction_of_positives"],
                "o-",
                label=f"{model_name} (ECE={cal['ece']:.3f})",
            )

        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title("Calibration Curves — Mortgage Risk Models", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = save_dir / "calibration_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved calibration plot → {path}")
    except ImportError:
        logger.warning("matplotlib not available — skipping calibration plot.")


def save_roc_plot(roc_data: dict, save_dir: Path):
    """Save ROC curves for all models."""
    try:
        import matplotlib.pyplot as plt

        save_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Random")

        for model_name, (fpr, tpr, auc_val) in roc_data.items():
            ax.plot(fpr, tpr, lw=2, label=f"{model_name} AUC={auc_val:.3f}")

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — Mortgage Risk Models", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = save_dir / "roc_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved ROC plot → {path}")
    except ImportError:
        logger.warning("matplotlib not available — skipping ROC plot.")


# ─── Main evaluation runner ───────────────────────────────────────────────────

def run_evaluation(seed: int = 42, save_plots: bool = False, verbose: bool = False):
    """
    Load all trained models and evaluate them on a reproducible test split.

    Uses the same seed and data loading logic as train.py to ensure
    the test split is identical to the one used during training.

    Args:
        seed: Random seed for reproducibility (must match training seed).
        save_plots: Whether to save calibration and ROC curve plots.
        verbose: Print full classification report per model.

    Returns:
        dict: Evaluation results per model.
    """
    logger.info("=" * 62)
    logger.info("  MORTGAGE AI — MODEL EVALUATION PIPELINE")
    logger.info(f"  Seed: {seed}  |  Models dir: {MODELS_DIR}")
    logger.info("=" * 62)

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading data splits...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(seed=seed)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    X_cv = pd.concat([X_train, X_val], axis=0) if isinstance(X_train, pd.DataFrame) else np.vstack([X_train, X_val])
    y_cv = pd.concat([y_train, y_val], axis=0) if isinstance(y_train, pd.Series) else np.concatenate([y_train, y_val])

    logger.info(f"  Test set: {len(X_test):,} samples | Default rate: {float(y_test.mean()):.1%}")

    # ── Check available model files ────────────────────────────────────────────
    available = []
    for name in AVAILABLE_MODELS:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            available.append(name)
        else:
            logger.warning(f"  [{name}] NOT FOUND — skipping. Run: python -m ml.training.train")

    if not available:
        logger.error("No trained models found. Run: python -m ml.training.train")
        sys.exit(1)

    results = {}
    roc_data = {}

    # ── Evaluate each model ────────────────────────────────────────────────────
    for model_name in available:
        logger.info(f"\n  Evaluating: {model_name.upper()}")
        path = MODELS_DIR / f"{model_name}.joblib"
        model = joblib.load(path)

        # Optimal threshold (find on val, apply to test)
        threshold = find_optimal_threshold(model, X_val, y_val)
        logger.info(f"    Optimal threshold (F1 on val): {threshold:.3f}")

        # Full metrics on test set
        metrics = compute_all_metrics(model, X_test, y_test, threshold=threshold, model_name=model_name)

        # Cross-validation AUC on combined train+val
        # allow_ensemble_cv=True: evaluate.py always runs full CV, including Ensemble nested CV
        try:
            cv_metrics = compute_cv_auc(
                model, model_name, X_cv, y_cv,
                n_splits=5, seed=seed,
                allow_ensemble_cv=True,
            )
            metrics.update(cv_metrics)
        except Exception as e:
            logger.warning(f"    CV AUC failed for {model_name}: {e}")

        # Calibration metrics
        try:
            cal_metrics = calibration_metrics(model, X_test, y_test)
            metrics["calibration"] = cal_metrics
        except Exception as e:
            logger.warning(f"    Calibration metrics failed for {model_name}: {e}")

        # ROC data for plotting
        try:
            from sklearn.metrics import roc_curve
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[model_name] = (fpr.tolist(), tpr.tolist(), metrics["roc_auc"])
        except Exception:
            pass

        results[model_name] = metrics

        # Print summary
        m = metrics
        logger.info(f"    Accuracy    : {m['accuracy']:.4f}")
        logger.info(f"    Precision   : {m['precision']:.4f}")
        logger.info(f"    Recall      : {m['recall']:.4f}")
        logger.info(f"    F1          : {m['f1']:.4f}")
        logger.info(f"    Specificity : {m['specificity']:.4f}")
        logger.info(f"    ROC-AUC     : {m['roc_auc']:.4f}")
        logger.info(f"    PR-AUC      : {m['pr_auc']:.4f}")
        logger.info(f"    Brier Score : {m['brier_score']:.4f}  (lower = better calibration)")
        if "cv_auc_mean" in m:
            logger.info(f"    CV-AUC      : {m['cv_auc_mean']:.4f} ± {m['cv_auc_std']:.4f}")
        if "calibration" in m:
            logger.info(f"    ECE         : {m['calibration']['ece']:.4f}")

        if verbose:
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
            print(classification_report(y_test, y_pred, target_names=["Non-Default", "Default"]))

    # ── Comparison summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 62)
    logger.info("  MODEL COMPARISON SUMMARY")
    logger.info("=" * 62)
    header = f"{'Model':<22} {'AUC':>7} {'PR-AUC':>8} {'Brier':>7} {'F1':>7} {'Recall':>7}"
    logger.info(header)
    logger.info("-" * len(header))
    for name, m in sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True):
        logger.info(
            f"{name:<22} {m['roc_auc']:>7.4f} {m['pr_auc']:>8.4f} "
            f"{m['brier_score']:>7.4f} {m['f1']:>7.4f} {m['recall']:>7.4f}"
        )

    # ── Save report ────────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_model = max(results, key=lambda k: results[k]["roc_auc"])
    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "seed": seed,
        "dataset": {
            "test_size": int(len(X_test)),
            "default_rate": round(float(y_test.mean()), 4),
        },
        "models": results,
        "best_model_by_roc_auc": best_model,
        "methodology_note": (
            "Metrics are computed on a held-out test set (20% of total data). "
            "CV-AUC is computed via 5-fold StratifiedKFold on train+val combined. "
            "Optimal threshold is selected by maximizing F1 on the validation set, "
            "then applied to the test set. All results are reproducible with the specified seed."
        ),
    }

    report_path = REPORTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\n  Full evaluation report → {report_path}")

    # ── Save plots ─────────────────────────────────────────────────────────────
    if save_plots:
        save_calibration_plot(results, PLOTS_DIR)
        save_roc_plot(roc_data, PLOTS_DIR)

    logger.info("=" * 62)
    logger.info(f"  Best model (by ROC-AUC): {best_model.upper()}")
    logger.info("=" * 62)

    return results


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all trained Mortgage AI models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (must match training seed for reproducibility).")
    parser.add_argument("--plot", action="store_true", help="Save calibration and ROC curve plots.")
    parser.add_argument("--verbose", action="store_true", help="Print full classification report per model.")
    args = parser.parse_args()
    run_evaluation(seed=args.seed, save_plots=args.plot, verbose=args.verbose)