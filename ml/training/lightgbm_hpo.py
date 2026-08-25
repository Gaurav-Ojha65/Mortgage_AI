"""
LightGBM Hyperparameter Optimization via Optuna — Mortgage AI
==============================================================
Systematic Bayesian hyperparameter optimization of LightGBM using Optuna.

CRITICAL DATA INTEGRITY RULES:
1. ONLY uses data/real_train.csv (N=99,856). NEVER accesses val.csv or test.csv.
2. In each trial, 5-fold StratifiedKFold cross-validation is performed.
3. SMOTE (ratio 0.5) is applied strictly inside each fold's training partition.
4. Fold validation partitions remain 100% real records at natural 6.76% default prevalence.
5. Primary optimization objective: Mean 5-fold validation ROC-AUC on real OOF predictions.
6. Pruning: MedianPruner active only after at least 3 folds are evaluated.
7. Baseline v3.0.0 hyperparameters are enqueued as Trial 0 for direct control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.inference.predict import MODEL_FEATURES
from ml.training.eval_utils import compute_calibration_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Reduce optuna verbose logs to WARNING/INFO
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = _PROJECT_ROOT / "ml" / "models"
DATA_DIR = _PROJECT_ROOT / "data"
REPORTS_DIR = _PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Baseline v3.0.0 LightGBM hyperparameters (Trial 0)
BASELINE_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 25,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.2,
    "reg_lambda": 2.0,
}


def compute_sha256(filepath: Path) -> str:
    """Compute sha256 hash of a file for provenance tracking."""
    if not filepath.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_sha() -> str:
    """Retrieve current Git commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "uncommitted_or_git_unavailable"


def load_real_train_only(data_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real training data only. Asserts no access to val.csv or test.csv.
    """
    d_dir = data_dir or DATA_DIR
    real_train_path = d_dir / "real_train.csv"

    if not real_train_path.exists():
        raise FileNotFoundError(f"Real train dataset not found at {real_train_path}")

    df = pd.read_csv(real_train_path)
    X = df[MODEL_FEATURES].values
    y = df["target"].values

    logger.info(f"Loaded real training data: N={len(df):,} samples | Defaults={int(y.sum()):,} ({y.mean():.2%})")
    return X, y


class LightGBMObjective:
    """
    Optuna objective function for LightGBM HPO with 5-fold Stratified CV.
    """

    def __init__(
        self,
        X_real: np.ndarray,
        y_real: np.ndarray,
        n_splits: int = 5,
        random_seed: int = 42,
    ):
        self.X_real = X_real
        self.y_real = y_real
        self.n_splits = n_splits
        self.random_seed = random_seed

        # Pre-compute fold splits for exact reproducibility across all trials
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        self.folds = list(skf.split(X_real, y_real))

    def sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters from bounded, search space."""
        n_estimators = trial.suggest_int("n_estimators", 300, 800)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.1, log=True)
        max_depth = trial.suggest_int("max_depth", 4, 10)

        # Ensure num_leaves is mathematically valid for max_depth (<= 2^max_depth - 1)
        max_possible_leaves = min(63, (1 << max_depth) - 1)
        min_leaves = min(15, max_possible_leaves)
        num_leaves = trial.suggest_int("num_leaves", min_leaves, max_possible_leaves)

        min_child_samples = trial.suggest_int("min_child_samples", 10, 80)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        # Log/zero-aware regularization search
        reg_alpha = trial.suggest_float("reg_alpha", 1e-5, 2.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-5, 5.0, log=True)

        params: Dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": self.random_seed,
            "verbose": -1,
            "n_jobs": -1,
        }

        # LightGBM requires subsample_freq > 0 for subsample to take effect
        if subsample < 1.0:
            params["subsample_freq"] = 1
        else:
            params["subsample_freq"] = 0

        return params

    def __call__(self, trial: optuna.Trial) -> float:
        t_start = time.time()
        params = self.sample_hyperparameters(trial)

        n_samples = len(self.X_real)
        oof_probs = np.zeros(n_samples, dtype=np.float64)
        fold_aucs: List[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.folds):
            # 1. Real fold partitions
            X_train_real = self.X_real[train_idx]
            y_train_real = self.y_real[train_idx]
            X_val_real = self.X_real[val_idx]
            y_val_real = self.y_real[val_idx]

            # 2. In-fold SMOTE augmentation on fold training data ONLY
            smote = SMOTE(
                sampling_strategy=0.5,
                random_state=self.random_seed + fold_idx + 1,
                k_neighbors=5,
            )
            X_train_smote, y_train_smote = smote.fit_resample(X_train_real, y_train_real)

            # Convert to DataFrame with feature names for LightGBM
            X_train_df = pd.DataFrame(X_train_smote, columns=MODEL_FEATURES)
            X_val_df = pd.DataFrame(X_val_real, columns=MODEL_FEATURES)

            # 3. Fit LightGBM on augmented fold training partition
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_df, y_train_smote)

            # 4. Predict on held-out REAL fold validation partition
            val_probs = model.predict_proba(X_val_df)[:, 1]
            oof_probs[val_idx] = val_probs

            fold_auc = float(roc_auc_score(y_val_real, val_probs))
            fold_aucs.append(fold_auc)

            # Report intermediate cumulative mean AUC for pruning
            cum_mean_auc = float(np.mean(fold_aucs))
            trial.report(cum_mean_auc, step=fold_idx)

            # Pruning rule: only prune after at least 3 folds are complete (step >= 2)
            if fold_idx >= 2 and trial.should_prune():
                logger.info(
                    f"Trial {trial.number} pruned at fold {fold_idx + 1}/{self.n_splits} "
                    f"(cumulative AUC = {cum_mean_auc:.4f})"
                )
                raise optuna.TrialPruned(
                    f"Pruned at fold {fold_idx + 1} with cumulative mean AUC={cum_mean_auc:.4f}"
                )

        t_elapsed = time.time() - t_start

        # Full trial completed — compute comprehensive validation metrics on real OOF predictions
        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        pr_auc = float(average_precision_score(self.y_real, oof_probs))
        brier = float(brier_score_loss(self.y_real, oof_probs))
        cal_res = compute_calibration_metrics(self.y_real, oof_probs, n_bins=10)

        # Store secondary metrics as trial user attributes
        trial.set_user_attr("mean_roc_auc", round(mean_auc, 5))
        trial.set_user_attr("std_roc_auc", round(std_auc, 5))
        trial.set_user_attr("fold_aucs", [round(a, 5) for a in fold_aucs])
        trial.set_user_attr("pr_auc", round(pr_auc, 5))
        trial.set_user_attr("brier_score", round(brier, 5))
        trial.set_user_attr("weighted_ece", cal_res["ece"])
        trial.set_user_attr("macro_ece", cal_res["macro_ece"])
        trial.set_user_attr("training_time_s", round(t_elapsed, 2))
        trial.set_user_attr("completed_folds", len(fold_aucs))

        return mean_auc


def run_hpo_study(
    n_trials: int = 75,
    random_seed: int = 42,
    study_name: str = "lightgbm_hpo_v3_study",
    db_filename: str = "optuna_study.db",
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """
    Run Optuna hyperparameter optimization study with SQLite persistence.
    """
    logger.info("=" * 70)
    logger.info("  STARTING LIGHTGBM HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    logger.info("=" * 70)

    # 1. Load real training data only
    X_real, y_real = load_real_train_only()

    # 2. Setup SQLite persistence
    db_path = REPORTS_DIR / db_filename
    storage_url = f"sqlite:///{db_path.resolve().as_posix()}"

    # Sampler & Pruner
    sampler = TPESampler(seed=random_seed, multivariate=True)
    # Prune only after at least 3 folds (warmup_steps=3)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=3, interval_steps=1)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    # 3. Enqueue baseline v3.0.0 as Trial 0 if not already present
    if len(study.trials) == 0:
        logger.info("Enqueuing baseline v3.0.0 parameters as Trial 0...")
        study.enqueue_trial(BASELINE_PARAMS)

    # 4. Objective instantiation
    objective = LightGBMObjective(X_real, y_real, n_splits=5, random_seed=random_seed)

    # 5. Execute optimization
    t0_study = time.time()
    logger.info(f"Launching Optuna search ({n_trials} target trials)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    t_study_elapsed = time.time() - t0_study

    logger.info("=" * 70)
    logger.info(f"Optuna HPO completed in {t_study_elapsed:.1f}s ({t_study_elapsed / 60:.1f} min)")
    logger.info(f"Total trials in study: {len(study.trials)}")
    logger.info("=" * 70)

    # 6. Analyze trials & extract summary
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    logger.info(f"Completed trials: {len(completed_trials)}")
    logger.info(f"Pruned trials:    {len(pruned_trials)}")
    logger.info(f"Failed trials:    {len(failed_trials)}")

    # Sort completed trials by value (mean ROC-AUC) descending
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    best_trial = completed_trials[0] if completed_trials else None

    # Identify Trial 0 (Baseline)
    baseline_trial = study.trials[0] if len(study.trials) > 0 else None
    baseline_val = baseline_trial.value if baseline_trial and baseline_trial.value else None

    # Calculate baseline rank
    baseline_rank = None
    if baseline_trial and baseline_trial.state == optuna.trial.TrialState.COMPLETE:
        baseline_rank = completed_trials.index(baseline_trial) + 1

    # Check improvement trajectory across last 20 trials
    last_20 = [t for t in study.trials[-20:] if t.state == optuna.trial.TrialState.COMPLETE]
    trajectory_note = "Optimization stabilized."
    if last_20 and best_trial:
        last_20_best = max(t.value for t in last_20)
        if last_20_best >= best_trial.value - 1e-4:
            trajectory_note = "Recent trials found near-optimal or optimal solutions; search converged smoothly."

    # Build Top 10 Trials list
    top_10_list: List[Dict[str, Any]] = []
    for rank, t in enumerate(completed_trials[:10], 1):
        top_10_list.append({
            "rank": rank,
            "trial_number": t.number,
            "mean_roc_auc": round(float(t.value), 5),
            "std_roc_auc": t.user_attrs.get("std_roc_auc"),
            "pr_auc": t.user_attrs.get("pr_auc"),
            "brier_score": t.user_attrs.get("brier_score"),
            "weighted_ece": t.user_attrs.get("weighted_ece"),
            "macro_ece": t.user_attrs.get("macro_ece"),
            "training_time_s": t.user_attrs.get("training_time_s"),
            "params": t.params,
        })

    # Prepare structured JSON report
    real_train_path = DATA_DIR / "real_train.csv"
    results_report = {
        "metadata": {
            "study_name": study_name,
            "study_database": str(db_path.resolve()),
            "git_commit_sha": get_git_commit_sha(),
            "environment": {
                "python_version": platform.python_version(),
                "os": platform.platform(),
                "optuna_version": optuna.__version__,
                "lightgbm_version": lgb.__version__,
                "joblib_version": joblib.__version__,
            },
            "data_provenance": {
                "real_train_csv_sha256": compute_sha256(real_train_path),
                "feature_schema_sha256": hashlib.sha256(",".join(MODEL_FEATURES).encode("utf-8")).hexdigest(),
                "sample_size": len(X_real),
                "positive_prevalence": round(float(y_real.mean()), 4),
            },
            "hpo_configuration": {
                "n_trials_requested": n_trials,
                "n_trials_total": len(study.trials),
                "random_seed": random_seed,
                "sampler": "TPESampler(multivariate=True)",
                "pruner": "MedianPruner(n_startup_trials=10, n_warmup_steps=3, interval_steps=1)",
                "pruning_rule": "Prune trial if cumulative mean ROC-AUC after fold 3, 4 is below median of completed trials at same step.",
                "search_space_definition": {
                    "n_estimators": "[300, 800] integer",
                    "learning_rate": "[0.005, 0.1] log-uniform",
                    "max_depth": "[4, 10] integer",
                    "num_leaves": "[15, min(63, 2^max_depth - 1)] integer",
                    "min_child_samples": "[10, 80] integer",
                    "subsample": "[0.6, 1.0] uniform (subsample_freq=1 when < 1.0)",
                    "colsample_bytree": "[0.5, 1.0] uniform",
                    "reg_alpha": "[1e-5, 2.0] log-uniform",
                    "reg_lambda": "[1e-5, 5.0] log-uniform",
                },
            },
            "created_at": datetime.now().isoformat(),
        },
        "study_summary": {
            "total_trials": len(study.trials),
            "completed_trials_count": len(completed_trials),
            "pruned_trials_count": len(pruned_trials),
            "failed_trials_count": len(failed_trials),
            "total_runtime_seconds": round(t_study_elapsed, 2),
            "trajectory_note": trajectory_note,
        },
        "baseline_trial_0": {
            "trial_number": 0,
            "params": BASELINE_PARAMS,
            "mean_roc_auc": round(float(baseline_val), 5) if baseline_val else None,
            "std_roc_auc": baseline_trial.user_attrs.get("std_roc_auc") if baseline_trial else None,
            "pr_auc": baseline_trial.user_attrs.get("pr_auc") if baseline_trial else None,
            "brier_score": baseline_trial.user_attrs.get("brier_score") if baseline_trial else None,
            "weighted_ece": baseline_trial.user_attrs.get("weighted_ece") if baseline_trial else None,
            "rank_in_study": baseline_rank,
        },
        "best_trial": {
            "trial_number": best_trial.number if best_trial else None,
            "mean_roc_auc": round(float(best_trial.value), 5) if best_trial else None,
            "std_roc_auc": best_trial.user_attrs.get("std_roc_auc") if best_trial else None,
            "pr_auc": best_trial.user_attrs.get("pr_auc") if best_trial else None,
            "brier_score": best_trial.user_attrs.get("brier_score") if best_trial else None,
            "weighted_ece": best_trial.user_attrs.get("weighted_ece") if best_trial else None,
            "macro_ece": best_trial.user_attrs.get("macro_ece") if best_trial else None,
            "training_time_s": best_trial.user_attrs.get("training_time_s") if best_trial else None,
            "params": best_trial.params if best_trial else None,
            "roc_auc_delta_vs_baseline": round(float(best_trial.value - (baseline_val or 0.0)), 5) if best_trial and baseline_val else None,
        },
        "top_10_trials": top_10_list,
    }

    out_json_path = METRICS_DIR / "optuna_results.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results_report, f, indent=2)

    logger.info(f"Saved Optuna results report to {out_json_path}")
    return study, results_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LightGBM HPO using Optuna")
    parser.add_argument("--trials", type=int, default=75, help="Number of trials to run (default: 75)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    run_hpo_study(n_trials=args.trials, random_seed=args.seed)
