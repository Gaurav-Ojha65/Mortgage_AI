"""
Promote v3.1 HPO Candidate to Canonical Baseline — Mortgage AI
===============================================================
Autonomously archives v3.0.0 artifacts and promotes v3.1 HPO candidate
to canonical production baseline.
"""

import os
import shutil
import json
import hashlib
import platform
import subprocess
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
ARCHIVE_DIR = MODELS_DIR / "archive" / "v3.0.0"
DATA_DIR = PROJECT_ROOT / "data"

def compute_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "uncommitted_or_git_unavailable"

def main():
    print("=" * 70)
    print("STEP 1: ARCHIVING v3.0.0 BASELINE ARTIFACTS")
    print("=" * 70)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    files_to_archive = [
        "lightgbm.joblib",
        "lightgbm_calibrated_pipeline.joblib",
        "lightgbm_calibrated_isotonic.joblib",
        "lightgbm_calibrated_sigmoid.joblib",
        "lightgbm_oof_calibrator_isotonic.joblib",
        "lightgbm_oof_calibrator_sigmoid.joblib",
        "frozen_policy_config.json",
        "training_metadata.json",
        "calibration_metadata.json",
    ]
    
    for filename in files_to_archive:
        src = MODELS_DIR / filename
        dst = ARCHIVE_DIR / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Archived: {src.name} -> {dst}")
        else:
            print(f"  Warning: {src.name} not found to archive.")
            
    # Write Version History
    version_history_md = """# Model & Policy Version History

## Active Canonical Baseline: v3.1
- **Model Version:** `v3.1` (HPO-optimized LightGBM, Trial #47)
- **Calibration Version:** `oof-iso-v3.1` (5-Fold Stratified OOF Isotonic Regression)
- **Policy Version:** `v3.1-policy-v1` (Approve <= 0.045, Reject >= 0.335, Review Rate: 24.09%)
- **Test Metrics (Untouched Test Set N=21,398):**
  - ROC-AUC: 0.8615
  - PR-AUC: 0.3995
  - Brier Score: 0.0492
  - Weighted ECE: 0.0012
  - Macro ECE: 0.0129
  - Expected Economic Cost: $4,062,100 ($189.84 / applicant)
- **Bootstrap Uncertainty (95% CI vs v3.0.0):**
  - Δ ROC-AUC: [+0.0002, +0.0032] (Statistically distinguishable)
  - Δ Brier: [-0.0005, -0.0001] (Statistically distinguishable)
  - Δ PR-AUC: [-0.0011, +0.0104] (Includes zero)
  - Δ Total Cost: [-$153,881, +$109,024] (Includes zero — cost improvement not statistically distinguishable)

---

## Historical Archived Baselines

### v3.0.0 (Archived in `ml/models/archive/v3.0.0/`)
- **Model Version:** `v3.0.0-oof-baseline`
- **Calibration Version:** `oof-iso-v3.0`
- **Policy Version:** `frozen_oof_3tier_policy` (Approve <= 0.055, Reject >= 0.405, Review Rate: 24.61%)
- **Hyperparameters:** `n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=31`
- **Test Metrics:**
  - ROC-AUC: 0.8599 | PR-AUC: 0.3947 | Brier: 0.0494 | wECE: 0.0018 | Expected Cost: $4,082,900
"""
    history_file = MODELS_DIR / "archive" / "VERSION_HISTORY.md"
    history_file.write_text(version_history_md, encoding="utf-8")
    print(f"  Created version history at: {history_file}")

    print("\n" + "=" * 70)
    print("STEP 2: PROMOTING v3.1 ARTIFACTS TO CANONICAL LOCATIONS")
    print("=" * 70)
    
    # 1. Load candidate model and calibrators
    cand_model_path = MODELS_DIR / "lightgbm_hpo_candidate.joblib"
    cand_iso_path = MODELS_DIR / "lightgbm_hpo_candidate_oof_calibrator_isotonic.joblib"
    cand_sig_path = MODELS_DIR / "lightgbm_hpo_candidate_oof_calibrator_sigmoid.joblib"
    
    cand_model = joblib.load(cand_model_path)
    cand_iso = joblib.load(cand_iso_path)
    cand_sig = joblib.load(cand_sig_path)
    
    # 2. Promote raw base model to lightgbm.joblib and best_model.joblib
    joblib.dump(cand_model, MODELS_DIR / "lightgbm.joblib")
    joblib.dump(cand_model, MODELS_DIR / "best_model.joblib")
    (MODELS_DIR / "best_model_name.txt").write_text("lightgbm", encoding="utf-8")
    print("  Promoted lightgbm_hpo_candidate.joblib -> lightgbm.joblib & best_model.joblib")
    
    # 3. Promote calibrators to canonical names
    joblib.dump(cand_iso, MODELS_DIR / "lightgbm_oof_calibrator_isotonic.joblib")
    joblib.dump(cand_sig, MODELS_DIR / "lightgbm_oof_calibrator_sigmoid.joblib")
    print("  Promoted calibrators -> lightgbm_oof_calibrator_isotonic.joblib & lightgbm_oof_calibrator_sigmoid.joblib")
    
    # 4. Construct and save promoted CalibratedPredictor pipeline
    from ml.training.calibrated_predictor import CalibratedPredictor
    promoted_pipeline = CalibratedPredictor(
        base_model=cand_model,
        calibrator=cand_iso,
        calibration_method="isotonic",
        model_name="lightgbm",
        version="v3.1-oof-calibrated",
    )
    joblib.dump(promoted_pipeline, MODELS_DIR / "lightgbm_calibrated_pipeline.joblib")
    joblib.dump(promoted_pipeline, MODELS_DIR / "lightgbm_calibrated_isotonic.joblib")
    print("  Created & saved canonical CalibratedPredictor -> lightgbm_calibrated_pipeline.joblib & lightgbm_calibrated_isotonic.joblib")

    print("\n" + "=" * 70)
    print("STEP 3: UPDATING CANONICAL FROZEN POLICY & PROVENANCE METADATA")
    print("=" * 70)
    
    from ml.inference.predict import MODEL_FEATURES
    
    # Load HPO metadata for exact trial 47 details
    hpo_meta = json.loads((MODELS_DIR / "hpo_metadata.json").read_text(encoding="utf-8"))
    
    git_sha = get_git_commit_sha()
    real_train_hash = compute_sha256(DATA_DIR / "real_train.csv")
    train_hash = compute_sha256(DATA_DIR / "train.csv")
    val_hash = compute_sha256(DATA_DIR / "val.csv")
    test_hash = compute_sha256(DATA_DIR / "test.csv")
    schema_hash = hashlib.sha256(",".join(MODEL_FEATURES).encode("utf-8")).hexdigest()
    now_iso = datetime.now().isoformat()
    
    # 1. Update frozen_policy_config.json
    frozen_policy_config = {
        "metadata": {
            "source": "validation_split_optimization",
            "model_version": "v3.1",
            "calibration_version": "oof-iso-v3.1",
            "policy_version": "v3.1-policy-v1",
            "val_sample_size": 21398,
            "val_default_prevalence": 0.0676,
            "optimized_at": now_iso,
            "git_commit_sha": git_sha,
            "training_data_sha256": real_train_hash,
            "feature_schema_sha256": schema_hash,
            "optuna_trial_number": 47,
            "random_seed": 42
        },
        "cost_model": {
            "cost_fn": 10000.0,
            "cost_fp": 1000.0,
            "cost_manual_review": 150.0,
            "is_demonstration": True
        },
        "frozen_thresholds": {
            "three_tier_economic": {
                "approve_threshold": 0.045,
                "reject_threshold": 0.335,
                "val_expected_cost": 3975900.0,
                "max_review_rate_constraint": 0.25
            },
            "f1_optimal": {
                "threshold": 0.22,
                "val_f1": 0.4449
            },
            "balanced_accuracy": {
                "threshold": 0.075,
                "val_balanced_acc": 0.783
            },
            "cost_sensitive_10_1": {
                "threshold": 0.085,
                "cost_ratio": "10:1"
            },
            "cost_sensitive_5_1": {
                "threshold": 0.185,
                "cost_ratio": "5:1"
            }
        }
    }
    (MODELS_DIR / "frozen_policy_config.json").write_text(json.dumps(frozen_policy_config, indent=2), encoding="utf-8")
    print("  Updated: ml/models/frozen_policy_config.json")
    
    # 2. Update training_metadata.json
    training_metadata = {
        "model_name": "LightGBM",
        "model_version": "v3.1",
        "calibration_version": "oof-iso-v3.1",
        "policy_version": "v3.1-policy-v1",
        "methodology": "5-Fold Stratified OOF Isotonic Calibration on Real Training Split (HPO-tuned via Optuna Trial 47)",
        "git_commit_sha": git_sha,
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "lightgbm_version": lgb.__version__,
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__
        },
        "dataset_provenance": {
            "real_train_csv_sha256": real_train_hash,
            "train_csv_sha256": train_hash,
            "val_csv_sha256": val_hash,
            "test_csv_sha256": test_hash,
            "feature_schema_sha256": schema_hash,
            "real_train_samples": 99856,
            "smote_train_samples": 139659,
            "val_samples": 21398,
            "test_samples": 21398,
            "natural_default_prevalence": 0.0676
        },
        "hpo_configuration": {
            "optuna_trial_number": 47,
            "random_seed": 42,
            "hyperparameters": hpo_meta["hyperparameters"],
            "depth_leaf_constraint": "num_leaves <= 2^(max_depth) enforced (20 <= 1024)",
            "feature_names": MODEL_FEATURES
        },
        "measured_test_metrics": hpo_meta["measured_metrics"]["test_metrics"],
        "frozen_policy": hpo_meta["measured_metrics"]["frozen_policy"],
        "statistical_uncertainty_vs_v3_baseline": {
            "bootstrap_samples": 1000,
            "roc_auc_delta_95_ci": [0.0002, 0.0032],
            "roc_auc_statistically_distinguishable": True,
            "brier_delta_95_ci": [-0.0005, -0.0001],
            "brier_statistically_distinguishable": True,
            "pr_auc_delta_95_ci": [-0.0011, 0.0104],
            "pr_auc_statistically_distinguishable": False,
            "total_cost_delta_95_ci": [-153881.0, 109024.0],
            "total_cost_statistically_distinguishable": False,
            "note": "Cost improvement observed on single test split is NOT statistically distinguishable under paired bootstrap resampling."
        },
        "saved_artifacts": {
            "raw_base_model": "ml/models/lightgbm.joblib",
            "oof_calibrator_isotonic": "ml/models/lightgbm_oof_calibrator_isotonic.joblib",
            "oof_calibrator_sigmoid": "ml/models/lightgbm_oof_calibrator_sigmoid.joblib",
            "calibrated_pipeline": "ml/models/lightgbm_calibrated_pipeline.joblib",
            "frozen_policy_config": "ml/models/frozen_policy_config.json"
        },
        "created_at": now_iso
    }
    (MODELS_DIR / "training_metadata.json").write_text(json.dumps(training_metadata, indent=2), encoding="utf-8")
    print("  Updated: ml/models/training_metadata.json")
    
    # 3. Update calibration_metadata.json
    calibration_metadata = {
        "model_name": "LightGBM",
        "model_version": "v3.1",
        "calibration_version": "oof-iso-v3.1",
        "policy_version": "v3.1-policy-v1",
        "methodology": "5-Fold Stratified OOF Calibration on Real Training Split",
        "git_commit_sha": git_sha,
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "lightgbm_version": lgb.__version__,
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__
        },
        "dataset_provenance": {
            "real_train_csv_sha256": real_train_hash,
            "feature_schema_sha256": schema_hash,
            "real_train_samples": 99856,
            "smote_train_samples": 139659,
            "val_samples": 21398,
            "test_samples": 21398,
            "natural_default_prevalence": 0.0676
        },
        "oof_configuration": {
            "n_splits": 5,
            "random_seed": 42,
            "optuna_trial_number": 47,
            "fold_balancing": "SMOTE(0.5) applied strictly to fold-training partition",
            "lightgbm_params": hpo_meta["hyperparameters"],
            "feature_names": MODEL_FEATURES
        },
        "calibrator_metrics": {
            "isotonic": {
                "test_roc_auc": 0.8615,
                "test_pr_auc": 0.3995,
                "test_brier": 0.0492,
                "test_weighted_ece": 0.0012,
                "test_macro_ece": 0.0129
            }
        },
        "saved_artifacts": {
            "raw_base_model": "ml/models/lightgbm.joblib",
            "oof_calibrator_isotonic": "ml/models/lightgbm_oof_calibrator_isotonic.joblib",
            "oof_calibrator_sigmoid": "ml/models/lightgbm_oof_calibrator_sigmoid.joblib",
            "calibrated_pipeline": "ml/models/lightgbm_calibrated_pipeline.joblib"
        },
        "created_at": now_iso
    }
    (MODELS_DIR / "calibration_metadata.json").write_text(json.dumps(calibration_metadata, indent=2), encoding="utf-8")
    print("  Updated: ml/models/calibration_metadata.json")

    print("\nPromotion complete successfully!")

if __name__ == "__main__":
    main()
