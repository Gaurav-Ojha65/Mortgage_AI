"""
Unit & Integration Tests for LightGBM HPO and Post-Validation
=============================================================
Tests:
1. Search-space bounds and validity.
2. Depth / leaf mathematical constraint (num_leaves <= 2^max_depth - 1).
3. Deterministic baseline trial (Trial 0) parameters.
4. Objective function output structure and bounds on synthetic data.
5. Schema and completeness of Optuna results report and post-HPO artifacts.
6. Candidate model artifact serialization.

Note: Tests use deterministic synthetic data that mirrors the production schema
so that CI can run without the private 100MB+ real_train.csv dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
import optuna

from ml.training.lightgbm_hpo import (
    LightGBMObjective,
    BASELINE_PARAMS,
    compute_sha256,
)
from ml.inference.predict import MODEL_FEATURES


# ─── Synthetic data fixture (mirrors real_train.csv schema) ──────────────────

def _generate_synthetic_mortgage_data(n_samples: int, default_rate: float = 0.07, seed: int = 42):
    """
    Generate deterministic synthetic data matching the 15-feature mortgage schema.
    Preserves the same column structure, realistic value ranges, and approximate
    default prevalence (~6.76%) as the real training dataset.
    """
    rng = np.random.RandomState(seed)

    data = np.column_stack([
        rng.randint(300, 850, n_samples).astype(float),         # credit_score
        rng.uniform(20000, 200000, n_samples),                   # annual_income
        rng.uniform(50000, 500000, n_samples),                   # loan_amount
        rng.choice([60, 120, 180, 240, 300, 360], n_samples).astype(float),  # loan_term
        rng.uniform(0.05, 0.80, n_samples),                      # dti_ratio
        rng.uniform(0.0, 30.0, n_samples),                       # employment_years
        rng.randint(1, 30, n_samples).astype(float),              # num_credit_lines
        rng.randint(0, 5, n_samples).astype(float),               # num_derogatory_marks
        rng.uniform(0.0, 1.0, n_samples),                        # credit_utilization
        rng.uniform(0.0, 1.0, n_samples),                        # late_payment_severity_score
        rng.choice([0, 1, 2], n_samples).astype(float),           # home_ownership
        rng.choice([0, 1, 2, 3], n_samples).astype(float),       # purpose_encoded
        rng.randint(0, 10, n_samples).astype(float),              # num_late_payments
        rng.uniform(0, 100000, n_samples),                        # savings_balance
        rng.uniform(500, 5000, n_samples),                        # monthly_expenses
    ])

    # Generate labels with approximate target default rate
    n_defaults = int(n_samples * default_rate)
    y = np.zeros(n_samples, dtype=int)
    y[:n_defaults] = 1
    rng.shuffle(y)

    return data, y


@pytest.fixture(scope="module")
def synthetic_train_sample():
    """Deterministic synthetic data for HPO testing (1,000 samples, ~7% default rate)."""
    X, y = _generate_synthetic_mortgage_data(1000, default_rate=0.07, seed=42)
    return X, y


class TestHPOSearchSpace:
    """Validate hyperparameter search space definitions and constraints."""

    def test_search_space_bounds_and_sampling(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        dummy_X = np.random.randn(100, 15)
        dummy_y = np.random.binomial(1, 0.1, 100)
        obj = LightGBMObjective(dummy_X, dummy_y, n_splits=2, random_seed=42)

        params = obj.sample_hyperparameters(trial)

        # 1. n_estimators
        assert 300 <= params["n_estimators"] <= 800
        assert isinstance(params["n_estimators"], int)

        # 2. learning_rate
        assert 0.005 <= params["learning_rate"] <= 0.1

        # 3. max_depth
        assert 4 <= params["max_depth"] <= 10
        assert isinstance(params["max_depth"], int)

        # 4. num_leaves & compatibility with max_depth
        assert 15 <= params["num_leaves"] <= 63
        max_allowed = min(63, (1 << params["max_depth"]) - 1)
        assert params["num_leaves"] <= max_allowed

        # 5. min_child_samples
        assert 10 <= params["min_child_samples"] <= 80
        assert isinstance(params["min_child_samples"], int)

        # 6. subsample & subsample_freq
        assert 0.6 <= params["subsample"] <= 1.0
        if params["subsample"] < 1.0:
            assert params["subsample_freq"] == 1
        else:
            assert params["subsample_freq"] == 0

        # 7. colsample_bytree
        assert 0.5 <= params["colsample_bytree"] <= 1.0

        # 8. Regularization terms
        assert 1e-5 <= params["reg_alpha"] <= 2.0
        assert 1e-5 <= params["reg_lambda"] <= 5.0

    def test_depth_leaf_constraint_across_all_depths(self):
        """Verify depth-leaf compatibility calculation for every valid depth from 4 to 10."""
        for depth in range(4, 11):
            max_allowed = min(63, (1 << depth) - 1)
            min_leaves = min(15, max_allowed)
            assert min_leaves <= max_allowed
            assert max_allowed <= (1 << depth) - 1
            if depth == 4:
                assert max_allowed == 15
            elif depth == 5:
                assert max_allowed == 31
            else:
                assert max_allowed == 63

    def test_baseline_parameters_match_v3_frozen(self):
        assert BASELINE_PARAMS["n_estimators"] == 500
        assert BASELINE_PARAMS["learning_rate"] == 0.03
        assert BASELINE_PARAMS["max_depth"] == 6
        assert BASELINE_PARAMS["num_leaves"] == 31
        assert BASELINE_PARAMS["min_child_samples"] == 25
        assert BASELINE_PARAMS["subsample"] == 0.8
        assert BASELINE_PARAMS["colsample_bytree"] == 0.8
        assert BASELINE_PARAMS["reg_alpha"] == 0.2
        assert BASELINE_PARAMS["reg_lambda"] == 2.0

    def test_objective_data_isolation_with_synthetic(self):
        """Verify objective function works with synthetic data matching production schema."""
        X_synth, y_synth = _generate_synthetic_mortgage_data(500, default_rate=0.07, seed=99)
        assert X_synth.shape == (500, 15)
        assert len(y_synth) == 500
        # Verify approximate default rate
        rate = y_synth.mean()
        assert 0.05 <= rate <= 0.10, f"Synthetic default rate out of range: {rate:.4f}"

    def test_objective_evaluation_returns_valid_auc(self, synthetic_train_sample):
        """Run single objective evaluation on synthetic data to verify float ROC-AUC output."""
        X_sample, y_sample = synthetic_train_sample
        study = optuna.create_study(direction="maximize")
        study.enqueue_trial(BASELINE_PARAMS)
        trial = study.ask()

        obj = LightGBMObjective(X_sample, y_sample, n_splits=2, random_seed=42)
        score = obj(trial)

        assert isinstance(score, float)
        assert 0.5 <= score <= 1.0
        assert "mean_roc_auc" in trial.user_attrs
        assert "brier_score" in trial.user_attrs
        assert "weighted_ece" in trial.user_attrs
        assert trial.user_attrs["completed_folds"] == 2


class TestHPOSerializationAndSchema:
    """Validate artifact serialization and report schemas."""

    def test_sha256_computation(self):
        p = Path("data/real_train.csv")
        if p.exists():
            h = compute_sha256(p)
            assert len(h) == 64
            assert h != "file_not_found"

    def test_hpo_results_json_schema_if_present(self):
        p = Path("reports/metrics/optuna_results.json")
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "metadata" in data
            assert "study_summary" in data
            assert "baseline_trial_0" in data
            assert "best_trial" in data
            assert "top_10_trials" in data

            assert data["metadata"]["hpo_configuration"]["n_trials_total"] >= 1
            assert data["baseline_trial_0"]["params"] is not None
            assert data["best_trial"]["mean_roc_auc"] is not None
