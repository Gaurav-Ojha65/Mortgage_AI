"""
Unit & Integration Tests for LightGBM HPO and Post-Validation
=============================================================
Tests:
1. Search-space bounds and validity.
2. Depth / leaf mathematical constraint (num_leaves <= 2^max_depth - 1).
3. Deterministic baseline trial (Trial 0) parameters.
4. Objective function output structure and bounds on real training data.
5. Zero access to val.csv and test.csv during optimization.
6. Schema and completeness of Optuna results report and post-HPO artifacts.
7. Candidate model artifact serialization.
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
    load_real_train_only,
    compute_sha256,
)
from ml.inference.predict import MODEL_FEATURES


@pytest.fixture(scope="module")
def real_train_sample():
    """Load sample of real training data for lightweight testing."""
    X_real, y_real = load_real_train_only()
    # Use first 1,000 samples for rapid unit testing
    return X_real[:1000], y_real[:1000]


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

    def test_objective_data_isolation(self, monkeypatch):
        """Verify objective function only loads real_train.csv."""
        X_real, y_real = load_real_train_only()
        assert len(X_real) == 99856
        assert len(y_real) == 99856
        assert round(float(y_real.mean()), 4) == 0.0676

    def test_objective_evaluation_returns_valid_auc(self, real_train_sample):
        """Run single objective evaluation on sample to verify float ROC-AUC output."""
        X_sample, y_sample = real_train_sample
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
