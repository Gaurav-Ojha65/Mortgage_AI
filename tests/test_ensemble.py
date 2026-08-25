"""
Unit tests for ML Ensemble Model
Fixed: removed nn_epochs parameter, removed 'nn' key assertions,
       aligned fixture with actual MortgageEnsembleModel API.
"""

import pytest
import numpy as np
from ml.inference.ensemble import MortgageEnsembleModel


class TestMortgageEnsembleModel:
    """Test suite for MortgageEnsembleModel."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        # Create decision boundary with some noise
        y = ((X[:, 0] + X[:, 1] + X[:, 2] * 0.5) > 0).astype(int)
        return X, y

    @pytest.fixture
    def trained_model(self, sample_data):
        """Create and train a model fixture."""
        X, y = sample_data
        model = MortgageEnsembleModel(random_state=42)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        model.fit(X, y, feature_names=feature_names)
        return model

    # ── Initialization ──────────────────────────────────────────────────────────

    def test_model_initialization(self):
        """Test model can be initialized with defaults."""
        model = MortgageEnsembleModel()
        assert model is not None
        assert not model.is_fitted

    def test_model_initialization_custom_params(self):
        """Test model accepts custom parameters."""
        model = MortgageEnsembleModel(random_state=123)
        assert model is not None

    # ── Fitting ─────────────────────────────────────────────────────────────────

    def test_model_fit(self, sample_data):
        """Test model fitting produces a fitted model."""
        X, y = sample_data
        model = MortgageEnsembleModel(random_state=42)
        model.fit(X, y)
        assert model.is_fitted
        assert "xgb" in model.models
        assert "lgb" in model.models

    def test_model_fit_with_feature_names(self, sample_data):
        """Test fitting with explicit feature names."""
        X, y = sample_data
        model = MortgageEnsembleModel(random_state=42)
        names = [f"feat_{i}" for i in range(X.shape[1])]
        model.fit(X, y, feature_names=names)
        assert model.feature_names == names

    # ── Prediction ──────────────────────────────────────────────────────────────

    def test_predict(self, trained_model, sample_data):
        """Test binary prediction output."""
        X, _ = sample_data
        predictions = trained_model.predict(X[:10])
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, trained_model, sample_data):
        """Test probability prediction output."""
        X, _ = sample_data
        probabilities = trained_model.predict_proba(X[:10])
        assert probabilities.shape == (10, 2)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
        # Probabilities should sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_returns_consistent_results(self, trained_model, sample_data):
        """Test that predictions are deterministic."""
        X, _ = sample_data
        preds1 = trained_model.predict(X[:20])
        preds2 = trained_model.predict(X[:20])
        assert np.array_equal(preds1, preds2)

    # ── Evaluation ──────────────────────────────────────────────────────────────

    def test_evaluate(self, trained_model, sample_data):
        """Test model evaluation returns all expected metrics."""
        X, y = sample_data
        metrics = trained_model.evaluate(X[:50], y[:50])
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics
        # Metrics should be in valid range
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    # ── SHAP Explanations ───────────────────────────────────────────────────────

    def test_explain(self, trained_model, sample_data):
        """Test SHAP explanation returns expected structure."""
        X, _ = sample_data
        explanations = trained_model.explain(X[:5])
        assert "base_value" in explanations
        assert "feature_importance" in explanations

    # ── Individual Predictions ──────────────────────────────────────────────────

    def test_individual_predictions(self, trained_model, sample_data):
        """Test individual model predictions (XGB, LGB, ensemble only — no NN)."""
        X, _ = sample_data
        individual_preds = trained_model.get_individual_predictions(X[:5])
        # Must have base models
        assert "xgb" in individual_preds
        assert "lgb" in individual_preds
        # Ensemble output
        assert "ensemble" in individual_preds
        # nn is NOT expected — ensemble.py has no neural net
        assert "nn" not in individual_preds, \
            "No neural net in MortgageEnsembleModel — remove this key from test"

    def test_individual_predictions_shape(self, trained_model, sample_data):
        """Test that individual predictions have correct shape."""
        X, _ = sample_data
        n = 5
        result = trained_model.get_individual_predictions(X[:n])
        for key in ["xgb", "lgb", "ensemble"]:
            assert key in result
            assert len(result[key]) == n

    # ── Persistence ─────────────────────────────────────────────────────────────

    def test_save_load(self, trained_model, sample_data, tmp_path):
        """Test model persistence — save and reload produces same predictions."""
        X, _ = sample_data
        model_path = tmp_path / "test_model.pkl"

        trained_model.save(str(model_path))
        assert model_path.exists()

        new_model = MortgageEnsembleModel()
        new_model.load(str(model_path))

        assert new_model.is_fitted
        assert new_model.feature_names == trained_model.feature_names

        preds_original = trained_model.predict(X[:10])
        preds_loaded   = new_model.predict(X[:10])
        assert np.array_equal(preds_original, preds_loaded)

    def test_save_creates_file(self, trained_model, tmp_path):
        """Test that save creates a non-empty file."""
        path = tmp_path / "model.pkl"
        trained_model.save(str(path))
        assert path.exists()
        assert path.stat().st_size > 0


class TestEdgeCases:
    """Edge case tests."""

    def test_unfitted_predict_raises(self):
        """Test that predicting on unfitted model raises ValueError."""
        model = MortgageEnsembleModel()
        X = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_unfitted_predict_proba_raises(self):
        """Test that predict_proba on unfitted model raises ValueError."""
        model = MortgageEnsembleModel()
        X = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="not fitted"):
            model.predict_proba(X)

    def test_imbalanced_data(self):
        """Test model handles imbalanced data (no SMOTE crash)."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.zeros(100, dtype=int)
        y[:10] = 1  # Only 10% positive

        model = MortgageEnsembleModel(random_state=42)
        # Should handle gracefully (may disable SMOTE if minority < k_neighbors)
        model.fit(X, y)
        assert model.is_fitted
        pred = model.predict(X)
        assert len(pred) == 100

    def test_single_feature(self):
        """Test with minimal feature set."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        model = MortgageEnsembleModel(random_state=42)
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert len(preds) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
