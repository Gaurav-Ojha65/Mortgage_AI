"""
Unit tests for ML Ensemble Model
"""

import pytest
import numpy as np
from ml.ensemble import MortgageEnsembleModel


class TestMortgageEnsembleModel:
    """Test suite for MortgageEnsembleModel."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        # Create simple decision boundary
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    @pytest.fixture
    def trained_model(self, sample_data):
        """Create and train a model fixture."""
        X, y = sample_data
        model = MortgageEnsembleModel(
            nn_epochs=10,  # Small for testing
            smote_ratio=0.5,
            random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        model.fit(X, y, feature_names=feature_names)
        return model

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = MortgageEnsembleModel()
        assert model is not None
        assert not model.is_fitted

    def test_model_fit(self, sample_data):
        """Test model fitting."""
        X, y = sample_data
        model = MortgageEnsembleModel(nn_epochs=5, smote_ratio=0.5)
        model.fit(X, y)
        assert model.is_fitted
        assert 'xgb' in model.models
        assert 'lgb' in model.models

    def test_predict(self, trained_model, sample_data):
        """Test prediction."""
        X, _ = sample_data
        predictions = trained_model.predict(X[:10])
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, trained_model, sample_data):
        """Test probability prediction."""
        X, _ = sample_data
        probabilities = trained_model.predict_proba(X[:10])
        assert probabilities.shape == (10, 2)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
        # Probabilities should sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_evaluate(self, trained_model, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        metrics = trained_model.evaluate(X[:50], y[:50])
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 'confusion_matrix' in metrics

    def test_explain(self, trained_model, sample_data):
        """Test SHAP explanation."""
        X, _ = sample_data
        explanations = trained_model.explain(X[:5])
        assert 'base_value' in explanations
        assert 'feature_importance' in explanations

    def test_save_load(self, trained_model, sample_data, tmp_path):
        """Test model persistence."""
        X, _ = sample_data
        model_path = tmp_path / "test_model.pkl"

        # Save
        trained_model.save(str(model_path))
        assert model_path.exists()

        # Load
        new_model = MortgageEnsembleModel()
        new_model.load(str(model_path))

        assert new_model.is_fitted
        assert new_model.feature_names == trained_model.feature_names

        # Predictions should match
        preds_original = trained_model.predict(X[:10])
        preds_loaded = new_model.predict(X[:10])
        assert np.array_equal(preds_original, preds_loaded)

    def test_individual_predictions(self, trained_model, sample_data):
        """Test individual model predictions."""
        X, _ = sample_data
        individual_preds = trained_model.get_individual_predictions(X[:5])
        assert 'xgb' in individual_preds
        assert 'lgb' in individual_preds
        assert 'nn' in individual_preds
        assert 'ensemble' in individual_preds


class TestEdgeCases:
    """Edge case tests."""

    def test_single_sample(self):
        """Test with single sample."""
        X = np.random.randn(1, 5)
        y = np.array([1])
        model = MortgageEnsembleModel(nn_epochs=5)
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == 1

    def test_all_same_class(self):
        """Test with imbalanced data."""
        X = np.random.randn(50, 5)
        y = np.ones(50)  # All same class
        model = MortgageEnsembleModel(nn_epochs=5)
        # Should handle gracefully
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == 50

    def test_invalid_input(self):
        """Test with invalid inputs."""
        model = MortgageEnsembleModel()
        X = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

        with pytest.raises(ValueError, match="not fitted"):
            model.predict_proba(X)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
