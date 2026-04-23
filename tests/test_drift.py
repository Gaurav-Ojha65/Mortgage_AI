"""
Unit tests for Drift Detection System
"""

import pytest
import numpy as np
import pandas as pd
from ml.drift_detector import (
    DataDriftDetector, ModelDriftDetector, DriftMonitor,
    create_drift_monitor_from_training
)


class TestDataDriftDetector:
    """Test suite for DataDriftDetector."""

    @pytest.fixture
    def reference_data(self):
        """Create reference data."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(100, 15, 1000),
            'feature2': np.random.normal(50, 10, 1000),
            'feature3': np.random.exponential(2, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })

    @pytest.fixture
    def detector(self, reference_data):
        """Create drift detector."""
        return DataDriftDetector(
            reference_data,
            psi_threshold=0.2,
            ks_threshold=0.05
        )

    def test_initialization(self, reference_data):
        """Test detector initialization."""
        detector = DataDriftDetector(reference_data)
        assert detector is not None
        assert len(detector.reference_stats) > 0

    def test_no_drift(self, detector, reference_data):
        """Test with same distribution (no drift)."""
        report = detector.detect(reference_data)
        assert not report.drift_detected
        assert report.drift_score < 0.1
        assert 'No drift' in report.recommendation

    def test_drift_detected(self, detector):
        """Test with drifted distribution."""
        # Create drifted data
        drifted_data = pd.DataFrame({
            'feature1': np.random.normal(150, 20, 1000),  # Shifted mean
            'feature2': np.random.normal(50, 10, 1000),
            'feature3': np.random.exponential(2, 1000),
        })

        report = detector.detect(drifted_data)
        assert report.drift_detected
        assert report.drift_score > 0.1
        assert 'Drift detected' in report.recommendation

    def test_psi_calculation(self, detector):
        """Test PSI calculation."""
        expected = np.random.normal(100, 15, 1000)
        actual = np.random.normal(100, 15, 1000)  # Same distribution
        psi = detector._calculate_psi(expected, actual)
        assert psi >= 0
        assert psi < 0.1  # Should be low for same distribution

    def test_ks_test(self, detector):
        """Test KS test."""
        reference = np.random.normal(100, 15, 1000)
        current = np.random.normal(100, 15, 1000)
        statistic, p_value = detector._ks_test(reference, current)
        assert 0 <= statistic <= 1
        assert 0 <= p_value <= 1

    def test_feature_drift_detection(self, detector):
        """Test individual feature drift."""
        current = pd.DataFrame({
            'feature1': np.random.normal(150, 20, 1000),  # Drifted
        })

        drift_info = detector.detect_feature_drift(current, 'feature1')
        assert drift_info['drift_detected']
        assert drift_info['psi'] > 0
        assert 'reference_mean' in drift_info
        assert 'current_mean' in drift_info

    def test_save_load(self, detector, tmp_path):
        """Test detector persistence."""
        path = tmp_path / "drift_detector.pkl"
        detector.save(str(path))

        new_detector = DataDriftDetector(pd.DataFrame())
        new_detector.load(str(path))

        assert new_detector.reference_stats == detector.reference_stats


class TestModelDriftDetector:
    """Test suite for ModelDriftDetector."""

    @pytest.fixture
    def detector(self):
        """Create model drift detector."""
        np.random.seed(42)
        predictions = np.random.random(1000)
        labels = np.random.randint(0, 2, 1000)
        return ModelDriftDetector(predictions, labels)

    def test_initialization(self):
        """Test detector initialization."""
        preds = np.random.random(100)
        labels = np.random.randint(0, 2, 100)
        detector = ModelDriftDetector(preds, labels)
        assert detector is not None
        assert detector.reference_auc > 0

    def test_no_drift(self, detector):
        """Test with similar performance."""
        current_preds = np.random.random(1000)
        current_labels = np.random.randint(0, 2, 1000)

        report = detector.detect(current_preds, current_labels)
        assert isinstance(report, dict)
        assert 'drift_detected' in report

    def test_performance_degradation(self, detector):
        """Test with degraded performance."""
        # Predictions that perform worse
        current_preds = np.random.random(100) * 0.3  # Low scores
        current_labels = np.ones(100)  # All positive, but low predictions

        report = detector.detect(current_preds, current_labels)
        assert isinstance(report, dict)


class TestDriftMonitor:
    """Test suite for DriftMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create drift monitor."""
        return DriftMonitor(alert_threshold=3)

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = DriftMonitor()
        assert monitor is not None
        assert monitor.consecutive_alerts == 0
        assert len(monitor.drift_history) == 0

    def test_check_no_detectors(self, monitor):
        """Test check with no detectors."""
        results = monitor.check()
        assert results == []

    def test_alert_threshold(self, monitor):
        """Test alerting threshold."""
        assert not monitor.should_alert()

        # Simulate alerts
        for _ in range(5):
            monitor.drift_history.append({'drift_detected': True, 'timestamp': pd.Timestamp.now()})
            monitor.consecutive_alerts += 1

        assert monitor.should_alert()

    def test_drift_summary(self, monitor):
        """Test drift summary."""
        summary = monitor.get_drift_summary(days=7)
        assert 'total_checks' in summary
        assert 'drift_detected_count' in summary
        assert 'consecutive_alerts' in summary


class TestIntegration:
    """Integration tests."""

    def test_create_drift_monitor_from_training(self):
        """Test creating monitor from training data."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(100, 15, 500),
            'feature2': np.random.normal(50, 10, 500),
        })
        y = pd.Series(np.random.randint(0, 2, 500))
        y_pred = np.random.random(500)

        monitor = create_drift_monitor_from_training(X, y, y_pred)
        assert monitor.data_detector is not None
        assert monitor.model_detector is not None

    def test_end_to_end_drift_detection(self):
        """Test complete drift detection workflow."""
        # Create reference data
        np.random.seed(42)
        ref_data = pd.DataFrame({
            'income': np.random.lognormal(10, 0.5, 1000),
            'credit_score': np.random.normal(650, 100, 1000),
        })

        detector = DataDriftDetector(ref_data)

        # Test with same data
        report1 = detector.detect(ref_data)
        assert not report1.drift_detected

        # Test with drifted data
        drifted = pd.DataFrame({
            'income': np.random.lognormal(11, 0.6, 1000),  # Higher income
            'credit_score': np.random.normal(580, 80, 1000),  # Lower scores
        })

        report2 = detector.detect(drifted)
        assert report2.drift_detected
        assert len(report2.feature_drifts) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
