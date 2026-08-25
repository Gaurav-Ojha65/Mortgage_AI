"""
Production Drift Detection System for Mortgage AI
Monitors data drift and model drift with statistical tests
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Drift detection report."""
    timestamp: str
    drift_detected: bool
    drift_score: float
    drift_type: str
    feature_drifts: Dict[str, Dict]
    threshold: float
    recommendation: str


class DataDriftDetector:
    """
    Detects data drift using statistical tests and distance metrics.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        feature_threshold: float = 0.05
    ):
        """
        Initialize drift detector with reference data.

        Args:
            reference_data: Training/reference data distribution
            psi_threshold: PSI threshold for drift alert (default 0.2)
            ks_threshold: KS test p-value threshold (default 0.05)
            feature_threshold: Per-feature drift threshold
        """
        self.reference_data = reference_data.copy()
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.feature_threshold = feature_threshold

        self.reference_stats = self._compute_statistics(reference_data)
        self.numerical_cols = reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.categorical_cols = reference_data.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

    def _compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute reference statistics."""
        stats_dict = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'quantiles': df[col].quantile([0.25, 0.5, 0.75]).to_dict()
            }
        return stats_dict

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI < 0.1: No significant change
        0.1 ≤ PSI < 0.2: Moderate change
        PSI ≥ 0.2: Significant change
        """
        # Create bins based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

        # Add small epsilon to avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        psi = np.sum((actual_percents - expected_percents) *
                     np.log(actual_percents / expected_percents))
        return psi

    def _ks_test(self, reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution equality.

        Returns:
            statistic: KS statistic
            p_value: p-value (drift if p < threshold)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return statistic, p_value

    def _wasserstein_distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Wasserstein distance between distributions."""
        return stats.wasserstein_distance(reference, current)

    def detect_feature_drift(
        self,
        current_data: pd.DataFrame,
        feature: str
    ) -> Dict[str, Any]:
        """Detect drift for a single feature."""
        if feature not in self.numerical_cols:
            return {'drift_detected': False, 'method': 'skipped'}

        reference = self.reference_data[feature].dropna()
        current = current_data[feature].dropna()

        # PSI
        psi = self._calculate_psi(reference.values, current.values)

        # KS test
        ks_stat, ks_pvalue = self._ks_test(reference.values, current.values)

        # Wasserstein distance (normalized)
        wasserstein = self._wasserstein_distance(reference.values, current.values)
        wasserstein_norm = wasserstein / (reference.std() + 1e-8)

        # Mean difference
        mean_diff = abs(current.mean() - reference.mean()) / (reference.std() + 1e-8)

        drift_detected = (
            psi >= self.psi_threshold or
            ks_pvalue < self.ks_threshold or
            mean_diff > 3.0  # 3 sigma rule
        )

        return {
            'feature': feature,
            'drift_detected': drift_detected,
            'psi': float(psi),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'wasserstein': float(wasserstein_norm),
            'mean_diff_sigma': float(mean_diff),
            'reference_mean': float(reference.mean()),
            'current_mean': float(current.mean()),
            'reference_std': float(reference.std()),
            'current_std': float(current.std())
        }

    def detect(self, current_data: pd.DataFrame) -> DriftReport:
        """
        Run full drift detection on current data.

        Returns:
            DriftReport with drift status and feature-level details
        """
        feature_drifts = {}
        drift_scores = []

        for feature in self.numerical_cols:
            if feature in current_data.columns:
                drift_info = self.detect_feature_drift(current_data, feature)
                feature_drifts[feature] = drift_info
                drift_scores.append(drift_info['psi'])

        # Overall drift score (average PSI across features)
        overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0

        # Determine if drift detected
        drift_detected = any(
            info['drift_detected'] for info in feature_drifts.values()
        )

        # Generate recommendation
        if overall_drift_score < 0.1:
            recommendation = "No drift detected. Continue monitoring."
        elif overall_drift_score < 0.2:
            recommendation = "Moderate drift detected. Monitor closely. Consider retraining if trend continues."
        else:
            recommendation = "Significant drift detected! Retrain model immediately."

        return DriftReport(
            timestamp=datetime.now().isoformat(),
            drift_detected=drift_detected,
            drift_score=float(overall_drift_score),
            drift_type='data_drift',
            feature_drifts=feature_drifts,
            threshold=self.psi_threshold,
            recommendation=recommendation
        )

    def save(self, path: str):
        """Save detector state."""
        state = {
            'reference_stats': self.reference_stats,
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'psi_threshold': self.psi_threshold,
            'ks_threshold': self.ks_threshold,
            'feature_threshold': self.feature_threshold
        }
        joblib.dump(state, path)
        logger.info(f"Drift detector saved to {path}")

    def load(self, path: str):
        """Load detector state."""
        state = joblib.load(path)
        self.reference_stats = state['reference_stats']
        self.numerical_cols = state['numerical_cols']
        self.categorical_cols = state['categorical_cols']
        logger.info(f"Drift detector loaded from {path}")


class ModelDriftDetector:
    """
    Detects model performance drift and prediction distribution drift.
    """

    def __init__(
        self,
        reference_predictions: np.ndarray,
        reference_labels: np.ndarray,
        performance_threshold: float = 0.05
    ):
        """
        Initialize model drift detector.

        Args:
            reference_predictions: Predictions on reference data
            reference_labels: True labels for reference data
            performance_threshold: Performance degradation threshold
        """
        self.reference_predictions = reference_predictions.copy()
        self.reference_labels = reference_labels.copy()
        self.performance_threshold = performance_threshold

        # Compute reference performance
        from sklearn.metrics import roc_auc_score, accuracy_score
        self.reference_auc = roc_auc_score(reference_labels, reference_predictions)
        self.reference_accuracy = accuracy_score(
            reference_labels, reference_predictions > 0.5
        )

    def detect(
        self,
        current_predictions: np.ndarray,
        current_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect model drift.

        Returns:
            Dictionary with drift status and metrics
        """
        from sklearn.metrics import roc_auc_score, accuracy_score

        current_auc = roc_auc_score(current_labels, current_predictions)
        current_accuracy = accuracy_score(current_labels, current_predictions > 0.5)

        # Performance drift
        auc_degradation = (self.reference_auc - current_auc) / self.reference_auc
        accuracy_degradation = (
            self.reference_accuracy - current_accuracy
        ) / (self.reference_accuracy + 1e-8)

        # Prediction distribution drift (JS divergence)
        # Bin predictions for distribution comparison
        ref_hist, _ = np.histogram(self.reference_predictions, bins=20, range=(0, 1))
        curr_hist, _ = np.histogram(current_predictions, bins=20, range=(0, 1))

        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()

        js_divergence = jensenshannon(ref_hist, curr_hist)

        drift_detected = (
            auc_degradation > self.performance_threshold or
            js_divergence > 0.1
        )

        return {
            'drift_detected': drift_detected,
            'drift_type': 'model_drift',
            'reference_auc': float(self.reference_auc),
            'current_auc': float(current_auc),
            'auc_degradation': float(auc_degradation),
            'reference_accuracy': float(self.reference_accuracy),
            'current_accuracy': float(current_accuracy),
            'accuracy_degradation': float(accuracy_degradation),
            'prediction_distribution_drift': float(js_divergence),
            'recommendation': (
                "Retrain model" if drift_detected else "Continue monitoring"
            )
        }


class DriftMonitor:
    """
    Production drift monitoring system with alerting.
    """

    def __init__(
        self,
        data_detector: Optional[DataDriftDetector] = None,
        model_detector: Optional[ModelDriftDetector] = None,
        alert_threshold: int = 3
    ):
        self.data_detector = data_detector
        self.model_detector = model_detector
        self.alert_threshold = alert_threshold
        self.drift_history: List[Dict] = []
        self.consecutive_alerts = 0

    def check(
        self,
        current_data: Optional[pd.DataFrame] = None,
        current_predictions: Optional[np.ndarray] = None,
        current_labels: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Run drift checks and return results.

        Returns:
            List of drift reports
        """
        results = []

        if self.data_detector and current_data is not None:
            report = self.data_detector.detect(current_data)
            results.append(report.__dict__)

            if report.drift_detected:
                self.consecutive_alerts += 1
            else:
                self.consecutive_alerts = 0

        if self.model_detector and current_predictions is not None:
            report = self.model_detector.detect(current_predictions, current_labels or [])
            results.append(report)

        self.drift_history.extend(results)

        return results

    def should_alert(self) -> bool:
        """Check if alerting threshold reached."""
        return self.consecutive_alerts >= self.alert_threshold

    def get_drift_summary(self, days: int = 7) -> Dict:
        """Get drift summary for last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        recent = [
            r for r in self.drift_history
            if datetime.fromisoformat(r['timestamp']) > cutoff
        ]

        return {
            'total_checks': len(recent),
            'drift_detected_count': sum(
                1 for r in recent if r.get('drift_detected', False)
            ),
            'avg_drift_score': np.mean([
                r.get('drift_score', 0) for r in recent
            ]) if recent else 0,
            'consecutive_alerts': self.consecutive_alerts,
            'alert_threshold': self.alert_threshold,
            'should_alert': self.should_alert()
        }

    def save_history(self, path: str):
        """Save drift history to file."""
        with open(path, 'w') as f:
            json.dump({
                'history': self.drift_history,
                'consecutive_alerts': self.consecutive_alerts
            }, f, indent=2, default=str)


# Convenience function for production use
def create_drift_monitor_from_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_pred_train: np.ndarray,
    psi_threshold: float = 0.2,
    performance_threshold: float = 0.05
) -> DriftMonitor:
    """
    Create a drift monitor from training data.

    Args:
        X_train: Training features
        y_train: Training labels
        y_pred_train: Training predictions
        psi_threshold: PSI threshold for data drift
        performance_threshold: Performance degradation threshold

    Returns:
        Configured DriftMonitor instance
    """
    data_detector = DataDriftDetector(
        X_train, psi_threshold=psi_threshold
    )

    model_detector = ModelDriftDetector(
        y_pred_train, y_train.values, performance_threshold
    )

    return DriftMonitor(
        data_detector=data_detector,
        model_detector=model_detector
    )


if __name__ == "__main__":
    print("Drift Detection System for Mortgage AI")
