"""
CalibratedPredictor Wrapper — Mortgage AI
==========================================
Scikit-learn compatible inference wrapper bundling a trained base model
with a fitted probability calibrator.
"""

from __future__ import annotations
from typing import Any, Union
import numpy as np
import pandas as pd
from ml.inference.predict import MODEL_FEATURES


class CalibratedPredictor:
    """
    Scikit-learn compatible inference wrapper bundling a trained base model
    with a fitted probability calibrator.

    Pipeline:
        X -> base_model.predict_proba(X)[:, 1] -> p_raw -> calibrator.predict(p_raw) -> p_cal
    """

    def __init__(
        self,
        base_model: Any,
        calibrator: Any,
        calibration_method: str = "isotonic",
        model_name: str = "lightgbm",
        version: str = "v3.0-oof-calibrated",
    ):
        self.base_model = base_model
        self.calibrator = calibrator
        self.calibration_method = calibration_method.lower()
        self.model_name = model_name
        self.version = version

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute calibrated class probabilities.

        Returns:
            2D numpy array of shape (N, 2) where column 1 is the calibrated
            probability of default P(target=1|X).
        """
        if isinstance(X, pd.DataFrame):
            raw_p = self.base_model.predict_proba(X)[:, 1]
        else:
            X_df = pd.DataFrame(X, columns=MODEL_FEATURES)
            raw_p = self.base_model.predict_proba(X_df)[:, 1]

        raw_p = np.asarray(raw_p, dtype=np.float64)

        if self.calibration_method == "isotonic":
            cal_p = self.calibrator.predict(raw_p)
        elif self.calibration_method in ("sigmoid", "platt"):
            cal_p = self.calibrator.predict_proba(raw_p.reshape(-1, 1))[:, 1]
        else:
            cal_p = raw_p

        # Bound strictly in [0.0, 1.0]
        cal_p = np.clip(cal_p, 0.0, 1.0)
        return np.column_stack([1.0 - cal_p, cal_p])

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Binary class prediction at specified threshold."""
        prob_default = self.predict_proba(X)[:, 1]
        return (prob_default >= threshold).astype(int)
