"""
Production Ensemble Model for Mortgage AI
XGBoost + LightGBM with Stacking
"""
import pickle
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MortgageEnsembleModel(ClassifierMixin, BaseEstimator):
    """
    Production ensemble model combining XGBoost and LightGBM via stacking.

    Architecture
    ------------
    * Base models: XGBoost + LightGBM, trained with 5-fold out-of-fold CV
      to generate meta-features without target leakage.
    * Meta-learner: LogisticRegression on the two base-model probability outputs.
    * Scaling: RobustScaler applied inside fit/predict — callers always pass raw features.
    * SHAP: TreeExplainer registered for XGBoost and LightGBM base models.

    Class imbalance is handled **upstream** by the data pipeline (SMOTE on training
    data before this model sees any data).  No class weighting is applied here.
    """

    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        lgb_params: Optional[Dict] = None,
        random_state: int = 42,
        use_shap: bool = True,
    ):
        self.random_state = random_state
        self.use_shap = use_shap
        self.xgb_params = xgb_params or self._default_xgb_params()
        self.lgb_params = lgb_params or self._default_lgb_params()
        self.models = {}
        self.meta_learner = None
        self.scaler = RobustScaler()
        self.feature_names = None
        self.shap_explainer = None
        self.is_fitted = False
        self.classes_ = None

    def _default_xgb_params(self) -> Dict:
        """Default XGBoost parameters optimized for credit risk."""
        return {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "n_jobs": -1,
            "eval_metric": "auc",
        }

    def _default_lgb_params(self) -> Dict:
        """Default LightGBM parameters optimized for credit risk."""
        return {
            "n_estimators": 300,
            "max_depth": -1,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

    def _create_meta_features(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, fit: bool = False
    ) -> np.ndarray:
        """Generate meta-features from base models using cross-validation."""
        n = X.shape[0]
        mf = np.zeros((n, 2))

        # Ensure consistent DataFrame with feature names for base models
        names = getattr(self, "feature_names", None) or [f"f{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=names) if not isinstance(X, pd.DataFrame) else X

        if fit:
            skf = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            )

            # XGBoost
            xm = xgb.XGBClassifier(**self.xgb_params)
            mf[:, 0] = cross_val_predict(xm, X_df, y, cv=skf, method="predict_proba")[
                :, 1
            ]
            self.models["xgb"] = xm.fit(X_df, y)

            # LightGBM
            lm = lgb.LGBMClassifier(**self.lgb_params)
            mf[:, 1] = cross_val_predict(lm, X_df, y, cv=skf, method="predict_proba")[
                :, 1
            ]
            self.models["lgb"] = lm.fit(X_df, y)
        else:
            mf[:, 0] = self.models["xgb"].predict_proba(X_df)[:, 1]
            mf[:, 1] = self.models["lgb"].predict_proba(X_df)[:, 1]

        return mf

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None
    ):
        """
        Fit the ensemble model.

        Args:
            X: Training features
            y: Training labels
            feature_names: Optional list of feature names for SHAP
        """
        self.feature_names = feature_names or [
            f"f{i}" for i in range(X.shape[1])
        ]

        # Scale features
        Xs = self.scaler.fit_transform(X)

        # Create meta-features
        mf = self._create_meta_features(Xs, y, fit=True)

        # Train meta-learner on out-of-fold probabilities from base models
        # No class_weight: training data is already SMOTE-balanced by the pipeline
        self.meta_learner = LogisticRegression(
            C=1.0, random_state=self.random_state
        )
        self.meta_learner.fit(mf, y)

        # Initialize SHAP explainers
        if self.use_shap:
            try:
                self.shap_explainer = {
                    "xgb": shap.TreeExplainer(self.models["xgb"]),
                    "lgb": shap.TreeExplainer(self.models["lgb"]),
                }
            except Exception as e:
                logger.warning(f"SHAP failed: {e}")

        self.classes_ = np.unique(y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        Xs = self.scaler.transform(X)
        mf = self._create_meta_features(Xs, fit=False)
        return self.meta_learner.predict(mf)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        Xs = self.scaler.transform(X)
        mf = self._create_meta_features(Xs, fit=False)
        return self.meta_learner.predict_proba(mf)

    def explain(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict:
        """
        Generate SHAP explanations for predictions.

        Returns:
            Dictionary with SHAP values and feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        if feature_names is None:
            feature_names = self.feature_names

        Xs = self.scaler.transform(X)
        explanations = {
            "base_value": 0.5,
            "shap_values": {},
            "feature_importance": {},
        }

        if self.shap_explainer:
            try:
                shap_values_xgb = self.shap_explainer["xgb"].shap_values(Xs)
                explanations["shap_values"]["xgb"] = shap_values_xgb.tolist()

                importance = np.abs(shap_values_xgb).mean(axis=0)
                explanations["feature_importance"] = {
                    name: float(imp) for name, imp in zip(feature_names, importance)
                }
            except Exception as e:
                logger.error(f"SHAP explanation failed: {e}")

        return explanations

    def get_individual_predictions(
        self, X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model in the ensemble.

        Args:
            X: Feature matrix

        Returns:
            Dictionary with predictions from XGBoost, LightGBM, and ensemble
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        Xs = self.scaler.transform(X)
        names = getattr(self, "feature_names", None) or [f"f{i}" for i in range(Xs.shape[1])]
        Xs_df = pd.DataFrame(Xs, columns=names) if not isinstance(Xs, pd.DataFrame) else Xs

        return {
            "xgb": self.models["xgb"].predict_proba(Xs_df)[:, 1],
            "lgb": self.models["lgb"].predict_proba(Xs_df)[:, 1],
            "ensemble": self.predict_proba(X)[:, 1],
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        p = self.predict(X)
        pr = self.predict_proba(X)[:, 1]

        return {
            "accuracy": accuracy_score(y, p),
            "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0),
            "roc_auc": roc_auc_score(y, pr),
            "confusion_matrix": confusion_matrix(y, p).tolist(),
        }

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "models": self.models,
            "meta_learner": self.meta_learner,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "classes_": self.classes_,
            "is_fitted": self.is_fitted,
            "config": {
                "xgb_params": self.xgb_params,
                "lgb_params": self.lgb_params,
                "random_state": self.random_state,
            },
        }

        joblib.dump(artifact, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        artifact = joblib.load(path)

        self.models = artifact["models"]
        self.meta_learner = artifact["meta_learner"]
        self.scaler = artifact["scaler"]
        self.feature_names = artifact["feature_names"]
        self.classes_ = artifact.get("classes_", np.array([0, 1]))
        self.is_fitted = artifact["is_fitted"]

        config = artifact["config"]
        self.xgb_params = config["xgb_params"]
        self.lgb_params = config["lgb_params"]
        # smote_ratio was removed (dead parameter) — ignore if present in old artifacts
        self.random_state = config["random_state"]

        logger.info(f"Model loaded from {path}")
        return self


if __name__ == "__main__":
    print("MortgageEnsembleModel ready")
