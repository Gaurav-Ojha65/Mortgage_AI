"""
Production Ensemble Model for Mortgage AI
XGBoost + LightGBM + Neural Network with Stacking
Includes SHAP explainability and MLflow tracking
"""

import os
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import mlflow
import mlflow.sklearn
import shap

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MortgageEnsembleModel(BaseEstimator, ClassifierMixin):
    """
    Production ensemble model combining XGBoost, LightGBM, and Neural Network.
    Uses stacking with logistic regression as meta-learner.
    Includes SMOTE for class imbalance and SHAP explainability.
    """

    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        lgb_params: Optional[Dict] = None,
        nn_epochs: int = 100,
        nn_batch_size: int = 256,
        smote_ratio: float = 0.5,
        random_state: int = 42,
        use_shap: bool = True
    ):
        self.xgb_params = xgb_params or self._default_xgb_params()
        self.lgb_params = lgb_params or self._default_lgb_params()
        self.nn_epochs = nn_epochs
        self.nn_batch_size = nn_batch_size
        self.smote_ratio = smote_ratio
        self.random_state = random_state
        self.use_shap = use_shap

        self.models = {}
        self.meta_learner = None
        self.scaler = RobustScaler()
        self.feature_names = None
        self.shap_explainer = None
        self.shap_values = None
        self.is_fitted = False

    def _default_xgb_params(self) -> Dict:
        """Default XGBoost parameters optimized for credit risk."""
        return {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'gamma': 0.1,
            'scale_pos_weight': 5,  # Handle imbalance
            'random_state': self.random_state,
            'n_jobs': -1,
            'eval_metric': 'auc',
            'early_stopping_rounds': 50
        }

    def _default_lgb_params(self) -> Dict:
        """Default LightGBM parameters optimized for credit risk."""
        return {
            'n_estimators': 500,
            'max_depth': -1,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 20,
            'scale_pos_weight': 5,
            'random_state': self.random_state,
            'n_jobs': -1,
            'metric': 'auc',
            'early_stopping_rounds': 50
        }

    def _build_neural_network(self, n_features: int) -> keras.Model:
        """Build neural network with regularization for credit risk."""
        model = models.Sequential([
            layers.Input(shape=(n_features,)),

            layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),

            layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),

            layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),

            layers.Dense(32, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['AUC', 'Precision', 'Recall']
        )
        return model

    def _create_meta_features(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        fit: bool = False
    ) -> np.ndarray:
        """Generate meta-features from base models using cross-validation."""
        n_samples = X.shape[0]
        meta_features = np.zeros((n_samples, 3))

        if fit:
            # Use cross-validation to generate meta-features
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

            # XGBoost
            logger.info("Training XGBoost...")
            xgb_model = xgb.XGBClassifier(**self.xgb_params)
            meta_features[:, 0] = cross_val_predict(
                xgb_model, X, y, cv=skf, method='predict_proba'
            )[:, 1]
            self.models['xgb'] = xgb_model.fit(X, y)

            # LightGBM
            logger.info("Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(**self.lgb_params)
            meta_features[:, 1] = cross_val_predict(
                lgb_model, X, y, cv=skf, method='predict_proba'
            )[:, 1]
            self.models['lgb'] = lgb_model.fit(X, y)

            # Neural Network
            logger.info("Training Neural Network...")
            nn_predictions = np.zeros(n_samples)
            nn_models = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                nn_model = self._build_neural_network(X.shape[1])
                early_stop = callbacks.EarlyStopping(
                    monitor='val_loss', patience=15, restore_best_weights=True
                )
                reduce_lr = callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
                )

                nn_model.fit(
                    X_train_fold, y_train_fold,
                    validation_data=(X_val_fold, y_val_fold),
                    epochs=self.nn_epochs,
                    batch_size=self.nn_batch_size,
                    callbacks=[early_stop, reduce_lr],
                    verbose=0
                )
                nn_predictions[val_idx] = nn_model.predict(X_val_fold, verbose=0).flatten()
                nn_models.append(nn_model)

            meta_features[:, 2] = nn_predictions
            self.models['nn'] = nn_models[0]  # Save first model for prediction

        else:
            # Predict using fitted models
            meta_features[:, 0] = self.models['xgb'].predict_proba(X)[:, 1]
            meta_features[:, 1] = self.models['lgb'].predict_proba(X)[:, 1]
            meta_features[:, 2] = self.models['nn'].predict(X, verbose=0).flatten()

        return meta_features

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit the ensemble model.

        Args:
            X: Training features
            y: Training labels
            feature_names: Optional list of feature names for SHAP
        """
        logger.info("Starting ensemble model training...")

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Scale features for neural network
        X_scaled = self.scaler.fit_transform(X)

        # Apply SMOTE for class imbalance
        logger.info("Applying SMOTE...")
        smote = SMOTE(sampling_strategy=self.smote_ratio, random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        logger.info(f"After SMOTE: {np.bincount(y_resampled)}")

        # Create meta-features
        meta_features = self._create_meta_features(X_resampled, y_resampled, fit=True)

        # Train meta-learner
        logger.info("Training meta-learner...")
        self.meta_learner = LogisticRegression(
            C=1.0, class_weight='balanced', random_state=self.random_state
        )
        self.meta_learner.fit(meta_features, y_resampled)

        # Initialize SHAP explainers
        if self.use_shap:
            logger.info("Initializing SHAP explainers...")
            self._init_shap_explainer(X_scaled[:100])  # Sample for background

        self.is_fitted = True
        logger.info("Ensemble training complete!")
        return self

    def _init_shap_explainer(self, X_background: np.ndarray):
        """Initialize SHAP explainers for each model."""
        try:
            # XGBoost SHAP
            self.shap_explainer = {
                'xgb': shap.TreeExplainer(self.models['xgb']),
                'lgb': shap.TreeExplainer(self.models['lgb'])
            }
        except Exception as e:
            logger.warning(f"Could not initialize TreeExplainer: {e}")
            self.shap_explainer = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        meta_features = self._create_meta_features(X_scaled, fit=False)
        return self.meta_learner.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        meta_features = self._create_meta_features(X_scaled, fit=False)
        return self.meta_learner.predict_proba(meta_features)

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

        X_scaled = self.scaler.transform(X)

        explanations = {
            'base_value': 0.5,
            'shap_values': {},
            'feature_importance': {}
        }

        if self.shap_explainer:
            try:
                # Get SHAP values from XGBoost
                shap_values_xgb = self.shap_explainer['xgb'].shap_values(X_scaled)
                explanations['shap_values']['xgb'] = shap_values_xgb.tolist()

                # Feature importance from mean absolute SHAP values
                importance = np.abs(shap_values_xgb).mean(axis=0)
                explanations['feature_importance'] = {
                    name: float(imp) for name, imp in zip(feature_names, importance)
                }
            except Exception as e:
                logger.error(f"SHAP explanation failed: {e}")

        return explanations

    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        X_scaled = self.scaler.transform(X)
        return {
            'xgb': self.models['xgb'].predict_proba(X_scaled)[:, 1],
            'lgb': self.models['lgb'].predict_proba(X_scaled)[:, 1],
            'nn': self.models['nn'].predict(X_scaled, verbose=0).flatten(),
            'ensemble': self.predict_proba(X_scaled)[:, 1]
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)[:, 1]

        return {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1': f1_score(y, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y, probabilities),
            'confusion_matrix': confusion_matrix(y, predictions).tolist()
        }

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            'models': {
                'xgb': self.models['xgb'],
                'lgb': self.models['lgb'],
                'nn': self.models['nn'],
                'meta_learner': self.meta_learner
            },
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'config': {
                'xgb_params': self.xgb_params,
               lgb_params': self.lgb_params,
                'nn_epochs': self.nn_epochs,
                'smote_ratio': self.smote_ratio,
                'random_state': self.random_state
            }
        }

        with open(path, 'wb') as f:
            pickle.dump(artifact, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            artifact = pickle.load(f)

        self.models = artifact['models']
        self.scaler = artifact['scaler']
        self.feature_names = artifact['feature_names']
        self.is_fitted = artifact['is_fitted']

        config = artifact['config']
        self.xgb_params = config['xgb_params']
        self.lgb_params = config['lgb_params']
        self.nn_epochs = config['nn_epochs']
        self.smote_ratio = config['smote_ratio']
        self.random_state = config['random_state']

        logger.info(f"Model loaded from {path}")
        return self


def train_with_mlflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    experiment_name: str = "mortgage_ensemble"
) -> MortgageEnsembleModel:
    """Train ensemble model with MLflow tracking."""

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'model_type': 'ensemble_xgb_lgb_nn',
            'n_features': X_train.shape[1],
            'n_train': len(X_train),
            'smote_ratio': 0.5
        })

        # Train model
        model = MortgageEnsembleModel(
            nn_epochs=100,
            smote_ratio=0.5,
            use_shap=True
        )
        model.fit(X_train, y_train, feature_names)

        # Evaluate
        train_metrics = model.evaluate(X_train, y_train)
        val_metrics = model.evaluate(X_val, y_val)

        # Log metrics
        for split, metrics in [('train', train_metrics), ('val', val_metrics)]:
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{split}_{metric}", value)

        # Log model
        mlflow.sklearn.log_model(model, "ensemble_model")

        logger.info(f"Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")

    return model


if __name__ == "__main__":
    # Example usage
    print("MortgageEnsembleModel - Production ensemble for credit risk")
