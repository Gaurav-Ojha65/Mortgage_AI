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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MortgageEnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_params=None, lgb_params=None, nn_epochs=100, smote_ratio=0.5, random_state=42, use_shap=True):
        self.xgb_params = xgb_params or {"n_estimators":300,"max_depth":6,"learning_rate":0.05,"subsample":0.8,"colsample_bytree":0.8,"random_state":42,"n_jobs":-1,"eval_metric":"auc"}
        self.lgb_params = lgb_params or {"n_estimators":300,"max_depth":-1,"learning_rate":0.05,"num_leaves":31,"subsample":0.8,"colsample_bytree":0.8,"random_state":42,"n_jobs":-1,"verbose":-1}
        self.smote_ratio = smote_ratio
        self.random_state = random_state
        self.use_shap = use_shap
        self.models = {}
        self.meta_learner = None
        self.scaler = RobustScaler()
        self.feature_names = None
        self.shap_explainer = None
        self.is_fitted = False

    def _create_meta_features(self, X, y=None, fit=False):
        n = X.shape[0]
        mf = np.zeros((n, 2))
        if fit:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            xm = xgb.XGBClassifier(**self.xgb_params)
            mf[:, 0] = cross_val_predict(xm, X, y, cv=skf, method="predict_proba")[:, 1]
            self.models["xgb"] = xm.fit(X, y)
            lm = lgb.LGBMClassifier(**self.lgb_params)
            mf[:, 1] = cross_val_predict(lm, X, y, cv=skf, method="predict_proba")[:, 1]
            self.models["lgb"] = lm.fit(X, y)
        else:
            mf[:, 0] = self.models["xgb"].predict_proba(X)[:, 1]
            mf[:, 1] = self.models["lgb"].predict_proba(X)[:, 1]
        return mf

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        Xs = self.scaler.fit_transform(X)
        try:
            sm = SMOTE(sampling_strategy=self.smote_ratio, random_state=self.random_state)
            Xr, yr = sm.fit_resample(Xs, y)
        except ValueError:
            sm = SMOTE(sampling_strategy="auto", random_state=self.random_state)
            Xr, yr = sm.fit_resample(Xs, y)
        mf = self._create_meta_features(Xr, yr, fit=True)
        self.meta_learner = LogisticRegression(C=1.0, class_weight="balanced", random_state=self.random_state)
        self.meta_learner.fit(mf, yr)
        if self.use_shap:
            try:
                self.shap_explainer = {"xgb": shap.TreeExplainer(self.models["xgb"]), "lgb": shap.TreeExplainer(self.models["lgb"])}
            except Exception as e:
                logger.warning(f"SHAP failed: {e}")
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.meta_learner.predict(self._create_meta_features(self.scaler.transform(X)))

    def predict_proba(self, X):
        return self.meta_learner.predict_proba(self._create_meta_features(self.scaler.transform(X)))

    def evaluate(self, X, y):
        p = self.predict(X)
        pr = self.predict_proba(X)[:, 1]
        return {"accuracy": accuracy_score(y,p), "precision": precision_score(y,p,zero_division=0), "recall": recall_score(y,p,zero_division=0), "f1": f1_score(y,p,zero_division=0), "roc_auc": roc_auc_score(y,pr), "confusion_matrix": confusion_matrix(y,p).tolist()}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"models":self.models,"meta_learner":self.meta_learner,"scaler":self.scaler,"feature_names":self.feature_names,"is_fitted":self.is_fitted,"config":{"xgb_params":self.xgb_params,"lgb_params":self.lgb_params,"smote_ratio":self.smote_ratio,"random_state":self.random_state}}, f)

    def load(self, path):
        with open(path, "rb") as f:
            a = pickle.load(f)
        self.models=a["models"]; self.meta_learner=a["meta_learner"]; self.scaler=a["scaler"]
        self.feature_names=a["feature_names"]; self.is_fitted=a["is_fitted"]
        return self

if __name__ == "__main__":
    print("MortgageEnsembleModel ready")
