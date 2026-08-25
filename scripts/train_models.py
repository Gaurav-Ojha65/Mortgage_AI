"""
Multi-Model Credit Scoring Trainer
Trains LogisticRegression (baseline), XGBoost, and LightGBM.
Saves all models + a comparison report to /models/
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix
)
from sklearn.datasets        import make_classification
import xgboost  as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

# ─── 1. DATA ─────────────────────────────────────────────────────────────────
# Replace this block with your real data loader:
#   df = pd.read_csv("your_data.csv")
#   X  = df[FEATURE_COLS]
#   y  = df["approved"]   # 1 = approved, 0 = rejected

print("Generating synthetic credit dataset (replace with your real data)...")

X_raw, y = make_classification(
    n_samples    = 10_000,
    n_features   = 15,
    n_informative= 10,
    n_redundant  = 3,
    weights      = [0.75, 0.25],   # realistic class imbalance
    random_state = 42,
)

FEATURE_NAMES = [
    "credit_score", "annual_income", "loan_amount", "loan_term",
    "dti_ratio", "employment_years", "num_credit_lines",
    "num_derogatory_marks", "credit_utilization", "late_payment_severity_score",
    "home_ownership", "purpose_encoded", "num_late_payments",
    "savings_balance", "monthly_expenses",
]
X = pd.DataFrame(X_raw, columns=FEATURE_NAMES)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"   Approval rate: {y.mean():.1%}")

# ─── 2. MODEL DEFINITIONS ────────────────────────────────────────────────────

models = {

    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter    = 1000,
            class_weight= "balanced",
            random_state= 42,
        )),
    ]),

    "XGBoost": xgb.XGBClassifier(
        n_estimators      = 400,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = (y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric       = "auc",
        early_stopping_rounds = 30,
        random_state      = 42,
        verbosity         = 0,
    ),

    "LightGBM": lgb.LGBMClassifier(
        n_estimators      = 400,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        class_weight      = "balanced",
        early_stopping_rounds = 30,
        random_state      = 42,
        verbose           = -1,
    ),
}

# ─── 3. TRAIN + EVALUATE ─────────────────────────────────────────────────────

results = {}
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTraining {name}...")
    t0 = time.time()

    # Fit (XGBoost + LightGBM support eval_set for early stopping)
    if name == "XGBoost":
        model.fit(
            X_train, y_train,
            eval_set        = [(X_test, y_test)],
            verbose         = False,
        )
    elif name == "LightGBM":
        model.fit(
            X_train, y_train,
            eval_set        = [(X_test, y_test)],
            callbacks       = [lgb.early_stopping(30, verbose=False),
                               lgb.log_evaluation(-1)],
        )
    else:
        model.fit(X_train, y_train)

    elapsed = time.time() - t0

    # Predictions
    y_pred     = model.predict(X_test)
    y_prob     = model.predict_proba(X_test)[:, 1]

    # Cross-val AUC — use a copy without early_stopping for CV compatibility
    if name == "XGBoost":
        cv_model = xgb.XGBClassifier(
            n_estimators=model.n_estimators, max_depth=model.max_depth,
            learning_rate=model.learning_rate, subsample=model.subsample,
            colsample_bytree=model.colsample_bytree,
            scale_pos_weight=model.scale_pos_weight,
            random_state=42, verbosity=0,
        )
    elif name == "LightGBM":
        cv_model = lgb.LGBMClassifier(
            n_estimators=model.n_estimators, max_depth=model.max_depth,
            learning_rate=model.learning_rate, subsample=model.subsample,
            colsample_bytree=model.colsample_bytree,
            class_weight="balanced", random_state=42, verbose=-1,
        )
    else:
        cv_model = model

    cv_aucs = cross_val_score(
        cv_model, X_train, y_train,
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )

    metrics = {
        "accuracy"  : round(accuracy_score(y_test,  y_pred),  4),
        "roc_auc"   : round(roc_auc_score(y_test,   y_prob),  4),
        "f1"        : round(f1_score(y_test,         y_pred),  4),
        "precision" : round(precision_score(y_test,  y_pred),  4),
        "recall"    : round(recall_score(y_test,     y_pred),  4),
        "cv_auc_mean": round(cv_aucs.mean(), 4),
        "cv_auc_std" : round(cv_aucs.std(),  4),
        "train_time_s": round(elapsed, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    results[name] = metrics
    print(f"   AUC: {metrics['roc_auc']:.4f}  |  F1: {metrics['f1']:.4f}  |  "
          f"CV-AUC: {metrics['cv_auc_mean']:.4f} +/- {metrics['cv_auc_std']:.4f}  |  "
          f"Time: {elapsed:.1f}s")

    # Save model
    joblib.dump(model, f"models/{name.lower().replace(' ', '_')}.joblib")

# ─── 4. FEATURE IMPORTANCE ───────────────────────────────────────────────────

importance_data = {}

for name, model in models.items():
    if name == "LogisticRegression":
        clf  = model.named_steps["clf"]
        imps = np.abs(clf.coef_[0])
    elif name == "XGBoost":
        imps = model.feature_importances_
    elif name == "LightGBM":
        imps = model.feature_importances_

    # Normalise to 0-100
    imps = 100 * imps / imps.sum()
    importance_data[name] = dict(zip(FEATURE_NAMES, imps.round(2).tolist()))

# ─── 5. PICK WINNER + SAVE REPORT ────────────────────────────────────────────

winner = max(results, key=lambda m: results[m]["roc_auc"])

report = {
    "winner"      : winner,
    "metrics"     : results,
    "importance"  : importance_data,
    "feature_names": FEATURE_NAMES,
}

with open("models/comparison_report.json", "w") as f:
    json.dump(report, f, indent=2)

# Save winner as the active model
best_model = models[winner]
joblib.dump(best_model, "models/best_model.joblib")
with open("models/best_model_name.txt", "w") as f:
    f.write(winner)

# ─── 6. PRINT SUMMARY ────────────────────────────────────────────────────────

print("\n" + "=" * 56)
print(f"  Winner: {winner}")
print("=" * 56)
print(f"{'Model':<22} {'AUC':>6} {'F1':>6} {'Precision':>10} {'Recall':>8}")
print("-" * 56)
for name, m in results.items():
    flag = " <- winner" if name == winner else ""
    print(f"{name:<22} {m['roc_auc']:>6.4f} {m['f1']:>6.4f} "
          f"{m['precision']:>10.4f} {m['recall']:>8.4f}{flag}")
print("=" * 56)
print(f"\nModels saved to /models/")
print(f"Report saved  to /models/comparison_report.json")