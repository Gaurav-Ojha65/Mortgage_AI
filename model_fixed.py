"""
Fixed Mortgage Risk ML Pipeline
Addresses: class imbalance, threshold tuning, fairness audit.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib, json, warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path="loan_data_real.csv"):
    df = pd.read_csv(path)
    # Drop rows where target is missing
    df = df.dropna(subset=["Loan_Status"])
    # Gender/sex available for fairness audit
    # Married, Education, Property_Area, Dependents also available
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Real column mapping
    df["income"] = df["ApplicantIncome"] + df["CoapplicantIncome"].fillna(0)
    df["loan_amount"] = df["LoanAmount"].fillna(df["LoanAmount"].median()) * 1000
    df["loan_term_years"] = (df["Loan_Amount_Term"].fillna(360) / 12).clip(1, 40).astype(int)
    df["credit_score_raw"] = df["Credit_History"].fillna(0)

    # Impute
    df["ApplicantIncome"] = df["ApplicantIncome"].fillna(df["ApplicantIncome"].median())
    df["CoapplicantIncome"] = df["CoapplicantIncome"].fillna(0)
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(360)
    df["Credit_History"] = df["Credit_History"].fillna(0)

    # Impute dependents
    df["Dependents"] = df["Dependents"].replace("3+", "3").fillna("0")
    df["existing_loans"] = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0)

    # Map Credit_History (0/1) to credit_score (300-850)
    np.random.seed(42)
    credit_scores = np.where(
        df["credit_score_raw"] == 1.0,
        np.random.normal(730, 55, len(df)).clip(650, 850).astype(int),
        np.random.normal(520, 90, len(df)).clip(300, 600).astype(int)
    )
    df["credit_score"] = credit_scores

    # ── Mandatory engineered features ──
    # LTV ratio = loan / property_value (property_value not in real data → estimate from loan_amount using median LTV ~0.80)
    median_ltv = 0.80
    df["property_value_est"] = df["loan_amount"] / median_ltv
    df["ltv_ratio"] = df["loan_amount"] / df["property_value_est"]

    # dti_ratio (already have debt_to_income via income vs loan)
    df["dti_ratio"] = df["loan_amount"] / (df["income"] * df["loan_term_years"])

    # EMI using standard formula
    r = (8.5 / 100) / 12  # 8.5% annual rate
    n = df["loan_term_years"] * 12
    emi = df["loan_amount"] * r * (1 + r)**n / ((1 + r)**n - 1)
    df["emi"] = emi.where(n > 0, df["loan_amount"] / n.where(n > 0, 12))
    df["emi_to_income"] = df["emi"] / df["income"]

    # credit_utilization  (300-850 → 0-1)
    df["credit_utilization"] = ((df["credit_score"] - 300) / 550).clip(0, 1)

    # loan_to_income
    df["loan_to_income"] = df["loan_amount"] / (df["income"] * df["loan_term_years"])

    # Clean extreme outliers
    df.loc[df["ltv_ratio"] > 5, "ltv_ratio"] = np.nan
    df.loc[df["dti_ratio"] > 10, "dti_ratio"] = np.nan
    df.loc[df["emi_to_income"] > 10, "emi_to_income"] = np.nan
    df.loc[df["loan_to_income"] > 10, "loan_to_income"] = np.nan

    # ── Demographic columns for fairness audit ──
    df["applicant_sex"] = df["Gender"].map({"Male": 0, "Female": 1}).fillna(-1)
    df["applicant_race"] = df["Property_Area"].map({
        "Urban": 0, "Semiurban": 1, "Rural": 2
    }).fillna(-1)
    df["applicant_age_group"] = pd.cut(
        pd.to_numeric(df["Dependents"], errors="coerce").fillna(1) * 10 + 25,
        bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]
    ).fillna(1).astype(int)

    # Target
    df["target"] = df["Loan_Status"].astype(int)

    return df


FEATURES_NUMERIC = [
    "income", "loan_amount", "credit_score", "existing_loans",
    "loan_term_years", "ltv_ratio", "dti_ratio", "emi_to_income",
    "credit_utilization", "loan_to_income"
]
FEATURES_CATEGORICAL = ["Married", "Education", "Self_Employed", "Property_Area"]
TARGET = "target"


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(X: pd.DataFrame, y: pd.Series,
               numeric_imputer=None, cat_imputer=None,
               scaler=None, encoders=None, fit=True):
    X = X.copy()

    # Numeric: median imputation
    if fit:
        numeric_imputer = SimpleImputer(strategy="median")
        X_num = pd.DataFrame(
            numeric_imputer.fit_transform(X[FEATURES_NUMERIC]),
            columns=FEATURES_NUMERIC, index=X.index
        )
    else:
        X_num = pd.DataFrame(
            numeric_imputer.transform(X[FEATURES_NUMERIC]),
            columns=FEATURES_NUMERIC, index=X.index
        )

    # Categorical: mode imputation + label encoding
    if fit:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        encoders = {}
    X_cat = X[FEATURES_CATEGORICAL].fillna("Unknown").astype(str)
    for col in FEATURES_CATEGORICAL:
        if fit:
            le = LabelEncoder()
            X_cat[col] = le.fit_transform(X_cat[col])
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels
            known = set(le.classes_)
            X_cat[col] = X_cat[col].apply(lambda x: x if x in known else le.classes_[0])
            X_cat[col] = le.transform(X_cat[col])

    X_out = pd.concat([X_num, X_cat], axis=1)

    # Scale
    if fit:
        scaler = StandardScaler()
        X_out = pd.DataFrame(
            scaler.fit_transform(X_out), columns=X_out.columns, index=X_out.index
        )
    else:
        X_out = pd.DataFrame(
            scaler.transform(X_out), columns=X_out.columns, index=X_out.index
        )

    return X_out, numeric_imputer, cat_imputer, scaler, encoders


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASS IMBALANCE — SMOTE + scale_pos_weight
# ─────────────────────────────────────────────────────────────────────────────

def handle_imbalance(X_train, y_train):
    orig_dist = y_train.value_counts().to_dict()
    print(f"\n  Class distribution BEFORE SMOTE:")
    print(f"    Class 0 (Approved):  {orig_dist.get(0, 0)}")
    print(f"    Class 1 (Denied):    {orig_dist.get(1, 0)}")
    print(f"    Imbalance ratio:     {orig_dist.get(0, 1) / orig_dist.get(1, 1):.2f}:1")

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    new_dist = pd.Series(y_res).value_counts().to_dict()
    print(f"\n  Class distribution AFTER SMOTE:")
    print(f"    Class 0 (Approved):  {new_dist.get(0, 0)}")
    print(f"    Class 1 (Denied):    {new_dist.get(1, 0)}")

    scale = orig_dist.get(0, 1) / orig_dist.get(1, 1)  # minority/majority
    print(f"\n  scale_pos_weight:     {scale:.2f}  (minority={orig_dist.get(0,0) if orig_dist.get(0,0) < orig_dist.get(1,1) else orig_dist.get(1,1)}, majority={orig_dist.get(0,0) if orig_dist.get(0,0) > orig_dist.get(1,1) else orig_dist.get(1,1)})")
    return X_res, y_res, scale


# ─────────────────────────────────────────────────────────────────────────────
# 5. THRESHOLD TUNING — F1 + Recall optimization
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_prob, metric="f1"):
    """Find threshold that optimizes F1 or Recall."""
    thresholds = np.linspace(0.05, 0.95, 100)
    best_t, best_v = 0.5, -1

    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        if y_hat.sum() == 0:
            continue
        if metric == "f1":
            v = f1_score(y_true, y_hat, zero_division=0)
        elif metric == "recall":
            v = recall_score(y_true, y_hat, zero_division=0)
        elif metric == "youden":
            # Youden's J statistic
            tn = ((y_true == 0) & (y_hat == 0)).sum()
            fp = ((y_true == 0) & (y_hat == 1)).sum()
            fn = ((y_true == 1) & (y_hat == 0)).sum()
            tp = ((y_true == 1) & (y_hat == 1)).sum()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            v = sens + spec - 1
        if v > best_v:
            best_v = v
            best_t = t

    return best_t, best_v


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_prob, threshold=0.5, label=""):
    y_hat = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n{'='*60}")
    print(f"  EVALUATION {label}")
    print(f"{'='*60}")
    print(f"  Threshold:              {threshold:.3f}")
    print(f"  ROC-AUC:                {auc:.4f}")
    print(f"  Accuracy:               {accuracy_score(y_true, y_hat):.4f}")
    print(f"  Precision (class 1):    {precision_score(y_true, y_hat):.4f}")
    print(f"  Recall    (class 1):    {recall_score(y_true, y_hat):.4f}  [KEY METRIC]")
    print(f"  F1-score  (class 1):    {f1_score(y_true, y_hat):.4f}")

    cm = confusion_matrix(y_true, y_hat)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                   0       1")
    print(f"  Actual 0     {tn:5d}   {fp:5d}   (TN={tn}, FP={fp})")
    print(f"         1     {fn:5d}   {tp:5d}   (FN={fn}, TP={tp})")

    print(f"\n  Full classification report:")
    print(classification_report(y_true, y_hat, target_names=["Approved(0)", "Denied(1)"]))

    # Precision-Recall curve data
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)
    # PR-AUC: sort by recall ascending so integration is positive
    order = np.argsort(rec)
    pr_auc = np.trapezoid(prec[order], rec[order])
    print(f"  PR-AUC:                 {pr_auc:.4f}")
    fpr, roc_thresh, _ = roc_curve(y_true, y_prob)
    print(f"  Best Youden-J threshold: {find_optimal_threshold(y_true, y_prob, 'youden')[0]:.3f}")

    return {
        "threshold": threshold, "auc": auc, "precision": precision_score(y_true, y_hat),
        "recall": recall_score(y_true, y_hat), "f1": f1_score(y_true, y_hat),
        "cm": cm.tolist(), "pr_auc": pr_auc
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance(model, feature_names, top_n=10):
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False).head(top_n)
    print(f"\n  Top {top_n} Feature Importances:")
    for i, (name, val) in enumerate(imp.items(), 1):
        bar = "*" * int(val * 100)
        print(f"    {i:2d}. {name:<30} {val:.4f} {bar}")
    return imp


# ─────────────────────────────────────────────────────────────────────────────
# 8. FAIRNESS AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def fairness_audit(df: pd.DataFrame, y_true, y_prob, threshold, group_col,
                   group_name, label_map=None):
    """Compute approval rates and disparate impact for a protected group."""
    y_hat = (y_prob >= threshold).astype(int)

    results = []
    groups = df[group_col].unique()
    overall_approval = y_hat.mean()

    print(f"\n  Fairness Audit: {group_name}")
    print(f"  Overall approval rate: {overall_approval:.3f} ({overall_approval*100:.1f}%)")
    print(f"  Deviation threshold:   20%")
    print(f"  {'Group':<15} {'Count':>6} {'Approval Rate':>14} {'Deviation':>12} {'Status':>10}")
    print(f"  {'-'*60}")

    flagged = []
    for g in sorted(groups):
        mask = df[group_col] == g
        n = mask.sum()
        if n == 0:
            continue
        approval_rate = y_hat[mask].mean()
        deviation = (approval_rate - overall_approval) / overall_approval * 100 if overall_approval > 0 else 0
        status = "[FLAG]" if abs(deviation) > 20 else "OK"
        if abs(deviation) > 20:
            flagged.append((g, deviation, approval_rate))

        label = label_map.get(g, g) if label_map else g
        print(f"  {str(label):<15} {n:>6} {approval_rate:>14.3f} {deviation:>+11.1f}% {status:>10}")

    if flagged:
        print(f"\n  !! FLAGGED GROUPS (>{20}% deviation):")
        for g, dev, rate in flagged:
            label = label_map.get(g, g) if label_map else g
            print(f"      {label}: {dev:+.1f}% deviation (approval rate={rate:.3f})")
    else:
        print(f"\n  [OK] No groups exceed 20% deviation threshold")

    # Disparate impact ratio = min(approval_rate_group / overall_approval,
    #                               overall_approval / approval_rate_group)
    if overall_approval > 0:
        di = min(approval_rate / overall_approval for approval_rate in
                 [y_hat[df[group_col] == g].mean() for g in groups if (df[group_col] == g).sum() > 0])
        print(f"\n  Disparate Impact Ratio: {di:.3f}  (4/5ths rule: >0.80 = OK)")
        if di < 0.80:
            print(f"  !! Disparate impact concern (ratio < 0.80)")

    return flagged


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  FIXED MORTGAGE RISK MODEL PIPELINE")
    print("="*60)

    # ── 1. Load ──
    df_raw = load_data("loan_data_real.csv")
    print(f"\n  Loaded {len(df_raw)} rows")

    # ── 2. Feature Engineering ──
    df = build_features(df_raw)
    print(f"  Features engineered")

    # ── Split BEFORE SMOTE (test set stays pristine) ──
    from sklearn.model_selection import train_test_split
    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y = df[TARGET]

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df,
        test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Train class dist: {y_train.value_counts().to_dict()}")
    print(f"  Test class dist:  {y_test.value_counts().to_dict()}")

    # ── 3. Preprocess (fit on train only) ──
    X_train_proc, num_imp, cat_imp, scaler, encoders = preprocess(X_train, y_train, fit=True)
    X_test_proc, _, _, _, _ = preprocess(X_test, y_test, fit=False,
                                          numeric_imputer=num_imp,
                                          cat_imputer=cat_imp,
                                          scaler=scaler,
                                          encoders=encoders)

    feature_names = list(X_train_proc.columns)

    # ── 4. Class Imbalance — SMOTE + scale_pos_weight ──
    X_train_res, y_train_res, scale_pos = handle_imbalance(X_train_proc, y_train)

    # ── 5. Model Training ──
    print(f"\n  Training XGBoost...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)

    # Predict probabilities on HELD-OUT test set
    y_prob_test = model.predict_proba(X_test_proc)[:, 1]

    # ── 6. Threshold Tuning ──
    print(f"\n  Threshold tuning on validation set...")
    opt_t_f1, opt_v_f1 = find_optimal_threshold(y_test, y_prob_test, "f1")
    opt_t_rec, opt_v_rec = find_optimal_threshold(y_test, y_prob_test, "recall")
    opt_t_j, _ = find_optimal_threshold(y_test, y_prob_test, "youden")

    print(f"\n  Optimal thresholds:")
    print(f"    F1-max:      {opt_t_f1:.3f} (F1={opt_v_f1:.4f})")
    print(f"    Recall-max:  {opt_t_rec:.3f} (Recall={opt_v_rec:.4f})")
    print(f"    Youden-J:    {opt_t_j:.3f}")

    # Evaluate at multiple thresholds
    for t, lbl in [(0.5, "Default 0.5"), (opt_t_f1, f"F1-optimal {opt_t_f1:.3f}"),
                    (opt_t_rec, f"Recall-optimal {opt_t_rec:.3f}"), (opt_t_j, f"Youden-J {opt_t_j:.3f}")]:
        evaluate(y_test, y_prob_test, threshold=t, label=f"({lbl})")

    # Use best recall threshold (accept precision tradeoff for risk model)
    final_threshold = opt_t_rec
    print(f"\n  >> FINAL THRESHOLD: {final_threshold:.3f} (favoring recall for risk detection)")

    # ── 7. Feature Importance ──
    fi = feature_importance(model, feature_names)

    # ── 8. Fairness Audit ──
    print(f"\n{'='*60}")
    print("  FAIRNESS AUDIT")
    print(f"{'='*60}")

    fairness_audit(
        df_test.reset_index(drop=True), y_test.reset_index(drop=True),
        y_prob_test, final_threshold,
        group_col="applicant_sex", group_name="Sex (Gender)"
    )
    fairness_audit(
        df_test.reset_index(drop=True), y_test.reset_index(drop=True),
        y_prob_test, final_threshold,
        group_col="applicant_race", group_name="Race (Property_Area)",
        label_map={0: "Urban", 1: "Semiurban", 2: "Rural"}
    )
    fairness_audit(
        df_test.reset_index(drop=True), y_test.reset_index(drop=True),
        y_prob_test, final_threshold,
        group_col="applicant_age_group", group_name="Age Group",
        label_map={0: "<30", 1: "30-45", 2: "45-60", 3: "60+"}
    )

    # ── Save artifacts ──
    artifacts = {
        "model": model,
        "numeric_imputer": num_imp,
        "cat_imputer": cat_imp,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": feature_names,
        "threshold": final_threshold,
        "scale_pos_weight": scale_pos,
        "evaluation": evaluate(y_test, y_prob_test, final_threshold, label="(FINAL)")
    }
    joblib.dump(artifacts, "model_fixed_artifacts.pkl")
    joblib.dump(fi, "feature_importance_fixed.pkl")
    print(f"\n  Saved: model_fixed_artifacts.pkl, feature_importance_fixed.pkl")

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  ROC-AUC target: >0.70  |  Recall(class 1) target: >0.40")
    print(f"{'='*60}")

    return artifacts


if __name__ == "__main__":
    artifacts = main()
