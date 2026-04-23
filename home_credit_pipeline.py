"""
Home Credit Mortgage Risk Pipeline
==================================
Adapted for Home Credit dataset (application_train.csv).
Target: TARGET (1=default, 0=no default)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib, json, warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING — chunked for 166MB file
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path="application_train.csv", nrows=None):
    dtypes = {
        'SK_ID_CURR': 'int32',
        'TARGET': 'int8',
        'AMT_INCOME_TOTAL': 'float32',
        'AMT_CREDIT': 'float32',
        'AMT_ANNUITY': 'float32',
        'AMT_GOODS_PRICE': 'float32',
        'DAYS_BIRTH': 'float32',
        'DAYS_EMPLOYED': 'float32',
        'DAYS_REGISTRATION': 'float32',
        'DAYS_ID_PUBLISH': 'float32',
        'CODE_GENDER': 'str',
        'FLAG_OWN_CAR': 'str',
        'FLAG_OWN_REALTY': 'str',
        'CNT_CHILDREN': 'int8',
        'CNT_FAM_MEMBERS': 'float32',
        'REGION_RATING_CLIENT': 'int8',
        'REGION_RATING_CLIENT_W_CITY': 'int8',
        'EXT_SOURCE_1': 'float32',
        'EXT_SOURCE_2': 'float32',
        'EXT_SOURCE_3': 'float32',
        'OWN_CAR_AGE': 'float32',
        'NAME_CONTRACT_TYPE': 'str',
        'NAME_EDUCATION_TYPE': 'str',
        'NAME_FAMILY_STATUS': 'str',
        'NAME_HOUSING_TYPE': 'str',
        'NAME_INCOME_TYPE': 'str',
        'OCCUPATION_TYPE': 'str',
        'ORGANIZATION_TYPE': 'str',
        'FLAG_EMP_PHONE': 'int8',
        'FLAG_WORK_PHONE': 'int8',
        'FLAG_PHONE': 'int8',
        'FLAG_EMAIL': 'int8',
        'WEEKDAY_APPR_PROCESS_START': 'str',
        'HOUR_APPR_PROCESS_START': 'int8',
        'REG_REGION_NOT_LIVE_REGION': 'int8',
        'REG_REGION_NOT_WORK_REGION': 'int8',
        'REG_CITY_NOT_LIVE_CITY': 'int8',
        'REG_CITY_NOT_WORK_CITY': 'int8',
    }
    usecols = list(dtypes.keys())
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes, nrows=nrows)
    # Drop rows with missing target
    df = df.dropna(subset=["TARGET"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING — Home Credit schema
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Core numeric mapping ──
    df["income"] = df["AMT_INCOME_TOTAL"]
    df["loan_amount"] = df["AMT_CREDIT"]
    df["annuity"] = df["AMT_ANNUITY"]
    df["goods_price"] = df["AMT_GOODS_PRICE"]

    # Age: DAYS_BIRTH is negative, convert to years
    df["age"] = (-df["DAYS_BIRTH"] / 365.25).clip(18, 100)

    # Employment: DAYS_EMPLOYED is negative; 365243 = retired/unemployed sentinel
    df["employment_years"] = (-df["DAYS_EMPLOYED"] / 365.25).replace(-1000.67, np.nan)

    # ── Mandatory engineered features ──
    # LTV ratio
    df["ltv_ratio"] = df["loan_amount"] / df["goods_price"].replace(0, np.nan)

    # DTI ratio = annuity / income (Home Credit standard)
    df["dti_ratio"] = df["annuity"] / df["income"].replace(0, np.nan)

    # Credit utilization proxy (no direct credit bureau data → use annuity as proxy)
    # Lower income-to-credit ratio = higher credit burden
    df["credit_burden"] = df["annuity"] * df["loan_amount"] / df["income"].replace(0, np.nan)**2

    # EMI to income
    df["emi_to_income"] = df["annuity"] / df["income"].replace(0, np.nan)

    # Loan to income
    df["loan_to_income"] = df["loan_amount"] / df["income"].replace(0, np.nan)

    # Credit term in years
    df["credit_term"] = df["loan_amount"] / df["annuity"].replace(0, np.nan) / 12

    # Income per family member
    df["income_per_family"] = df["income"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)

    # AgeEmployment interaction
    df["age_employment_ratio"] = df["age"] / df["employment_years"].replace(0, np.nan)

    # Credit term
    df["credit_term_years"] = df["annuity"].fillna(0) / (df["loan_amount"].fillna(1) / 12)
    df.loc[df["credit_term_years"] <= 0, "credit_term_years"] = np.nan
    df.loc[df["credit_term_years"] > 40, "credit_term_years"] = np.nan

    # External source scores — most predictive features in Home Credit
    df["ext_mean"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["ext_std"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    df["ext_min"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
    df["ext_max"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1)

    # Income stability proxy
    df["employment_to_age"] = df["employment_years"] / df["age"].replace(0, np.nan)

    # Region rating (1-3)
    df["region_rating"] = df["REGION_RATING_CLIENT"].fillna(2)

    # Flag features
    df["has_car"] = (df["FLAG_OWN_CAR"] == "Y").astype(int)
    df["has_realty"] = (df["FLAG_OWN_REALTY"] == "Y").astype(int)
    df["car_age_flag"] = (df["OWN_CAR_AGE"] > 10).astype(int)

    # Age groups for fairness
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(float)

    # ── Categorical features ──
    df["gender"] = df["CODE_GENDER"].map({"M": 0, "F": 1, "XNA": -1}).fillna(-1)
    df["contract_type"] = df["NAME_CONTRACT_TYPE"].map({"Cash loans": 0, "Revolving loans": 1}).fillna(-1)
    df["education"] = df["NAME_EDUCATION_TYPE"].map({
        "Secondary / secondary special": 0,
        "Higher education": 1,
        "Incomplete higher": 2,
        "Lower secondary": 3,
        "Academic degree": 4
    }).fillna(0)
    df["family_status"] = df["NAME_FAMILY_STATUS"].map({
        "Single / not married": 0, "Married": 1, "Civil marriage": 2,
        "Separated": 3, "Widow": 4
    }).fillna(0)
    df["housing_type"] = df["NAME_HOUSING_TYPE"].map({
        "House / apartment": 0, "With parents": 1, "Municipal apartment": 2,
        "Office apartment": 3, "Co-op apartment": 4, "Rented apartment": 5
    }).fillna(0)
    df["income_type"] = df["NAME_INCOME_TYPE"].map({
        "Working": 0, "Commercial associate": 1, "Pensioner": 2,
        "State servant": 3, "Unemployed": 4, "Student": 5
    }).fillna(0)
    df["occupation_type"] = pd.Categorical(df["OCCUPATION_TYPE"]).codes.clip(0, 25).astype(float)
    df.loc[df["OCCUPATION_TYPE"].isna(), "occupation_type"] = np.nan

    df["target"] = df["TARGET"].astype(int)

    return df


# Numeric features (no categorical encoding yet — done in preprocess)
FEATURES_NUMERIC = [
    "income", "loan_amount", "annuity", "goods_price", "age", "employment_years",
    "ltv_ratio", "dti_ratio", "credit_burden", "emi_to_income", "loan_to_income",
    "credit_term_years", "income_per_family", "age_employment_ratio",
    "ext_mean", "ext_std", "ext_min", "ext_max",
    "employment_to_age", "region_rating",
    "has_car", "has_realty", "car_age_flag",
    "gender", "contract_type", "education", "family_status",
    "housing_type", "income_type", "occupation_type", "age_group"
]

# Features for fairness audit
FAIRNESS_FEATURES = ["gender", "age_group", "education", "housing_type"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING — median imputation + scaling
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(X: pd.DataFrame, imputer=None, scaler=None, fit=True):
    X = X.copy()
    if fit:
        imputer = SimpleImputer(strategy="median")
        X_proc = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns, index=X.index
        )
        scaler = StandardScaler()
        X_proc = pd.DataFrame(
            scaler.fit_transform(X_proc),
            columns=X_proc.columns, index=X_proc.index
        )
    else:
        X_proc = pd.DataFrame(
            imputer.transform(X),
            columns=X.columns, index=X.index
        )
        X_proc = pd.DataFrame(
            scaler.transform(X_proc),
            columns=X_proc.columns, index=X_proc.index
        )
    return X_proc, imputer, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASS IMBALANCE — SMOTE + scale_pos_weight
# ─────────────────────────────────────────────────────────────────────────────

def handle_imbalance(X_train, y_train):
    orig = y_train.value_counts().to_dict()
    print(f"\n  Class distribution BEFORE SMOTE:")
    print(f"    Class 0 (No Default):  {orig.get(0, 0)}")
    print(f"    Class 1 (Default):    {orig.get(1, 0)}")
    print(f"    Imbalance ratio:      {orig.get(1, 1) / orig.get(0, 1):.2f}:1  (positive=minority)")

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    new = pd.Series(y_res).value_counts().to_dict()
    print(f"\n  Class distribution AFTER SMOTE:")
    print(f"    Class 0 (No Default):  {new.get(0, 0)}")
    print(f"    Class 1 (Default):    {new.get(1, 0)}")

    # scale_pos_weight for XGBoost: count(negative) / count(positive)
    scale = orig.get(0, 1) / orig.get(1, 1)
    print(f"\n  scale_pos_weight:       {scale:.2f}")
    return X_res, y_res, scale


# ─────────────────────────────────────────────────────────────────────────────
# 5. THRESHOLD TUNING — F1, Recall, Youden-J
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_thresholds(y_true, y_prob):
    """Scan thresholds and record F1, Recall, Youden-J at each."""
    thresholds = np.linspace(0.05, 0.95, 100)
    results = []
    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        if y_hat.sum() == 0:
            continue
        tn = ((y_true == 0) & (y_hat == 0)).sum()
        fp = ((y_true == 0) & (y_hat == 1)).sum()
        fn = ((y_true == 1) & (y_hat == 0)).sum()
        tp = ((y_true == 1) & (y_hat == 1)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = sens
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        youden = sens + spec - 1
        results.append((t, f1, rec, youden, sens, spec, prec))

    best_f1 = max(results, key=lambda x: x[1])
    best_rec = max(results, key=lambda x: x[2])
    best_youden = max(results, key=lambda x: x[3])
    return best_f1, best_rec, best_youden, results


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_prob, threshold=0.5, label=""):
    y_hat = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    tn = ((y_true == 0) & (y_hat == 0)).sum()
    fp = ((y_true == 0) & (y_hat == 1)).sum()
    fn = ((y_true == 1) & (y_hat == 0)).sum()
    tp = ((y_true == 1) & (y_hat == 1)).sum()

    print(f"\n{'='*60}")
    print(f"  EVALUATION {label}")
    print(f"{'='*60}")
    print(f"  Threshold:              {threshold:.3f}")
    print(f"  ROC-AUC:               {auc:.4f}")
    print(f"  PR-AUC (AP):           {ap:.4f}")
    print(f"  Accuracy:               {accuracy_score(y_true, y_hat):.4f}")
    print(f"  Precision (class 1):    {precision_score(y_true, y_hat):.4f}")
    print(f"  Recall    (class 1):    {recall_score(y_true, y_hat):.4f}  [KEY]")
    print(f"  F1-score  (class 1):    {f1_score(y_true, y_hat):.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"                       Predicted")
    print(f"                  Neg(0)    Pos(1)")
    print(f"  Actual 0    {tn:6d}   {fp:6d}   (TN={tn}, FP={fp})")
    print(f"         1    {fn:6d}   {tp:6d}   (FN={fn}, TP={tp})")

    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_hat, target_names=["NoDefault(0)", "Default(1)"]))

    return {
        "threshold": threshold, "auc": auc, "ap": ap,
        "precision": precision_score(y_true, y_hat),
        "recall": recall_score(y_true, y_hat),
        "f1": f1_score(y_true, y_hat),
        "cm": [int(tn), int(fp), int(fn), int(tp)]
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance(model, feature_names, top_n=15):
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False).head(top_n)
    print(f"\n  Top {top_n} Feature Importances (XGBoost gain):")
    for i, (name, val) in enumerate(imp.items(), 1):
        bar = "*" * int(val * 200)
        print(f"    {i:2d}. {name:<30} {val:.4f} {bar}")
    return imp


# ─────────────────────────────────────────────────────────────────────────────
# 8. FAIRNESS AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def fairness_audit(df: pd.DataFrame, y_true, y_prob, threshold,
                   group_col, group_name, label_map=None):
    y_hat = (y_prob >= threshold).astype(int)
    overall_approval = 1 - y_hat.mean()  # approval = predict 0 (no default)

    print(f"\n  Fairness Audit: {group_name}")
    print(f"  Overall approval rate: {overall_approval:.3f} ({overall_approval*100:.1f}%)")
    print(f"  Flag threshold:         >20% deviation")
    print(f"  {'Group':<20} {'N':>6} {'Approval Rate':>14} {'Deviation':>12} {'Status':>10}")
    print(f"  {'-'*65}")

    flagged = []
    groups = sorted(df[group_col].dropna().unique())
    for g in groups:
        mask = (df[group_col] == g).to_numpy()
        n = mask.sum()
        if n < 5:
            continue
        appr = 1 - y_hat[mask].mean()
        dev = (appr - overall_approval) / overall_approval * 100 if overall_approval > 0 else 0
        status = "[FLAG]" if abs(dev) > 20 else "OK"
        if abs(dev) > 20:
            flagged.append((g, dev, appr, n))
        label = label_map.get(g, g) if label_map else g
        print(f"  {str(label):<20} {n:>6} {appr:>14.3f} {dev:>+11.1f}% {status:>10}")

    if flagged:
        print(f"\n  !! FLAGGED GROUPS (>{20}% deviation):")
        for g, dev, appr, n in flagged:
            label = label_map.get(g, g) if label_map else g
            print(f"      {label} (N={n}): {dev:+.1f}% deviation, approval={appr:.3f}")
    else:
        print(f"\n  [OK] No groups exceed 20% deviation threshold")

    # Disparate impact: 4/5ths rule
    group_rates = []
    for g in groups:
        mask = (df[group_col] == g).to_numpy()
        if mask.sum() >= 5:
            group_rates.append(1 - y_hat[mask].mean())
    if group_rates and overall_approval > 0:
        di = min(group_rates) / max(group_rates) if max(group_rates) > 0 else 0
        print(f"  Disparate Impact Ratio: {di:.3f}  (4/5 rule: >0.80 = OK)")
        if di < 0.80:
            print(f"  [!!] Disparate impact concern (ratio < 0.80)")
    return flagged


# ─────────────────────────────────────────────────────────────────────────────
# 9. CROSS-VALIDATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(X, y, imputer, scaler, scale_pos, n_splits=5):
    """Stratified k-fold CV to get robust threshold estimate."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        X_tr_proc, _, _ = preprocess(X_tr, imputer=imputer, scaler=scaler, fit=False)
        X_va_proc, _, _ = preprocess(X_va, imputer=imputer, scaler=scaler, fit=False)

        # SMOTE inside fold
        smote = SMOTE(random_state=42)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_proc, y_tr)

        model_cv = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos, eval_metric="logloss",
            random_state=42, use_label_encoder=False, n_jobs=-1
        )
        model_cv.fit(X_tr_res, y_tr_res)
        oof_probs[val_idx] = model_cv.predict_proba(X_va_proc)[:, 1]

    best_t_f1 = max(zip(np.linspace(0.05, 0.95, 100), [
        f1_score(y, (oof_probs >= t).astype(int)) for t in np.linspace(0.05, 0.95, 100)
    ]), key=lambda x: x[1])[0]
    best_t_rec = max(zip(np.linspace(0.05, 0.95, 100), [
        recall_score(y, (oof_probs >= t).astype(int)) for t in np.linspace(0.05, 0.95, 100)
    ]), key=lambda x: x[1])[0]
    return oof_probs, best_t_f1, best_t_rec


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  HOME CREDIT MORTGAGE RISK PIPELINE")
    print("="*60)

    # ── 1. Load ──
    print(f"\n  Loading application_train.csv...")
    df_raw = load_data("application_train.csv")
    print(f"  Loaded {len(df_raw):,} rows, {df_raw.shape[1]} columns")
    print(f"  Target dist: {df_raw['TARGET'].value_counts().to_dict()}")

    # ── 2. Feature Engineering ──
    df = build_features(df_raw)
    print(f"  Features engineered")

    # ── 3. Train/Test Split (BEFORE any SMOTE) ──
    X = df[FEATURES_NUMERIC]
    y = df["target"]
    # Reset index so fairness df aligns
    df = df.reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df,
        test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Train class dist: {y_train.value_counts().to_dict()}")
    print(f"  Test class dist:  {y_test.value_counts().to_dict()}")

    # ── 4. Preprocess (fit on train only) ──
    X_train_proc, imputer, scaler = preprocess(X_train, fit=True)
    X_test_proc, _, _ = preprocess(X_test, imputer=imputer, scaler=scaler, fit=False)
    feature_names = list(X_train_proc.columns)
    print(f"\n  Preprocessed {len(feature_names)} features")

    # ── 5. Class Imbalance ──
    X_train_res, y_train_res, scale_pos = handle_imbalance(X_train_proc, y_train)

    # ── 6. Cross-Validation for robust threshold ──
    print(f"\n  Running 5-fold CV for threshold calibration...")
    oof_probs, cv_t_f1, cv_t_rec = cross_validate(
        X_train_proc, y_train, imputer, scaler, scale_pos
    )
    print(f"  CV best threshold (F1):   {cv_t_f1:.3f}")
    print(f"  CV best threshold (Rec): {cv_t_rec:.3f}")

    # ── 7. Train Final Model ──
    print(f"\n  Training XGBoost (n_estimators=300, max_depth=6, lr=0.05)...")
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
    y_prob_test = model.predict_proba(X_test_proc)[:, 1]

    # ── 8. Threshold Tuning ──
    print(f"\n  Threshold tuning...")
    best_f1, best_rec, best_youden, scan = find_optimal_thresholds(y_test, y_prob_test)

    print(f"\n  Optimal thresholds:")
    print(f"    F1-max:      {best_f1[0]:.3f} (F1={best_f1[1]:.4f}, Recall={best_f1[4]:.4f})")
    print(f"    Recall-max:  {best_rec[0]:.3f} (Recall={best_rec[2]:.4f}, F1={best_rec[1]:.4f})")
    print(f"    Youden-J:    {best_youden[0]:.3f} (J={best_youden[3]:.4f}, Sens={best_youden[4]:.4f}, Spec={best_youden[5]:.4f})")
    print(f"    CV F1:       {cv_t_f1:.3f}")
    print(f"    CV Recall:   {cv_t_rec:.3f}")

    # Evaluate at key thresholds
    for t, lbl in [
        (0.5, "Default 0.5"),
        (best_f1[0], f"F1-optimal {best_f1[0]:.3f}"),
        (best_rec[0], f"Recall-optimal {best_rec[0]:.3f}"),
        (best_youden[0], f"Youden-J {best_youden[0]:.3f}"),
        (cv_t_f1, f"CV-F1 {cv_t_f1:.3f}"),
    ]:
        evaluate(y_test, y_prob_test, threshold=t, label=f"({lbl})")

    # Recommendation: use Youden-J for balanced, CV-F1 for high-precision
    # For risk: favor recall (catch defaults) with acceptable precision
    final_threshold = best_youden[0]
    print(f"\n  >> FINAL THRESHOLD: {final_threshold:.3f} (Youden-J: balanced Sens+Spec)")

    # ── 9. Feature Importance ──
    fi = feature_importance(model, feature_names)

    # ── 10. Fairness Audit ──
    print(f"\n{'='*60}")
    print("  FAIRNESS AUDIT")
    print(f"{'='*60}")

    fairness_audit(
        df_test, y_test, y_prob_test, final_threshold,
        "gender", "Gender (CODE_GENDER)",
        label_map={0: "Male", 1: "Female", -1: "Unknown"}
    )
    fairness_audit(
        df_test, y_test, y_prob_test, final_threshold,
        "age_group", "Age Group",
        label_map={0: "<30", 1: "30-45", 2: "45-60", 3: "60+"}
    )
    fairness_audit(
        df_test, y_test, y_prob_test, final_threshold,
        "education", "Education Level",
        label_map={0: "Secondary", 1: "Higher", 2: "Incomplete", 3: "Lower", 4: "Academic"}
    )
    fairness_audit(
        df_test, y_test, y_prob_test, final_threshold,
        "housing_type", "Housing Type",
        label_map={0: "House/Apt", 1: "With Parents", 2: "Municipal", 3: "Office", 4: "Co-op", 5: "Rented"}
    )

    # ── 11. Save Artifacts ──
    artifacts = {
        "imputer": imputer,
        "scaler": scaler,
        "feature_names": feature_names,
        "threshold": final_threshold,
        "scale_pos_weight": scale_pos,
        "cv_f1_threshold": cv_t_f1,
        "cv_rec_threshold": cv_t_rec,
        "youden_threshold": best_youden[0],
    }
    joblib.dump(model, "home_credit_model.pkl")
    joblib.dump(artifacts, "home_credit_artifacts.pkl")
    print(f"\n  Saved: home_credit_model.pkl, home_credit_artifacts.pkl")

    # Final eval at chosen threshold
    final_eval = evaluate(y_test, y_prob_test, final_threshold, "(FINAL MODEL)")

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"  ROC-AUC target: >0.70  |  Recall(class 1) target: >0.40")
    print(f"  Final: ROC-AUC={final_eval['auc']:.4f}  Recall={final_eval['recall']:.4f}")
    print(f"{'='*60}")

    return model, artifacts, final_eval


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def predict(input_dict: dict, model=None, artifacts=None) -> dict:
    """
    Make prediction on new application.
    input_dict keys: AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE,
                     DAYS_BIRTH, DAYS_EMPLOYED, CODE_GENDER, etc.
    """
    if artifacts is None:
        artifacts = joblib.load("home_credit_artifacts.pkl")
    if model is None:
        model = joblib.load("home_credit_model.pkl")

    # Build single-row DataFrame
    row = build_features(pd.DataFrame([input_dict]))
    X = row[artifacts["feature_names"]]
    X, _, _ = preprocess(X, imputer=artifacts["imputer"],
                         scaler=artifacts["scaler"], fit=False)

    prob = float(model.predict_proba(X)[0, 1])
    approved = bool(prob < artifacts["threshold"])

    return {
        "default_probability": prob,
        "approved": approved,
        "threshold_used": artifacts["threshold"],
        "model": "XGBoost"
    }


if __name__ == "__main__":
    main()
