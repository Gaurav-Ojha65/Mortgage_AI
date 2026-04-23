"""
Production Credit Risk Model — Home Credit Default Risk
=====================================================
LightGBM + Fairness-aware + SHAP + Business Cost Optimization
================================================================
FAIRNESS COMPLIANT: CODE_GENDER fully removed from training and inference.
Proxy bias checked via SHAP audit.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap
import joblib, json, time

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "application_train.csv"
MODEL_PATH  = "credit_risk_fair_model.pkl"
ARTIFACTS_PATH = "credit_risk_fair_artifacts.pkl"
SHAP_PATH   = "credit_risk_fair_shap.pkl"

COST_FP = 1.0
COST_FN = 5.0

PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=8,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    is_unbalance=True,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path=DATA_PATH):
    dtypes = {
        'SK_ID_CURR': 'int32', 'TARGET': 'int8',
        'AMT_INCOME_TOTAL': 'float32', 'AMT_CREDIT': 'float32',
        'AMT_ANNUITY': 'float32', 'AMT_GOODS_PRICE': 'float32',
        'DAYS_BIRTH': 'float32', 'DAYS_EMPLOYED': 'float32',
        'DAYS_REGISTRATION': 'float32', 'DAYS_ID_PUBLISH': 'float32',
        'CNT_CHILDREN': 'int8', 'CNT_FAM_MEMBERS': 'float32',
        'REGION_RATING_CLIENT': 'int8', 'REGION_RATING_CLIENT_W_CITY': 'int8',
        'EXT_SOURCE_1': 'float32', 'EXT_SOURCE_2': 'float32', 'EXT_SOURCE_3': 'float32',
        'FLAG_OWN_CAR': 'str', 'FLAG_OWN_REALTY': 'str',
        'NAME_CONTRACT_TYPE': 'str', 'NAME_EDUCATION_TYPE': 'str',
        'NAME_FAMILY_STATUS': 'str', 'NAME_HOUSING_TYPE': 'str',
        'NAME_INCOME_TYPE': 'str', 'OCCUPATION_TYPE': 'str',
        'ORGANIZATION_TYPE': 'str',
        # Bureau
        'AMT_REQ_CREDIT_BUREAU_HOUR': 'float32',
        'AMT_REQ_CREDIT_BUREAU_DAY': 'float32',
        'AMT_REQ_CREDIT_BUREAU_WEEK': 'float32',
        'AMT_REQ_CREDIT_BUREAU_MON': 'float32',
        'AMT_REQ_CREDIT_BUREAU_QRT': 'float32',
        'AMT_REQ_CREDIT_BUREAU_YEAR': 'float32',
        'OWN_CAR_AGE': 'float32',
        'REG_REGION_NOT_LIVE_REGION': 'int8',
        'REG_REGION_NOT_WORK_REGION': 'int8',
        'LIVE_REGION_NOT_WORK_REGION': 'int8',
        'REG_CITY_NOT_LIVE_CITY': 'int8',
        'REG_CITY_NOT_WORK_CITY': 'int8',
        'LIVE_CITY_NOT_WORK_CITY': 'int8',
        'HOUR_APPR_PROCESS_START': 'int8',
        'FLAG_EMP_PHONE': 'int8', 'FLAG_WORK_PHONE': 'int8',
        'FLAG_PHONE': 'int8', 'FLAG_EMAIL': 'int8',
    }
    df = pd.read_csv(path, dtype=dtypes)
    df = df.dropna(subset=['TARGET'])
    print(f"  Loaded {len(df):,} rows | Target: {df['TARGET'].value_counts().to_dict()}")
    print(f"  Columns: {df.shape[1]} | Default rate: {df['TARGET'].mean()*100:.2f}%")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING — NO CODE_GENDER
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering.
    CODE_GENDER: NEVER used — not encoded, not included in model features.
    Protected attributes kept ONLY in audit dataframe.
    """
    df = df.copy()

    # ── Core numerics ──
    df['income']     = df['AMT_INCOME_TOTAL']
    df['credit']     = df['AMT_CREDIT']
    df['annuity']    = df['AMT_ANNUITY'].fillna(df['AMT_CREDIT'] / 120)
    df['goods']      = df['AMT_GOODS_PRICE'].fillna(df['AMT_CREDIT'])
    df['age_years']  = -df['DAYS_BIRTH'] / 365.25
    df['emp_years']  = -df['DAYS_EMPLOYED'].replace({365243: np.nan}) / 365.25
    df['reg_years']  = -df['DAYS_REGISTRATION'] / 365.25
    df['id_years']   = -df['DAYS_ID_PUBLISH'] / 365.25

    # ── Core ratio features ──
    df['credit_term']       = df['credit'] / df['annuity'].replace(0, np.nan)
    df['ltv']              = df['credit'] / df['goods'].replace(0, np.nan)
    df['dti']              = df['annuity'] / df['income'].replace(0, np.nan)
    df['emi_to_income']    = df['annuity'] / df['income'].replace(0, np.nan)
    df['income_per_person']= df['income'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)
    df['emp_ratio']        = df['emp_years'] / df['age_years'].replace(0, np.nan)
    df['goods_to_income']  = df['goods'] / df['income'].replace(0, np.nan)
    df['annuity_per_credit']= df['annuity'] / df['credit'].replace(0, np.nan)

    for c in ['credit_term','ltv','dti','emi_to_income','income_per_person',
              'emp_ratio','goods_to_income','annuity_per_credit']:
        df[c] = df[c].clip(df[c].quantile(0.01), df[c].quantile(0.99))

    # ── EXT_SOURCE — strongest predictors ──
    ext = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
    df['ext_mean']     = df[ext].mean(axis=1)
    df['ext_std']      = df[ext].std(axis=1)
    df['ext_min']      = df[ext].min(axis=1)
    df['ext_max']      = df[ext].max(axis=1)
    df['ext_prod']     = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['ext_1_2']      = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['ext_2_3']      = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['ext_weighted'] = 0.15*df['EXT_SOURCE_1'].fillna(0) + 0.5*df['EXT_SOURCE_2'].fillna(0) + 0.35*df['EXT_SOURCE_3'].fillna(0)

    # ── Bureau ──
    bureau_cols = [c for c in df.columns if 'AMT_REQ_CREDIT_BUREAU' in c]
    if bureau_cols:
        df['bureau_total_req'] = df[bureau_cols].sum(axis=1)

    # ── Behavioral flags ──
    df['has_car']       = (df['FLAG_OWN_CAR']    == 'Y').astype(int)
    df['has_realty']    = (df['FLAG_OWN_REALTY'] == 'Y').astype(int)
    df['car_age_flag']  = (df['OWN_CAR_AGE'] > 10).astype(int)
    df['children_ratio']= df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)
    df['region_mismatch'] = (
        (df['REG_REGION_NOT_LIVE_REGION'] == 1) |
        (df['REG_REGION_NOT_WORK_REGION'] == 1) |
        (df['REG_CITY_NOT_LIVE_CITY'] == 1) |
        (df['REG_CITY_NOT_WORK_CITY'] == 1)
    ).astype(int)
    df['documents_provided'] = (
        df['FLAG_EMP_PHONE'] + df['FLAG_WORK_PHONE'] +
        df['FLAG_PHONE'] + df['FLAG_EMAIL']
    )

    # ── Categorical: label encode ──
    # NOTE: CODE_GENDER deliberately omitted — never encoded, never used
    cat_map = {
        'NAME_CONTRACT_TYPE':  {'Cash loans': 0, 'Revolving loans': 1},
        'NAME_FAMILY_STATUS':  {'Single / not married': 0, 'Married': 1,
                                'Civil marriage': 2, 'Separated': 3, 'Widow': 4},
        'NAME_INCOME_TYPE':   {'Working': 0, 'Commercial associate': 1,
                               'Pensioner': 2, 'State servant': 3},
    }
    for col, mp in cat_map.items():
        df[col] = df[col].map(mp).fillna(-1).astype(float)

    # Occupation ordinal by risk tier
    occ_map = {
        'Laborers': 0, 'Drivers': 0, 'Security staff': 0,
        'Sales staff': 1, 'Core staff': 1, 'Accountants': 1,
        'Managers': 2, 'High skill tech staff': 2, 'IT staff': 2,
    }
    df['OCCUPATION_TYPE_ORD'] = df['OCCUPATION_TYPE'].map(occ_map).fillna(0).astype(float)

    # ── Protected attributes — FOR AUDIT ONLY, NOT model features ──
    # Gender: keep raw for audit only, never pass to model
    df['_gender_raw'] = df['CODE_GENDER'] if 'CODE_GENDER' in df.columns else 'Unknown'

    df['age_group']     = pd.cut(df['age_years'],
                                  bins=[0, 25, 35, 45, 55, 65, 100],
                                  labels=[0, 1, 2, 3, 4, 5]).astype(float)
    df['education_raw'] = df['NAME_EDUCATION_TYPE']
    df['housing_raw']   = df['NAME_HOUSING_TYPE']

    # Income quintile groups for fairness audit
    df['income_quintile'] = pd.qcut(df['income'].rank(method='first'),
                                     q=5, labels=[0, 1, 2, 3, 4]).astype(float)

    df['target'] = df['TARGET'].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FEATURES — NO protected attributes
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FEATURES = [
    # Core financials
    'income', 'credit', 'annuity', 'goods',
    # Derived ratios
    'credit_term', 'ltv', 'dti', 'emi_to_income', 'income_per_person',
    'emp_ratio', 'goods_to_income', 'annuity_per_credit',
    # Age & employment
    'age_years', 'emp_years', 'reg_years', 'id_years',
    # External scores
    'ext_mean', 'ext_std', 'ext_min', 'ext_max',
    'ext_prod', 'ext_1_2', 'ext_2_3', 'ext_weighted',
    # Bureau
    'bureau_total_req',
    # Behavioral
    'has_car', 'has_realty', 'car_age_flag', 'children_ratio',
    'region_mismatch', 'documents_provided',
    # Encoded categoricals (NO CODE_GENDER)
    'NAME_CONTRACT_TYPE', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE',
    'OCCUPATION_TYPE_ORD',
    # Region
    'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
    'HOUR_APPR_PROCESS_START',
]

# Audit features (separate from model — used only for fairness evaluation)
AUDIT_FEATURES  = ['age_group', 'education_raw', 'housing_raw', 'income_quintile']
GENDER_COL      = '_gender_raw'   # only for post-training fairness audit


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(X_train, X_test=None):
    imp = SimpleImputer(strategy='median')
    imp.fit(X_train)
    X_train_imp = pd.DataFrame(imp.transform(X_train), columns=X_train.columns, index=X_train.index)
    if X_test is not None:
        X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_test.columns, index=X_test.index)
        return X_train_imp, X_test_imp, imp
    return X_train_imp, imp


# ─────────────────────────────────────────────────────────────────────────────
# 4. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def train_cv(X, y, n_splits=5, params=PARAMS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs  = np.zeros(len(X))
    fold_models = []
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        cb = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

        model = lgb.train({**params, 'metric': 'auc'}, dtrain,
                           num_boost_round=params['n_estimators'],
                           valid_sets=[dval], callbacks=cb)

        oof_probs[va_idx] = model.predict(X_va)
        auc = roc_auc_score(y_va, oof_probs[va_idx])
        ap  = average_precision_score(y_va, oof_probs[va_idx])
        fold_scores.append({'fold': fold+1, 'auc': auc, 'ap': ap,
                           'n_iter': model.best_iteration})
        fold_models.append(model)
        print(f"    Fold {fold+1}: AUC={auc:.4f}  AP={ap:.4f}  iter={model.best_iteration}  ({time.time()-t0:.1f}s)")

    cv_auc = np.mean([s['auc'] for s in fold_scores])
    cv_ap  = np.mean([s['ap']  for s in fold_scores])
    print(f"\n  CV ROC-AUC: {cv_auc:.4f} (+/- {np.std([s['auc'] for s in fold_scores]):.4f})")
    print(f"  CV PR-AUC:  {cv_ap:.4f}")
    return oof_probs, fold_models, fold_scores


# ─────────────────────────────────────────────────────────────────────────────
# 5. PROBABILITY CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_probabilities(oof_probs, y_train, X_test_sc, fold_models, X_train_sc):
    print(f"\n  Calibrating with isotonic regression...")
    calibrated_probs = np.zeros(len(X_test_sc))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_sc, y_train)):
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(oof_probs[tr_idx], y_train.iloc[tr_idx].values)
        calibrated_probs += iso.predict(fold_models[fold].predict(X_test_sc)) / 5

    iso_full = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso_full.fit(oof_probs, y_train.values)
    print(f"  Calibration done")
    return calibrated_probs, iso_full


# ─────────────────────────────────────────────────────────────────────────────
# 6. THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_thresholds(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 300)
    rows = []
    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        if y_hat.sum() == 0:
            continue
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0,1]).ravel()
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0
        rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        cost = fp*COST_FP + fn*COST_FN
        rows.append({'t':t,'prec':prec,'rec':rec,'f1':f1,
                     'tn':tn,'fp':fp,'fn':fn,'tp':tp,'cost':cost})

    df = pd.DataFrame(rows)
    best_f1   = df.loc[df['f1'].idxmax()]
    best_cost = df.loc[df['cost'].idxmin()]

    prec_p, rec_p, thresh_p = precision_recall_curve(y_true, y_prob)
    f1_pr = 2*prec_p*rec_p/(prec_p+rec_p+1e-10)
    best_pr_t = thresh_p[np.argmax(f1_pr[:-1])]

    print(f"\n  Threshold Optimization:")
    print(f"    F1-max:       {best_f1['t']:.3f}  F1={best_f1['f1']:.4f}  Prec={best_f1['prec']:.4f}  Rec={best_f1['rec']:.4f}")
    print(f"    Cost-min:     {best_cost['t']:.3f}  Cost={best_cost['cost']:,.0f}  FP={int(best_cost['fp'])}  FN={int(best_cost['fn'])}")
    print(f"    PR-curve:     {best_pr_t:.3f}")
    return df, best_f1, best_cost, best_pr_t


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_prob, threshold, label=""):
    y_hat = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0,1]).ravel()
    prec = precision_score(y_true, y_hat)
    rec  = recall_score(y_true, y_hat)
    f1   = f1_score(y_true, y_hat)
    cost = fp*COST_FP + fn*COST_FN

    print(f"\n{'='*60}")
    print(f"  EVALUATION {label}")
    print(f"{'='*60}")
    print(f"  Threshold:        {threshold:.4f}")
    print(f"  ROC-AUC:         {auc:.4f}")
    print(f"  PR-AUC (AP):      {ap:.4f}")
    print(f"  Accuracy:         {accuracy_score(y_true, y_hat):.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:          {rec:.4f}")
    print(f"  F1-score:        {f1:.4f}")
    print(f"  Business Cost:    {cost:,.0f}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"              NoDefault   Default")
    print(f"  Actual 0  {tn:8d}  {fp:8d}  (TN={tn}, FP={fp})")
    print(f"       1    {fn:8d}  {tp:8d}  (FN={fn}, TP={tp})")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_hat, target_names=["NoDefault(0)","Default(1)"]))

    return {'threshold':threshold,'auc':auc,'ap':ap,'precision':prec,
            'recall':rec,'f1':f1,'cost':cost,
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def show_importance(model, feature_names, top_n=15):
    imp = pd.Series(
        model.booster_.feature_importance(importance_type='gain'),
        index=feature_names
    ).sort_values(ascending=False).head(top_n)
    print(f"\n  Top {top_n} Features (LightGBM gain):")
    mx = imp.iloc[0]
    for i,(name,val) in enumerate(imp.items(),1):
        bar = "*" * int(val/mx*40)
        print(f"    {i:2d}. {name:<30} {val:8.1f}  {bar}")
    return imp


# ─────────────────────────────────────────────────────────────────────────────
# 9. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(model, X_sample, feature_names, top_n=15):
    print(f"\n  Computing SHAP on {len(X_sample)} samples...")
    sv = shap.TreeExplainer(model).shap_values(X_sample)
    sv_use = sv[1] if isinstance(sv, list) else sv
    mean_abs = (pd.Series(np.abs(sv_use).mean(axis=0), index=feature_names)
                .sort_values(ascending=False).head(top_n))
    print(f"\n  Top {top_n} SHAP Drivers (mean |SHAP|):")
    mx = mean_abs.iloc[0]
    for i,(name,val) in enumerate(mean_abs.items(),1):
        bar = "*" * int(val/mx*40)
        print(f"    {i:2d}. {name:<30} {val:.4f}  {bar}")

    # ── Proxy bias check ──
    proxy_keywords = ['gender', 'sex', 'marital', 'family', 'children', 'race', 'religion']
    print(f"\n  Proxy Bias Check:")
    proxy_features = [n for n in feature_names if any(p in n.lower() for p in proxy_keywords)]
    proxy_shap = mean_abs[mean_abs.index.isin(proxy_features)]
    if len(proxy_shap) > 0:
        print(f"    WARNING: Proxy features detected in top SHAP:")
        for n, v in proxy_shap.items():
            print(f"      {n}: {v:.4f}")
    else:
        print(f"    [OK] No proxy features in top SHAP drivers")

    joblib.dump({'shap_values': sv_use, 'mean_abs': mean_abs}, SHAP_PATH)
    return mean_abs


# ─────────────────────────────────────────────────────────────────────────────
# 10. FAIRNESS AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def fairness_audit(df_audit, y_true, y_prob, threshold,
                   group_col, group_name, label_map=None):
    y_hat  = (y_prob >= threshold).astype(int)
    overall_appr = 1 - y_hat.mean()
    print(f"\n  Fairness: {group_name}")
    print(f"  Overall approval: {overall_appr:.3f} ({overall_appr*100:.1f}%)")
    print(f"  {'Group':<25} {'N':>7} {'Approval':>9} {'Dev%':>9} {'DI':>7} {'Status':>10}")
    print(f"  {'-'*70}")
    di_rates = []
    for g in sorted(df_audit[group_col].dropna().unique()):
        mask = df_audit[group_col] == g
        n = mask.sum()
        if n < 10:
            continue
        appr = 1 - y_hat[mask].mean()
        dev   = (appr - overall_appr)/overall_appr*100 if overall_appr > 0 else 0
        di    = min(appr/overall_appr, 1.0) if overall_appr > 0 else 1.0
        di_rates.append((g, di, appr, n, dev))
        lbl = (label_map.get(g,g) if label_map else g)
        flag = "[FLAG]" if abs(dev) > 20 else "OK"
        print(f"  {str(lbl):<25} {n:>7} {appr:>9.3f} {dev:>+8.1f}% {di:>7.3f} {flag:>10}")

    if not di_rates:
        return {'di_ratio': 1.0, 'max_dev': 0.0, 'flagged': []}

    di_ratio = min(d for _, d, _, _, _ in di_rates) / max(d for _, d, _, _, _ in di_rates) \
                if di_rates else 1.0
    max_dev  = max(abs(dev) for _, _, _, _, dev in di_rates)
    flagged  = [(g, dev, appr, n) for g, di, appr, n, dev in di_rates if abs(dev) > 20]

    print(f"  DI Ratio: {di_ratio:.3f} (>=0.80 = pass)  Max Dev: {max_dev:.1f}% (>20% = fail)")
    if flagged:
        print(f"  !! FLAGGED GROUPS:")
        for g, dev, appr, n in flagged:
            lbl = (label_map.get(g,g) if label_map else g)
            print(f"      {lbl} (N={n}): {dev:+.1f}% deviation, approval={appr:.3f}")
    return {'di_ratio': di_ratio, 'max_dev': max_dev, 'flagged': flagged}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("="*60)
    print("  CREDIT RISK — FAIR (NO CODE_GENDER)")
    print("="*60)

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Feature engineering
    print(f"\n  Engineering features...")
    df = engineer_features(df)
    print(f"  Model features: {len(MODEL_FEATURES)} | Audit features: {len(AUDIT_FEATURES)}")

    # Verify CODE_GENDER is not in model features
    assert 'CODE_GENDER' not in MODEL_FEATURES, "CODE_GENDER still in MODEL_FEATURES!"
    assert '_gender_raw' not in MODEL_FEATURES, "_gender_raw leaked into MODEL_FEATURES!"
    print(f"  [VERIFIED] CODE_GENDER not in model features")

    # 3. Prepare data
    X = df[MODEL_FEATURES].copy()
    y = df['target'].copy()
    df_audit = df[AUDIT_FEATURES + [GENDER_COL]].copy()
    X.index = y.index = df_audit.index = range(len(X))

    # 4. Train/test split
    X_tr, X_te, y_tr, y_te, df_aud_tr, df_aud_te = train_test_split(
        X, y, df_audit, test_size=0.2, random_state=42, stratify=y)
    print(f"\n  Train: {len(X_tr):,} | Test: {len(X_te):,}")
    print(f"  Test default rate: {y_te.mean()*100:.2f}%")

    # 5. Preprocess
    X_tr_imp, X_te_imp, imputer = preprocess(X_tr, X_te)
    print(f"  Preprocessed {len(MODEL_FEATURES)} features")

    # 6. Class distribution
    orig = y_tr.value_counts().to_dict()
    print(f"\n  Class dist: 0={orig.get(0,0):,}  1={orig.get(1,0):,}  ratio={orig.get(0,1)/orig.get(1,1):.2f}:1")

    # 7. Cross-validation
    print(f"\n  5-Fold CV (LightGBM, is_unbalance=True):")
    oof_probs, fold_models, fold_scores = train_cv(X_tr_imp, y_tr, n_splits=5, params=PARAMS)

    # 8. Probability calibration
    cal_probs_test, iso_full = calibrate_probabilities(
        oof_probs, y_tr, X_te_imp, fold_models, X_tr_imp)

    # 9. Threshold optimization on calibrated OOF
    cal_oof = iso_full.predict(oof_probs)
    print(f"\n  Threshold optimization on calibrated OOF...")
    scan_df, best_f1, best_cost, best_pr_t = find_optimal_thresholds(y_tr.values, cal_oof)

    # 10. Train final model on full training data + SMOTE
    print(f"\n  Training final model (full data + SMOTE)...")
    smote = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_imp, y_tr)
    print(f"  After SMOTE: {len(X_tr_sm):,} samples")

    final_model = lgb.LGBMClassifier(**{**PARAMS})
    final_model.fit(X_tr_sm, y_tr_sm)
    raw_test_probs  = final_model.predict_proba(X_te_imp)[:,1]
    cal_test_probs  = iso_full.predict(raw_test_probs)

    print(f"  Final model trained")

    # 11. Multi-threshold evaluation
    thresholds = [
        (best_f1['t'],           f"F1-optimal ({best_f1['t']:.3f})"),
        (best_cost['t'],         f"Cost-optimal ({best_cost['t']:.3f})"),
        (best_pr_t,              f"PR-curve ({best_pr_t:.3f})"),
        (0.5,                    "Default 0.5"),
    ]
    best_result = None
    print(f"\n{'='*60}")
    print("  MULTI-THRESHOLD EVALUATION")
    print(f"{'='*60}")
    for t, lbl in thresholds:
        r = evaluate(y_te.values, cal_test_probs, t, f"({lbl})")
        if best_result is None or r['f1'] > best_result['f1']:
            best_result = {**r, 'label': lbl}

    # 12. Feature importance
    fi = show_importance(final_model, MODEL_FEATURES)

    # 13. SHAP
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_te_imp), size=min(3000, len(X_te_imp)), replace=False)
    X_shap = X_te_imp.iloc[sample_idx]
    shap_drivers = compute_shap(final_model, X_shap, MODEL_FEATURES)

    # 14. Fairness audit
    print(f"\n{'='*60}")
    print("  FAIRNESS AUDIT")
    print(f"{'='*60}")
    df_aud_te = df_aud_te.reset_index(drop=True)
    y_te_vals = y_te.reset_index(drop=True).values
    cal_test_reset = cal_test_probs.copy()

    fairness_results = {}
    for feat, name, lmap in [
        ('age_group', 'Age Group',
         {0:'<25',1:'25-35',2:'35-45',3:'45-55',4:'55-65',5:'65+'}),
        ('education_raw', 'Education',
         {'Secondary / secondary special':'Secondary','Higher education':'Higher',
          'Incomplete higher':'Incomplete','Lower secondary':'Lower','Academic degree':'Academic'}),
        ('housing_raw', 'Housing Type',
         {'House / apartment':'House/Apt','With parents':'Parents',
          'Municipal apartment':'Municipal','Office apartment':'Office',
          'Co-op apartment':'Co-op','Rented apartment':'Rented'}),
        ('income_quintile', 'Income Quintile',
         {0:'Q1 (lowest)',1:'Q2',2:'Q3',3:'Q4',4:'Q5 (highest)'}),
    ]:
        fairness_results[name] = fairness_audit(
            df_aud_te, y_te_vals, cal_test_reset,
            best_result['threshold'], feat, name, lmap)

    # Gender audit (post-training only — not used in model)
    gender_map = {'M': 'Male', 'F': 'Female', 'XNA': 'Unknown', 'Unknown': 'Unknown'}
    fairness_results['Gender'] = fairness_audit(
        df_aud_te, y_te_vals, cal_test_reset,
        best_result['threshold'], '_gender_raw', 'Gender (CODE_GENDER — audit only, NOT in model)',
        gender_map)

    # 15. Save
    artifacts = {
        'model': final_model,
        'imputer': imputer,
        'iso_calibrator': iso_full,
        'features': MODEL_FEATURES,
        'audit_features': AUDIT_FEATURES,
        'params': PARAMS,
        'threshold': best_result['threshold'],
        'cv_auc': np.mean([s['auc'] for s in fold_scores]),
        'cv_ap':  np.mean([s['ap']  for s in fold_scores]),
        'fairness_results': fairness_results,
        'shap_top10': shap_drivers.head(10).to_dict(),
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)
    joblib.dump(final_model, MODEL_PATH)
    print(f"\n  Saved: {MODEL_PATH}, {ARTIFACTS_PATH}")

    # 16. Comparison vs previous
    print(f"\n{'='*60}")
    print("  PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Fair Model':<15} {'Previous':<15} {'Target':<12} {'Δ'}")
    print(f"  {'-'*65}")
    prev = {'auc':0.7596,'ap':0.2473,'precision':0.2497,'recall':0.4268,'f1':0.3150,'cost':20598}
    for k in ['auc','ap','precision','recall','f1']:
        cur = best_result[k]
        pv  = prev.get(k, 0)
        tgt = {'auc':0.70,'ap':None,'precision':0.30,'recall':0.60,'f1':None}.get(k,'')
        delta = cur - pv
        flag = '✓' if (tgt == '' or (isinstance(tgt,float) and cur >= tgt)) else '✗'
        print(f"  {k.capitalize():<20} {cur:<15.4f} {pv:<15.4f} {str(tgt):<12} {delta:+.4f} {flag}")

    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:              LightGBM (NO CODE_GENDER)")
    print(f"  Features:           {len(MODEL_FEATURES)}")
    print(f"  CODE_GENDER:        EXCLUDED from model")
    print(f"  CV ROC-AUC:        {np.mean([s['auc'] for s in fold_scores]):.4f}")
    print(f"  CV PR-AUC:         {np.mean([s['ap']  for s in fold_scores]):.4f}")
    print(f"  Final threshold:    {best_result['threshold']:.3f}")
    print(f"  Precision:         {best_result['precision']:.4f}")
    print(f"  Recall:            {best_result['recall']:.4f}")
    print(f"  F1:                 {best_result['f1']:.4f}")
    print(f"  Business Cost:       {best_result['cost']:,.0f}")
    print(f"\n  Fairness Status:")
    all_pass = True
    for name, res in fairness_results.items():
        status = "PASS" if res['di_ratio'] >= 0.80 and res['max_dev'] <= 20 else "FAIL"
        if status == "FAIL": all_pass = False
        print(f"    {name:<20} DI={res['di_ratio']:.3f}  MaxDev={res['max_dev']:.1f}%  [{status}]")
    print(f"\n  Total time: {time.time()-t0:.1f}s")
    print(f"{'='*60}")
    return artifacts, best_result


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict(input_dict: dict, artifacts=None) -> dict:
    if artifacts is None:
        artifacts = joblib.load(ARTIFACTS_PATH)
    row = pd.DataFrame([input_dict])
    row.columns = [c.upper() for c in row.columns]
    # engineer_features handles any incoming dict — CODE_GENDER not used
    df = engineer_features(row)
    X  = df[artifacts['features']]
    X  = pd.DataFrame(artifacts['imputer'].transform(X), columns=X.columns)
    raw_prob = artifacts['model'].predict_proba(X)[0,1]
    prob = artifacts['iso_calibrator'].predict([raw_prob])[0]
    approved = bool(prob < artifacts['threshold'])
    return {
        'default_probability': float(prob),
        'approved': approved,
        'threshold': artifacts['threshold'],
        'model': 'LightGBM-Fair',
        'gender_in_model': False,
    }


if __name__ == "__main__":
    main()
