"""
Production Credit Risk Pipeline v2 — Home Credit Default Risk
============================================================
Targets: Precision >= 0.30, Recall >= 0.60, ROC-AUC >= 0.75
Fairness: No group deviation > 20%, DI >= 0.80
Drops: CODE_GENDER, NAME_FAMILY_STATUS (proxy bias)
LightGBM with early stopping, threshold optimized on PR-curve.
"""

import numpy as np
import pandas as pd
import warnings, json, time, joblib
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

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH     = "application_train.csv"
MODEL_PATH    = "optimized_model.pkl"
ARTIFACTS_PATH = "preprocessing_pipeline.pkl"
THRESHOLD_PATH = "threshold.json"
EVAL_PATH     = "evaluation_report.txt"
FAIR_PATH     = "fairness_report.txt"
SHAP_PATH     = "shap_summary.pkl"

COST_FP = 1.0
COST_FN = 5.0

PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
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
    print(f"  Loaded {len(df):,} rows | Default rate: {df['TARGET'].mean()*100:.2f}%")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING — v2: richer ratios + no sensitive features
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Core numerics ──
    df['income']       = df['AMT_INCOME_TOTAL']
    df['credit']       = df['AMT_CREDIT']
    df['annuity']      = df['AMT_ANNUITY'].fillna(df['AMT_CREDIT'] / 120)
    df['goods']       = df['AMT_GOODS_PRICE'].fillna(df['AMT_CREDIT'])
    df['age_years']   = -df['DAYS_BIRTH'] / 365.25
    df['emp_years']   = -df['DAYS_EMPLOYED'].replace({365243: np.nan}) / 365.25
    df['reg_years']   = -df['DAYS_REGISTRATION'] / 365.25
    df['id_years']    = -df['DAYS_ID_PUBLISH'] / 365.25

    # ── REQUIRED FEATURES (per spec) ──
    # income_to_credit = AMT_INCOME_TOTAL / AMT_CREDIT
    df['income_to_credit'] = df['income'] / df['credit'].replace(0, np.nan)
    # annuity_to_income = AMT_ANNUITY / AMT_INCOME_TOTAL
    df['annuity_to_income'] = df['annuity'] / df['income'].replace(0, np.nan)
    # credit_term = AMT_CREDIT / AMT_ANNUITY (months)
    df['credit_term_months'] = df['credit'] / df['annuity'].replace(0, np.nan)
    df['credit_term_years']  = df['credit_term_months'] / 12

    # ── Additional ratios ──
    df['ltv']               = df['credit'] / df['goods'].replace(0, np.nan)
    df['dti']               = df['annuity'] / df['income'].replace(0, np.nan)
    df['emi_to_income']     = df['annuity'] / df['income'].replace(0, np.nan)
    df['income_per_person'] = df['income'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)
    df['emp_ratio']         = df['emp_years'] / df['age_years'].replace(0, np.nan)
    df['goods_to_income']  = df['goods'] / df['income'].replace(0, np.nan)
    df['annuity_per_credit']= df['annuity'] / df['credit'].replace(0, np.nan)
    df['loan_to_income']    = df['credit'] / (df['income'] * df['credit_term_years'].replace(0, np.nan))

    # Clean extreme outliers
    for c in ['income_to_credit','annuity_to_income','credit_term_months','ltv','dti',
              'emi_to_income','income_per_person','emp_ratio','goods_to_income',
              'annuity_per_credit','loan_to_income']:
        q01, q99 = df[c].quantile([0.01, 0.99])
        df[c] = df[c].clip(q01, q99)

    # ── EXT_SOURCE (per spec) ──
    ext = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
    df['ext_mean']     = df[ext].mean(axis=1)
    df['ext_std']      = df[ext].std(axis=1)
    df['ext_min']      = df[ext].min(axis=1)
    df['ext_max']      = df[ext].max(axis=1)
    df['ext_prod']     = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['ext_1_2']     = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['ext_2_3']     = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['ext_weighted'] = 0.15*df['EXT_SOURCE_1'].fillna(0) + 0.5*df['EXT_SOURCE_2'].fillna(0) + 0.35*df['EXT_SOURCE_3'].fillna(0)

    # ── Bureau ──
    bureau_cols = [c for c in df.columns if 'AMT_REQ_CREDIT_BUREAU' in c]
    if bureau_cols:
        df['bureau_total_req'] = df[bureau_cols].sum(axis=1)

    # ── Behavioral ──
    df['has_car']        = (df['FLAG_OWN_CAR']    == 'Y').astype(int)
    df['has_realty']     = (df['FLAG_OWN_REALTY'] == 'Y').astype(int)
    df['car_age_flag']   = (df['OWN_CAR_AGE'] > 10).astype(int)
    df['children_ratio'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)
    df['region_mismatch']= (
        (df['REG_REGION_NOT_LIVE_REGION'] == 1) |
        (df['REG_REGION_NOT_WORK_REGION'] == 1) |
        (df['REG_CITY_NOT_LIVE_CITY'] == 1) |
        (df['REG_CITY_NOT_WORK_CITY'] == 1)
    ).astype(int)
    df['documents_provided'] = (
        df['FLAG_EMP_PHONE'] + df['FLAG_WORK_PHONE'] +
        df['FLAG_PHONE'] + df['FLAG_EMAIL']
    )

    # ── Categorical: encode (NO CODE_GENDER, NO NAME_FAMILY_STATUS) ──
    cat_map = {
        'NAME_CONTRACT_TYPE': {'Cash loans': 0, 'Revolving loans': 1},
        'NAME_INCOME_TYPE':  {'Working': 0, 'Commercial associate': 1,
                              'Pensioner': 2, 'State servant': 3},
    }
    for col, mp in cat_map.items():
        df[col] = df[col].map(mp).fillna(-1).astype(float)

    occ_map = {
        'Laborers': 0, 'Drivers': 0, 'Security staff': 0,
        'Sales staff': 1, 'Core staff': 1, 'Accountants': 1,
        'Managers': 2, 'High skill tech staff': 2, 'IT staff': 2,
    }
    df['OCCUPATION_TYPE_ORD'] = df['OCCUPATION_TYPE'].map(occ_map).fillna(0).astype(float)

    # ── Protected attributes (AUDIT ONLY — not model features) ──
    df['_gender_raw']      = df['CODE_GENDER'] if 'CODE_GENDER' in df.columns else 'Unknown'
    df['age_group']         = pd.cut(df['age_years'],
                                      bins=[0, 25, 35, 45, 55, 65, 100],
                                      labels=[0, 1, 2, 3, 4, 5]).astype(float)
    df['education_raw']     = df['NAME_EDUCATION_TYPE']
    df['housing_raw']       = df['NAME_HOUSING_TYPE']
    df['income_quintile']   = pd.qcut(df['income'].rank(method='first'),
                                       q=5, labels=[0,1,2,3,4]).astype(float)

    df['target'] = df['TARGET'].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FEATURES — NO sensitive or proxy variables
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FEATURES = [
    # Spec-required features
    'income_to_credit', 'annuity_to_income', 'credit_term_months',
    # Core financials
    'income', 'credit', 'annuity', 'goods',
    # Derived ratios
    'ltv', 'dti', 'emi_to_income', 'income_per_person',
    'emp_ratio', 'goods_to_income', 'annuity_per_credit', 'loan_to_income',
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
    # Encoded categoricals (NO CODE_GENDER, NO NAME_FAMILY_STATUS)
    'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE',
    'OCCUPATION_TYPE_ORD',
    # Region
    'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
    'HOUR_APPR_PROCESS_START',
]

AUDIT_FEATURES = ['age_group', 'education_raw', 'housing_raw', 'income_quintile', '_gender_raw']
DROP_FEATURES  = ['CODE_GENDER', 'NAME_FAMILY_STATUS']


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
# 4. CROSS-VALIDATION with early stopping
# ─────────────────────────────────────────────────────────────────────────────

def train_cv(X, y, n_splits=5, params=PARAMS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs   = np.zeros(len(X))
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
# 5. CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_probabilities(oof_probs, y_train, X_test_sc, fold_models, X_train_sc):
    calibrated = np.zeros(len(X_test_sc))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_sc, y_train)):
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(oof_probs[tr_idx], y_train.iloc[tr_idx].values)
        calibrated += iso.predict(fold_models[fold].predict(X_test_sc)) / 5

    iso_full = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso_full.fit(oof_probs, y_train.values)
    return calibrated, iso_full


# ─────────────────────────────────────────────────────────────────────────────
# 6. THRESHOLD OPTIMIZATION — target Precision >= 0.30, maximize Recall
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_prob, target_prec=0.30):
    """
    Find threshold where precision >= target_prec AND recall is maximized.
    Falls back to best F1 if no threshold meets precision target.
    """
    thresholds = np.linspace(0.05, 0.95, 500)
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

    df_rows = pd.DataFrame(rows)

    # Find all thresholds meeting precision target
    viable = df_rows[df_rows['prec'] >= target_prec]
    if len(viable) > 0:
        # Maximize recall among viable thresholds
        best = viable.loc[viable['rec'].idxmax()]
        print(f"\n  Precision-target threshold found: {best['t']:.3f}")
        print(f"    Prec={best['prec']:.4f}  Rec={best['rec']:.4f}  F1={best['f1']:.4f}  Cost={int(best['cost']):,}")
        return best, 'precision_target'
    else:
        # Fallback: best F1
        best = df_rows.loc[df_rows['f1'].idxmax()]
        print(f"\n  No threshold achieves Prec>=0.30. Best F1 fallback:")
        print(f"    Prec={best['prec']:.4f}  Rec={best['rec']:.4f}  F1={best['f1']:.4f}")
        return best, 'f1_fallback'


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

    print(f"\n{'='*58}")
    print(f"  EVALUATION {label}")
    print(f"{'='*58}")
    print(f"  Threshold:        {threshold:.4f}")
    print(f"  ROC-AUC:         {auc:.4f}")
    print(f"  PR-AUC (AP):      {ap:.4f}")
    print(f"  Accuracy:         {accuracy_score(y_true, y_hat):.4f}")
    print(f"  Precision:        {prec:.4f}  [target >= 0.30] {'OK' if prec >= 0.30 else 'LOW'}")
    print(f"  Recall:          {rec:.4f}  [target >= 0.60] {'OK' if rec >= 0.60 else 'LOW'}")
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
    imp = pd.Series(model.booster_.feature_importance(importance_type='gain'),
                    index=feature_names).sort_values(ascending=False).head(top_n)
    print(f"\n  Top {top_n} Features (gain):")
    mx = imp.iloc[0]
    for i,(name,val) in enumerate(imp.items(),1):
        bar = "*" * int(val/mx*40)
        print(f"    {i:2d}. {name:<32} {val:8.1f}  {bar}")
    return imp


# ─────────────────────────────────────────────────────────────────────────────
# 9. SHAP
# ─────────────────────────────────────────────────────────────────────────────

PROXY_KEYWORDS = ['gender','sex','marital','family','children','race','religion']

def compute_shap(model, X_sample, feature_names, top_n=15):
    print(f"\n  Computing SHAP on {len(X_sample)} samples...")
    sv = shap.TreeExplainer(model).shap_values(X_sample)
    sv_use = sv[1] if isinstance(sv, list) else sv
    mean_abs = (pd.Series(np.abs(sv_use).mean(axis=0), index=feature_names)
                .sort_values(ascending=False).head(top_n))
    print(f"\n  Top {top_n} SHAP Drivers:")
    mx = mean_abs.iloc[0]
    for i,(name,val) in enumerate(mean_abs.items(),1):
        bar = "*" * int(val/mx*40)
        print(f"    {i:2d}. {name:<32} {val:.4f}  {bar}")

    # Proxy check
    proxy_feat = [n for n in feature_names if any(p in n.lower() for p in PROXY_KEYWORDS)]
    proxy_shap = mean_abs[mean_abs.index.isin(proxy_feat)]
    if len(proxy_shap) > 0:
        print(f"\n  PROXY BIAS WARNING:")
        for n, v in proxy_shap.items():
            print(f"    {n}: {v:.4f}")
    else:
        print(f"\n  [OK] No proxy features in top SHAP")

    joblib.dump({'mean_abs': mean_abs.to_dict()}, SHAP_PATH)
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
    print(f"  {'Group':<25} {'N':>7} {'Approval':>9} {'Dev%':>9} {'DI':>7} {'Status':>8}")
    print(f"  {'-'*68}")
    rows_out = []
    for g in sorted(df_audit[group_col].dropna().unique()):
        mask = df_audit[group_col] == g
        n = mask.sum()
        if n < 10:
            continue
        appr = 1 - y_hat[mask].mean()
        dev   = (appr - overall_appr)/overall_appr*100 if overall_appr > 0 else 0
        di    = min(appr/overall_appr, 1.0) if overall_appr > 0 else 1.0
        lbl   = (label_map.get(g,g) if label_map else g)
        flag  = "[FLAG]" if abs(dev) > 20 else "OK"
        print(f"  {str(lbl):<25} {n:>7} {appr:>9.3f} {dev:>+8.1f}% {di:>7.3f} {flag:>8}")
        rows_out.append({'group': str(lbl), 'n': int(n), 'approval': float(appr),
                         'deviation': float(dev), 'di': float(di), 'flagged': abs(dev) > 20})

    if not rows_out:
        return {'di_ratio': 1.0, 'max_dev': 0.0, 'groups': []}

    di_ratio = min(r['di'] for r in rows_out) / max(r['di'] for r in rows_out) \
               if rows_out else 1.0
    max_dev  = max(abs(r['deviation']) for r in rows_out)
    flagged  = [r['group'] for r in rows_out if r['flagged']]
    print(f"  DI Ratio: {di_ratio:.3f} (>=0.80 = pass)  Max Dev: {max_dev:.1f}% (>20% = fail)")
    return {'di_ratio': float(di_ratio), 'max_dev': float(max_dev),
            'groups': rows_out, 'flagged_groups': flagged}


# ─────────────────────────────────────────────────────────────────────────────
# 11. REPORT WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def write_evaluation_report(results, fold_scores, shap_imp, threshold_result, eval_path=EVAL_PATH):
    cv_auc = np.mean([s['auc'] for s in fold_scores])
    cv_ap  = np.mean([s['ap']  for s in fold_scores])
    prev = {'auc':'0.7596','ap':'0.2473','precision':'0.2497','recall':'0.4268','f1':'0.3150','cost':'20,598'}

    lines = [
        "=" * 60,
        "  EVALUATION REPORT — Credit Risk Pipeline v2",
        "=" * 60,
        "",
        f"  ROC-AUC:   {results['auc']:.4f}  (target >= 0.75)  {'PASS' if results['auc'] >= 0.75 else 'FAIL'}",
        f"  PR-AUC:    {results['ap']:.4f}",
        f"  Precision: {results['precision']:.4f}  (target >= 0.30)  {'PASS' if results['precision'] >= 0.30 else 'FAIL'}",
        f"  Recall:    {results['recall']:.4f}  (target >= 0.60)  {'PASS' if results['recall'] >= 0.60 else 'LOW'}",
        f"  F1-score: {results['f1']:.4f}",
        f"  Business Cost: {results['cost']:,}  (previous: {prev['cost']})",
        "",
        "  Confusion Matrix:",
        f"    TN={results['tn']:,}  FP={results['fp']:,}",
        f"    FN={results['fn']:,}  TP={results['tp']:,}",
        "",
        f"  Threshold: {results['threshold']:.4f}  (optimized: {threshold_result})",
        f"  CV ROC-AUC: {cv_auc:.4f}  CV PR-AUC: {cv_ap:.4f}",
        "",
        "  Top 10 SHAP Drivers:",
    ]
    for i,(name,val) in enumerate(shap_imp.items(),1):
        lines.append(f"    {i:2d}. {name:<32} {val:.4f}")

    with open(eval_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Saved: {eval_path}")


def write_fairness_report(fairness_results, fair_path=FAIR_PATH):
    lines = [
        "=" * 60,
        "  FAIRNESS REPORT — Credit Risk Pipeline v2",
        "=" * 60,
        "",
        "  Constraint: No group deviation > 20%, DI >= 0.80",
        "",
    ]
    all_pass = True
    for name, res in fairness_results.items():
        status = "PASS" if res['di_ratio'] >= 0.80 and res['max_dev'] <= 20 else "FAIL"
        if status == "FAIL": all_pass = False
        lines.append(f"  {name}:")
        lines.append(f"    DI Ratio: {res['di_ratio']:.3f}  Max Dev: {res['max_dev']:.1f}%  [{status}]")
        if res.get('flagged_groups'):
            lines.append(f"    Flagged: {', '.join(res['flagged_groups'])}")
        lines.append("")

    lines.append(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME GROUPS FAIL'}")
    with open(fair_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {fair_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("="*60)
    print("  CREDIT RISK PIPELINE v2 — Production Optimized")
    print("="*60)

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Feature engineering
    print(f"\n  Engineering features...")
    df = engineer_features(df)

    # Verify drops
    for f in DROP_FEATURES:
        assert f not in MODEL_FEATURES, f"{f} still in MODEL_FEATURES!"
    print(f"  Features: {len(MODEL_FEATURES)} | Dropped: {DROP_FEATURES}")

    # 3. Prepare data
    X = df[MODEL_FEATURES].copy()
    y = df['target'].copy()
    df_audit = df[AUDIT_FEATURES].copy()
    X.index = y.index = df_audit.index = range(len(X))

    # 4. Train/test split
    X_tr, X_te, y_tr, y_te, df_aud_tr, df_aud_te = train_test_split(
        X, y, df_audit, test_size=0.2, random_state=42, stratify=y)
    print(f"\n  Train: {len(X_tr):,} | Test: {len(X_te):,}")
    print(f"  Default rate: {y_te.mean()*100:.2f}%")

    # 5. Preprocess
    X_tr_imp, X_te_imp, imputer = preprocess(X_tr, X_te)

    # 6. Class imbalance
    orig = y_tr.value_counts().to_dict()
    scale = orig.get(0,1) / orig.get(1,1)
    print(f"\n  Class dist: 0={orig.get(0,0):,}  1={orig.get(1,0):,}  scale_pos_weight={scale:.2f}")

    # 7. Cross-validation
    print(f"\n  5-Fold CV (LightGBM, early stopping 50):")
    oof_probs, fold_models, fold_scores = train_cv(X_tr_imp, y_tr, n_splits=5, params=PARAMS)

    # 8. Train final SMOTE model with higher capacity
    print(f"\n  Training final SMOTE model...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_imp, y_tr)
    print(f"  After SMOTE: {len(X_tr_sm):,}")
    # Use more boosting rounds, lower learning rate for better recall
    final_params = {**PARAMS, 'n_estimators': 1200, 'learning_rate': 0.02,
                    'num_leaves': 48, 'max_depth': 7}
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_tr_sm, y_tr_sm)
    raw_probs = final_model.predict_proba(X_te_imp)[:, 1]

    # 9. Recalibrate using SMOTE OOF
    print(f"\n  Recalibrating SMOTE model...")
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
    smote_oof = np.zeros(len(X_tr_imp))
    for tr2_idx, va2_idx in skf2.split(X_tr_imp, y_tr):
        X_tr2, X_va2 = X_tr_imp.iloc[tr2_idx], X_tr_imp.iloc[va2_idx]
        y_tr2 = y_tr.iloc[tr2_idx]
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr2s, y_tr2s = sm.fit_resample(X_tr2, y_tr2)
        m2 = lgb.LGBMClassifier(**{**PARAMS, 'n_estimators': 600, 'learning_rate': 0.02,
                                   'num_leaves': 48, 'max_depth': 7, 'verbose': -1})
        m2.fit(X_tr2s, y_tr2s)
        smote_oof[va2_idx] = m2.predict_proba(X_va2)[:, 1]

    iso_final = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso_final.fit(smote_oof, y_tr.values)
    cal_test = iso_final.predict(raw_probs)
    print(f"  Recalibration done")

    # 10. Find threshold: report both precision-targeted and recall-targeted options
    # The data fundamentally cannot achieve Rec>=0.60 at Prec>=0.30
    # Show the full tradeoff so business can make informed decision
    cal_oof = iso_final.predict(smote_oof)
    print(f"\n  Full Precision-Recall Tradeoff Analysis:")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Cost':>12}")
    print(f"  {'-'*55}")
    tradeoff_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    for t in tradeoff_thresholds:
        y_hat = (cal_oof >= t).astype(int)
        if y_hat.sum() == 0: continue
        prec = precision_score(y_tr, y_hat); rec = recall_score(y_tr, y_hat)
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        tn, fp, fn, tp = confusion_matrix(y_tr, y_hat, labels=[0,1]).ravel()
        cost = fp*COST_FP + fn*COST_FN
        marker = " <<<" if (abs(prec - 0.30) < 0.02 or abs(rec - 0.60) < 0.02) else ""
        print(f"  {t:>10.3f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {cost:>12,}{marker}")

    # Select best F1 threshold as final (best balance)
    thresholds_all = np.linspace(0.05, 0.95, 500)
    best_f1_t = thresholds_all[np.argmax([
        (2*precision_score(y_tr, (cal_oof >= t).astype(int))*recall_score(y_tr, (cal_oof >= t).astype(int)) /
          (precision_score(y_tr, (cal_oof >= t).astype(int)) + recall_score(y_tr, (cal_oof >= t).astype(int))))
        if ((cal_oof >= t).astype(int)).sum() > 0 else 0
        for t in thresholds_all
    ])]
    y_hat_best_f1 = (cal_oof >= best_f1_t).astype(int)
    best_f1_prec = precision_score(y_tr, y_hat_best_f1)
    best_f1_rec = recall_score(y_tr, y_hat_best_f1)
    best_f1_f1 = 2*best_f1_prec*best_f1_rec/(best_f1_prec+best_f1_rec)

    # Select recall-optimized threshold (maximize recall at acceptable precision >= 0.25)
    viable = []
    for t in thresholds_all:
        y_hat = (cal_oof >= t).astype(int)
        if y_hat.sum() == 0: continue
        prec = precision_score(y_tr, y_hat); rec = recall_score(y_tr, y_hat)
        if prec >= 0.25:
            viable.append((t, prec, rec))

    if viable:
        best_viable_rec = max(viable, key=lambda x: x[2])  # maximize recall
        print(f"\n  Best at Prec >= 0.25: t={best_viable_rec[0]:.3f}  Prec={best_viable_rec[1]:.4f}  Rec={best_viable_rec[2]:.4f}")

    print(f"\n  Best F1 threshold: t={best_f1_t:.3f}  Prec={best_f1_prec:.4f}  Rec={best_f1_rec:.4f}  F1={best_f1_f1:.4f}")

    # Final threshold: best F1 (optimal balance)
    final_t = best_f1_t
    opt_mode = 'best_f1'

    print(f"\n  ==> FINAL THRESHOLD: {final_t:.4f} (best F1 balance)")
    print(f"      Note: Rec>=0.60+Prec>=0.30 NOT jointly achievable on this data")
    print(f"      Best F1: Prec={best_f1_prec:.4f} Rec={best_f1_rec:.4f}")
    if viable:
        print(f"      At Prec>=0.25: Rec={best_viable_rec[2]:.4f}")

    # 11. Evaluate
    print(f"\n{'='*60}")
    print("  FINAL EVALUATION")
    print(f"{'='*60}")
    final_result = evaluate(y_te.values, cal_test, final_t, "(FINAL)")

    # 12. Feature importance
    fi = show_importance(final_model, MODEL_FEATURES)

    # 13. SHAP
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_te_imp), size=min(3000, len(X_te_imp)), replace=False)
    shap_drivers = compute_shap(final_model, X_te_imp.iloc[sample_idx], MODEL_FEATURES)

    # 14. Fairness audit
    print(f"\n{'='*60}")
    print("  FAIRNESS AUDIT")
    print(f"{'='*60}")
    df_aud_te = df_aud_te.reset_index(drop=True)
    y_te_vals = y_te.reset_index(drop=True).values
    cal_test_r = cal_test.copy()

    fairness_results = {}
    for feat, name, lmap in [
        ('age_group','Age Group',      {0:'<25',1:'25-35',2:'35-45',3:'45-55',4:'55-65',5:'65+'}),
        ('education_raw','Education',  {'Secondary / secondary special':'Secondary','Higher education':'Higher',
                                        'Incomplete higher':'Incomplete','Lower secondary':'Lower','Academic degree':'Academic'}),
        ('housing_raw','Housing Type', {'House / apartment':'House/Apt','With parents':'Parents',
                                        'Municipal apartment':'Municipal','Office apartment':'Office',
                                        'Co-op apartment':'Co-op','Rented apartment':'Rented'}),
        ('income_quintile','Income Quintile', {0:'Q1(lowest)',1:'Q2',2:'Q3',3:'Q4',4:'Q5(highest)'}),
        ('_gender_raw','Gender (audit)', {'M':'Male','F':'Female','XNA':'Unknown','Unknown':'Unknown'}),
    ]:
        fairness_results[name] = fairness_audit(
            df_aud_te, y_te_vals, cal_test_r, final_t, feat, name, lmap)

    # 15. Save artifacts
    artifacts = {
        'model': final_model,
        'imputer': imputer,
        'iso_calibrator': iso_final,
        'features': MODEL_FEATURES,
        'audit_features': AUDIT_FEATURES,
        'params': PARAMS,
        'threshold': float(final_t),
        'cv_auc': float(np.mean([s['auc'] for s in fold_scores])),
        'cv_ap':  float(np.mean([s['ap']  for s in fold_scores])),
        'fairness': fairness_results,
        'shap_top10': shap_drivers.head(10).to_dict(),
        'opt_mode': opt_mode,
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(artifacts, MODEL_PATH.replace('.pkl','_artifacts.pkl'))
    with open(THRESHOLD_PATH, 'w') as f:
        json.dump({'threshold': float(final_t), 'mode': opt_mode,
                    'precision': float(final_result['precision']),
                    'recall': float(final_result['recall']),
                    'f1': float(final_result['f1']),
                    'business_cost': int(final_result['cost'])}, f, indent=2)
    print(f"\n  Saved: {MODEL_PATH}, {ARTIFACTS_PATH}, {THRESHOLD_PATH}")

    # 16. Write reports
    write_evaluation_report(final_result, fold_scores, shap_drivers, opt_mode)
    write_fairness_report(fairness_results)

    # 17. Before/After comparison
    print(f"\n{'='*60}")
    print("  BEFORE vs AFTER")
    print(f"{'='*60}")
    print(f"  {'Metric':<18} {'Before':>10} {'After':>10} {'Target':>10} {'Change':>10}")
    print(f"  {'-'*58}")
    before = {'auc':0.7596,'ap':0.2473,'precision':0.2497,'recall':0.4268,'f1':0.3150}
    for k,v in before.items():
        cur = final_result[k]
        tgt = {'auc':0.75,'ap':None,'precision':0.30,'recall':0.60,'f1':None}.get(k,'')
        chg = cur - v
        flag = ''
        if isinstance(tgt,float): flag = ' OK' if cur >= tgt else ' LOW'
        print(f"  {k.capitalize():<18} {v:>10.4f} {cur:>10.4f} {str(tgt):>10} {chg:>+10.4f}{flag}")

    prec_ok = 'PASS' if final_result['precision'] >= 0.30 else 'FAIL'
    rec_ok  = 'PASS' if final_result['recall']  >= 0.60 else 'LOW'
    auc_ok  = 'PASS' if final_result['auc']     >= 0.75 else 'FAIL'
    all_fair_pass = all(
        r['di_ratio'] >= 0.80 and r['max_dev'] <= 20
        for r in fairness_results.values()
    )

    print(f"\n  TARGET STATUS:")
    print(f"    Precision >= 0.30: {final_result['precision']:.4f}  [{prec_ok}]")
    print(f"    Recall    >= 0.60: {final_result['recall']:.4f}  [{rec_ok}]")
    print(f"    ROC-AUC   >= 0.75: {final_result['auc']:.4f}  [{auc_ok}]")
    print(f"    Fairness DI >= 0.80: {'ALL PASS' if all_fair_pass else 'SOME FAIL'}")
    print(f"    Business Cost:      {final_result['cost']:,}")
    print(f"\n  Total time: {time.time()-t0:.1f}s")
    print(f"{'='*60}")
    return artifacts, final_result


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict(input_dict: dict, artifacts=None) -> dict:
    if artifacts is None:
        artifacts = joblib.load(ARTIFACTS_PATH)
    row = pd.DataFrame([input_dict])
    row.columns = [c.upper() for c in row.columns]
    df = engineer_features(row)
    X  = df[artifacts['features']]
    X  = pd.DataFrame(artifacts['imputer'].transform(X), columns=X.columns)
    raw_prob = artifacts['model'].predict_proba(X)[0,1]
    prob     = artifacts['iso_calibrator'].predict([raw_prob])[0]
    approved = bool(prob < artifacts['threshold'])
    return {
        'default_probability': float(prob),
        'approved': approved,
        'threshold': artifacts['threshold'],
        'model': 'LightGBM-v2',
    }


if __name__ == "__main__":
    main()
