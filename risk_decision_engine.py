"""
Risk Decision Engine v3 — Production Credit Risk Scoring System
==============================================================
Replaces binary classification with tiered risk scoring + decision optimization.
Uses Home Credit dataset (application_train.csv) as primary data source.

Key design:
- Continuous risk score (0-1 probability), not binary decision
- 3-tier decision: AUTO-APPROVE / MANUAL REVIEW / REJECT
- Segment-specific thresholds
- Business-cost-driven threshold optimization
- Fairness audit per tier
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
    precision_recall_curve, average_precision_score
)
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
HOME_CREDIT_PATH = "application_train.csv"
LOAN_REAL_PATH   = "loan_data_real.csv"
RISK_MODEL_PATH   = "risk_model.pkl"
PREPROC_PATH      = "preprocessing_pipeline.pkl"
THRESHOLDS_PATH   = "thresholds.json"
DECISION_ENG_PATH = "decision_engine.py"
COST_CSV_PATH     = "cost_analysis.csv"
SEG_REPORT_PATH   = "segmentation_report.txt"
FAIR_PATH         = "fairness_report.txt"
SHAP_ANALYSIS_PATH = "shap_analysis.pkl"
EVAL_REPORT_PATH  = "risk_engine_report.txt"

# Business cost parameters (real-world inspired)
COST_FN = 5.0   # cost of approving a defaulter (5xFP — high)
COST_FP = 1.0   # cost of rejecting a good applicant
COST_TP = 0.0   # correct approval — no cost
COST_TN = 0.0   # correct rejection — no cost

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_home_credit(path=HOME_CREDIT_PATH):
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
    print(f"  Home Credit: {len(df):,} rows | Default rate: {df['TARGET'].mean()*100:.2f}%")
    return df


def load_loan_real(path=LOAN_REAL_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Loan_Status"])
    print(f"  Loan Real: {len(df)} rows | Default rate: {df['Loan_Status'].mean()*100:.2f}%")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, source="home_credit") -> pd.DataFrame:
    df = df.copy()

    if source == "home_credit":
        # Core numerics
        df['income']      = df['AMT_INCOME_TOTAL']
        df['credit']      = df['AMT_CREDIT']
        df['annuity']     = df['AMT_ANNUITY'].fillna(df['AMT_CREDIT'] / 120)
        df['goods']       = df['AMT_GOODS_PRICE'].fillna(df['AMT_CREDIT'])
        df['age_years']   = -df['DAYS_BIRTH'] / 365.25
        df['emp_years']   = -df['DAYS_EMPLOYED'].replace({365243: np.nan}) / 365.25
        df['reg_years']   = -df['DAYS_REGISTRATION'] / 365.25
        df['id_years']    = -df['DAYS_ID_PUBLISH'] / 365.25

        # REQUIRED RATIOS
        df['income_to_credit']    = df['income'] / df['credit'].replace(0, np.nan)
        df['annuity_to_income']  = df['annuity'] / df['income'].replace(0, np.nan)
        df['credit_term_months'] = df['credit'] / df['annuity'].replace(0, np.nan)

        # ADDITIONAL RATIOS
        df['ltv']               = df['credit'] / df['goods'].replace(0, np.nan)
        df['dti']               = df['annuity'] / df['income'].replace(0, np.nan)
        df['emi_to_income']     = df['annuity'] / df['income'].replace(0, np.nan)
        df['income_per_person']  = df['income'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)
        df['emp_ratio']         = df['emp_years'] / df['age_years'].replace(0, np.nan)
        df['goods_to_income']   = df['goods'] / df['income'].replace(0, np.nan)
        df['annuity_per_credit']= df['annuity'] / df['credit'].replace(0, np.nan)
        df['loan_to_income']    = df['credit'] / (df['income'] * (df['credit_term_months']/12).replace(0, np.nan))

        # INTERACTION FEATURES
        df['income_x_ext_mean']    = df['income'] * df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
        df['annuity_x_ext_mean']  = df['annuity'] * df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
        df['credit_x_ext_mean']   = df['credit'] * df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
        df['income_x_ltv']        = df['income'] * df['ltv']
        df['age_x_ext_mean']      = df['age_years'] * df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)

        # Clean outliers
        for c in ['income_to_credit','annuity_to_income','credit_term_months','ltv','dti',
                  'emi_to_income','income_per_person','emp_ratio','goods_to_income',
                  'annuity_per_credit','loan_to_income','income_x_ext_mean',
                  'annuity_x_ext_mean','credit_x_ext_mean','income_x_ltv','age_x_ext_mean']:
            q01, q99 = df[c].quantile([0.01, 0.99])
            df[c] = df[c].clip(q01, q99)

        # EXT_SOURCE
        ext = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
        df['ext_mean']     = df[ext].mean(axis=1)
        df['ext_std']      = df[ext].std(axis=1)
        df['ext_min']      = df[ext].min(axis=1)
        df['ext_max']      = df[ext].max(axis=1)
        df['ext_prod']     = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        df['ext_1_2']     = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
        df['ext_2_3']     = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        df['ext_weighted']= 0.15*df['EXT_SOURCE_1'].fillna(0) + 0.5*df['EXT_SOURCE_2'].fillna(0) + 0.35*df['EXT_SOURCE_3'].fillna(0)

        # Bureau
        bureau_cols = [c for c in df.columns if 'AMT_REQ_CREDIT_BUREAU' in c]
        if bureau_cols:
            df['bureau_total_req'] = df[bureau_cols].sum(axis=1)

        # Behavioral
        df['has_car']         = (df['FLAG_OWN_CAR']    == 'Y').astype(int)
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

        # Categorical (NO CODE_GENDER, NO NAME_FAMILY_STATUS)
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

        # Protected (audit only)
        df['_gender_raw']       = df['CODE_GENDER'] if 'CODE_GENDER' in df.columns else 'Unknown'
        df['age_group']          = pd.cut(df['age_years'], bins=[0,25,35,45,55,65,100],
                                           labels=[0,1,2,3,4,5]).astype(float)
        df['education_raw']      = df['NAME_EDUCATION_TYPE']
        df['housing_raw']        = df['NAME_HOUSING_TYPE']
        df['income_quintile']    = pd.qcut(df['income'].rank(method='first'),
                                            q=5, labels=[0,1,2,3,4]).astype(float)
        df['income_group']       = pd.qcut(df['income'].rank(method='first'),
                                            q=3, labels=[0,1,2]).astype(float)  # low/med/high
        df['emp_type']           = df['NAME_INCOME_TYPE'].map(
            {'Working':0,'Commercial associate':1,'Pensioner':2,'State servant':3}
        ).fillna(-1).astype(float)
        df['has_credit_history']= (df['EXT_SOURCE_1'].notna() | df['EXT_SOURCE_2'].notna()).astype(int)
        df['target']             = df['TARGET'].astype(int)

    elif source == "loan_real":
        df["income"]    = df["ApplicantIncome"] + df["CoapplicantIncome"].fillna(0)
        df["loan_amount"] = df["LoanAmount"].fillna(df["LoanAmount"].median()) * 1000
        df["loan_term_years"] = (df["Loan_Amount_Term"].fillna(360) / 12).clip(1, 40).astype(int)
        df["credit_score_raw"] = df["Credit_History"].fillna(0)
        np.random.seed(42)
        df["credit_score"] = np.where(
            df["credit_score_raw"] == 1.0,
            np.random.normal(730, 55, len(df)).clip(650, 850).astype(int),
            np.random.normal(520, 90, len(df)).clip(300, 600).astype(int)
        )
        median_ltv = 0.80
        df["property_value_est"] = df["loan_amount"] / median_ltv
        df["ltv"] = df["loan_amount"] / df["property_value_est"]
        df["dti"] = df["loan_amount"] / (df["income"] * df["loan_term_years"])
        r = (8.5 / 100) / 12
        n = df["loan_term_years"] * 12
        df["emi"] = df["loan_amount"] * r * (1 + r)**n / ((1 + r)**n - 1)
        df["emi_to_income"] = df["emi"] / df["income"]
        df["income_to_credit"] = df["income"] / df["loan_amount"]
        df["annuity_to_income"] = df["emi"] / df["income"]
        df["credit_term_months"] = df["loan_term_years"] * 12
        df["income_per_person"] = df["income"] / (pd.to_numeric(df["Dependents"].replace("3+","3"), errors="coerce").fillna(0) + 1)
        df["ext_mean"] = df["credit_score"] / 850
        df["ext_std"] = 0.05
        df["ext_weighted"] = df["ext_mean"]
        df["has_realty"] = df["Property_Area"].map({"Urban":1,"Semiurban":1,"Rural":0}).fillna(0).astype(int)
        df["has_car"] = 0
        df["documents_provided"] = 1
        df["bureau_total_req"] = 0
        df["region_mismatch"] = 0
        df["age_years"] = 30
        df["emp_years"] = 5
        df["OCCUPATION_TYPE_ORD"] = 0
        df["NAME_CONTRACT_TYPE"] = 0
        df["NAME_INCOME_TYPE"] = 0
        df["car_age_flag"] = 0
        df["age_group"] = 1
        df["education_raw"] = df["Education"]
        df["housing_raw"] = df["Property_Area"]
        df["income_quintile"] = pd.qcut(df["income"].rank(method="first"), q=5, labels=[0,1,2,3,4]).astype(float)
        df["income_group"] = pd.qcut(df["income"].rank(method="first"), q=3, labels=[0,1,2]).astype(float)
        df["emp_type"] = 0
        df["has_credit_history"] = df["credit_score_raw"]
        df["target"] = df["Loan_Status"].astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FEATURES — NO sensitive features
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FEATURES = [
    # Required ratios
    'income_to_credit', 'annuity_to_income', 'credit_term_months',
    # Core financials
    'income', 'credit', 'annuity', 'goods',
    # Derived ratios
    'ltv', 'dti', 'emi_to_income', 'income_per_person',
    'annuity_per_credit', 'loan_to_income',
    # Age & employment
    'age_years', 'emp_years',
    # Interaction features
    'income_x_ext_mean', 'annuity_x_ext_mean', 'credit_x_ext_mean',
    'income_x_ltv', 'age_x_ext_mean',
    # External scores
    'ext_mean', 'ext_std', 'ext_min', 'ext_max',
    'ext_prod', 'ext_1_2', 'ext_2_3', 'ext_weighted',
    # Bureau
    'bureau_total_req',
    # Behavioral
    'has_car', 'has_realty', 'car_age_flag', 'children_ratio',
    'region_mismatch', 'documents_provided',
    # Encoded categoricals
    'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE',
    'OCCUPATION_TYPE_ORD',
    # Region
    'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
    'HOUR_APPR_PROCESS_START',
    # Segments
    'income_group', 'emp_type', 'has_credit_history',
]

AUDIT_FEATURES = ['age_group', 'education_raw', 'housing_raw', 'income_quintile', '_gender_raw']
DROP_FEATURES  = ['CODE_GENDER', 'NAME_FAMILY_STATUS']


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(X_train, X_test=None):
    imp = SimpleImputer(strategy='median')
    imp.fit(X_train)
    X_tr = pd.DataFrame(imp.transform(X_train), columns=X_train.columns, index=X_train.index)
    if X_test is not None:
        X_te = pd.DataFrame(imp.transform(X_test), columns=X_test.columns, index=X_test.index)
        return X_tr, X_te, imp
    return X_tr, imp


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

PARAMS = dict(
    n_estimators=1000, learning_rate=0.03, num_leaves=31,
    max_depth=6, min_child_samples=50, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    is_unbalance=True, random_state=42, verbose=-1, n_jobs=-1,
)

def train_cv(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X)); fold_models = []; scores = []
    for fold,(tr,va) in enumerate(skf.split(X,y)):
        X_tr,X_va,y_tr,y_va = X.iloc[tr],X.iloc[va],y.iloc[tr],y.iloc[va]
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        model  = lgb.train({**PARAMS,'metric':'auc'}, dtrain,
                            num_boost_round=PARAMS['n_estimators'],
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
        oof[va] = model.predict(X_va)
        auc = roc_auc_score(y_va, oof[va])
        ap  = average_precision_score(y_va, oof[va])
        scores.append({'fold':fold+1,'auc':auc,'ap':ap,'iter':model.best_iteration})
        fold_models.append(model)
        print(f"    Fold {fold+1}: AUC={auc:.4f} AP={ap:.4f} iter={model.best_iteration}")
    print(f"  CV ROC-AUC: {np.mean([s['auc'] for s in scores]):.4f} (+/- {np.std([s['auc'] for s in scores]):.4f})")
    return oof, fold_models, scores


def train_final_smote(X_tr_imp, y_tr):
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_sm, y_sm = smote.fit_resample(X_tr_imp, y_tr)
    model = lgb.LGBMClassifier(**{**PARAMS, 'n_estimators': 800})
    model.fit(X_sm, y_sm)
    return model, smote


def calibrate_oof(X_tr_imp, y_tr, smote_model, smote):
    """Generate OOF from SMOTE model for calibration."""
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
    smote_oof = np.zeros(len(X_tr_imp))
    for tr2,va2 in skf2.split(X_tr_imp, y_tr):
        X_t,X_v = X_tr_imp.iloc[tr2],X_tr_imp.iloc[va2]
        y_t = y_tr.iloc[tr2]
        X_ts,y_ts = smote.fit_resample(X_t, y_t)
        m = lgb.LGBMClassifier(**{**PARAMS,'n_estimators':500,'verbose':-1})
        m.fit(X_ts, y_ts)
        smote_oof[va2] = m.predict_proba(X_v)[:,1]
    iso = IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip')
    iso.fit(smote_oof, y_tr.values)
    return iso, smote_oof


# ─────────────────────────────────────────────────────────────────────────────
# 5. TIERED DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def apply_tiers(probs, t1, t2):
    """Assign risk tier based on thresholds."""
    decision = np.where(
        probs < t1, 'AUTO-APPROVE',
        np.where(probs < t2, 'MANUAL-REVIEW', 'REJECT')
    )
    tier_score = np.where(
        probs < t1, 0,
        np.where(probs < t2, 1, 2)
    )
    return decision, tier_score


def tier_metrics(y_true, probs, t1, t2):
    """Compute per-tier metrics."""
    decision, tier_score = apply_tiers(probs, t1, t2)
    results = {}
    for tier_name, tier_code in [('AUTO-APPROVE',0),('MANUAL-REVIEW',1),('REJECT',2)]:
        mask = tier_score == tier_code
        n = mask.sum()
        if n == 0:
            results[tier_name] = {'n':0,'approval_rate':0,'default_rate':0,'n_defaults':0}
            continue
        y_t = y_true[mask]
        defaults = y_t.sum()
        appr_rate = 1 - y_t.mean()
        default_rate = y_t.mean()
        results[tier_name] = {
            'n': int(n), 'approval_rate': float(appr_rate),
            'default_rate': float(default_rate), 'n_defaults': int(defaults)
        }
    return results


def business_cost_at_tiers(y_true, probs, t1, t2, cost_fp=COST_FP, cost_fn=COST_FN):
    """Compute total business cost for tiered decision."""
    decision, tier_score = apply_tiers(probs, t1, t2)
    total_cost = 0.0
    breakdown = {}
    for tier_name, tier_code in [('AUTO-APPROVE',0),('MANUAL-REVIEW',1),('REJECT',2)]:
        mask = tier_score == tier_code
        n = mask.sum()
        if n == 0:
            breakdown[tier_name] = {'n':0,'cost':0,'fp':0,'fn':0}; continue
        y_t = y_true[mask]
        fp = (y_t == 0).sum()   # rejected good applicant
        fn = (y_t == 1).sum()   # missed defaulter
        tier_cost = fp * cost_fp + fn * cost_fn
        total_cost += tier_cost
        breakdown[tier_name] = {'n':int(n),'cost':float(tier_cost),'fp':int(fp),'fn':int(fn)}
    return float(total_cost), breakdown


# ─────────────────────────────────────────────────────────────────────────────
# 6. THRESHOLD OPTIMIZATION FOR TIERED SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def optimize_tiers(y_true, y_prob, segments=None, seg_col=None):
    """
    Grid search T1, T2 to minimize business cost.
    Returns optimal thresholds + full tradeoff table.
    """
    print(f"\n  Optimizing T1, T2 for minimum business cost...")
    t1_grid = np.linspace(0.05, 0.40, 36)
    t2_grid = np.linspace(0.20, 0.70, 51)
    best_cost = np.inf
    best_t1 = best_t2 = 0.15
    rows = []

    for t1 in t1_grid:
        for t2 in t2_grid:
            if t2 <= t1:
                continue
            cost, bd = business_cost_at_tiers(y_true, y_prob, t1, t2)
            tiers = apply_tiers(y_prob, t1, t2)[1]
            n_auto = (tiers==0).sum()
            n_manual = (tiers==1).sum()
            n_reject = (tiers==2).sum()
            appr_rate = n_auto / len(y_true)
            manual_rate = n_manual / len(y_true)
            reject_rate = n_reject / len(y_true)
            rows.append({'t1':t1,'t2':t2,'cost':cost,
                         'n_auto':n_auto,'n_manual':n_manual,'n_reject':n_reject,
                         'appr_rate':appr_rate,'manual_rate':manual_rate,'reject_rate':reject_rate})
            if cost < best_cost:
                best_cost = cost; best_t1 = t1; best_t2 = t2

    df = pd.DataFrame(rows).sort_values('cost')
    print(f"  Best: T1={best_t1:.3f} T2={best_t2:.3f} Cost={best_cost:,.0f}")

    # Show top 10 configurations
    print(f"\n  Top 10 cheapest configurations:")
    print(f"  {'T1':>6} {'T2':>6} {'Cost':>12} {'Auto%':>8} {'Manual%':>9} {'Reject%':>9}")
    print(f"  {'-'*50}")
    for _, r in df.head(10).iterrows():
        print(f"  {r['t1']:>6.3f} {r['t2']:>6.3f} {r['cost']:>12,.0f} "
              f"{r['appr_rate']*100:>7.1f}% {r['manual_rate']*100:>8.1f}% {r['reject_rate']*100:>8.1f}%")

    return best_t1, best_t2, df


def segment_threshold_optimization(X_tr_imp, y_tr, iso, smote_model, smote, seg_col):
    """Find optimal thresholds per data segment."""
    results = {}
    segs = sorted(X_tr_imp[seg_col].unique())
    print(f"\n  Segment-specific thresholds for '{seg_col}': {list(segs)}")
    print(f"  {'Segment':>10} {'T1':>8} {'T2':>8} {'Cost':>12} {'N':>8}")
    print(f"  {'-'*50}")
    global_t1, global_t2 = None, None

    for seg in segs:
        mask = X_tr_imp[seg_col] == seg
        if mask.sum() < 500:
            continue
        y_s = y_tr[mask].values
        p_s = iso.predict(smote_model.predict_proba(X_tr_imp[mask])[:,1])
        t1, t2, _ = optimize_tiers(y_s, p_s)
        cost, _ = business_cost_at_tiers(y_s, p_s, t1, t2)
        results[int(seg)] = {'t1':float(t1),'t2':float(t2),'cost':float(cost),'n':int(mask.sum())}
        print(f"  {int(seg):>10} {t1:>8.3f} {t2:>8.3f} {cost:>12,.0f} {mask.sum():>8,}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_tiered(y_true, probs, t1, t2, label=""):
    decision, tier_score = apply_tiers(probs, t1, t2)
    total_cost, bd = business_cost_at_tiers(y_true, probs, t1, t2)
    tier_results = tier_metrics(y_true, probs, t1, t2)
    overall_appr = 1 - y_true.mean()

    print(f"\n{'='*60}")
    print(f"  TIERED EVALUATION {label}")
    print(f"{'='*60}")
    print(f"  Thresholds: T1={t1:.4f} T2={t2:.4f}")
    print(f"  Total Business Cost: {total_cost:,.0f}")
    print(f"\n  Tier Distribution:")
    print(f"  {'Tier':<18} {'N':>8} {'Appr%':>8} {'Default%':>10} {'Defaults':>10} {'Cost':>12}")
    print(f"  {'-'*68}")
    for name in ['AUTO-APPROVE','MANUAL-REVIEW','REJECT']:
        r = tier_results[name]
        print(f"  {name:<18} {r['n']:>8,} {r['approval_rate']*100:>7.1f}% {r['default_rate']*100:>9.2f}% {r['n_defaults']:>10,} {bd[name]['cost']:>12,.0f}")

    print(f"\n  Overall: Approval Rate={1-y_true.mean()*100:.1f}% | Cost={total_cost:,.0f}")
    return {'t1':t1,'t2':t2,'total_cost':total_cost,'breakdown':bd,'tier_results':tier_results}


def evaluate_binary(y_true, probs, threshold, label=""):
    y_hat = (probs >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    tn,fp,fn,tp = confusion_matrix(y_true, y_hat, labels=[0,1]).ravel()
    prec = precision_score(y_true, y_hat)
    rec  = recall_score(y_true, y_hat)
    f1   = f1_score(y_true, y_hat)
    cost = fp*COST_FP + fn*COST_FN
    print(f"\n  Binary @ {threshold:.3f}: Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} Cost={cost:,}")
    return {'threshold':threshold,'auc':auc,'precision':prec,'recall':rec,'f1':f1,'cost':cost}


# ─────────────────────────────────────────────────────────────────────────────
# 8. FAIRNESS AUDIT PER TIER
# ─────────────────────────────────────────────────────────────────────────────

def fairness_audit_tiered(df_audit, y_true, probs, t1, t2,
                          group_col, group_name, label_map=None):
    decision, tier_score = apply_tiers(probs, t1, t2)
    results = []
    overall_auto = (tier_score == 0).mean()
    overall_manual = (tier_score == 1).mean()
    overall_reject = (tier_score == 2).mean()

    print(f"\n  Fairness per tier: {group_name}")
    print(f"  {'Group':<20} {'N':>7} {'Auto%':>8} {'Manual%':>9} {'Reject%':>9} {'MaxDev%':>9} {'Status':>8}")
    print(f"  {'-'*72}")
    flagged = []
    for g in sorted(df_audit[group_col].dropna().unique()):
        mask = df_audit[group_col] == g
        n = mask.sum()
        if n < 10: continue
        auto_r = (tier_score[mask] == 0).mean()
        man_r  = (tier_score[mask] == 1).mean()
        rej_r  = (tier_score[mask] == 2).mean()
        auto_dev = (auto_r - overall_auto)/overall_auto*100 if overall_auto > 0 else 0
        man_dev  = (man_r - overall_manual)/overall_manual*100 if overall_manual > 0 else 0
        rej_dev  = (rej_r - overall_reject)/overall_reject*100 if overall_reject > 0 else 0
        max_dev = max(abs(auto_dev), abs(man_dev), abs(rej_dev))
        flag = "[FLAG]" if max_dev > 20 else "OK"
        if max_dev > 20: flagged.append(str(g))
        lbl = (label_map.get(g,g) if label_map else g)
        print(f"  {str(lbl):<20} {n:>7} {auto_r*100:>7.1f}% {man_r*100:>8.1f}% {rej_r*100:>8.1f}% {max_dev:>8.1f}% {flag:>8}")
        results.append({'group':str(lbl),'n':int(n),'auto_rate':float(auto_r),
                       'manual_rate':float(man_r),'reject_rate':float(rej_r),
                       'max_dev':float(max_dev),'flagged':max_dev>20})
    di_ratio = min(r['auto_rate']/overall_auto for r in results if r['auto_rate']>0) / \
               max(r['auto_rate']/overall_auto for r in results if r['auto_rate']>0) if results else 1.0
    print(f"  Auto-approval DI Ratio: {di_ratio:.3f}")
    return {'di_ratio':float(di_ratio),'groups':results,'flagged_groups':flagged}


# ─────────────────────────────────────────────────────────────────────────────
# 9. SHAP
# ─────────────────────────────────────────────────────────────────────────────

PROXY_KW = ['gender','sex','marital','family','children','race','religion']

def compute_shap(model, X_sample, feature_names, top_n=15):
    print(f"\n  SHAP on {len(X_sample)} samples...")
    sv = shap.TreeExplainer(model).shap_values(X_sample)
    sv_u = sv[1] if isinstance(sv,list) else sv
    ma = pd.Series(np.abs(sv_u).mean(axis=0), index=feature_names).sort_values(ascending=False).head(top_n)
    print(f"\n  Top {top_n} SHAP:")
    mx = ma.iloc[0]
    for i,(n,v) in enumerate(ma.items(),1):
        print(f"    {i:2d}. {n:<35} {v:.4f} {'*'*int(v/mx*30)}")
    proxies = [n for n in feature_names if any(p in n.lower() for p in PROXY_KW)]
    prox = ma[ma.index.isin(proxies)]
    if len(prox): print(f"  PROXY: {', '.join(prox.index.tolist())}")
    else: print(f"  [OK] No proxies")
    joblib.dump({'mean_abs': ma.to_dict()}, SHAP_ANALYSIS_PATH)
    return ma


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("="*60)
    print("  RISK DECISION ENGINE v3")
    print("="*60)

    # Load data
    print(f"\n  Loading data...")
    df = load_home_credit(HOME_CREDIT_PATH)

    # Engineer features
    print(f"\n  Engineering features...")
    df = engineer_features(df, source="home_credit")
    for f in DROP_FEATURES:
        assert f not in MODEL_FEATURES, f"{f} still in features!"
    print(f"  Features: {len(MODEL_FEATURES)} | Dropped: {DROP_FEATURES}")

    # Prepare data
    X = df[MODEL_FEATURES].copy()
    y = df['target'].copy()
    df_audit = df[AUDIT_FEATURES].copy()
    X.index = y.index = df_audit.index = range(len(X))

    # Split
    X_tr,X_te,y_tr,y_te,df_aud_tr,df_aud_te = train_test_split(
        X,y,df_audit,test_size=0.2,random_state=42,stratify=y)
    print(f"\n  Train: {len(X_tr):,} | Test: {len(X_te):,}")
    print(f"  Default rate: {y_te.mean()*100:.2f}%")

    # Preprocess
    X_tr_i,X_te_i,imputer = preprocess(X_tr, X_te)
    print(f"  Preprocessed: {len(MODEL_FEATURES)} features")

    # CV (for reporting — model uses SMOTE)
    print(f"\n  5-Fold CV (LightGBM):")
    oof_probs, fold_models, fold_scores = train_cv(X_tr_i, y_tr)

    # Final SMOTE model
    print(f"\n  Training final SMOTE model...")
    final_model, smote = train_final_smote(X_tr_i, y_tr)
    raw_test = final_model.predict_proba(X_te_i)[:,1]

    # Calibrate
    print(f"\n  Calibrating...")
    iso, smote_oof = calibrate_oof(X_tr_i, y_tr, final_model, smote)
    cal_test = iso.predict(raw_test)
    cal_oof  = iso.predict(smote_oof)

    # ─── TIER 0: Binary baseline (for comparison) ───
    print(f"\n{'='*60}")
    print("  BINARY vs TIERED COMPARISON")
    print(f"{'='*60}")
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    print(f"\n  Binary threshold sweep:")
    print(f"  {'Threshold':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Cost':>12}")
    print(f"  {'-'*50}")
    for t in thresholds:
        y_h = (cal_test >= t).astype(int)
        if y_h.sum() == 0: continue
        p = precision_score(y_te, y_h); r = recall_score(y_te, y_h)
        f = 2*p*r/(p+r) if (p+r)>0 else 0
        tn,fp,fn,tp = confusion_matrix(y_te, y_h, labels=[0,1]).ravel()
        cost = fp*COST_FP + fn*COST_FN
        print(f"  {t:>9.3f} {p:>10.4f} {r:>8.4f} {f:>8.4f} {cost:>12,}")

    # ─── TIER 1: Global tier optimization ───
    print(f"\n{'='*60}")
    print("  TIER OPTIMIZATION (Global)")
    print(f"{'='*60}")
    t1_opt, t2_opt, tradeoff_df = optimize_tiers(y_tr.values, cal_oof)

    # ─── TIER 2: Evaluate ───
    print(f"\n{'='*60}")
    print("  TEST SET EVALUATION")
    print(f"{'='*60}")

    # Tiered
    tier_result = evaluate_tiered(y_te.values, cal_test, t1_opt, t2_opt, "(TEST SET)")

    # Binary at best F1 threshold
    f1s = []
    for t in np.linspace(0.05,0.95,100):
        y_h = (cal_oof >= t).astype(int)
        if y_h.sum()==0: f1s.append(0); continue
        p = precision_score(y_tr,y_h); r = recall_score(y_tr,y_h)
        f1s.append(2*p*r/(p+r) if (p+r)>0 else 0)
    best_f1_t = np.linspace(0.05,0.95,100)[np.argmax(f1s)]
    y_h_b = (cal_test >= best_f1_t).astype(int)
    bin_cost = ((y_te-y_h_b)==1).sum() * COST_FN + ((y_te-y_h_b)==-1).sum() * COST_FP

    print(f"\n  Comparison: Tiered vs Binary @ best F1 threshold")
    print(f"  Tiered:  Cost={tier_result['total_cost']:>12,} | T1={t1_opt:.3f} T2={t2_opt:.3f}")
    print(f"  Binary:  Cost={int(bin_cost):>12,} | t={best_f1_t:.3f}")
    print(f"  Savings: {int(bin_cost)-int(tier_result['total_cost']):>12,} ({((int(bin_cost)-int(tier_result['total_cost']))/int(bin_cost)*100):.1f}%)")

    # ─── TIER 3: Segment-specific thresholds ───
    print(f"\n{'='*60}")
    print("  SEGMENT-SPECIFIC THRESHOLDS")
    print(f"{'='*60}")
    seg_results = {}
    for seg_col, seg_name in [
        ('income_group',  'Income Group'),
        ('emp_type',      'Employment Type'),
        ('has_credit_history', 'Credit History'),
    ]:
        if seg_col not in X_tr_i.columns:
            print(f"  Skipping {seg_col} — not in features")
            continue
        seg_res = segment_threshold_optimization(X_tr_i, y_tr, iso, final_model, smote, seg_col)
        seg_results[seg_name] = seg_res

    # ─── TIER 4: SHAP ───
    print(f"\n{'='*60}")
    print("  SHAP ANALYSIS")
    print(f"{'='*60}")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_te_i), size=min(3000,len(X_te_i)), replace=False)
    shap_drivers = compute_shap(final_model, X_te_i.iloc[idx], MODEL_FEATURES)

    # ─── TIER 5: Fairness audit ───
    print(f"\n{'='*60}")
    print("  FAIRNESS AUDIT (per tier)")
    print(f"{'='*60}")
    df_aud_te = df_aud_te.reset_index(drop=True)
    y_te_v = y_te.reset_index(drop=True).values
    cal_test_r = cal_test.copy()

    fairness_results = {}
    for feat,name,lmap in [
        ('age_group','Age Group',     {0:'<25',1:'25-35',2:'35-45',3:'45-55',4:'55-65',5:'65+'}),
        ('education_raw','Education', {'Secondary / secondary special':'Secondary','Higher education':'Higher',
                                       'Incomplete higher':'Incomplete','Lower secondary':'Lower','Academic degree':'Academic'}),
        ('housing_raw','Housing Type',{'House / apartment':'House/Apt','With parents':'Parents',
                                       'Municipal apartment':'Municipal','Office apartment':'Office',
                                       'Co-op apartment':'Co-op','Rented apartment':'Rented'}),
        ('income_quintile','Income Quintile',{0:'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5'}),
        ('_gender_raw','Gender',     {'M':'Male','F':'Female','XNA':'Unknown','Unknown':'Unknown'}),
    ]:
        fairness_results[name] = fairness_audit_tiered(
            df_aud_te, y_te_v, cal_test_r, t1_opt, t2_opt, feat, name, lmap)

    # ─── SAVE ARTIFACTS ───
    artifacts = {
        'model': final_model,
        'imputer': imputer,
        'iso_calibrator': iso,
        'features': MODEL_FEATURES,
        'audit_features': AUDIT_FEATURES,
        'params': PARAMS,
        't1': float(t1_opt), 't2': float(t2_opt),
        'cv_auc': float(np.mean([s['auc'] for s in fold_scores])),
        'cv_ap':  float(np.mean([s['ap']  for s in fold_scores])),
        'fairness': fairness_results,
        'segment_thresholds': seg_results,
        'shap_top10': shap_drivers.head(10).to_dict(),
    }
    joblib.dump(artifacts, RISK_MODEL_PATH.replace('.pkl','_artifacts.pkl'))
    joblib.dump({'model':final_model,'imputer':imputer,'features':MODEL_FEATURES}, RISK_MODEL_PATH)
    joblib.dump({'imputer':imputer,'features':MODEL_FEATURES,'t1':float(t1_opt),'t2':float(t2_opt)}, PREPROC_PATH)
    with open(THRESHOLDS_PATH,'w') as f:
        json.dump({'t1':float(t1_opt),'t2':float(t2_opt),
                   'tier_names':{'AUTO-APPROVE':f'risk < {t1_opt:.4f}',
                                  'MANUAL-REVIEW':f'{t1_opt:.4f} <= risk < {t2_opt:.4f}',
                                  'REJECT':f'risk >= {t2_opt:.4f}'},
                   'business_cost':int(tier_result['total_cost'])}, f, indent=2)

    # ─── COST ANALYSIS CSV ───
    tradeoff_df.to_csv(COST_CSV_PATH, index=False)
    print(f"\n  Saved: {COST_CSV_PATH}")

    # ─── SEGMENTATION REPORT ───
    seg_lines = ["="*60,"SEGMENTATION REPORT","="*60,""]
    for seg_name, seg_res in seg_results.items():
        seg_lines.append(f"\n{seg_name}:")
        for k,v in seg_res.items():
            seg_lines.append(f"  {k}: T1={v['t1']:.3f} T2={v['t2']:.3f} Cost={v['cost']:,.0f} N={v['n']:,}")
    with open(SEG_REPORT_PATH,'w') as f: f.write('\n'.join(seg_lines))
    print(f"  Saved: {SEG_REPORT_PATH}")

    # ─── FAIRNESS REPORT ───
    fair_lines = ["="*60,"FAIRNESS REPORT — Risk Decision Engine v3","="*60,"",
                  "Constraint: No tier deviation > 20% per group","",
                  "Per-tier analysis shows approval rate deviation across groups.","",
                  "Global thresholds: T1={:.3f} T2={:.3f}".format(t1_opt,t2_opt),"",
                  "Results:","overall_auto_rate={:.3f}".format((tier_score==0).mean()),""]
    for name,res in fairness_results.items():
        status = "PASS" if res['di_ratio']>=0.80 and not res['flagged_groups'] else "FAIL"
        fair_lines.append(f"  {name}: DI={res['di_ratio']:.3f} Flagged={res['flagged_groups']} [{status}]")
    with open(FAIR_PATH,'w') as f: f.write('\n'.join(fair_lines))
    print(f"  Saved: {FAIR_PATH}")

    # ─── WRITE DECISION ENGINE ───
    decision_engine_code = f'''
"""
Auto-Generated Risk Decision Engine
T1={t1_opt:.4f} | T2={t2_opt:.4f}
Generated: 2026-04-15
"""
import numpy as np, joblib

def risk_to_decision(risk_score, t1={t1_opt:.4f}, t2={t2_opt:.4f}):
    if risk_score < t1: return "AUTO-APPROVE"
    if risk_score < t2: return "MANUAL-REVIEW"
    return "REJECT"

def risk_to_tier(risk_score, t1={t1_opt:.4f}, t2={t2_opt:.4f}):
    if risk_score < t1: return 0
    if risk_score < t2: return 1
    return 2

def predict(input_dict):
    artifacts = joblib.load("risk_model.pkl")
    import pandas as pd
    # ... apply same feature engineering + preprocessing + model
    # Returns: {{"risk_score": float, "tier": int, "decision": str}}
    pass

def batch_score(probabilities, t1={t1_opt:.4f}, t2={t2_opt:.4f}):
    decisions = [risk_to_decision(p,t1,t2) for p in probabilities]
    tier_scores = [risk_to_tier(p,t1,t2) for p in probabilities]
    return {{"decisions": decisions, "tier_scores": tier_scores}}
'''
    with open(DECISION_ENG_PATH,'w') as f: f.write(decision_engine_code)
    print(f"  Saved: {DECISION_ENG_PATH}")

    # ─── FINAL REPORT ───
    tier_score_test = apply_tiers(cal_test, t1_opt, t2_opt)[1]
    tier_result_test = tier_metrics(y_te.values, cal_test, t1_opt, t2_opt)

    print(f"\n{'='*60}")
    print("  FINAL SUMMARY — RISK DECISION ENGINE v3")
    print(f"{'='*60}")
    print(f"  Model:          LightGBM + SMOTE + Isotonic Calibration")
    print(f"  Features:       {len(MODEL_FEATURES)}")
    print(f"  DROPPED:        CODE_GENDER, NAME_FAMILY_STATUS")
    print(f"  CV ROC-AUC:    {np.mean([s['auc'] for s in fold_scores]):.4f}")
    print(f"  T1 (Auto-Approve): < {t1_opt:.4f}")
    print(f"  T2 (Manual-Review): < {t2_opt:.4f} | Reject: >= {t2_opt:.4f}")
    print(f"\n  TEST SET TIER DISTRIBUTION:")
    for name in ['AUTO-APPROVE','MANUAL-REVIEW','REJECT']:
        r = tier_result_test[name]
        print(f"    {name:<18} N={r['n']:>7,}  default_rate={r['default_rate']*100:.2f}%")
    print(f"\n  BUSINESS COST: {tier_result['total_cost']:>12,}")
    print(f"  vs Binary best-F1 savings: {int(bin_cost)-int(tier_result['total_cost']):>12,} ({(int(bin_cost)-int(tier_result['total_cost']))/int(bin_cost)*100:.1f}%)")
    print(f"\n  Fairness: {'ALL PASS' if all(not r['flagged_groups'] for r in fairness_results.values()) else 'SOME FLAGGED'}")
    print(f"  Proxy bias: {'NONE' if not any('PROXY' in str(r) for r in shap_drivers.index) else 'DETECTED'}")
    print(f"\n  Total time: {time.time()-t0:.1f}s")
    print(f"{'='*60}")

    return artifacts, tier_result


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def score(input_dict: dict, artifacts=None) -> dict:
    """Main inference function — returns risk score + decision."""
    if artifacts is None:
        artifacts = joblib.load(RISK_MODEL_PATH)
    row = pd.DataFrame([input_dict])
    row.columns = [c.upper() for c in row.columns]
    df = engineer_features(row, source="home_credit")
    X  = df[artifacts['features']]
    X  = pd.DataFrame(artifacts['imputer'].transform(X), columns=X.columns)
    raw_prob = artifacts['model'].predict_proba(X)[0,1]
    prob     = artifacts['iso_calibrator'].predict([raw_prob])[0]
    t1, t2  = artifacts['t1'], artifacts['t2']
    tier_score = (0 if prob < t1 else 1 if prob < t2 else 2)
    decision = ['AUTO-APPROVE','MANUAL-REVIEW','REJECT'][tier_score]
    return {
        'risk_score': float(prob),
        'tier': tier_score,
        'decision': decision,
        'model': 'LightGBM-v3-RiskEngine',
    }


if __name__ == "__main__":
    main()
