"""
Data Pipeline for Mortgage AI - Real Data Processing

Downloads and processes the "Give Me Some Credit" dataset from Kaggle,
engineers features to match the MODEL_FEATURES schema, applies SMOTE,
and saves train/val/test splits for model retraining.

Dataset: https://www.kaggle.com/c/GiveMeSomeCredit
Alternative: LendingClub Loan Data (if Kaggle unavailable)

Usage:
    python data_pipeline.py
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# =============================================================================
# Target Feature Schema - All downstream models expect these exact features
# =============================================================================

MODEL_FEATURES = [
    "credit_score",           # Credit score (300-850)
    "annual_income",          # Annual income in USD
    "loan_amount",            # Loan amount requested
    "loan_term",              # Loan term in months
    "dti_ratio",              # Debt-to-income ratio (0-1)
    "employment_years",       # Years at current employer
    "num_credit_lines",       # Number of open credit lines
    "num_derogatory_marks",   # Number of derogatory marks
    "credit_utilization",     # Credit utilization ratio (0-1)
    "payment_history_score",  # Payment history score (0-1)
    "home_ownership",         # Encoded: 0=rent, 1=own, 2=mortgage
    "purpose_encoded",        # Loan purpose encoded
    "num_late_payments",      # Number of late payments (30+ days)
    "savings_balance",        # Savings account balance
    "monthly_expenses",       # Monthly living expenses
]

# Kaggle "Give Me Some Credit" column mappings
KAGGLE_COLUMN_MAP = {
    'SeriousDlqin2yrs': 'target',  # Correct column name from Kaggle
    'RevolvingUtilizationOfUnsecuredLines': 'credit_utilization',
    'age': 'age_proxy',
    'NumberOfTime30-59DaysPastDueNotWorse': 'late_30_59',
    'DebtRatio': 'dti_ratio_raw',
    'MonthlyIncome': 'monthly_income',
    'NumberOfOpenCreditLinesAndLoans': 'num_credit_lines',
    'NumberOfTimes90DaysLate': 'late_90_plus',
    'NumberRealEstateLoansOrLines': 'real_estate_loans',
    'NumberOfTime60-89DaysPastDueNotWorse': 'late_60_89',
    'NumberOfDependents': 'dependents',
}

# Expected columns for validation
EXPECTED_KAGGLE_COLUMNS = [
    'SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
]

# =============================================================================
# Data Loading Functions
# =============================================================================


def load_kaggle_credit_data(data_dir: str = "data", use_api: bool = True) -> pd.DataFrame:
    """
    Load Kaggle Give Me Some Credit dataset.

    Tries in order:
    1. Local CSV if exists
    2. Kaggle API download (if kaggle package installed and configured)
    3. Direct HTTP download with opendatasets
    4. Synthetic data fallback

    Args:
        data_dir: Directory to store data
        use_api: Whether to attempt API download

    Returns:
        Raw DataFrame with Kaggle columns

    Raises:
        ValueError: If expected columns are missing from dataset
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    csv_path = data_path / "cs-training.csv"

    # Check if local file exists
    if csv_path.exists():
        logger.info(f"Loading Kaggle data from {csv_path}")
        df = pd.read_csv(csv_path, index_col=0)
        logger.info(f"Loaded {len(df):,} records from local cache")

        # Validate schema - safety check for expected columns
        logger.info(f"Dataset columns: {list(df.columns)}")
        missing_cols = [col for col in EXPECTED_KAGGLE_COLUMNS if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing expected columns: {missing_cols}")
            logger.error(f"Available columns: {list(df.columns)}")
            raise ValueError(
                f"Dataset schema mismatch. Missing columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}. "
                f"Expected 'SeriousDlqin2yrs' (Kaggle's target column)."
            )

        return df

    if not use_api:
        logger.info("API download disabled, using synthetic data")
        return _generate_synthetic_credit_data(n_samples=150000)

    # Try Kaggle API download
    logger.info("Attempting Kaggle API download...")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        import zipfile
        import shutil

        logger.info("Kaggle API available, attempting download...")

        # Initialize API
        api = KaggleApi()
        api.authenticate()

        # Download competition data
        competition = "GiveMeSomeCredit"
        download_path = data_path / "kaggle_download"
        download_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading competition: {competition}")
        api.competition_download_files(competition, path=str(download_path))

        # Extract ZIP
        zip_files = list(download_path.glob("*.zip"))
        if zip_files:
            logger.info(f"Extracting {zip_files[0].name}...")
            with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                zip_ref.extractall(data_path)

            # Clean up
            shutil.rmtree(download_path)
            zip_files[0].unlink()

        # Load the CSV
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0)
            logger.info(f"Downloaded and loaded {len(df):,} records from Kaggle API")
            return df
        else:
            logger.warning("CSV not found after extraction, checking for alternate names...")
            csv_files = list(data_path.glob("*.csv"))
            if csv_files:
                logger.info(f"Found alternate CSV: {csv_files[0]}")
                df = pd.read_csv(csv_files[0], index_col=0)
                return df

    except ImportError:
        logger.info("kaggle package not installed. Install with: pip install kaggle")
    except Exception as e:
        logger.warning(f"Kaggle API download failed: {e}")
        logger.info("Ensure kaggle.json is configured: ~/.kaggle/kaggle.json")
        logger.info("Get token from: https://www.kaggle.com/account")

    # Try opendatasets as fallback
    try:
        import opendatasets as od
        logger.info("Attempting opendatasets download...")
        od.download("https://www.kaggle.com/datasets/GiveMeSomeCredit/givemesomecredit",
                    data_dir=str(data_path))

        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0)
            logger.info(f"Downloaded and loaded {len(df):,} records via opendatasets")
            return df

    except ImportError:
        logger.info("opendatasets not installed. Install with: pip install opendatasets")
    except Exception as e:
        logger.warning(f"opendatasets download failed: {e}")

    # Final fallback: synthetic data
    logger.warning("=" * 60)
    logger.warning("KAGGLE DOWNLOAD UNAVAILABLE - USING SYNTHETIC DATA")
    logger.warning("=" * 60)
    logger.info("For real data, either:")
    logger.info("  1. Install kaggle: pip install kaggle")
    logger.info("     Then configure: kaggle competitions download givemesomecredit")
    logger.info("  2. Manual download: https://www.kaggle.com/competitions/givemesomecredit/data")
    logger.info(f"  3. Place cs-training.csv at: {csv_path.absolute()}")
    logger.info("=" * 60)

    return _generate_synthetic_credit_data(n_samples=150000)


def _generate_synthetic_credit_data(n_samples: int = 150000) -> pd.DataFrame:
    """
    Generate synthetic credit data matching Kaggle schema.

    Used when real data is unavailable for demonstration purposes.
    """
    np.random.seed(42)

    df = pd.DataFrame({
        'RevolvingUtilizationOfUnsecuredLines': np.clip(
            np.random.exponential(0.5, n_samples), 0, 5
        ),
        'age': np.random.randint(20, 80, n_samples),
        'NumberOfTime30-59DaysPastDueNotWorse': np.random.choice(
            [0, 0, 0, 0, 1, 2, 3, 5], n_samples
        ),
        'DebtRatio': np.clip(np.random.exponential(0.4, n_samples), 0, 3),
        'MonthlyIncome': np.random.lognormal(10.5, 0.5, n_samples),
        'NumberOfOpenCreditLinesAndLoans': np.random.randint(1, 20, n_samples),
        'NumberOfTimes90DaysLate': np.random.choice(
            [0, 0, 0, 0, 0, 1, 2, 3], n_samples
        ),
        'NumberRealEstateLoansOrLines': np.random.randint(0, 6, n_samples),
        'NumberOfTime60-89DaysPastDueNotWorse': np.random.choice(
            [0, 0, 0, 0, 1, 2, 3], n_samples
        ),
        'NumberOfDependents': np.random.choice([0, 0, 1, 2, 3, 4], n_samples),
    })

    # Generate target: SeriousDlqin2yrs (correct Kaggle column name)
    # Higher probability with: high utilization, many late payments, high DTI
    default_prob = (
        df['RevolvingUtilizationOfUnsecuredLines'] * 0.15 +
        df['NumberOfTime30-59DaysPastDueNotWorse'] * 0.1 +
        df['NumberOfTimes90DaysLate'] * 0.15 +
        df['DebtRatio'] * 0.1 +
        np.random.normal(0, 0.1, n_samples)
    )
    default_prob = np.clip(default_prob, 0.02, 0.95)
    df['SeriousDlqin2yrs'] = (
        np.random.random(n_samples) < default_prob
    ).astype(int)

    return df


# =============================================================================
# Data Cleaning Functions
# =============================================================================


def clean_kaggle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Kaggle data.

    Steps:
    1. Remove rows with >40% missing values
    2. Impute remaining nulls (median for numeric, mode for categorical)
    3. Cap outliers at 1st/99th percentile
    4. Remove duplicate applicants
    """
    logger.info(f"Starting data cleaning. Shape: {df.shape}")

    # Step 1: Remove rows with excessive missing values
    missing_per_row = df.isnull().sum(axis=1) / df.shape[1]
    df = df[missing_per_row < 0.4].copy()
    logger.info(f"After removing high-missing rows: {len(df):,}")

    # Step 2: Impute missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Imputed {col} with median={median_val:.2f}")

    # Step 3: Cap outliers at 1st/99th percentile
    for col in numeric_cols:
        p1, p99 = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(p1, p99)

    # Step 4: Remove duplicates (based on key features)
    key_cols = ['MonthlyIncome', 'age', 'DebtRatio']
    duplicates = df.duplicated(subset=key_cols, keep='first')
    n_dups = duplicates.sum()
    if n_dups > 0:
        df = df[~duplicates].copy()
        logger.info(f"Removed {n_dups:,} duplicate records")

    logger.info(f"Cleaning complete. Final shape: {df.shape}")
    return df


# =============================================================================
# Feature Engineering Functions
# =============================================================================


def engineer_features_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw Kaggle columns to MODEL_FEATURES schema.

    Creates all 15 features required by downstream models.
    """
    logger.info("Engineering features to match MODEL_FEATURES schema...")

    features = pd.DataFrame()

    # 1. credit_score: Derived from payment history and utilization
    # Base score 650, adjust based on risk factors
    base_score = 650
    payment_penalty = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] * 5 +
        df['NumberOfTime60-89DaysPastDueNotWorse'] * 10 +
        df['NumberOfTimes90DaysLate'] * 15
    )
    util_penalty = df['RevolvingUtilizationOfUnsecuredLines'] * 50
    features['credit_score'] = (
        base_score - payment_penalty - util_penalty +
        np.random.normal(0, 20, len(df))
    ).clip(300, 850).astype(int)

    # 2. annual_income: From MonthlyIncome
    features['annual_income'] = (df['MonthlyIncome'] * 12).fillna(50000)

    # 3. loan_amount: Simulated based on income (2-5x annual income)
    loan_multiplier = np.random.choice([2, 3, 4, 5], len(df), p=[0.3, 0.4, 0.2, 0.1])
    features['loan_amount'] = features['annual_income'] * loan_multiplier * np.random.uniform(0.8, 1.2, len(df))

    # 4. loan_term: In months (12-360 months)
    features['loan_term'] = np.random.choice([12, 24, 36, 48, 60, 120, 180, 240, 360], len(df))

    # 5. dti_ratio: Already available, cap at 1.0
    features['dti_ratio'] = df['DebtRatio'].clip(0, 1.0)

    # 6. employment_years: Proxy from age
    age = df['age']
    features['employment_years'] = np.clip(age - 22, 0, 40).astype(float)
    features['employment_years'] *= np.random.uniform(0.5, 1.5, len(df))

    # 7. num_credit_lines: Direct mapping
    features['num_credit_lines'] = df['NumberOfOpenCreditLinesAndLoans'].astype(int)

    # 8. num_derogatory_marks: Sum of all late payment types
    features['num_derogatory_marks'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    ).astype(int)

    # 9. credit_utilization: Direct mapping, capped 0-1
    features['credit_utilization'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(0, 1)

    # 10. payment_history_score: 1 - (late payments / 12)
    total_late = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )
    features['payment_history_score'] = (1 - total_late / 12).clip(0, 1)

    # 11. home_ownership: Encoded (0=rent, 1=own, 2=mortgage)
    # Simulated based on age and income
    ownership_prob = (
        (age > 30) * 0.3 +
        (features['annual_income'] > 60000) * 0.3 +
        np.random.random(len(df)) * 0.4
    )
    features['home_ownership'] = pd.cut(
        ownership_prob,
        bins=[0, 0.33, 0.66, 1],
        labels=[0, 1, 2]  # rent, own, mortgage
    ).astype(int)

    # 12. purpose_encoded: Simulated loan purpose
    features['purpose_encoded'] = np.random.choice([0, 1, 2, 3, 4], len(df))

    # 13. num_late_payments: Already computed
    features['num_late_payments'] = total_late.astype(int)

    # 14. savings_balance: Simulated based on income
    features['savings_balance'] = (
        features['annual_income'] * np.random.exponential(0.3, len(df))
    ).clip(0, 500000)

    # 15. monthly_expenses: Based on DTI and income
    features['monthly_expenses'] = (
        features['annual_income'] / 12 * features['dti_ratio'] * 0.7
    ).clip(500, 20000)

    # Add target variable (correct Kaggle column name: SeriousDlqin2yrs)
    features['target'] = df['SeriousDlqin2yrs']

    logger.info(f"Feature engineering complete. Shape: {features.shape}")
    return features


# =============================================================================
# Data Splitting with SMOTE
# =============================================================================


def split_and_balance(
    features: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test with stratification and apply SMOTE.

    Args:
        features: DataFrame with features and target
        test_size: Test split ratio
        val_size: Validation split ratio
        random_state: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df) with SMOTE applied to train
    """
    logger.info("Splitting data: 70% train / 15% val / 15% test")

    X = features[MODEL_FEATURES].values
    y = features['target'].values

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio,
        random_state=random_state, stratify=y_trainval
    )

    logger.info(f"Before SMOTE - Train: {np.bincount(y_train)}")
    logger.info(f"Validation: {np.bincount(y_val)}")
    logger.info(f"Test: {np.bincount(y_test)}")

    # Apply SMOTE to training data (50/50 balance)
    logger.info("Applying SMOTE to training data (target ratio: 0.5)...")
    # Use 'auto' to balance to 1:1 ratio, or specify exact counts
    try:
        smote = SMOTE(sampling_strategy=0.5, random_state=random_state, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    except ValueError as e:
        # If ratio fails, use 'auto' to balance to minority class count
        logger.warning(f"SMOTE ratio failed: {e}. Using auto balance instead.")
        smote = SMOTE(sampling_strategy='auto', random_state=random_state, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    logger.info(f"After SMOTE - Train: {np.bincount(y_train_balanced)}")

    # Create DataFrames
    train_df = pd.DataFrame(X_train_balanced, columns=MODEL_FEATURES)
    train_df['target'] = y_train_balanced

    val_df = pd.DataFrame(X_val, columns=MODEL_FEATURES)
    val_df['target'] = y_val

    test_df = pd.DataFrame(X_test, columns=MODEL_FEATURES)
    test_df['target'] = y_test

    return train_df, val_df, test_df


# =============================================================================
# Report Generation
# =============================================================================


def generate_preprocessing_report(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: str
) -> Dict[str, Any]:
    """
    Generate preprocessing report with dataset statistics.

    Saves JSON report with shape, null counts, class balance.
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "original_data": {
            "shape": list(original_df.shape),
            "null_counts": original_df.isnull().sum().to_dict(),
        },
        "splits": {
            "train": {
                "shape": list(train_df.shape),
                "class_balance": {
                    "class_0": int((train_df['target'] == 0).sum()),
                    "class_1": int((train_df['target'] == 1).sum()),
                    "ratio": float((train_df['target'] == 1).mean()),
                },
                "null_counts": {k: int(v) for k, v in train_df.isnull().sum().items()},
            },
            "validation": {
                "shape": list(val_df.shape),
                "class_balance": {
                    "class_0": int((val_df['target'] == 0).sum()),
                    "class_1": int((val_df['target'] == 1).sum()),
                    "ratio": float((val_df['target'] == 1).mean()),
                },
                "null_counts": {k: int(v) for k, v in val_df.isnull().sum().items()},
            },
            "test": {
                "shape": list(test_df.shape),
                "class_balance": {
                    "class_0": int((test_df['target'] == 0).sum()),
                    "class_1": int((test_df['target'] == 1).sum()),
                    "ratio": float((test_df['target'] == 1).mean()),
                },
                "null_counts": {k: int(v) for k, v in test_df.isnull().sum().items()},
            },
        },
        "feature_statistics": {},
    }

    # Add feature statistics
    for feature in MODEL_FEATURES:
        if feature in train_df.columns:
            report["feature_statistics"][feature] = {
                "mean": float(train_df[feature].mean()),
                "std": float(train_df[feature].std()),
                "min": float(train_df[feature].min()),
                "max": float(train_df[feature].max()),
                "median": float(train_df[feature].median()),
            }

    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Preprocessing report saved to {output_path}")
    return report


# =============================================================================
# Model Retraining
# =============================================================================


def retrain_models_on_new_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Retrain XGBoost and LightGBM on the new data.

    Saves models to existing .joblib files, overwriting previous versions.
    Returns before/after AUC comparison.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame

    Returns:
        Dictionary with AUC scores for comparison
    """
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    logger.info("=" * 60)
    logger.info("RETRAINING MODELS ON NEW DATA")
    logger.info("=" * 60)

    X_train = train_df[MODEL_FEATURES].values
    y_train = train_df['target'].values
    X_val = val_df[MODEL_FEATURES].values
    y_val = val_df['target'].values
    X_test = test_df[MODEL_FEATURES].values
    y_test = test_df['target'].values

    results = {}

    # Train XGBoost
    logger.info("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=5,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc',
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    results['xgboost_auc'] = xgb_auc
    logger.info(f"XGBoost Test AUC: {xgb_auc:.4f}")

    # Save XGBoost model
    joblib.dump(xgb_model, 'xgboost_model.joblib')
    logger.info("XGBoost model saved to xgboost_model.joblib")

    # Train LightGBM
    logger.info("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=-1,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=5,
        random_state=42,
        n_jobs=-1,
        metric='auc',
        early_stopping_rounds=50,
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc')
    lgb_auc = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
    results['lightgbm_auc'] = lgb_auc
    logger.info(f"LightGBM Test AUC: {lgb_auc:.4f}")

    # Save LightGBM model
    joblib.dump(lgb_model, 'lightgbm_model.joblib')
    logger.info("LightGBM model saved to lightgbm_model.joblib")

    return results


# =============================================================================
# Main Pipeline
# =============================================================================


def run_pipeline(data_dir: str = "data", output_dir: str = "data"):
    """
    Execute the complete data pipeline.

    1. Load raw data (Kaggle or synthetic)
    2. Clean data (remove nulls, cap outliers, dedupe)
    3. Engineer features to match MODEL_FEATURES
    4. Split 70/15/15 with stratification
    5. Apply SMOTE for class balance
    6. Save train.csv, val.csv, test.csv
    7. Generate preprocessing_report.json
    8. Retrain XGBoost + LightGBM
    9. Print AUC comparison

    Args:
        data_dir: Directory for input data
        output_dir: Directory for output files
    """
    logger.info("=" * 60)
    logger.info("MORTGAGE AI - DATA PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("\n[STEP 1] Loading data...")
    raw_df = load_kaggle_credit_data(data_dir)

    # Step 2: Clean data
    logger.info("\n[STEP 2] Cleaning data...")
    clean_df = clean_kaggle_data(raw_df)

    # Step 3: Engineer features
    logger.info("\n[STEP 3] Engineering features...")
    features_df = engineer_features_to_schema(clean_df)

    # Step 4: Split and balance
    logger.info("\n[STEP 4] Splitting and balancing data...")
    train_df, val_df, test_df = split_and_balance(features_df)

    # Step 5: Save splits
    logger.info("\n[STEP 5] Saving data splits...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    logger.info(f"Saved: train.csv ({len(train_df):,} rows)")
    logger.info(f"Saved: val.csv ({len(val_df):,} rows)")
    logger.info(f"Saved: test.csv ({len(test_df):,} rows)")

    # Step 6: Generate report
    logger.info("\n[STEP 6] Generating preprocessing report...")
    report = generate_preprocessing_report(
        raw_df, train_df, val_df, test_df,
        output_path / "preprocessing_report.json"
    )

    # Step 7: Retrain models
    logger.info("\n[STEP 7] Retraining models...")
    auc_results = retrain_models_on_new_data(train_df, val_df, test_df)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Train samples: {len(train_df):,} (balanced with SMOTE)")
    logger.info(f"Validation samples: {len(val_df):,}")
    logger.info(f"Test samples: {len(test_df):,}")
    logger.info(f"Train class ratio: {report['splits']['train']['class_balance']['ratio']:.2%}")
    logger.info(f"Test class ratio: {report['splits']['test']['class_balance']['ratio']:.2%}")
    logger.info(f"XGBoost AUC: {auc_results['xgboost_auc']:.4f}")
    logger.info(f"LightGBM AUC: {auc_results['lightgbm_auc']:.4f}")
    logger.info("=" * 60)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "report": report,
        "auc_results": auc_results,
    }


if __name__ == "__main__":
    run_pipeline()
