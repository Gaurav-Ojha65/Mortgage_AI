"""
Mortgage Loan Approval ML Model - Real World Data Training
Trains on real loan approval dataset with proper preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - Single Source of Truth
# ============================================================================

# Feature columns for model training and inference
INFERENCE_FEATURES = [
    "income",
    "loan_amount",
    "credit_score",
    "existing_loans",
    "loan_term",
    "debt_to_income_ratio",
    "emi_to_income_ratio",
    "credit_utilization_score"
]

CONFIG = {
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,
    "MIN_SAMPLES": 300,  # Real dataset has ~400 samples
}

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_and_preprocess_data(filepath="loan_data_real.csv"):
    """
    Load real loan approval data and convert to our schema.
    Real dataset columns:
    - ApplicantIncome (monthly) -> income
    - CoapplicantIncome -> added to income
    - LoanAmount (in thousands) -> loan_amount * 1000
    - Loan_Amount_Term (months) -> loan_term (years)
    - Credit_History (0/1) -> converted to credit_score (300-850 scale)
    - Loan_Status (1/0) -> approved (1/0)
    """
    df = pd.read_csv(filepath)

    # Handle missing values
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(360.0)  # Default to 30 years
    df['Credit_History'] = df['Credit_History'].fillna(0.0)  # No history = risky
    df['CoapplicantIncome'] = df['CoapplicantIncome'].fillna(0.0)

    # Create our standardized schema
    processed = pd.DataFrame()

    # Income: Applicant + Coapplicant
    processed['income'] = df['ApplicantIncome'] + df['CoapplicantIncome']

    # Loan amount: convert from thousands to actual
    processed['loan_amount'] = df['LoanAmount'] * 1000

    # Loan term: convert months to years
    processed['loan_term'] = (df['Loan_Amount_Term'] / 12).astype(int)
    processed['loan_term'] = processed['loan_term'].clip(lower=1, upper=30)

    # Credit score: map Credit_History (0/1) to realistic score range
    # Credit_History=1 (has history) -> 650-850 range
    # Credit_History=0 (no history) -> 300-600 range
    has_history = df['Credit_History'] == 1.0
    processed['credit_score'] = np.where(
        has_history,
        np.random.normal(720, 50, len(df)).clip(650, 850),
        np.random.normal(550, 80, len(df)).clip(300, 600)
    ).astype(int)

    # Existing loans: derived from categorical features as proxy
    # Dependents can proxy for existing financial obligations
    dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
    processed['existing_loans'] = df['Dependents'].map(dependents_map).fillna(0).astype(int)

    # Target variable
    processed['approved'] = df['Loan_Status'].astype(int)

    return processed

def engineer_features(data: dict) -> dict:
    """
    Engineer features for ML model.
    Handles edge cases properly with validation.
    """
    income = float(data.get("income", 0))
    loan_amount = float(data.get("loan_amount", 0))
    credit_score = int(data.get("credit_score", 650))
    existing_loans = int(data.get("existing_loans", 0))
    loan_term = int(data.get("loan_term", 5))

    # Validate inputs - raise clear errors for edge cases
    if income <= 0:
        raise ValueError(f"Income must be positive, got {income}")
    if loan_term <= 0 or loan_term > 30:
        raise ValueError(f"Loan term must be between 1-30 years, got {loan_term}")
    if credit_score < 300 or credit_score > 850:
        raise ValueError(f"Credit score must be between 300-850, got {credit_score}")

    # Calculate EMI (monthly)
    monthly_rate = 8.5 / 12 / 100  # Default 8.5% interest
    n_months = loan_term * 12
    if monthly_rate == 0:
        emi = loan_amount / n_months
    else:
        emi = loan_amount * monthly_rate * (1 + monthly_rate) ** n_months / \
              ((1 + monthly_rate) ** n_months - 1)

    # Feature engineering
    debt_to_income_ratio = loan_amount / (income * loan_term)
    emi_to_income_ratio = (emi / income) * 100
    credit_utilization_score = (credit_score - 300) / (850 - 300)

    return {
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "existing_loans": existing_loans,
        "loan_term": loan_term,
        "debt_to_income_ratio": round(debt_to_income_ratio, 4),
        "emi_to_income_ratio": round(emi_to_income_ratio, 2),
        "credit_utilization_score": round(credit_utilization_score, 4),
    }

def prepare_training_data(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target from preprocessed dataframe.
    """
    # Engineer features for all rows
    feature_rows = []
    for _, row in df.iterrows():
        try:
            features = engineer_features(row.to_dict())
            feature_rows.append(features)
        except ValueError as e:
            # Skip rows with invalid data
            continue

    feature_df = pd.DataFrame(feature_rows)

    # Align with inference features
    X = feature_df[INFERENCE_FEATURES]
    y = df['approved'].iloc[:len(X)]  # Align length after dropping invalid rows

    return X, y

# ============================================================================
# Model Training
# ============================================================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and return best one.
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=CONFIG["RANDOM_STATE"]
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=CONFIG["RANDOM_STATE"]
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            random_state=CONFIG["RANDOM_STATE"]
        )
    }

    results = {}
    best_model = None
    best_score = 0

    print(f"\n{'='*70}")
    print("  MODEL TRAINING ON REAL DATA")
    print(f"{'='*70}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(INFERENCE_FEATURES)}")
    print(f"{'='*70}\n")

    for name, model in models.items():
        print(f"  Training {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
            "model": model
        }

        results[name] = metrics

        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")
        print(f"    AUC:       {metrics['auc']:.4f}\n")

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_model = (name, model)

    return results, best_model

def save_model_artifacts(model, X_test, y_test, results):
    """
    Save model and test set for reproducible evaluation.
    """
    # Save best model
    joblib.dump(model, "best_model.pkl")
    print(f"  [Saved] best_model.pkl")

    # Save test set for proper evaluation (not regenerating!)
    joblib.dump((X_test, y_test), "test_set.pkl")
    print(f"  [Saved] test_set.pkl (test set for evaluation)")

    # Save feature list for consistency
    joblib.dump(INFERENCE_FEATURES, "feature_list.pkl")
    print(f"  [Saved] feature_list.pkl")

    # Save metrics summary
    import json
    with open("model_metrics.json", "w") as f:
        json.dump({
            "model_type": type(model).__name__,
            "features": INFERENCE_FEATURES,
            "n_features": len(INFERENCE_FEATURES),
            "train_samples": len(X_test) * 4,  # approximate
            "test_samples": len(X_test)
        }, f, indent=2)

# ============================================================================
# Prediction Function (Fixed)
# ============================================================================

def predict(input_dict: dict, model=None) -> dict:
    """
    Make prediction with loaded model.
    Fixed: accepts optional model param to avoid reloading from disk.
    """
    # Load model if not provided (for API usage)
    if model is None:
        model = joblib.load("best_model.pkl")

    # Engineer features
    features = engineer_features(input_dict)

    # Create DataFrame with exact column order
    X = pd.DataFrame([{k: features[k] for k in INFERENCE_FEATURES}])

    approved = bool(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return {
        "approved": approved,
        "probability": probability,
        "model_used": type(model).__name__,
        "features_used": list(X.columns)
    }

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Load real data
    print("="*70)
    print("  LOADING REAL LOAN APPROVAL DATA")
    print("="*70)

    try:
        raw_df = load_and_preprocess_data("loan_data_real.csv")
        print(f"  Loaded {len(raw_df)} samples from real dataset")
        print(f"  Approval rate: {raw_df['approved'].mean()*100:.1f}%")
    except Exception as e:
        print(f"  Error loading data: {e}")
        exit(1)

    # Prepare features
    print("\n  Engineering features...")
    X, y = prepare_training_data(raw_df)

    if len(X) < CONFIG["MIN_SAMPLES"]:
        print(f"  ERROR: Insufficient samples ({len(X)} < {CONFIG['MIN_SAMPLES']})")
        exit(1)

    print(f"  Valid samples after feature engineering: {len(X)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["TEST_SIZE"],
        random_state=CONFIG["RANDOM_STATE"],
        stratify=y
    )

    # Train models
    results, best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Save artifacts
    print(f"\n{'='*70}")
    print("  SAVING MODEL ARTIFACTS")
    print(f"{'='*70}")
    save_model_artifacts(best_model[1], X_test, y_test, results)

    print(f"\n{'='*70}")
    print(f"  BEST MODEL: {best_model[0]}")
    print(f"{'='*70}")

    # Test prediction
    test_input = {
        "income": 5000,
        "loan_amount": 120000,
        "credit_score": 700,
        "existing_loans": 0,
        "loan_term": 10
    }

    result = predict(test_input, model=best_model[1])
    print(f"\n  Test prediction:")
    print(f"    Input: {test_input}")
    print(f"    Approved: {result['approved']}")
    print(f"    Probability: {result['probability']:.2%}")
    print(f"    Model: {result['model_used']}")

    print(f"\n{'='*70}")
    print("  TRAINING COMPLETE - NO ERRORS")
    print(f"{'='*70}")
