"""
Mortgage Loan Approval ML Model Comparison
Compares Logistic Regression, Random Forest, and XGBoost for loan approval prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier
from features import engineer_features, get_feature_names
from emi import calculate_emi
import joblib


# Configuration thresholds for loan approval rules
CONFIG = {
    "CREDIT_SCORE_MIN": 650,
    "EMI_TO_INCOME_MAX": 40.0,
    "EXISTING_LOANS_MAX": 3,
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,
    "N_SAMPLES": 500,
}


def generate_synthetic_data(n_samples: int = CONFIG["N_SAMPLES"]) -> pd.DataFrame:
    """
    Generate realistic synthetic loan application data.

    Approval rules (with noise added to avoid perfect separation):
    - credit_score > 650
    - emi_to_income_ratio < 40
    - existing_loans < 3
    """
    np.random.seed(CONFIG["RANDOM_STATE"])

    data = []
    for _ in range(n_samples):
        income = np.random.lognormal(10.5, 0.5)
        credit_score = np.random.normal(700, 100)
        credit_score = np.clip(credit_score, 300, 850)

        loan_term_months = np.random.choice([12, 24, 36, 48, 60, 72, 84, 120])
        loan_amount = np.random.lognormal(11.5, 0.8)
        interest_rate = np.random.normal(8.5, 2.0)
        interest_rate = max(5.0, min(15.0, interest_rate))

        existing_loans = np.random.poisson(1.5)
        existing_loans = max(0, min(6, existing_loans))

        emi = calculate_emi(loan_amount, interest_rate, loan_term_months / 12)
        emi_to_income_ratio = (emi / income) * 100

        # Base approval rule
        approved = (
            credit_score > CONFIG["CREDIT_SCORE_MIN"]
            and emi_to_income_ratio < CONFIG["EMI_TO_INCOME_MAX"]
            and existing_loans < CONFIG["EXISTING_LOANS_MAX"]
        )

        # Add realistic noise (misclassification rate ~8-12%)
        noise = np.random.random()
        if approved and noise > 0.88:
            approved = False
        elif not approved and noise > 0.90:
            approved = True

        data.append({
            "income": income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "existing_loans": existing_loans,
            "loan_term": loan_term_months,
            "emi": emi,
            "approved": 1 if approved else 0,
        })

    df = pd.DataFrame(data)

    # Engineer features
    enriched_data = []
    for _, row in df.iterrows():
        enriched = engineer_features(row.to_dict())
        enriched_data.append(enriched)

    return pd.DataFrame(enriched_data)


def train_and_evaluate_models(df: pd.DataFrame):
    """
    Train all three models and return evaluation results.
    """
    feature_cols = [
        "income", "loan_amount", "credit_score", "existing_loans",
        "loan_term", "debt_to_income_ratio", "emi_to_income_ratio",
        "credit_utilization_score"
    ]

    X = df[feature_cols]
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["TEST_SIZE"], random_state=CONFIG["RANDOM_STATE"]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=CONFIG["RANDOM_STATE"]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=CONFIG["RANDOM_STATE"]),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            random_state=CONFIG["RANDOM_STATE"]
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"  {name.upper()}")
        print(f"{'=' * 60}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\nAccuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")

        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Reject", "Approve"]))

        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:")
        print(f"                Predicted")
        print(f"              Reject  Approve")
        print(f"Actual Reject   {cm[0][0]:4d}    {cm[0][1]:4d}")
        print(f"Actual Approve {cm[1][0]:4d}    {cm[1][1]:4d}")

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "model": model,
        }

    return results, X_test, y_test


def print_comparison_table(results: dict):
    """
    Print a clean comparison table of all models.
    """
    print(f"\n{'=' * 80}")
    print(f"  MODEL COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print(f"{'-' * 80}")

    best_model_name = None
    best_f1 = -1

    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['auc']:<12.4f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name

    print(f"{'-' * 80}")

    # Determine winner based on F1 score
    print(f"\n*** WINNER: {best_model_name} (Best F1-Score: {best_f1:.4f}) ***")

    return best_model_name


def save_best_model(results: dict, best_model_name: str):
    """
    Save the best performing model to disk.
    """
    best_model = results[best_model_name]["model"]
    joblib.dump(best_model, "best_model.pkl")
    print(f"\nBest model saved as 'best_model.pkl' ({best_model_name})")


def predict(input_dict: dict) -> dict:
    """
    Load best_model.pkl and make a prediction.

    Args:
        input_dict: Dictionary with loan application data

    Returns:
        {"approved": True/False, "probability": float, "model_used": str}
    """
    model = joblib.load("best_model.pkl")

    feature_cols = [
        "income", "loan_amount", "credit_score", "existing_loans",
        "loan_term", "debt_to_income_ratio", "emi_to_income_ratio",
        "credit_utilization_score"
    ]

    enriched = engineer_features(input_dict)
    X = pd.DataFrame([enriched])[feature_cols]

    approved = bool(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return {
        "approved": approved,
        "probability": probability,
        "model_used": "XGBoost" if "XGB" in type(model).__name__ else type(model).__name__
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  GENERATING SYNTHETIC DATASET")
    print("=" * 60)
    print(f"  {CONFIG['N_SAMPLES']} samples with realistic distributions")

    df = generate_synthetic_data()

    print(f"\nDataset shape: {df.shape}")
    print(f"Approved: {df['approved'].sum()} ({df['approved'].mean()*100:.1f}%)")
    print(f"Rejected: {(df['approved'] == 0).sum()} ({(1-df['approved'].mean())*100:.1f}%)")

    print("\nSample data:")
    print(df[get_feature_names() + ["approved"]].head())

    results, X_test, y_test = train_and_evaluate_models(df)

    best_model_name = print_comparison_table(results)

    save_best_model(results, best_model_name)

    print("\n" + "=" * 60)
    print("  PREDICTION TEST")
    print("=" * 60)

    test_application = {
        "income": 50000,
        "loan_amount": 200000,
        "credit_score": 650,
        "existing_loans": 2,
        "loan_term": 60,
        "emi": 4103.31,
    }

    result = predict(test_application)
    print(f"\nTest Application: {test_application}")
    print(f"Prediction: {result}")