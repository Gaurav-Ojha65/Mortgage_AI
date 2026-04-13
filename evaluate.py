"""
Professional Model Evaluation Report for Mortgage Approval Model
Generates metrics, visualizations, and model documentation.
Run standalone: python evaluate.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import joblib

from model import generate_synthetic_data
from features import engineer_features, get_feature_names


def load_model_and_data():
    """Load trained model and generate fresh dataset for evaluation."""
    model = joblib.load("best_model.pkl")
    df = generate_synthetic_data()

    feature_cols = [
        "income", "loan_amount", "credit_score", "existing_loans",
        "loan_term", "debt_to_income_ratio", "emi_to_income_ratio",
        "credit_utilization_score"
    ]

    X = df[feature_cols]
    y = df["approved"]

    return model, X, y


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """
    Confusion Matrix Heatmap

    What it shows: How many predictions were correct vs incorrect for each class.
    - Rows = actual classes (Reject/Approve)
    - Columns = predicted classes (Reject/Approve)
    - Diagonal = correct predictions (True Negatives, True Positives)
    - Off-diagonal = errors (False Positives, False Negatives)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Reject", "Approve"],
        yticklabels=["Reject", "Approve"],
        annot_kws={"size": 16}
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix\nMortgage Approval Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_roc_curve(model, X, y, save_path="roc_curve.png"):
    """
    ROC Curve (Receiver Operating Characteristic)

    What it shows: Trade-off between True Positive Rate (sensitivity)
    and False Positive Rate as the classification threshold changes.

    AUC (Area Under Curve) = 1.0 means perfect, 0.5 means random guessing.
    Higher AUC = better model at distinguishing approved vs rejected.
    """
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")
    plt.fill_between(fpr, tpr, alpha=0.3, color="darkorange")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve\nMortgage Approval Model", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")
    return roc_auc


def plot_precision_recall_curve(model, X, y, save_path="precision_recall_curve.png"):
    """
    Precision-Recall Curve

    What it shows: Trade-off between precision (accuracy of positive predictions)
    and recall (coverage of actual positives) at different thresholds.

    Used when classes are imbalanced or when False Positives are costly.
    """
    y_prob = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    avg_precision = average_precision_score(y, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="green", lw=2, label=f"PR Curve (AP = {avg_precision:.3f})")
    plt.fill_between(recall, precision, alpha=0.3, color="green")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12)
    plt.title("Precision-Recall Curve\nMortgage Approval Model", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_feature_importance(model, feature_cols, save_path="feature_importance.png"):
    """
    Feature Importance (Top 10)

    What it shows: Which input features had the most influence on predictions.
    Higher bar = more important for mortgage approval decisions.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print("  [Skipped] Model does not support feature importance")
        return

    indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_cols[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(top_features)), top_importances, color="steelblue", align="center")
    plt.yticks(range(len(top_features)), top_features, fontsize=11)
    plt.xlabel("Importance Score", fontsize=12)
    plt.title("Top 10 Feature Importance\nMortgage Approval Model", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()

    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{top_importances[i]:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


def run_cross_validation(model, X, y, cv=5):
    """
    5-Fold Cross-Validation

    What it shows: How model performance varies across different data splits.
    Splits data into 5 parts, trains on 4, tests on 1, repeats 5 times.
    Mean ± Std gives confidence interval - lower std = more stable model.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    metrics = {
        "accuracy": cross_val_score(model, X, y, cv=skf, scoring="accuracy"),
        "precision": cross_val_score(model, X, y, cv=skf, scoring="precision"),
        "recall": cross_val_score(model, X, y, cv=skf, scoring="recall"),
        "f1": cross_val_score(model, X, y, cv=skf, scoring="f1"),
    }

    print(f"\n{'=' * 60}")
    print("  5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<15} {'Mean':<12} {'Std':<12}")
    print(f"{'-' * 40}")

    cv_results = {}
    for metric_name, scores in metrics.items():
        mean_val = scores.mean()
        std_val = scores.std()
        print(f"{metric_name.capitalize():<15} {mean_val:<12.4f} {std_val:<12.4f}")
        cv_results[metric_name] = {"mean": round(mean_val, 4), "std": round(std_val, 4)}

    return cv_results


def print_classification_report(y_true, y_pred):
    """
    Classification Report

    What each metric means in plain English:
    - Precision: Of all applications we predicted as "Approve", how many actually were?
    - Recall: Of all actual "Approve" applications, how many did we correctly identify?
    - F1-Score: Harmonic mean of Precision and Recall (balance between the two)
    - Support: Number of actual samples in each class
    """
    print(f"\n{'=' * 60}")
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=["Reject", "Approve"]))


def print_model_card(model, X, y, metrics, cv_results, feature_cols):
    """
    Model Card - Summary documentation for the mortgage approval model.
    """
    y_pred = model.predict(X)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)

    model_type = type(model).__name__

    print(f"\n{'=' * 60}")
    print("  MODEL CARD")
    print("=" * 60)
    print(f"""
Model Type:           {model_type}
Training Data Size:   {len(X)} samples
Features ({len(feature_cols)}):   {', '.join(feature_cols)}

--- Performance Metrics ---
Accuracy:             {acc:.4f}
Precision:            {prec:.4f}
Recall:               {rec:.4f}
F1-Score:             {f1:.4f}
ROC-AUC:              {auc:.4f}

--- Cross-Validation (5-Fold) ---
Accuracy:             {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}
Precision:            {cv_results['precision']['mean']:.4f} ± {cv_results['precision']['std']:.4f}
Recall:               {cv_results['recall']['mean']:.4f} ± {cv_results['recall']['std']:.4f}
F1-Score:             {cv_results['f1']['mean']:.4f} ± {cv_results['f1']['std']:.4f}

--- Known Limitations ---
1. Synthetic training data may not fully represent real-world edge cases
2. Model does not account for macroeconomic factors (interest rate changes, unemployment)
3. No feature for co-applicant income or collateral information
4. Credit score range may not cover all demographic variations
5. Threshold for approval is static; real underwriting is dynamic

--- Recommended Use Cases ---
1. Pre-screening tool for initial loan application review
2. Risk stratification for portfolio management
3. Benchmarking against manual underwriting decisions
4. Identifying borderline cases requiring human review

--- NOT Recommended For ---
1. Final approval authority without human review
2. Legal or compliance determination
3. Scenarios with missing or incomplete application data
""")
    print("=" * 60)

    return {
        "model_type": model_type,
        "training_samples": len(X),
        "features": feature_cols,
        "performance": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
        },
        "cross_validation": cv_results,
        "limitations": [
            "Synthetic training data",
            "No macroeconomic factors",
            "No co-applicant consideration",
            "Static thresholds"
        ],
        "use_cases": [
            "Pre-screening",
            "Risk stratification",
            "Decision benchmarking",
            "Borderline case identification"
        ]
    }


def save_metrics_json(metrics, cv_results, model_card, save_path="model_metrics.json"):
    """Save all evaluation metrics to JSON for programmatic access."""
    all_metrics = {
        "test_set_metrics": metrics,
        "cross_validation": cv_results,
        "model_card": model_card
    }

    with open(save_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n  [Saved] {save_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  MORTGAGE APPROVAL MODEL - EVALUATION REPORT")
    print("=" * 60)

    # Load model and data
    print("\nLoading model and data...")
    model, X, y = load_model_and_data()
    print(f"  Loaded {len(X)} samples with {len(X.columns)} features")

    # Split for evaluation (use same split as training for consistency)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_eval, y_eval = X_test, y_test

    # Generate predictions
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    metrics = {
        "accuracy": round(accuracy_score(y_eval, y_pred), 4),
        "precision": round(precision_score(y_eval, y_pred), 4),
        "recall": round(recall_score(y_eval, y_pred), 4),
        "f1_score": round(f1_score(y_eval, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_eval, y_prob), 4),
    }

    # Generate visualizations
    print("\n" + "=" * 60)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 60)

    feature_cols = list(X.columns)
    plot_confusion_matrix(y_eval, y_pred)
    auc = plot_roc_curve(model, X_eval, y_eval)
    plot_precision_recall_curve(model, X_eval, y_eval)
    plot_feature_importance(model, feature_cols)

    # Classification report
    print_classification_report(y_eval, y_pred)

    # Cross-validation
    cv_results = run_cross_validation(model, X, y)

    # Model Card
    model_card = print_model_card(model, X_eval, y_eval, metrics, cv_results, feature_cols)

    # Save metrics to JSON
    save_metrics_json(metrics, cv_results, model_card)

    print("\nEvaluation complete!")