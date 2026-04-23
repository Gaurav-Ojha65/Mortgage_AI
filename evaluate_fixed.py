"""
Model Evaluation - Fixed version using saved test set.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib


def load_model_and_test_set():
    """Load trained model and saved test set (not regenerated data!)."""
    model = joblib.load("best_model.pkl")
    X_test, y_test = joblib.load("test_set.pkl")
    feature_list = joblib.load("feature_list.pkl")

    print(f"  Loaded model: {type(model).__name__}")
    print(f"  Test set size: {len(X_test)}")
    print(f"  Features: {len(feature_list)}")

    return model, X_test, y_test, feature_list


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Reject", "Approve"],
        yticklabels=["Reject", "Approve"],
        annot_kws={"size": 16}
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix - Real Data", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_roc_curve(model, X, y, save_path="roc_curve.png"):
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")
    plt.fill_between(fpr, tpr, alpha=0.3, color="darkorange")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Real Data Model", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")
    return roc_auc


def plot_precision_recall_curve(model, X, y, save_path="precision_recall_curve.png"):
    y_prob = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_prob)
    avg_precision = average_precision_score(y, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="green", lw=2, label=f"PR Curve (AP = {avg_precision:.3f})")
    plt.fill_between(recall, precision, alpha=0.3, color="green")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12)
    plt.title("Precision-Recall Curve - Real Data", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_feature_importance(model, feature_cols, save_path="feature_importance.png"):
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
    plt.title("Top 10 Feature Importance - Real Data Model", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()

    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{top_importances[i]:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


def print_classification_report(y_true, y_pred):
    print(f"\n{'='*60}")
    print("  CLASSIFICATION REPORT - REAL DATA")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=["Reject", "Approve"]))


def save_metrics_json(metrics, save_path="model_metrics.json"):
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [Saved] {save_path}")


if __name__ == "__main__":
    print("="*70)
    print("  MODEL EVALUATION - REAL DATA (FIXED)")
    print("="*70)

    # Load model and saved test set (NOT regenerated data!)
    print("\nLoading model and saved test set...")
    model, X_test, y_test, features = load_model_and_test_set()

    # Generate predictions on held-out test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "n_test_samples": len(y_test),
        "features": features
    }

    # Print results
    print(f"\n{'='*60}")
    print("  TEST SET METRICS (REAL DATA)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Generate visualizations
    print(f"\n{'='*60}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(model, X_test, y_test)
    plot_precision_recall_curve(model, X_test, y_test)
    plot_feature_importance(model, features)

    # Classification report
    print_classification_report(y_test, y_pred)

    # Save metrics
    save_metrics_json(metrics)

    print("\n" + "="*70)
    print("  EVALUATION COMPLETE - NO ERRORS")
    print("="*70)
