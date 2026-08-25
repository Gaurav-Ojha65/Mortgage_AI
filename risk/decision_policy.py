"""
Economic & Cost-Sensitive Decision Policy Engine — Mortgage AI
==============================================================
Provides policy versioning, configurable cost models, two-threshold
three-tier decision routing (APPROVE / MANUAL_REVIEW / REJECT),
and out-of-sample optimization on validation splits.

IMPORTANT DISCLAIMER:
Cost parameters (cost_fn, cost_fp, cost_manual_review) are configurable
demonstration/illustrative parameters designed to simulate asymmetric credit
risk economics. They do not represent real-world bank underwriting financials.

Decision Architecture:
- Calibrated Default Probability p in [0.0, 1.0]
- If p <= approve_threshold:
      decision = APPROVE
- If p >= reject_threshold:
      decision = REJECT
- If approve_threshold < p < reject_threshold:
      decision = MANUAL_REVIEW (sent to human underwriters)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


class DecisionState(str, Enum):
    APPROVE = "APPROVE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REJECT = "REJECT"


class RiskTier(str, Enum):
    VERY_LOW = "VERY_LOW"      # p <= 0.02
    LOW = "LOW"                # 0.02 < p <= 0.05
    MODERATE = "MODERATE"      # 0.05 < p <= 0.15
    HIGH = "HIGH"              # 0.15 < p <= 0.35
    SEVERE = "SEVERE"          # p > 0.35


@dataclass
class CostModel:
    """
    Configurable cost matrix parameters (Demonstration/Illustrative values).

    Attributes:
        cost_fn: Cost incurred when a defaulting borrower is approved (False Negative).
                 Default: $10,000 (illustrative Loss Given Default).
        cost_fp: Opportunity loss incurred when a good borrower is rejected (False Positive).
                 Default: $1,000 (illustrative lifetime net interest margin loss).
        cost_manual_review: Cost of human underwriting manual review.
                 Default: $150 (illustrative operational underwriting cost).
        currency: Currency symbol or code. Default: 'USD'.
        is_demonstration: Flag confirming costs are illustrative demo values.
    """
    cost_fn: float = 10000.0
    cost_fp: float = 1000.0
    cost_manual_review: float = 150.0
    currency: str = "USD"
    is_demonstration: bool = True

    def __post_init__(self):
        if self.cost_fn < 0 or self.cost_fp < 0 or self.cost_manual_review < 0:
            raise ValueError("All cost matrix parameters must be non-negative numbers.")


@dataclass
class DecisionPolicy:
    """
    Configurable decision policy with two thresholds and audit logging.

    Attributes:
        policy_name: Human-readable policy identifier.
        policy_version: Version string for compliance tracking.
        approve_threshold: Max default probability for automatic approval (p <= thresh).
        reject_threshold: Min default probability for automatic rejection (p >= thresh).
        cost_model: Cost matrix parameters.
        description: Description of the policy objective.
        created_at: ISO timestamp of policy definition.
    """
    policy_name: str = "optimized_3tier_economic_policy"
    policy_version: str = "v3.1-policy-v1"
    approve_threshold: float = 0.045
    reject_threshold: float = 0.335
    cost_model: CostModel = field(default_factory=CostModel)
    description: str = "3-tier cost-sensitive policy optimized on validation data (Approve <= 0.045, Reject >= 0.335)"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate threshold ordering and range constraints."""
        if not (0.0 <= self.approve_threshold <= 1.0):
            raise ValueError(
                f"approve_threshold ({self.approve_threshold}) must be between 0.0 and 1.0"
            )
        if not (0.0 <= self.reject_threshold <= 1.0):
            raise ValueError(
                f"reject_threshold ({self.reject_threshold}) must be between 0.0 and 1.0"
            )
        if self.approve_threshold > self.reject_threshold:
            raise ValueError(
                f"Threshold ordering violation: approve_threshold ({self.approve_threshold}) "
                f"cannot exceed reject_threshold ({self.reject_threshold})"
            )

    def decide(self, calibrated_prob: float) -> Dict[str, Any]:
        """
        Evaluate an applicant's calibrated default probability against this policy.

        Args:
            calibrated_prob: Calibrated default probability in [0.0, 1.0].

        Returns:
            Dict containing decision, risk_tier, expected_cost, and policy metadata.
        """
        if not (0.0 <= calibrated_prob <= 1.0):
            raise ValueError(f"calibrated_prob must be in [0.0, 1.0], got {calibrated_prob}")

        # 1. Determine decision state and expected economic cost
        if calibrated_prob <= self.approve_threshold:
            decision = DecisionState.APPROVE
            expected_cost = calibrated_prob * self.cost_model.cost_fn
        elif calibrated_prob >= self.reject_threshold:
            decision = DecisionState.REJECT
            expected_cost = (1.0 - calibrated_prob) * self.cost_model.cost_fp
        else:
            decision = DecisionState.MANUAL_REVIEW
            # Direct triage operational cost for human underwriting review
            expected_cost = self.cost_model.cost_manual_review

        # 2. Risk tier classification
        if calibrated_prob <= 0.02:
            risk_tier = RiskTier.VERY_LOW
        elif calibrated_prob <= 0.05:
            risk_tier = RiskTier.LOW
        elif calibrated_prob <= 0.15:
            risk_tier = RiskTier.MODERATE
        elif calibrated_prob <= 0.35:
            risk_tier = RiskTier.HIGH
        else:
            risk_tier = RiskTier.SEVERE

        return {
            "decision": decision.value,
            "risk_tier": risk_tier.value,
            "calibrated_default_probability": round(float(calibrated_prob), 4),
            "expected_economic_cost": round(float(expected_cost), 2),
            "policy_version": self.policy_version,
            "policy_name": self.policy_name,
            "policy_metadata": {
                "approve_threshold": self.approve_threshold,
                "reject_threshold": self.reject_threshold,
                "cost_fn": self.cost_model.cost_fn,
                "cost_fp": self.cost_model.cost_fp,
                "cost_manual_review": self.cost_model.cost_manual_review,
                "is_demonstration": self.cost_model.is_demonstration,
                "description": self.description,
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "approve_threshold": self.approve_threshold,
            "reject_threshold": self.reject_threshold,
            "cost_model": asdict(self.cost_model),
            "description": self.description,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DecisionPolicy:
        """Construct a DecisionPolicy instance from a dictionary."""
        cost_data = data.get("cost_model", {})
        if isinstance(cost_data, dict):
            cost_model = CostModel(**cost_data)
        elif isinstance(cost_data, CostModel):
            cost_model = cost_data
        else:
            cost_model = CostModel()
        return cls(
            policy_name=data.get("policy_name", "custom_policy"),
            policy_version=data.get("policy_version", "v1.0"),
            approve_threshold=float(data["approve_threshold"]),
            reject_threshold=float(data["reject_threshold"]),
            cost_model=cost_model,
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


# ─── Global Active Policy Registry ────────────────────────────────────────────

_ACTIVE_POLICY = DecisionPolicy()
_POLICY_AUDIT_LOG: List[Dict[str, Any]] = [
    {
        "timestamp": datetime.now().isoformat(),
        "action": "INITIALIZE_DEFAULT_POLICY",
        "policy": _ACTIVE_POLICY.to_dict(),
        "user": "system_startup"
    }
]


def get_active_policy() -> DecisionPolicy:
    """Return a copy of the current active frozen decision policy."""
    return _ACTIVE_POLICY


def update_active_policy(new_policy: DecisionPolicy, user: str = "admin") -> DecisionPolicy:
    """
    Update the active decision policy with audit logging.
    Validates threshold constraints before committing.
    """
    global _ACTIVE_POLICY
    new_policy.validate()
    _ACTIVE_POLICY = new_policy
    _POLICY_AUDIT_LOG.append({
        "timestamp": datetime.now().isoformat(),
        "action": "UPDATE_POLICY",
        "policy": new_policy.to_dict(),
        "user": user,
    })
    logger.info(
        f"[Policy Audit] Active policy updated to '{new_policy.policy_name}' "
        f"(v: {new_policy.policy_version}) by '{user}'"
    )
    return _ACTIVE_POLICY


def get_policy_audit_log() -> List[Dict[str, Any]]:
    """Return the audit trail of all policy modifications."""
    return list(_POLICY_AUDIT_LOG)


# ─── Out-of-Sample Policy Optimization (Validation Split Only) ───────────────


def optimize_f1_threshold(y_true_val, p_val) -> Tuple[float, float]:
    """Find single binary threshold maximizing F1 on validation split."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.01, 0.99, 0.005):
        f = f1_score(y_true_val, p_val >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(float(best_t), 3), round(float(best_f1), 4)


def optimize_balanced_accuracy_threshold(y_true_val, p_val) -> Tuple[float, float]:
    """Find single binary threshold maximizing Balanced Accuracy on validation split."""
    from sklearn.metrics import balanced_accuracy_score
    best_t, best_bacc = 0.5, 0.0
    for t in np.arange(0.01, 0.99, 0.005):
        bacc = balanced_accuracy_score(y_true_val, p_val >= t)
        if bacc > best_bacc:
            best_bacc, best_t = bacc, t
    return round(float(best_t), 3), round(float(best_bacc), 4)


def optimize_cost_sensitive_binary_threshold(
    y_true_val,
    p_val,
    cost_fn: float = 10000.0,
    cost_fp: float = 1000.0,
) -> Tuple[float, float]:
    """
    Find single binary threshold minimizing total financial cost on validation split.

    Cost = FN * cost_fn + FP * cost_fp
    """
    from sklearn.metrics import confusion_matrix
    y_true_val = np.asarray(y_true_val)
    best_t, min_cost = 0.5, float("inf")

    for t in np.arange(0.01, 0.99, 0.005):
        pred = (p_val >= t).astype(int)
        cm = confusion_matrix(y_true_val, pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        cost = fn * cost_fn + fp * cost_fp
        if cost < min_cost:
            min_cost, best_t = cost, t

    return round(float(best_t), 3), round(float(min_cost), 2)


def optimize_three_tier_policy(
    y_true_val,
    p_val,
    cost_model: CostModel,
    target_review_rate_max: float = 0.25,
) -> DecisionPolicy:
    """
    Optimize two thresholds (approve_threshold, reject_threshold) on validation split.

    Goal: Minimize total expected cost while bounding manual review volume to a realistic operational limit.
    """
    y_val = np.asarray(y_true_val)
    p_val = np.asarray(p_val)
    n = len(y_val)

    best_approve_t = 0.03
    best_reject_t = 0.15
    min_total_cost = float("inf")

    # Grid search candidate thresholds on validation split
    for t_app in np.arange(0.01, 0.10, 0.005):
        for t_rej in np.arange(t_app + 0.02, 0.35, 0.01):
            is_approve = p_val <= t_app
            is_reject = p_val >= t_rej
            is_review = (~is_approve) & (~is_reject)

            review_rate = np.sum(is_review) / n
            if review_rate > target_review_rate_max:
                continue

            # Financial cost evaluation
            # 1. Approved defaults (False Negatives)
            fn_count = np.sum((y_val == 1) & is_approve)
            cost_approved_defaults = fn_count * cost_model.cost_fn

            # 2. Rejected non-defaults (False Positives)
            fp_count = np.sum((y_val == 0) & is_reject)
            cost_rejected_good = fp_count * cost_model.cost_fp

            # 3. Manual review triage cost
            review_count = np.sum(is_review)
            cost_review_total = review_count * cost_model.cost_manual_review

            total_cost = cost_approved_defaults + cost_rejected_good + cost_review_total

            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_approve_t = t_app
                best_reject_t = t_rej

    return DecisionPolicy(
        policy_name="optimized_3tier_economic_policy",
        policy_version="v3.1-policy-v1",
        approve_threshold=round(float(best_approve_t), 3),
        reject_threshold=round(float(best_reject_t), 3),
        cost_model=cost_model,
        description=f"Val-optimized 3-tier policy: auto-approve <= {best_approve_t:.3f}, auto-reject >= {best_reject_t:.3f}",
    )
