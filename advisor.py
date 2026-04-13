# pip install ollama  (ensure Ollama is running: ollama serve)
"""
Production-grade Mortgage Advisor
Orchestrates: features → EMI → ML prediction → risk → Monte Carlo → Ollama reasoning.
"""

import json
import time
import logging
import re
from datetime import datetime

import ollama
from ollama import chat

from features import engineer_features
from emi import calculate_emi
from model import predict as ml_predict
from risk import calculate_risk
from monte_carlo import simulate as mc_simulate


# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("advisor")


# =============================================================================
# Input Validation
# =============================================================================

REQUIRED_FIELDS = ["income", "loan_amount", "interest_rate", "loan_term", "credit_score"]
OPTIONAL_FIELDS = ["existing_loans"]

LOAN_TERM_MONTHS_MAX = 360
CREDIT_SCORE_MIN, CREDIT_SCORE_MAX = 300, 850
INCOME_MIN, INCOME_MAX = 1000, 10_000_000
LOAN_AMOUNT_MIN, LOAN_AMOUNT_MAX = 1000, 100_000_000
RATE_MIN, RATE_MAX = 0.1, 30.0


def validate_input(loan_input: dict) -> dict:
    """Validate and normalise loan application input. Raises ValueError on bad data."""
    if not isinstance(loan_input, dict):
        raise ValueError("loan_input must be a dictionary")

    missing = [f for f in REQUIRED_FIELDS if f not in loan_input or loan_input[f] is None]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    data = {**loan_input}

    # loan_term: accept months or years, normalise to years
    term = data.get("loan_term", 0)
    if isinstance(term, (int, float)):
        if term > LOAN_TERM_MONTHS_MAX:
            data["loan_term"] = int(term / 12)   # treat as months → convert to years
        else:
            data["loan_term"] = int(term)         # treat as years (keep as-is)
    else:
        raise ValueError("loan_term must be a number")

    if not (INCOME_MIN <= data["income"] <= INCOME_MAX):
        raise ValueError(f"income must be between {INCOME_MIN:,} and {INCOME_MAX:,}")
    if not (LOAN_AMOUNT_MIN <= data["loan_amount"] <= LOAN_AMOUNT_MAX):
        raise ValueError(f"loan_amount must be between {LOAN_AMOUNT_MIN:,} and {LOAN_AMOUNT_MAX:,}")
    if not (RATE_MIN <= data["interest_rate"] <= RATE_MAX):
        raise ValueError(f"interest_rate must be between {RATE_MIN} and {RATE_MAX}%")
    if not (1 <= data["loan_term"] <= 30):
        raise ValueError("loan_term must be between 1 and 30 years")
    if not (CREDIT_SCORE_MIN <= data["credit_score"] <= CREDIT_SCORE_MAX):
        raise ValueError(f"credit_score must be between {CREDIT_SCORE_MIN} and {CREDIT_SCORE_MAX}")
    data["existing_loans"] = int(data.get("existing_loans", 0))
    if data["existing_loans"] < 0 or data["existing_loans"] > 20:
        raise ValueError("existing_loans must be between 0 and 20")

    return data


# =============================================================================
# Ollama Reasoning
# =============================================================================

OLLAMA_MODEL = "qwen3-coder:30b"


def build_prompt(data: dict, emi: float, risk_level: str, default_prob: float,
                  approval_prob: float, feature_values: dict, mc_summary: dict) -> str:
    """Build the full prompt sent to Ollama."""
    annual_income = data["income"] * 12
    total_interest = emi * data["loan_term"] * 12 - data["loan_amount"]
    dti = data["loan_amount"] / (annual_income + 1e-9)
    eti = (emi / data["income"]) * 100

    return f"""You are a senior mortgage underwriter. Analyse this application and respond ONLY with a valid JSON object — no markdown, no explanation, no extra text.

Application:
- Loan Amount: Rs.{data['loan_amount']:,.0f}
- Interest Rate: {data['interest_rate']}% p.a.
- Loan Term: {data['loan_term']} years ({data['loan_term']*12} months)
- Monthly EMI: Rs.{emi:,.2f}
- Total Interest Payable: Rs.{total_interest:,.0f}
- Monthly Income: Rs.{data['income']:,.2f}
- Annual Income: Rs.{annual_income:,.0f}
- Credit Score: {data['credit_score']}
- Existing Loans: {data['existing_loans']}

Derived Metrics:
- Loan-to-Income Ratio: {dti:.2f}x
- EMI-to-Income Ratio: {eti:.1f}%
- Debt-to-Income Ratio: {feature_values.get('debt_to_income_ratio', 0):.4f}
- Credit Utilisation Score: {feature_values.get('credit_utilization_score', 0):.4f}
- Loan Burden Index: {feature_values.get('loan_burden_index', 0):.4f}
- Affordability Score: {feature_values.get('affordability_score', 0):.4f}

ML Model:
- Approval Probability: {approval_prob:.2%}
- Model Used: {feature_values.get('model_used', 'N/A')}

Risk Assessment:
- Risk Level: {risk_level}
- Monte Carlo Default Probability ({mc_summary.get('n_simulations', 1000)} runs): {default_prob:.2%}
- Worst-case EMI (95th pct): Rs.{mc_summary.get('worst_case_emi', 0):,.2f}
- Safe Income Threshold: Rs.{mc_summary.get('safe_income_threshold', 0):,.2f}
- Scenarios — Stable: {mc_summary.get('scenario_breakdown', {}).get('stable', 0)}, Stressed: {mc_summary.get('scenario_breakdown', {}).get('stressed', 0)}, Crisis: {mc_summary.get('scenario_breakdown', {}).get('crisis', 0)}

Decision rules you must apply:
1. APPROVE if: risk_level == LOW AND default_prob < 0.15 AND approval_prob > 0.70
2. REJECT if: risk_level == HIGH OR default_prob > 0.35 OR approval_prob < 0.30
3. CONDITIONAL otherwise (treat as REJECT for this exercise)
4. EMI-to-income ratio > 40% is a hard reject
5. Credit score < 600 is a hard reject

Respond with this exact JSON structure (no keys outside this schema):
{{
    "decision": "APPROVED" or "REJECTED",
    "confidence": 0.0 to 1.0,
    "rating": 1 to 10,
    "reasons": ["reason 1", "reason 2", "reason 3"],
    "improvement_tips": ["tip 1", "tip 2", "tip 3"]
}}"""


def parse_ollama_response(raw: str) -> dict:
    """Parse JSON from Ollama response, handling markdown code fences."""
    text = raw.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    parsed = json.loads(text)
    # Validate and default missing keys
    parsed.setdefault("decision", "REJECTED")
    parsed.setdefault("confidence", 0.5)
    parsed.setdefault("rating", 5)
    parsed.setdefault("reasons", ["Automated rule-based decision"])
    parsed.setdefault("improvement_tips", ["Improve credit score", "Reduce existing debt", "Increase income"])
    return parsed


def get_ollama_advice(data: dict, emi: float, risk_level: str,
                      default_prob: float, approval_prob: float,
                      feature_values: dict, mc_summary: dict) -> dict:
    """Call Ollama and return structured advice dict."""
    prompt = build_prompt(data, emi, risk_level, default_prob, approval_prob, feature_values, mc_summary)

    try:
        log.info("Calling Ollama (%s) …", OLLAMA_MODEL)
        response = chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.3, "num_predict": 512}
        )
        raw = response.message.content
        log.info("Ollama response received (%d chars)", len(raw))
        parsed = parse_ollama_response(raw)
        return {"advice": parsed, "ai_available": True}

    except Exception as e:
        log.warning("Ollama call failed — using rule-based fallback: %s", e)
        return {
            "advice": {
                "decision": "REJECTED" if risk_level == "HIGH" or default_prob > 0.35 else "APPROVED",
                "confidence": round(approval_prob, 4),
                "rating": int(min(max(round(approval_prob * 10), 1), 10)),
                "reasons": _rule_based_reasons(data, emi, risk_level, default_prob, approval_prob),
                "improvement_tips": _rule_based_tips(data, emi, risk_level, default_prob, approval_prob),
            },
            "ai_available": False,
            "error": str(e)
        }


def _rule_based_reasons(data, emi, risk_level, default_prob, approval_prob) -> list:
    reasons = []
    eti = (emi / data["income"]) * 100
    if eti > 40:
        reasons.append(f"EMI-to-income ratio ({eti:.1f}%) exceeds 40% safety threshold")
    if risk_level == "HIGH":
        reasons.append(f"Risk assessment is HIGH — borrower profile is high-risk")
    if default_prob > 0.25:
        reasons.append(f"Monte Carlo default probability ({default_prob:.1%}) indicates elevated default risk")
    if data["credit_score"] < 650:
        reasons.append(f"Credit score ({data['credit_score']}) is below 650 — marginal borrower")
    if len(reasons) < 3:
        reasons.append(f"Approval probability from ML model is {approval_prob:.1%}")
    return reasons[:3]


def _rule_based_tips(data, emi, risk_level, default_prob, approval_prob) -> list:
    tips = []
    if data["credit_score"] < 750:
        tips.append(f"Improve credit score above 750 to qualify for lower interest rates")
    if (emi / data["income"]) * 100 > 30:
        tips.append("Reduce loan amount or increase down payment to lower EMI burden")
    if data["existing_loans"] >= 2:
        tips.append("Pay off or consolidate existing loans before applying")
    if default_prob > 0.20:
        tips.append("Consider a shorter loan term to reduce default exposure")
    return tips[:3] or ["Focus on improving credit score", "Reduce total debt", "Increase income"]


# =============================================================================
# Main Orchestrator
# =============================================================================

def analyze(loan_input: dict) -> dict:
    """
    Full mortgage decision pipeline.
    Returns structured JSON with decision, metrics, reasons, and tips.
    """
    t0 = time.perf_counter()
    log.info("=== Mortgage Advisor: Starting analysis ===")
    log.info("Input: %s", {k: v for k, v in loan_input.items() if k != "existing_loans"})

    # ── Step 1: Validation ────────────────────────────────────────────────────
    step = "validation"
    log.info("[%s] Validating input …", step)
    t_val = time.perf_counter()
    data = validate_input(loan_input)
    log.info("[%s] Done (%.1fms)", step, (time.perf_counter() - t_val) * 1000)

    # ── Step 2: EMI Calculation (must come before features) ────────────────────
    step = "emi"
    log.info("[%s] Calculating EMI …", step)
    t_emi = time.perf_counter()
    emi = calculate_emi(data["loan_amount"], data["interest_rate"], data["loan_term"])
    data["emi"] = emi
    log.info("[%s] EMI = Rs.%.2f (%.1fms)", step, emi, (time.perf_counter() - t_emi) * 1000)

    # ── Step 3: Feature Engineering (emi must be set first) ─────────────────────
    step = "features"
    log.info("[%s] Engineering features …", step)
    t_feat = time.perf_counter()
    enriched = engineer_features(data)
    feature_cols = [
        "income", "loan_amount", "credit_score", "existing_loans",
        "loan_term", "debt_to_income_ratio", "emi_to_income_ratio",
        "credit_utilization_score"
    ]
    log.info("[%s] Done (%.1fms)", step, (time.perf_counter() - t_feat) * 1000)

    # ── Step 4: ML Prediction ──────────────────────────────────────────────────
    step = "ml"
    log.info("[%s] Running ML prediction …", step)
    t_ml = time.perf_counter()
    ml_result = ml_predict(data)
    approval_prob = ml_result["probability"]
    model_used = ml_result["model_used"]
    log.info("[%s] approved=%s prob=%.2f model=%s (%.1fms)",
             step, ml_result["approved"], approval_prob, model_used,
             (time.perf_counter() - t_ml) * 1000)

    # ── Step 5: Risk Assessment ───────────────────────────────────────────────
    step = "risk"
    log.info("[%s] Assessing risk …", step)
    t_risk = time.perf_counter()
    risk_level = calculate_risk(
        data["income"], data["loan_amount"],
        data["credit_score"], data["existing_loans"]
    )
    log.info("[%s] risk_level=%s (%.1fms)",
             step, risk_level, (time.perf_counter() - t_risk) * 1000)

    # ── Step 6: Monte Carlo Simulation ────────────────────────────────────────
    step = "monte_carlo"
    log.info("[%s] Running Monte Carlo (1000 sims) …", step)
    t_mc = time.perf_counter()
    mc_result = mc_simulate({
        "income": data["income"],
        "loan_amount": data["loan_amount"],
        "interest_rate": data["interest_rate"],
        "loan_term": data["loan_term"],
        "credit_score": data["credit_score"]
    }, n_simulations=1000)
    default_prob = mc_result["default_probability"]
    mc_summary = {
        "default_probability": default_prob,
        "risk_label": mc_result.get("risk_label", risk_level),
        "worst_case_emi": mc_result.get("worst_case_emi", emi),
        "safe_income_threshold": mc_result.get("safe_income_threshold", data["income"]),
        "scenario_breakdown": mc_result.get("scenario_breakdown", {}),
        "mean_emi_ratio": mc_result.get("mean_emi_ratio", 0),
        "n_simulations": 1000,
    }
    log.info("[%s] default_prob=%.2f scenarios=%s (%.1fms)",
             step, default_prob, mc_summary["scenario_breakdown"],
             (time.perf_counter() - t_mc) * 1000)

    # ── Step 7: Ollama Reasoning ────────────────────────────────────────────────
    step = "ollama"
    log.info("[%s] Generating AI advice …", step)
    t_ai = time.perf_counter()
    advice_result = get_ollama_advice(
        data, emi, risk_level, default_prob, approval_prob,
        {**enriched, "model_used": model_used}, mc_summary
    )
    ai_elapsed = (time.perf_counter() - t_ai) * 1000
    ai_status = "✓" if advice_result["ai_available"] else "✗ fallback"
    log.info("[%s] Done (%s, %.1fms)", step, ai_status, ai_elapsed)

    advice_data = advice_result["advice"]

    # ── Step 8: Build Final Response ──────────────────────────────────────────
    total_ms = int((time.perf_counter() - t0) * 1000)
    log.info("=== Analysis complete in %dms ===", total_ms)

    return {
        "decision": advice_data["decision"],
        "confidence": float(advice_data.get("confidence", approval_prob)),
        "rating": int(advice_data.get("rating", 5)),
        "emi": round(emi, 2),
        "risk_level": risk_level,
        "default_probability": round(default_prob, 4),
        "approval_probability": round(approval_prob, 4),
        "feature_values": {
            "debt_to_income_ratio": enriched.get("debt_to_income_ratio"),
            "emi_to_income_ratio": enriched.get("emi_to_income_ratio"),
            "credit_utilization_score": enriched.get("credit_utilization_score"),
            "loan_burden_index": enriched.get("loan_burden_index"),
            "affordability_score": enriched.get("affordability_score"),
        },
        "monte_carlo_summary": mc_summary,
        "reasons": advice_data.get("reasons", [])[:3],
        "improvement_tips": advice_data.get("improvement_tips", [])[:3],
        "model_used": model_used,
        "processing_time_ms": total_ms,
        "ai_available": advice_result["ai_available"],
        "_log": {
            "timestamp": datetime.now().isoformat(),
            "total_ms": total_ms,
            "steps": {
                "validation_ms": int((time.perf_counter() - t0) * 1000),
            }
        }
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    test_case = {
        "income": 75000,
        "loan_amount": 300000,
        "interest_rate": 8.75,
        "loan_term": 5,         # years (not months — auto-detect uses <=360 as years)
        "credit_score": 720,
        "existing_loans": 1,
    }

    print("\n" + "=" * 70)
    print("  MORTGAGE ADVISOR — TEST RUN")
    print("=" * 70)
    print(f"\nInput: {json.dumps(test_case, indent=2)}\n")

    result = analyze(test_case)

    print("=" * 70)
    print("  RESULT")
    print("=" * 70)
    print(f"\n  Decision:              {result['decision']}")
    print(f"  Confidence:            {result['confidence']:.2%}")
    print(f"  Rating:                {result['rating']}/10")
    print(f"  EMI:                   Rs.{result['emi']:,.2f}")
    print(f"  Risk Level:            {result['risk_level']}")
    print(f"  Default Probability:   {result['default_probability']:.2%}")
    print(f"  Approval Probability:  {result['approval_probability']:.2%}")
    print(f"  Model Used:            {result['model_used']}")
    print(f"  Processing Time:       {result['processing_time_ms']}ms")
    print(f"  AI Available:          {result['ai_available']}")

    print(f"\n  Feature Values:")
    for k, v in result["feature_values"].items():
        print(f"    {k}: {v}")

    print(f"\n  Monte Carlo Summary:")
    print(f"    Default Probability:  {result['monte_carlo_summary']['default_probability']:.2%}")
    print(f"    Risk Label:           {result['monte_carlo_summary']['risk_label']}")
    print(f"    Worst-case EMI:       Rs.{result['monte_carlo_summary']['worst_case_emi']:,.2f}")
    print(f"    Safe Income Threshold: Rs.{result['monte_carlo_summary']['safe_income_threshold']:,.2f}")
    print(f"    Scenarios:            {result['monte_carlo_summary']['scenario_breakdown']}")

    print(f"\n  Reasons:")
    for i, r in enumerate(result["reasons"], 1):
        print(f"    {i}. {r}")

    print(f"\n  Improvement Tips:")
    for i, t in enumerate(result["improvement_tips"], 1):
        print(f"    {i}. {t}")

    print("\n" + "=" * 70)