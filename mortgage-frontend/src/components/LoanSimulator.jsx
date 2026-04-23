import React, { useState, useEffect, useCallback } from "react";

const API = process.env.REACT_APP_API_URL || "http://localhost:8001";

const S = {
  container: {
    background: "#0f172a",
    borderRadius: 12,
    padding: "1.5rem",
    border: "1px solid #334155",
    color: "#e2e8f0",
    fontFamily: "'IBM Plex Mono', monospace",
  },
  title: {
    fontSize: "1.1rem",
    fontWeight: 700,
    marginBottom: "1rem",
    color: "#f8fafc",
  },
  sliderContainer: {
    marginBottom: "1.25rem",
  },
  label: {
    display: "flex",
    justifyContent: "space-between",
    marginBottom: "0.5rem",
    fontSize: "0.85rem",
    color: "#94a3b8",
  },
  slider: {
    width: "100%",
    height: 6,
    WebkitAppearance: "none",
    background: "#334155",
    borderRadius: 3,
    outline: "none",
  },
  value: {
    fontWeight: 600,
    color: "#60a5fa",
  },
  result: {
    marginTop: "1.5rem",
    padding: "1rem",
    background: "#1e293b",
    borderRadius: 8,
    border: "1px solid #475569",
  },
  probability: {
    fontSize: "2rem",
    fontWeight: 800,
    textAlign: "center",
    marginBottom: "0.5rem",
  },
  decision: {
    textAlign: "center",
    padding: "0.5rem 1rem",
    borderRadius: 6,
    fontWeight: 700,
    fontSize: "0.9rem",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    marginTop: "0.5rem",
  },
  gauge: {
    width: "100%",
    height: 8,
    background: "#334155",
    borderRadius: 4,
    marginTop: "1rem",
    position: "relative",
    overflow: "hidden",
  },
  gaugeFill: {
    height: "100%",
    borderRadius: 4,
    transition: "width 0.3s ease, background-color 0.3s ease",
  },
  threshold: {
    position: "absolute",
    top: 0,
    bottom: 0,
    width: 2,
    background: "#fbbf24",
    left: "50%",
  },
};

const SLIDERS = [
  { key: "income", label: "Annual Income", min: 20000, max: 200000, step: 1000, format: (v) => `$${v.toLocaleString()}` },
  { key: "loan_amount", label: "Loan Amount", min: 5000, max: 100000, step: 1000, format: (v) => `$${v.toLocaleString()}` },
  { key: "credit_score", label: "Credit Score", min: 300, max: 850, step: 10, format: (v) => v },
  { key: "interest_rate", label: "Interest Rate", min: 2, max: 20, step: 0.5, format: (v) => `${v}%` },
  { key: "loan_term", label: "Loan Term (years)", min: 1, max: 30, step: 1, format: (v) => `${v} years` },
  { key: "existing_loans", label: "Existing Loans", min: 0, max: 5, step: 1, format: (v) => v },
];

export default function LoanSimulator() {
  const [values, setValues] = useState({
    income: 60000,
    loan_amount: 25000,
    credit_score: 650,
    interest_rate: 8.5,
    loan_term: 5,
    existing_loans: 0,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const calculate = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [values]);

  useEffect(() => {
    const timer = setTimeout(calculate, 300);
    return () => clearTimeout(timer);
  }, [calculate]);

  const handleChange = (key, value) => {
    setValues((prev) => ({ ...prev, [key]: Number(value) }));
  };

  const getDecisionColor = (prob) => {
    if (prob >= 0.7) return "#22d3a5";
    if (prob >= 0.5) return "#fbbf24";
    return "#f87171";
  };

  const getDecisionText = (prob) => {
    if (prob >= 0.7) return "Approved";
    if (prob >= 0.5) return "Review";
    return "Declined";
  };

  return (
    <div style={S.container}>
      <h2 style={S.title}>Loan Simulator</h2>

      {SLIDERS.map(({ key, label, min, max, step, format }) => (
        <div key={key} style={S.sliderContainer}>
          <div style={S.label}>
            <span>{label}</span>
            <span style={S.value}>{format(values[key])}</span>
          </div>
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={values[key]}
            onChange={(e) => handleChange(key, e.target.value)}
            style={S.slider}
          />
        </div>
      ))}

      {loading && (
        <div style={{ textAlign: "center", padding: "2rem", color: "#64748b" }}>
          Calculating...
        </div>
      )}

      {error && (
        <div style={{ color: "#f87171", padding: "1rem", textAlign: "center" }}>
          Error: {error}
        </div>
      )}

      {result && !loading && (
        <div style={S.result}>
          <div
            style={{
              ...S.probability,
              color: getDecisionColor(result.approval_probability || result.data?.approval_probability),
            }}
          >
            {((result.approval_probability || result.data?.approval_probability || 0) * 100).toFixed(1)}%
          </div>
          <div style={{ textAlign: "center", color: "#64748b", fontSize: "0.85rem" }}>
            Approval Probability
          </div>

          <div
            style={{
              ...S.decision,
              background: `${getDecisionColor(result.approval_probability || result.data?.approval_probability)}20`,
              border: `1px solid ${getDecisionColor(result.approval_probability || result.data?.approval_probability)}60`,
              color: getDecisionColor(result.approval_probability || result.data?.approval_probability),
            }}
          >
            {getDecisionText(result.approval_probability || result.data?.approval_probability)}
          </div>

          <div style={S.gauge}>
            <div
              style={{
                ...S.gaugeFill,
                width: `${(result.approval_probability || result.data?.approval_probability || 0) * 100}%`,
                background: getDecisionColor(result.approval_probability || result.data?.approval_probability),
              }}
            />
            <div style={S.threshold} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: "0.5rem", fontSize: "0.75rem", color: "#64748b" }}>
            <span>0%</span>
            <span>50% threshold</span>
            <span>100%</span>
          </div>

          {result.emi && (
            <div style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid #334155" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                <span style={{ color: "#94a3b8" }}>Monthly EMI:</span>
                <span style={{ fontWeight: 600 }}>${result.emi?.toFixed(2) || result.data?.emi?.toFixed(2)}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "#94a3b8" }}>Risk Level:</span>
                <span style={{ fontWeight: 600, color: getDecisionColor(result.approval_probability || result.data?.approval_probability) }}>
                  {result.risk_level || result.data?.risk_level}
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
