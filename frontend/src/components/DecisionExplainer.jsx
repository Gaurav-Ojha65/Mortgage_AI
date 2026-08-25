import React, { useState, useEffect, useCallback } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8001";

const pct    = (n) => `${(+n * 100).toFixed(1)}%`;
const signed = (n) => (n >= 0 ? "+" : "") + (+n * 100).toFixed(2) + "%";
const clamp  = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function WaterfallChart({ baseValue, factors, finalProb }) {
  const W = 640, ROW = 44, PAD = 16, LABEL = 200, BAR_AREA = W - LABEL - PAD * 2;
  const top5 = factors.slice(0, 7);
  const H = (top5.length + 2) * ROW + PAD * 3;

  let running = baseValue;
  const rows = top5.map((f) => {
    const start = running;
    running = clamp(running + f.shap_value, 0, 1);
    return { ...f, start, end: running };
  });

  const toX = (p) => LABEL + PAD + clamp(p, 0, 1) * BAR_AREA;
  const GREEN  = "#22d3a5";
  const RED    = "#f87171";
  const GRAY   = "#334155";

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${W} ${H}`}
      style={{ fontFamily: "IBM Plex Mono, monospace", overflow: "visible" }}
      aria-label="SHAP waterfall chart"
    >
      <text x={LABEL - 6} y={PAD + ROW * 0.6}
        textAnchor="end" fontSize={11} fill="#64748b">Base rate</text>
      <rect x={LABEL + PAD} y={PAD + 8} width={clamp(baseValue, 0, 1) * BAR_AREA}
        height={ROW - 18} rx={3} fill={GRAY} />
      <text x={toX(baseValue) + 6} y={PAD + ROW * 0.6}
        fontSize={11} fill="#94a3b8">{pct(baseValue)}</text>

      {rows.map((f, i) => {
        const y     = PAD + ROW * (i + 1);
        const isPos = f.shap_value >= 0;
        const x1    = toX(Math.min(f.start, f.end));
        const x2    = toX(Math.max(f.start, f.end));
        const barW  = Math.max(x2 - x1, 2);
        const color = isPos ? GREEN : RED;

        return (
          <g key={f.feature}>
            <text x={LABEL - 6} y={y + ROW * 0.55}
              textAnchor="end" fontSize={11} fill="#cbd5e1">
              {f.label.length > 22 ? f.label.slice(0, 21) + "..." : f.label}
            </text>
            <rect x={LABEL + PAD} y={y + 8}
              width={clamp(f.end, 0, 1) * BAR_AREA} height={ROW - 18}
              rx={3} fill={GRAY} opacity={0.4} />
            <rect x={x1} y={y + 8} width={barW} height={ROW - 18} rx={3} fill={color} opacity={0.9}>
              <animate attributeName="width" from="0" to={barW} dur="0.5s"
                begin={`${i * 0.08}s`} fill="freeze" calcMode="spline"
                keySplines="0.4 0 0.2 1" />
            </rect>
            <text
              x={isPos ? x2 + 5 : x1 - 5}
              y={y + ROW * 0.6}
              textAnchor={isPos ? "start" : "end"}
              fontSize={10} fill={color} fontWeight="700"
            >
              {signed(f.shap_value)}
            </text>
            {i < rows.length - 1 && (
              <line x1={toX(f.end)} y1={y + ROW - 8}
                x2={toX(f.end)} y2={y + ROW + 8}
                stroke={GRAY} strokeWidth={1} strokeDasharray="3 2" />
            )}
          </g>
        );
      })}

      {(() => {
        const y     = PAD + ROW * (top5.length + 1);
        const color = finalProb >= 0.5 ? GREEN : RED;
        return (
          <g>
            <rect x={LABEL + PAD - 1} y={y + 6} width={clamp(finalProb, 0, 1) * BAR_AREA + 2}
              height={ROW - 14} rx={4} fill={color} opacity={0.2}
              stroke={color} strokeWidth={1} />
            <text x={LABEL - 6} y={y + ROW * 0.6}
              textAnchor="end" fontSize={11} fill="#f1f5f9" fontWeight="700">
              Final
            </text>
            <text x={toX(finalProb) + 6} y={y + ROW * 0.6}
              fontSize={13} fill={color} fontWeight="800">{pct(finalProb)}</text>
          </g>
        );
      })()}

      <line x1={toX(0.5)} y1={PAD} x2={toX(0.5)} y2={H - PAD}
        stroke="#475569" strokeWidth={1} strokeDasharray="4 3" />
      <text x={toX(0.5)} y={PAD - 4}
        textAnchor="middle" fontSize={9} fill="#475569">50% threshold</text>
    </svg>
  );
}

function ReasonCards({ reasons, decision }) {
  const isApproved = decision === "approved";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {reasons.map((r, i) => (
        <div key={i} style={{
          background: "#0f172a", borderRadius: 8, padding: "0.75rem 1rem",
          borderLeft: `3px solid ${isApproved ? "#22d3a5" : "#f87171"}`,
          fontSize: "0.82rem", color: "#cbd5e1", lineHeight: 1.6,
        }}>
          {r}
        </div>
      ))}
    </div>
  );
}

function FactorRow({ factor }) {
  const isPos = factor.shap_value >= 0;
  const pctW  = clamp(Math.abs(factor.shap_value) * 300, 2, 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 0",
      borderBottom: "1px solid #1e293b" }}>
      <div style={{ width: 180, fontSize: "0.75rem", color: "#94a3b8",
        whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
        {factor.label}
      </div>
      <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{ flex: 1, background: "#1e293b", borderRadius: 3, height: 8, overflow: "hidden" }}>
          <div style={{
            width: `${pctW}%`, height: "100%", borderRadius: 3,
            background: isPos ? "#22d3a5" : "#f87171",
            transition: "width 0.4s ease",
          }} />
        </div>
        <span style={{ width: 64, textAlign: "right", fontFamily: "monospace",
          fontSize: "0.72rem", color: isPos ? "#22d3a5" : "#f87171", fontWeight: 700 }}>
          {signed(factor.shap_value)}
        </span>
      </div>
    </div>
  );
}

const WHAT_IF_FIELDS = [
  { key: "credit_score",          label: "Credit Score",       min: 300,  max: 850,  step: 10  },
  { key: "annual_income",         label: "Annual Income ($)",  min: 10000,max: 300000,step: 5000},
  { key: "dti_ratio",             label: "DTI Ratio",          min: 0,    max: 1,    step: 0.01},
  { key: "credit_utilization",          label: "Credit Utilization", min: 0,    max: 1,    step: 0.01},
  { key: "late_payment_severity_score", label: "Late Payment Severity", min: 0, max: 1,    step: 0.01},
  { key: "num_late_payments",           label: "Late Payments",      min: 0,    max: 20,   step: 1   },
  { key: "num_derogatory_marks",  label: "Derogatory Marks",   min: 0,    max: 10,   step: 1   },
  { key: "savings_balance",       label: "Savings ($)",        min: 0,    max: 100000,step: 1000},
];

function WhatIfSimulator({ originalApplicant, modelName, originalProb }) {
  const [changes,  setChanges]  = useState({});
  const [result,   setResult]   = useState(null);
  const [loading,  setLoading]  = useState(false);

  const handleChange = (key, val) => {
    setChanges((p) => ({ ...p, [key]: +val }));
  };

  const runWhatIf = useCallback(async () => {
    if (Object.keys(changes).length === 0) return;
    setLoading(true);
    try {
      const res = await fetch(
        `${API}/api/explain/what-if?model=${modelName}`,
        {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ applicant: originalApplicant, changes }),
        }
      );
      setResult(await res.json());
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [changes, originalApplicant, modelName]);

  const delta    = result ? result.probability_delta : 0;
  const newProb  = result ? result.modified.approval_probability : originalProb;

  return (
    <div>
      <p style={{ color: "#64748b", fontSize: "0.8rem", marginTop: 0, marginBottom: "1rem" }}>
        Adjust values below to see how the decision would change.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginBottom: "1rem" }}>
        {WHAT_IF_FIELDS.map(({ key, label, min, max, step }) => {
          const orig = originalApplicant[key] ?? 0;
          const curr = changes[key] ?? orig;
          const changed = changes[key] !== undefined && changes[key] !== orig;
          return (
            <label key={key} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: "0.7rem", color: changed ? "#fbbf24" : "#64748b",
                textTransform: "uppercase", letterSpacing: "0.05em" }}>
                {label} {changed && "*"}
              </span>
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <input type="range" min={min} max={max} step={step} value={curr}
                  onChange={(e) => handleChange(key, e.target.value)}
                  style={{ flex: 1 }} />
                <span style={{ fontFamily: "monospace", fontSize: "0.75rem",
                  color: "#e2e8f0", minWidth: 52, textAlign: "right" }}>
                  {step < 1 ? (+curr).toFixed(2) : (+curr).toLocaleString()}
                </span>
              </div>
            </label>
          );
        })}
      </div>

      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", marginBottom: "1rem" }}>
        <button onClick={runWhatIf} disabled={loading || Object.keys(changes).length === 0}
          style={{
            background: "rgba(96,165,250,0.15)", border: "1px solid rgba(96,165,250,0.5)",
            color: "#93c5fd", padding: "0.5rem 1.25rem", borderRadius: 8,
            fontFamily: "monospace", fontSize: "0.85rem", fontWeight: 700,
            cursor: loading ? "wait" : "pointer", opacity: Object.keys(changes).length === 0 ? 0.4 : 1,
          }}>
          {loading ? "Calculating..." : "Apply changes"}
        </button>
        {Object.keys(changes).length > 0 && (
          <button onClick={() => { setChanges({}); setResult(null); }}
            style={{ background: "transparent", border: "1px solid #334155",
              color: "#64748b", padding: "0.5rem 1rem", borderRadius: 8,
              fontFamily: "monospace", fontSize: "0.8rem", cursor: "pointer" }}>
            Reset
          </button>
        )}
      </div>

      {result && (
        <div style={{ background: "#0f172a", borderRadius: 10, padding: "1rem",
          border: `1px solid ${result.decision_changed ? "#fbbf24" : "#334155"}` }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
            <div>
              <div style={{ fontSize: "0.7rem", color: "#64748b", textTransform: "uppercase",
                letterSpacing: "0.05em", marginBottom: 4 }}>Probability shift</div>
              <div style={{ fontSize: "1.4rem", fontFamily: "monospace", fontWeight: 800,
                color: delta > 0 ? "#22d3a5" : "#f87171" }}>
                {delta > 0 ? "+" : ""}{(delta * 100).toFixed(1)}pp
              </div>
              <div style={{ fontSize: "0.8rem", color: "#64748b" }}>
                {pct(originalProb)} → {pct(newProb)}
              </div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: "0.7rem", color: "#64748b", textTransform: "uppercase",
                letterSpacing: "0.05em", marginBottom: 4 }}>New decision</div>
              <div style={{
                fontSize: "1rem", fontWeight: 800, textTransform: "uppercase",
                color: result.modified.decision === "approved" ? "#22d3a5" : "#f87171",
              }}>
                {result.modified.decision}
              </div>
              {result.decision_changed && (
                <div style={{ fontSize: "0.75rem", color: "#fbbf24", marginTop: 2 }}>
                  Decision changed!
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const DEFAULT_APPLICANT = {
  credit_score: 680, annual_income: 60000, loan_amount: 20000,
  loan_term: 36, dti_ratio: 0.38, employment_years: 2,
  num_credit_lines: 3, num_derogatory_marks: 1,
  credit_utilization: 0.45, late_payment_severity_score: 0.95,
  home_ownership: 0, purpose_encoded: 0, num_late_payments: 2,
  savings_balance: 3000, monthly_expenses: 2800,
};

const S = {
  page:     { minHeight: "100vh", background: "#0f1117", color: "#e2e8f0",
               fontFamily: "'IBM Plex Mono', monospace", padding: "2rem" },
  title:    { fontSize: "1.4rem", fontWeight: 700, color: "#f8fafc", margin: 0 },
  subtitle: { color: "#475569", fontSize: "0.8rem", margin: "0.25rem 0 0" },
  card:     { background: "#1e293b", border: "1px solid #334155",
               borderRadius: 12, padding: "1.25rem" },
  cardNote: { color: "#64748b", fontSize: "0.78rem", margin: "0 0 1rem",
               lineHeight: 1.5 },
  spinner:  { width: 36, height: 36, border: "3px solid #1e293b",
               borderTopColor: "#60a5fa", borderRadius: "50%",
               animation: "spin 0.7s linear infinite" },
};

export default function DecisionExplainer({ applicant: propApplicant, modelName = "xgboost" }) {
  const applicant              = propApplicant || DEFAULT_APPLICANT;
  const [data,    setData]     = useState(null);
  const [loading, setLoading]  = useState(true);
  const [error,   setError]    = useState(null);
  const [tab,     setTab]      = useState("waterfall");

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API}/api/explain?model=${modelName}`, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify(applicant),
        });
        if (!res.ok) throw new Error(`API ${res.status}`);
        setData(await res.json());
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [applicant, modelName]);

  if (loading) return (
    <div style={{ ...S.page, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 12 }}>
      <div style={S.spinner} />
      <p style={{ color: "#64748b", fontFamily: "monospace" }}>Computing SHAP values...</p>
    </div>
  );

  if (error) return (
    <div style={{ ...S.page, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <p style={{ color: "#f87171", fontFamily: "monospace" }}>Error: {error} — is your API running?</p>
    </div>
  );

  if (!data) return null;

  const rawDec = (data.decision || '').toUpperCase().replace(/_/g, ' ');
  const isApproved = rawDec === 'APPROVE' || rawDec === 'APPROVED';
  const isRejected = rawDec === 'REJECT' || rawDec === 'REJECTED';
  const displayDecision = isApproved ? 'APPROVE' : (isRejected ? 'REJECT' : 'MANUAL REVIEW');
  const accentColor = isApproved ? "#22d3a5" : (isRejected ? "#f87171" : "#fbbf24");
  const defaultProb = data.calibrated_default_probability !== undefined 
    ? data.calibrated_default_probability 
    : (1 - data.approval_probability);

  return (
    <div style={S.page}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12, marginBottom: "1.5rem" }}>
        <div>
          <div style={{ fontSize: "11px", color: "#38bdf8", fontWeight: "700", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "4px" }}>
            Model Explanation & Feature Attribution
          </div>
          <h1 style={S.title}>Decision Explainability</h1>
          <p style={S.subtitle}>Model: {modelName === 'lightgbm' ? 'LightGBM v3.1 (Canonical)' : modelName} · SHAP Waterfall Attribution</p>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: "11px", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "4px" }}>
            Economic Policy Decision
          </div>
          <div style={{
            display: "inline-block", padding: "0.4rem 1.25rem", borderRadius: 8,
            background: `${accentColor}20`, border: `1px solid ${accentColor}60`,
            color: accentColor, fontWeight: 800, fontSize: "1rem", textTransform: "uppercase",
            letterSpacing: "0.08em",
          }}>
            {displayDecision}
          </div>
          <div style={{ color: "#94a3b8", fontSize: "0.8rem", marginTop: 6, textAlign: "right" }}>
            <span style={{ fontFamily: "monospace", color: "#f8fafc", fontWeight: "700" }}>{pct(defaultProb)}</span> Calibrated Default Risk
          </div>
          <div style={{ color: "#64748b", fontSize: "0.75rem", textAlign: "right" }}>
            ({pct(data.approval_probability)} approval probability)
          </div>
        </div>
      </div>

      <div style={{ display: "flex", gap: 4, marginBottom: "1.25rem", borderBottom: "1px solid #1e293b", paddingBottom: 0 }}>
        {[
          ["waterfall", "Waterfall chart"],
          ["reasons",   "Plain-English reasons"],
          ["factors",   "All factors"],
          ["whatif",    "What-If simulator"],
        ].map(([key, label]) => (
          <button key={key} onClick={() => setTab(key)} style={{
            background: "transparent", border: "none",
            borderBottom: tab === key ? `2px solid ${accentColor}` : "2px solid transparent",
            color: tab === key ? accentColor : "#64748b",
            padding: "0.5rem 1rem", fontFamily: "monospace", fontSize: "0.82rem",
            cursor: "pointer", marginBottom: -1, fontWeight: tab === key ? 700 : 400,
            transition: "all 0.15s",
          }}>
            {label}
          </button>
        ))}
      </div>

      {tab === "waterfall" && (
        <div style={S.card}>
          <p style={S.cardNote}>
            Each bar shows how much a feature shifted the prediction up or down from the base rate of {pct(data.base_value)}.
          </p>
          <WaterfallChart
            baseValue={data.base_value}
            factors={data.top_factors}
            finalProb={data.approval_probability}
          />
        </div>
      )}

      {tab === "reasons" && (
        <div style={S.card}>
          <p style={S.cardNote}>
            The top factors driving this {displayDecision} decision, explained in plain English.
          </p>
          <ReasonCards reasons={data.plain_english} decision={displayDecision.toLowerCase()} />
        </div>
      )}

      {tab === "factors" && (
        <div style={S.card}>
          <p style={S.cardNote}>All 15 features ranked by impact magnitude.</p>
          {(data.all_factors || data.top_factors).map((f) => (
            <FactorRow key={f.feature} factor={f} />
          ))}
        </div>
      )}

      {tab === "whatif" && (
        <div style={S.card}>
          <WhatIfSimulator
            originalApplicant={applicant}
            modelName={modelName}
            originalProb={data.approval_probability}
          />
        </div>
      )}
    </div>
  );
}