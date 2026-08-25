import React, { useState, useRef, useEffect } from 'react';
import GaugeChart from './GaugeChart';
import './DecisionResult.css';

/* ─── Helpers ─────────────────────────────────────────── */

/**
 * Derive top-3 risk factors from feature_values + raw data.
 * Returns array of { label, reason, impact: 'high'|'medium'|'low', icon }
 */
function deriveTopFactors(decision) {
  const fv = decision.feature_values || {};
  const factors = [];

  const dti = (fv.debt_to_income_ratio || 0) * 100;
  const emi = fv.emi_to_income_ratio || 0;
  const util = (fv.credit_utilization_score || 0) * 100;
  const burden = fv.loan_burden_index || 0;
  const afford = fv.affordability_score || 0;
  const defProb = (decision.default_probability || 0) * 100;

  if (dti > 43)
    factors.push({ label: 'Debt-to-Income Ratio', reason: `At ${dti.toFixed(1)}%, DTI exceeds the 43% recommended ceiling — high monthly obligations relative to income.`, impact: dti > 60 ? 'high' : 'medium', icon: '📊' });
  else if (dti < 20)
    factors.push({ label: 'Debt-to-Income Ratio', reason: `Excellent DTI of ${dti.toFixed(1)}% — income comfortably covers obligations, a strong positive signal.`, impact: 'low', icon: '✅' });

  if (emi > 35)
    factors.push({ label: 'EMI Burden', reason: `EMI consumes ${emi.toFixed(1)}% of monthly income. Underwriters prefer this below 35% for sustainable repayment.`, impact: emi > 50 ? 'high' : 'medium', icon: '💳' });
  else
    factors.push({ label: 'EMI Burden', reason: `EMI-to-income at ${emi.toFixed(1)}% is within healthy limits, indicating manageable monthly payments.`, impact: 'low', icon: '✅' });

  if (defProb > 30)
    factors.push({ label: 'Monte Carlo Default Probability', reason: `Simulation across 5,000 scenarios puts default probability at ${defProb.toFixed(1)}% — above the 30% caution threshold.`, impact: defProb > 50 ? 'high' : 'medium', icon: '⚠️' });
  else
    factors.push({ label: 'Monte Carlo Default Probability', reason: `Stress-tested default probability of ${defProb.toFixed(1)}% is within acceptable bounds across economic scenarios.`, impact: 'low', icon: '✅' });

  if (burden > 0.5)
    factors.push({ label: 'Loan Burden Index', reason: `Loan burden index of ${burden.toFixed(2)} suggests the loan amount is large relative to applicant's financial capacity.`, impact: burden > 0.8 ? 'high' : 'medium', icon: '🏋️' });

  if (afford < 0.4)
    factors.push({ label: 'Affordability Score', reason: `Affordability score of ${afford.toFixed(2)} indicates limited financial slack after loan repayments.`, impact: 'medium', icon: '💰' });

  // Return top 3 sorted by impact
  const impactOrder = { high: 0, medium: 1, low: 2 };
  return factors.sort((a, b) => impactOrder[a.impact] - impactOrder[b.impact]).slice(0, 3);
}

/**
 * Derive actionable suggestions.
 */
function deriveSuggestions(decision) {
  const fv = decision.feature_values || {};
  const suggestions = [];

  const dti = (fv.debt_to_income_ratio || 0) * 100;
  const emi = fv.emi_to_income_ratio || 0;
  const defProb = (decision.default_probability || 0) * 100;

  if (dti > 43) suggestions.push('Reduce existing debt obligations before applying — paying off high-interest loans or credit cards could lower your DTI below the 43% threshold.');
  if (emi > 35) suggestions.push('Consider requesting a lower loan amount or extending the repayment term to bring the EMI-to-income ratio below 35%.');
  if (defProb > 30) suggestions.push('A larger down payment would reduce the principal and improve your risk profile significantly in our stress-test simulations.');
  if (decision.risk_level === 'HIGH') suggestions.push('Providing collateral or a co-applicant with strong credit history can offset high-risk indicators and strengthen the application.');

  if (suggestions.length === 0) {
    suggestions.push('Application profile is strong. Ensuring all documentation is complete will expedite underwriter review.');
  }

  return suggestions.slice(0, 2);
}

/**
 * Compute confidence tier from approval_probability.
 */
function getConfidence(prob) {
  if (prob === null || prob === undefined) return { label: 'Unknown', color: '#94A3B8', bg: 'rgba(148,163,184,0.1)' };
  const p = prob * 100;
  if (p >= 75 || p <= 25) return { label: 'High', color: '#22C55E', bg: 'rgba(34,197,94,0.1)' };
  if (p >= 60 || p <= 40) return { label: 'Medium', color: '#F59E0B', bg: 'rgba(245,158,11,0.1)' };
  return { label: 'Low', color: '#EF4444', bg: 'rgba(239,68,68,0.1)' };
}

/* ─── Impact Pill ─────────────────────────────────────── */
function ImpactPill({ impact }) {
  const map = {
    high:   { label: 'High Impact',   color: '#EF4444', bg: 'rgba(239,68,68,0.1)'   },
    medium: { label: 'Medium Impact', color: '#F59E0B', bg: 'rgba(245,158,11,0.1)' },
    low:    { label: 'Low Impact',    color: '#22C55E', bg: 'rgba(34,197,94,0.1)'   },
  };
  const cfg = map[impact] || map.medium;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '4px',
      padding: '2px 10px', borderRadius: '999px', fontSize: '11px', fontWeight: 700,
      background: cfg.bg, color: cfg.color, border: `1px solid ${cfg.color}40`
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: cfg.color, display: 'inline-block' }} />
      {cfg.label}
    </span>
  );
}

/* ─── Factor Card ─────────────────────────────────────── */
function FactorCard({ factor, index }) {
  return (
    <div className="xp-factor-card" style={{ animationDelay: `${index * 80}ms` }}>
      <div className="xp-factor-header">
        <span className="xp-factor-num">#{index + 1}</span>
        <span className="xp-factor-icon">{factor.icon}</span>
        <span className="xp-factor-label">{factor.label}</span>
        <ImpactPill impact={factor.impact} />
      </div>
      <p className="xp-factor-reason">{factor.reason}</p>
    </div>
  );
}

/* ─── Manual Review Banner ───────────────────────────── */
function ManualReviewBanner() {
  return (
    <div className="xp-manual-banner">
      <div className="xp-manual-icon">🔍</div>
      <div>
        <div className="xp-manual-title">Flagged for Manual Review</div>
        <div className="xp-manual-sub">Calibrated risk falls within the 4.5% – 33.5% policy review band. A human underwriter must review this application before a final lending commitment is issued.</div>
      </div>
    </div>
  );
}

/* ─── Main DecisionResult ────────────────────────────── */
const DecisionResult = ({ decision }) => {
  const [showExplain, setShowExplain] = useState(true);
  if (!decision) return null;

  const getDecisionConfig = (d) => {
    const norm = (d || '').toUpperCase().replace(/_/g, ' ');
    switch (norm) {
      case 'APPROVE':
      case 'APPROVED':
        return {
          color: '#22C55E', borderColor: 'rgba(34,197,94,0.4)',
          bg: 'rgba(34,197,94,0.06)', icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
          label: 'APPROVE', description: 'Application meets all auto-approval criteria (p ≤ 0.045)'
        };
      case 'REJECT':
      case 'REJECTED':
        return {
          color: '#EF4444', borderColor: 'rgba(239,68,68,0.4)',
          bg: 'rgba(239,68,68,0.06)', icon: 'M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z',
          label: 'REJECT', description: 'Application exceeds risk ceiling (p ≥ 0.335)'
        };
      default:
        return {
          color: '#F59E0B', borderColor: 'rgba(245,158,11,0.4)',
          bg: 'rgba(245,158,11,0.06)', icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
          label: 'MANUAL REVIEW', description: 'Application routed to human underwriting triage (0.045 < p < 0.335)'
        };
    }
  };

  const formatCurrency = (v) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(v || 0);
  const formatPct = (v) => `${((v || 0) * 100).toFixed(1)}%`;

  const cfg = getDecisionConfig(decision.decision);
  const factors = deriveTopFactors(decision);
  const suggestions = deriveSuggestions(decision);
  const confidence = getConfidence(decision.approval_probability);
  const defProb = decision.calibrated_default_probability !== undefined 
    ? decision.calibrated_default_probability 
    : (decision.default_probability || 0);
  const defPct = defProb * 100;
  const normDec = (decision.decision || '').toUpperCase().replace(/_/g, ' ');
  const needsManualReview = normDec === 'MANUAL REVIEW' || normDec === 'REVIEW' || (defPct > 4.5 && defPct < 33.5);

  return (
    <div className="dr-root">

      {/* ── Manual Review Banner ──────────────────────── */}
      {needsManualReview && <ManualReviewBanner />}

      {/* ── Main Decision Card ────────────────────────── */}
      <div className="dr-hero" style={{ borderColor: cfg.borderColor, background: cfg.bg }}>
        <div className="dr-hero-left">
          <div className="dr-hero-icon" style={{ borderColor: cfg.borderColor, background: `${cfg.color}18` }}>
            <svg width="36" height="36" fill="none" stroke={cfg.color} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={cfg.icon} />
            </svg>
          </div>
          <div>
            <h2 className="dr-hero-label" style={{ color: cfg.color }}>{cfg.label}</h2>
            <p className="dr-hero-desc">{cfg.description}</p>
            <div className="dr-hero-badges">
              <span className="dr-badge" style={{ color: cfg.color, background: `${cfg.color}14`, border: `1px solid ${cfg.color}40` }}>
                {decision.risk_level} RISK
              </span>
              <span className="dr-badge" style={{ color: confidence.color, background: confidence.bg, border: `1px solid ${confidence.color}40` }}>
                Confidence: {confidence.label}
              </span>
              {decision.model_used && (
                <span className="dr-badge" style={{ color: '#A78BFA', background: 'rgba(167,139,250,0.1)', border: '1px solid rgba(167,139,250,0.3)' }}>
                  🤖 {decision.model_used}
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="dr-hero-right">
          <div className="dr-emi-label">Monthly EMI</div>
          <div className="dr-emi-val">{formatCurrency(decision.emi)}</div>
          <div className="dr-emi-sub">Estimated payment</div>
        </div>
      </div>

      {/* ── Metrics Row ───────────────────────────────── */}
      <div className="dr-metrics-grid">
        <div className="dr-metric-card">
          <div className="dr-metric-label">Default Risk</div>
          <GaugeChart value={defPct} max={100} size={130} strokeWidth={11} label="Risk" suffix="%" />
        </div>
        <div className="dr-metric-card">
          <div className="dr-metric-label">Approval Confidence</div>
          <GaugeChart value={(decision.approval_probability || 0) * 100} max={100} size={130} strokeWidth={11} label="Confidence" suffix="%" />
        </div>
        <div className="dr-metric-card">
          <div className="dr-metric-label">Safe Income Threshold</div>
          <div className="dr-metric-big">{formatCurrency(decision.monte_carlo?.safe_income_threshold)}</div>
          <div className="dr-metric-sub">Minimum recommended income</div>
          <div className="dr-metric-bar-wrap">
            <div className="dr-metric-bar-track"><div className="dr-metric-bar-fill" style={{ width: '70%', background: '#E8A020' }} /></div>
          </div>
        </div>
        <div className="dr-metric-card">
          <div className="dr-metric-label">Worst-Case EMI</div>
          <div className="dr-metric-big" style={{ color: '#EF4444' }}>{formatCurrency(decision.monte_carlo?.worst_case_emi)}</div>
          <div className="dr-metric-sub">95th percentile scenario</div>
          <div className="dr-metric-bar-wrap">
            <div className="dr-metric-bar-track"><div className="dr-metric-bar-fill" style={{ width: '80%', background: '#EF4444' }} /></div>
          </div>
        </div>
      </div>

      {/* ── DTI Bar ───────────────────────────────────── */}
      <div className="dr-dti-card">
        <div className="dr-dti-header">
          <span className="dr-dti-label">Debt-to-Income Ratio</span>
          <span className="dr-dti-val">{((decision.feature_values?.debt_to_income_ratio || 0) * 100).toFixed(1)}%</span>
        </div>
        <div className="dr-dti-track">
          <div className="dr-dti-fill" style={{ width: `${Math.min((decision.feature_values?.debt_to_income_ratio || 0) * 100, 100)}%` }} />
          <div className="dr-dti-marker" style={{ left: '43%' }} />
        </div>
        <div className="dr-dti-labels">
          <span>0% — Healthy</span>
          <span style={{ color: '#F59E0B' }}>43% Max Recommended</span>
          <span>100% — Critical</span>
        </div>
      </div>

      {/* ── Explainability Panel ──────────────────────── */}
      <div className="xp-panel">
        <div className="xp-panel-header" onClick={() => setShowExplain(v => !v)}>
          <div className="xp-panel-title">
            <span className="xp-panel-icon">🔬</span>
            AI Explainability — Why this decision?
          </div>
          <button className="xp-toggle-btn">{showExplain ? '▲ Collapse' : '▼ Expand'}</button>
        </div>

        {showExplain && (
          <div className="xp-panel-body">

            {/* Top Factors */}
            <div className="xp-section">
              <h4 className="xp-section-title">Top 3 Contributing Factors</h4>
              <div className="xp-factors-grid">
                {factors.map((f, i) => <FactorCard key={i} factor={f} index={i} />)}
              </div>
            </div>

            {/* Suggestions */}
            <div className="xp-section">
              <h4 className="xp-section-title">Actionable Improvements</h4>
              <div className="xp-suggestions">
                {suggestions.map((s, i) => (
                  <div key={i} className="xp-suggestion-item">
                    <div className="xp-suggestion-num">{i + 1}</div>
                    <p className="xp-suggestion-text">{s}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Feature Metrics */}
            {decision.feature_values && (
              <div className="xp-section">
                <h4 className="xp-section-title">Calculated Risk Metrics</h4>
                <div className="xp-metrics-grid">
                  {Object.entries(decision.feature_values).map(([key, value]) => (
                    <div key={key} className="xp-metric-item">
                      <div className="xp-metric-key">{key.replace(/_/g, ' ')}</div>
                      <div className="xp-metric-value">{typeof value === 'number' ? value.toFixed(3) : value}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Advice */}
            {decision.advice && (
              <div className="xp-section">
                <h4 className="xp-section-title">Underwriter Notes</h4>
                <div className="xp-advice">
                  {decision.advice.split(';').map((item, i) => (
                    <div key={i} className="xp-advice-item">
                      <span className="xp-advice-dot" />
                      <span>{item.trim()}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Disclaimer ───────────────────────────────── */}
      <div className="dr-disclaimer">
        <span className="dr-disclaimer-icon">⚖️</span>
        <span><strong>Disclaimer:</strong> This is an AI-assisted risk assessment. Final decisions require human underwriter approval. This output is a recommendation only and does not constitute a formal lending offer or denial.</span>
      </div>
    </div>
  );
};

export default DecisionResult;
