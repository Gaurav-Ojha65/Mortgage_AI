import React, { useState, useMemo } from 'react';
import './EligibilityCheck.css';

const CRITERIA = [
  { key: 'credit', label: 'Credit Score', min: 620, ideal: 740, unit: '', desc: 'FICO credit score' },
  { key: 'dti',    label: 'Debt-to-Income Ratio', max: 43, ideal: 30, unit: '%', desc: 'Total monthly debt / gross monthly income' },
  { key: 'income', label: 'Monthly Income', min: 3000, unit: '$', desc: 'Gross monthly income before taxes' },
  { key: 'employ', label: 'Employment History', min: 2, unit: 'yrs', desc: 'Years at current/recent employer' },
];

function checkEligibility(data) {
  const checks = [];
  const cs = +data.credit || 0;
  const dti = +data.dti || 0;
  const income = +data.income || 0;
  const employ = +data.employ || 0;
  const loan = +data.loanAmount || 0;

  // Credit check
  if (cs >= 740) checks.push({ label: 'Credit Score', status: 'pass', msg: `Excellent score of ${cs} — qualifies for best rates.`, score: 100 });
  else if (cs >= 620) checks.push({ label: 'Credit Score', status: 'warn', msg: `Score of ${cs} meets minimum, but below ideal of 740 for best rates.`, score: 65 });
  else if (cs > 0) checks.push({ label: 'Credit Score', status: 'fail', msg: `Score of ${cs} is below the 620 minimum threshold for most lenders.`, score: 20 });
  else checks.push({ label: 'Credit Score', status: 'missing', msg: 'Credit score not provided. This field is required for eligibility.', score: 0 });

  // DTI
  if (dti > 0 && dti <= 30) checks.push({ label: 'Debt-to-Income', status: 'pass', msg: `DTI of ${dti}% is well within the recommended 30% ceiling.`, score: 100 });
  else if (dti <= 43) checks.push({ label: 'Debt-to-Income', status: 'warn', msg: `DTI of ${dti}% is acceptable but exceeds the recommended 30% target.`, score: 55 });
  else if (dti > 43) checks.push({ label: 'Debt-to-Income', status: 'fail', msg: `DTI of ${dti}% exceeds the 43% maximum. Reduce existing debt obligations.`, score: 15 });
  else checks.push({ label: 'Debt-to-Income', status: 'missing', msg: 'DTI ratio not provided. Required for assessment.', score: 0 });

  // Income
  if (income >= 5000) checks.push({ label: 'Monthly Income', status: 'pass', msg: `Income of $${income.toLocaleString()} meets comfortable lending thresholds.`, score: 100 });
  else if (income >= 3000) checks.push({ label: 'Monthly Income', status: 'warn', msg: `Income of $${income.toLocaleString()} meets minimum but may limit loan size.`, score: 60 });
  else if (income > 0) checks.push({ label: 'Monthly Income', status: 'fail', msg: `Income of $${income.toLocaleString()} may be insufficient for most mortgage products.`, score: 25 });
  else checks.push({ label: 'Monthly Income', status: 'missing', msg: 'Monthly income not provided. Required for assessment.', score: 0 });

  // Employment
  if (employ >= 3) checks.push({ label: 'Employment', status: 'pass', msg: `${employ} years of stable employment is a strong positive signal.`, score: 100 });
  else if (employ >= 1) checks.push({ label: 'Employment', status: 'warn', msg: `${employ} year(s) of employment may require additional documentation.`, score: 55 });
  else if (employ > 0) checks.push({ label: 'Employment', status: 'fail', msg: `Less than 1 year of employment may disqualify from conventional loans.`, score: 20 });
  else checks.push({ label: 'Employment', status: 'missing', msg: 'Employment history not provided.', score: 0 });

  // Loan affordability
  if (income > 0 && loan > 0) {
    const ratio = loan / (income * 12);
    if (ratio <= 3) checks.push({ label: 'Affordability', status: 'pass', msg: `Loan-to-annual-income ratio of ${ratio.toFixed(1)}x is within safe bounds.`, score: 100 });
    else if (ratio <= 5) checks.push({ label: 'Affordability', status: 'warn', msg: `Loan-to-income ratio of ${ratio.toFixed(1)}x is acceptable but may limit flexibility.`, score: 55 });
    else checks.push({ label: 'Affordability', status: 'fail', msg: `Loan-to-income ratio of ${ratio.toFixed(1)}x exceeds recommended limits.`, score: 20 });
  }

  const validChecks = checks.filter(c => c.status !== 'missing');
  const overallScore = validChecks.length ? Math.round(validChecks.reduce((a, c) => a + c.score, 0) / validChecks.length) : 0;
  const passCt = checks.filter(c => c.status === 'pass').length;
  const failCt = checks.filter(c => c.status === 'fail').length;
  const missCt = checks.filter(c => c.status === 'missing').length;

  let verdict = 'Likely Eligible';
  let verdictColor = '#22C55E';
  if (failCt >= 2 || overallScore < 40) { verdict = 'Unlikely Eligible'; verdictColor = '#EF4444'; }
  else if (failCt >= 1 || overallScore < 65) { verdict = 'Borderline — Needs Review'; verdictColor = '#F59E0B'; }

  return { checks, overallScore, passCt, failCt, missCt, verdict, verdictColor };
}

export default function EligibilityCheck() {
  const [data, setData] = useState({ credit: '', dti: '', income: '', employ: '', loanAmount: '' });
  const [submitted, setSubmitted] = useState(false);

  const result = useMemo(() => submitted ? checkEligibility(data) : null, [data, submitted]);

  const update = (k, v) => { setData(d => ({ ...d, [k]: v })); if (submitted) setSubmitted(false); };

  const statusIcon = { pass: '✅', warn: '⚠️', fail: '❌', missing: '❓' };
  const statusColor = { pass: '#22C55E', warn: '#F59E0B', fail: '#EF4444', missing: '#64748B' };

  return (
    <div className="el-page">
      <div className="el-header">
        <h1 className="el-title">Eligibility Pre-Check</h1>
        <p className="el-sub">Quick self-assessment before submitting a full application. No credit pull required.</p>
      </div>

      <div className="el-main-grid">
        {/* Form */}
        <div className="el-form-card">
          <h3 className="el-form-title">Applicant Details</h3>
          {[
            { key: 'credit', label: 'Credit Score', type: 'number', placeholder: '720', min: 300, max: 850 },
            { key: 'income', label: 'Monthly Income ($)', type: 'number', placeholder: '5000' },
            { key: 'dti',    label: 'Debt-to-Income Ratio (%)', type: 'number', placeholder: '28', step: '0.1' },
            { key: 'employ', label: 'Employment History (years)', type: 'number', placeholder: '3' },
            { key: 'loanAmount', label: 'Desired Loan Amount ($)', type: 'number', placeholder: '250000' },
          ].map(f => (
            <div key={f.key} className="el-field">
              <label className="el-label">{f.label}</label>
              <input className="el-input" type={f.type} placeholder={f.placeholder} value={data[f.key]}
                onChange={e => update(f.key, e.target.value)} min={f.min} max={f.max} step={f.step} />
            </div>
          ))}
          <button className="el-submit" onClick={() => setSubmitted(true)}>Check Eligibility</button>
          <p className="el-note">🔒 No data is stored. This check is instant and private.</p>
        </div>

        {/* Results */}
        <div className="el-results-card">
          {!result ? (
            <div className="el-empty">
              <div className="el-empty-icon">📋</div>
              <p>Enter your details and hit "Check Eligibility" to see your assessment.</p>
            </div>
          ) : (
            <>
              {/* Verdict banner */}
              <div className="el-verdict" style={{ borderColor: `${result.verdictColor}60`, background: `${result.verdictColor}08` }}>
                <div className="el-verdict-score" style={{ color: result.verdictColor }}>
                  <svg width="80" height="80">
                    <circle cx="40" cy="40" r="34" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="6" />
                    <circle cx="40" cy="40" r="34" fill="none" stroke={result.verdictColor} strokeWidth="6"
                      strokeDasharray={2*Math.PI*34} strokeDashoffset={2*Math.PI*34*(1-result.overallScore/100)}
                      strokeLinecap="round" style={{ transform: 'rotate(-90deg)', transformOrigin: 'center', transition: 'stroke-dashoffset 1s ease' }} />
                    <text x="40" y="44" textAnchor="middle" fill={result.verdictColor} fontSize="18" fontWeight="800" fontFamily="JetBrains Mono">{result.overallScore}</text>
                  </svg>
                </div>
                <div>
                  <div className="el-verdict-label" style={{ color: result.verdictColor }}>{result.verdict}</div>
                  <div className="el-verdict-sub">{result.passCt} passed · {result.failCt} failed · {result.missCt} missing</div>
                </div>
              </div>

              {/* Check Items */}
              <div className="el-checks">
                {result.checks.map((c, i) => (
                  <div key={i} className="el-check-item" style={{ borderLeft: `3px solid ${statusColor[c.status]}` }}>
                    <div className="el-check-header">
                      <span className="el-check-icon">{statusIcon[c.status]}</span>
                      <span className="el-check-label">{c.label}</span>
                      <span className="el-check-status" style={{ color: statusColor[c.status] }}>{c.status.toUpperCase()}</span>
                    </div>
                    <p className="el-check-msg">{c.msg}</p>
                  </div>
                ))}
              </div>

              <div className="el-disclaimer">
                ⚖️ This is an AI-assisted pre-screening tool — not a formal lending decision. Final eligibility requires full documentation review and human underwriter approval. Rates vary — contact your loan officer for specifics.
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
