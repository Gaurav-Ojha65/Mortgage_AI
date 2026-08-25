import React, { useEffect, useState, useMemo } from 'react';
import { api } from '../api';
import './HistoryPage.css';

/* ─── Smart Summarizer (Client-side rule engine) ──────── */
function generateSummary(history) {
  if (!history || history.length < 2) return null;
  const total = history.length;
  const approved = history.filter(h => h.decision?.toLowerCase() === 'approve').length;
  const rejected = history.filter(h => h.decision?.toLowerCase() === 'reject').length;
  const rate = (approved / total) * 100;

  // Split into first half vs second half to detect trend
  const half = Math.floor(total / 2);
  const recentHalf = history.slice(0, half);
  const olderHalf  = history.slice(half);
  const recentRate = recentHalf.filter(h => h.decision?.toLowerCase() === 'approve').length / Math.max(recentHalf.length, 1) * 100;
  const olderRate  = olderHalf.filter(h => h.decision?.toLowerCase() === 'approve').length / Math.max(olderHalf.length, 1) * 100;
  const delta = recentRate - olderRate;

  let trend;
  if (Math.abs(delta) < 5) trend = 'stable';
  else if (delta > 0)      trend = 'improving';
  else                     trend = 'declining';

  const avgDefaultRisk = history.reduce((a, h) => a + (h.default_probability || 0) * 100, 0) / total;

  // Most notable recent change
  const latest = history[0];
  const previous = history[1];
  let notableChange = 'No significant change detected between the most recent applications.';
  if (latest && previous) {
    const riskDiff = ((latest.default_probability || 0) - (previous.default_probability || 0)) * 100;
    if (Math.abs(riskDiff) > 10) {
      notableChange = `The most recent application showed a ${Math.abs(riskDiff).toFixed(1)}% ${riskDiff > 0 ? 'increase' : 'decrease'} in default risk compared to the prior submission.`;
    } else if (latest.decision !== previous.decision) {
      notableChange = `The most recent decision changed from ${previous.decision} to ${latest.decision}, suggesting a shift in applicant profile.`;
    }
  }

  return {
    trend,
    rate: rate.toFixed(1),
    total,
    approved,
    rejected,
    avgDefaultRisk: avgDefaultRisk.toFixed(1),
    sentence1: `The portfolio of ${total} application${total !== 1 ? 's' : ''} shows a ${trend} risk profile with an overall approval rate of ${rate.toFixed(1)}% and average default risk of ${avgDefaultRisk.toFixed(1)}%.`,
    sentence2: notableChange,
    delta: delta.toFixed(1),
  };
}

/* ─── Mini Risk Distribution (SVG bars) ───────────────── */
function RiskDistribution({ history }) {
  const buckets = useMemo(() => {
    const low  = history.filter(h => h.risk_level === 'LOW').length;
    const med  = history.filter(h => h.risk_level === 'MEDIUM').length;
    const high = history.filter(h => h.risk_level === 'HIGH').length;
    const total = history.length || 1;
    return [
      { label: 'Low',    count: low,  pct: (low  / total * 100).toFixed(0), color: '#22C55E' },
      { label: 'Medium', count: med,  pct: (med  / total * 100).toFixed(0), color: '#F59E0B' },
      { label: 'High',   count: high, pct: (high / total * 100).toFixed(0), color: '#EF4444' },
    ];
  }, [history]);

  return (
    <div className="hp-dist">
      {buckets.map(b => (
        <div key={b.label} className="hp-dist-row">
          <span className="hp-dist-label">{b.label}</span>
          <div className="hp-dist-track">
            <div className="hp-dist-fill" style={{ width: `${b.pct}%`, background: b.color }} />
          </div>
          <span className="hp-dist-val" style={{ color: b.color }}>{b.count}</span>
        </div>
      ))}
    </div>
  );
}

/* ─── Approval Timeline sparkline ─────────────────────── */
function ApprovalTimeline({ history }) {
  const points = useMemo(() => {
    const chunks = [];
    const chunkSize = Math.max(1, Math.ceil(history.length / 10));
    for (let i = 0; i < history.length; i += chunkSize) {
      const slice = history.slice(i, i + chunkSize);
      const rate = slice.filter(h => h.decision?.toLowerCase() === 'approve').length / slice.length * 100;
      chunks.push(rate);
    }
    return chunks;
  }, [history]);

  if (points.length < 2) return null;

  const W = 280, H = 70;
  const max = 100, min = 0;
  const pts = points.map((v, i) => {
    const x = (i / (points.length - 1)) * W;
    const y = H - ((v - min) / (max - min)) * H;
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="hp-timeline-wrap">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
        <defs>
          <linearGradient id="hpGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#22C55E" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#22C55E" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={`M0,${H} ${pts.split(' ').map(p => `L${p}`).join(' ')} L${W},${H} Z`} fill="url(#hpGrad)" />
        <polyline points={pts} fill="none" stroke="#22C55E" strokeWidth="2" strokeLinecap="round" />
        {points.map((v, i) => {
          const x = (i / (points.length - 1)) * W;
          const y = H - (v / 100) * H;
          return <circle key={i} cx={x} cy={y} r="3" fill="#22C55E" />;
        })}
      </svg>
    </div>
  );
}

/* ─── Summary Card ─────────────────────────────────────── */
function SummaryCard({ summary }) {
  if (!summary) return null;
  const trendIcon = summary.trend === 'improving' ? '📈' : summary.trend === 'declining' ? '📉' : '📊';
  const trendColor = summary.trend === 'improving' ? '#22C55E' : summary.trend === 'declining' ? '#EF4444' : '#F59E0B';

  return (
    <div className="hp-summary-card">
      <div className="hp-summary-header">
        <div className="hp-summary-icon">🤖</div>
        <div>
          <div className="hp-summary-title">AI Portfolio Summary</div>
          <span className="hp-summary-badge" style={{ color: trendColor, background: `${trendColor}18`, border: `1px solid ${trendColor}40` }}>
            {trendIcon} {summary.trend.charAt(0).toUpperCase() + summary.trend.slice(1)} Profile
          </span>
        </div>
      </div>
      <p className="hp-summary-text">{summary.sentence1}</p>
      <p className="hp-summary-text hp-summary-text2">{summary.sentence2}</p>
      <div className="hp-summary-disclaimer">
        ⚖️ This is an AI-assisted assessment. Keep it factual — no financial advice implied. Final decisions require human underwriter approval.
      </div>
    </div>
  );
}

/* ─── Decision Badge ──────────────────────────────────── */
function DecisionBadge({ decision }) {
  const d = (decision || '').toUpperCase().replace(/_/g, ' ');
  if (d === 'APPROVE' || d === 'APPROVED') {
    return <span style={{ display: 'inline-block', padding: '3px 11px', borderRadius: '999px', fontSize: '11px', fontWeight: 700, color: '#4ade80', background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.3)' }}>✓ Approve</span>;
  }
  if (d === 'REJECT' || d === 'REJECTED') {
    return <span style={{ display: 'inline-block', padding: '3px 11px', borderRadius: '999px', fontSize: '11px', fontWeight: 700, color: '#f87171', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)' }}>✗ Reject</span>;
  }
  return <span style={{ display: 'inline-block', padding: '3px 11px', borderRadius: '999px', fontSize: '11px', fontWeight: 700, color: '#fbbf24', background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.3)' }}>~ Manual Review</span>;
}

function RiskPill({ level }) {
  const map = {
    LOW:      { color: '#4ade80', bg: 'rgba(34,197,94,0.1)'  },
    VERY_LOW: { color: '#4ade80', bg: 'rgba(34,197,94,0.1)'  },
    MEDIUM:   { color: '#fbbf24', bg: 'rgba(245,158,11,0.1)' },
    MODERATE: { color: '#fbbf24', bg: 'rgba(245,158,11,0.1)' },
    HIGH:     { color: '#f87171', bg: 'rgba(239,68,68,0.1)'  },
    SEVERE:   { color: '#ef4444', bg: 'rgba(239,68,68,0.15)' },
  };
  const c = map[level?.toUpperCase()] || map.MEDIUM;
  return (
    <span style={{ display: 'inline-block', padding: '2px 9px', borderRadius: '999px', fontSize: '11px', fontWeight: 600, color: c.color, background: c.bg }}>
      {(level || 'MEDIUM').replace(/_/g, ' ')}
    </span>
  );
}

/* ─── Main Page ────────────────────────────────────────── */
export default function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter]   = useState('all');
  const [riskFilter, setRiskFilter] = useState('all');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');

  useEffect(() => {
    api.history(50).then(res => {
      setHistory(res || []);
    }).catch(console.error).finally(() => setLoading(false));
  }, []);

  const summary = useMemo(() => generateSummary(history), [history]);

  const filtered = useMemo(() => {
    let result = history;
    
    if (filter !== 'all') {
      result = result.filter(h => {
        const d = (h.decision || '').toLowerCase().replace(/_/g, '-');
        const r = (h.risk_level || '').toLowerCase();
        if (filter === 'approve') return d === 'approve' || d === 'approved';
        if (filter === 'reject') return d === 'reject' || d === 'rejected';
        if (filter === 'review' || filter === 'manual-review') return d.includes('review') || d.includes('conditional');
        return d === filter || r === filter;
      });
    }
    
    if (riskFilter !== 'all') {
      result = result.filter(h => h.risk_level?.toLowerCase() === riskFilter);
    }
    
    if (dateFrom) {
      const from = new Date(dateFrom).getTime();
      result = result.filter(h => new Date(h.timestamp).getTime() >= from);
    }
    
    if (dateTo) {
      const to = new Date(dateTo);
      to.setHours(23, 59, 59, 999);
      result = result.filter(h => new Date(h.timestamp).getTime() <= to.getTime());
    }
    
    return result;
  }, [history, filter, riskFilter, dateFrom, dateTo]);

  const kpis = useMemo(() => {
    const total    = history.length;
    const approved = history.filter(h => h.decision?.toLowerCase() === 'approve').length;
    const high     = history.filter(h => h.risk_level === 'HIGH').length;
    const avgCredit = total ? Math.round(history.reduce((a,h) => a+(h.credit_score||0), 0) / total) : 0;
    return { total, approved, high, avgCredit, rate: total ? ((approved/total)*100).toFixed(1) : 0 };
  }, [history]);

  const handleExport = () => {
    const headers = ['ID', 'Timestamp', 'Applicant Name', 'Loan Amount', 'Risk Score', 'Risk Level', 'Status'];
    const rows = filtered.map(row => {
      const id = history.length - history.indexOf(row);
      const timestamp = new Date(row.timestamp).toLocaleString().replace(/,/g, '');
      const applicantName = 'N/A';
      const loanAmount = row.loan_amount || 0;
      const riskScore = row.credit_score || 'N/A';
      const riskLevel = row.risk_level || 'N/A';
      const status = row.decision || 'N/A';
      return [id, timestamp, applicantName, loanAmount, riskScore, riskLevel, status].join(',');
    });
    
    const csvContent = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'application_history_filtered.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="hp-page">
      <div className="hp-top-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="hp-page-title">Application History</h1>
          <p className="hp-page-sub">Full audit trail of all processed applications with analytics</p>
        </div>
        <button 
          onClick={handleExport}
          style={{ padding: '8px 16px', background: '#3b82f6', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 600, fontSize: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
          Export CSV
        </button>
      </div>

      {/* ── KPI Strip ──────────────────────────────────── */}
      <div className="hp-kpi-strip">
        {[
          { label: 'Total Applications', value: kpis.total, color: '#A78BFA' },
          { label: 'Approval Rate',       value: `${kpis.rate}%`, color: '#22C55E' },
          { label: 'High Risk',           value: kpis.high, color: '#EF4444' },
          { label: 'Avg Credit Score',    value: kpis.avgCredit, color: '#E8A020' },
        ].map(k => (
          <div key={k.label} className="hp-kpi-card">
            <div className="hp-kpi-val" style={{ color: k.color }}>{loading ? '—' : k.value}</div>
            <div className="hp-kpi-label">{k.label}</div>
          </div>
        ))}
      </div>

      {/* ── Analytics Row ───────────────────────────────── */}
      <div className="hp-analytics-row">
        <div className="hp-analytics-card">
          <div className="hp-analytics-title">Risk Distribution</div>
          {loading ? <div className="hp-skeleton" style={{ height: 80 }} /> : <RiskDistribution history={history} />}
        </div>
        <div className="hp-analytics-card">
          <div className="hp-analytics-title">Approval Rate Trend</div>
          {loading ? <div className="hp-skeleton" style={{ height: 80 }} /> : <ApprovalTimeline history={history} />}
          <div className="hp-spark-caption">Approval rate across recent batches</div>
        </div>
        <SummaryCard summary={summary} />
      </div>

      {/* ── Table ────────────────────────────────────────── */}
      <div className="hp-table-card">
        <div className="hp-table-header">
          <h2 className="hp-table-title">All Applications</h2>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
            <div className="hp-filter-tabs" style={{ margin: 0 }}>
              {[
                { id: 'all', label: 'All' },
                { id: 'approve', label: 'Approve' },
                { id: 'manual-review', label: 'Manual Review' },
                { id: 'reject', label: 'Reject' }
              ].map(f => (
                <button key={f.id} onClick={() => setFilter(f.id)} className={`hp-filter-tab ${filter === f.id ? 'active' : ''}`}>
                  {f.label}
                </button>
              ))}
            </div>
            
            <select 
              value={riskFilter} 
              onChange={e => setRiskFilter(e.target.value)}
              style={{ padding: '6px 12px', borderRadius: '6px', border: '1px solid #e2e8f0', background: '#fff', fontSize: '13px', color: '#334155', outline: 'none', cursor: 'pointer' }}
            >
              <option value="all">All Risks</option>
              <option value="low">Low Risk</option>
              <option value="medium">Medium Risk</option>
              <option value="high">High Risk</option>
            </select>

            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: '#fff', padding: '2px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <input 
                type="date" 
                value={dateFrom} 
                onChange={e => setDateFrom(e.target.value)}
                style={{ padding: '4px 8px', border: 'none', fontSize: '13px', color: '#334155', outline: 'none', background: 'transparent' }}
              />
              <span style={{ fontSize: '12px', color: '#94a3b8', fontWeight: 500 }}>to</span>
              <input 
                type="date" 
                value={dateTo} 
                onChange={e => setDateTo(e.target.value)}
                style={{ padding: '4px 8px', border: 'none', fontSize: '13px', color: '#334155', outline: 'none', background: 'transparent' }}
              />
            </div>
          </div>
        </div>

        <div className="hp-table-wrap">
          <table className="hp-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Timestamp</th>
                <th className="align-right">Loan Amount</th>
                <th className="align-center">Credit Score</th>
                <th className="align-center">Risk Level</th>
                <th className="align-right">Default Risk</th>
                <th className="align-center">Decision</th>
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: 8 }).map((_, i) => (
                    <tr key={i}>
                      {Array.from({ length: 7 }).map((_, j) => <td key={j}><div className="hp-skeleton" style={{ height: 14, borderRadius: 4 }} /></td>)}
                    </tr>
                  ))
                : filtered.length === 0
                  ? <tr><td colSpan={7} style={{ textAlign: 'center', color: '#475569', padding: '40px' }}>No records match this filter.</td></tr>
                  : filtered.map((row, i) => (
                      <tr key={i} className="hp-row">
                        <td className="hp-id">{history.length - history.indexOf(row)}</td>
                        <td className="hp-time">
                          <div>{new Date(row.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</div>
                          <div style={{ fontSize: 10, color: '#475569' }}>{new Date(row.timestamp).toLocaleDateString()}</div>
                        </td>
                        <td className="align-right mono hp-loan">${(row.loan_amount || 0).toLocaleString()}</td>
                        <td className="align-center">
                          <span className={`hp-credit ${row.credit_score >= 700 ? 'credit-good' : row.credit_score >= 600 ? 'credit-fair' : 'credit-poor'}`}>
                            {row.credit_score || 'N/A'}
                          </span>
                        </td>
                        <td className="align-center"><RiskPill level={row.risk_level} /></td>
                        <td className="align-right">
                          <div className="hp-risk-bar-wrap">
                            <div className="hp-risk-bar-track">
                              <div className="hp-risk-bar-fill" style={{
                                width: `${(row.default_probability || 0) * 100}%`,
                                background: (row.default_probability || 0) > 0.5 ? '#EF4444' : (row.default_probability || 0) > 0.3 ? '#F59E0B' : '#22C55E'
                              }} />
                            </div>
                            <span className="hp-risk-pct" style={{ color: (row.default_probability || 0) > 0.5 ? '#f87171' : (row.default_probability || 0) > 0.3 ? '#fbbf24' : '#4ade80' }}>
                              {((row.default_probability || 0) * 100).toFixed(0)}%
                            </span>
                          </div>
                        </td>
                        <td className="align-center"><DecisionBadge decision={row.decision} /></td>
                      </tr>
                    ))
              }
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Audit Trail Disclaimer ──────────────────────── */}
      <div className="hp-audit-note">
        <span>🔒</span>
        <span>This is a full audit log. All entries are read-only and timestamped at submission. Data is retained for compliance and human underwriter review purposes.</span>
      </div>
    </div>
  );
}
