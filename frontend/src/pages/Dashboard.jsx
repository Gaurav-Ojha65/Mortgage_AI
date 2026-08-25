import React, { useEffect, useState, useRef } from 'react';
import { api } from '../api';
import GaugeChart from '../components/GaugeChart';
import './Dashboard.css';

/* ─── Animated Counter ───────────────────────────────────────────── */
function AnimatedCounter({ target, suffix = '', decimals = 0, duration = 1200 }) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let start = null;
    const step = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      setVal(+(target * ease).toFixed(decimals));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [target, duration, decimals]);
  return <>{decimals ? val.toFixed(decimals) : val}{suffix}</>;
}

/* ─── Mini Sparkline (SVG) ───────────────────────────────────────── */
function Sparkline({ data, color = '#E8A020', height = 40, width = 120 }) {
  if (!data || data.length < 2) return null;
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x},${y}`;
  }).join(' ');
  const area = `M0,${height} L${pts.split(' ').map(p => p).join(' L')} L${width},${height} Z`;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: 'visible' }}>
      <defs>
        <linearGradient id={`sg-${color.replace('#','')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#sg-${color.replace('#','')})`} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/* ─── Decision Badge ─────────────────────────────────────────────── */
function DecisionBadge({ decision }) {
  const d = (decision || '').toUpperCase().replace(/_/g, ' ');
  if (d === 'APPROVE' || d === 'APPROVED') {
    return <span className="db-badge badge-approve">✓ Approve</span>;
  }
  if (d === 'REJECT' || d === 'REJECTED') {
    return <span className="db-badge badge-reject">✗ Reject</span>;
  }
  return <span className="db-badge badge-conditional">~ Manual Review</span>;
}

/* ─── Risk Bar ───────────────────────────────────────────────────── */
function RiskBar({ score }) {
  const pct = Math.min(100, Math.max(0, score));
  const color = pct < 30 ? '#22C55E' : pct < 65 ? '#F59E0B' : '#EF4444';
  return (
    <div className="risk-bar-wrap">
      <div className="risk-bar-track">
        <div className="risk-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="risk-bar-val" style={{ color }}>{Math.round(pct)}</span>
    </div>
  );
}

/* ─── KPI Card ───────────────────────────────────────────────────── */
function KpiCard({ icon, label, value, suffix, decimals = 0, color, sparkData, trend, trendDir, loading }) {
  return (
    <div className="kpi-card">
      <div className="kpi-header">
        <div className="kpi-icon" style={{ background: `${color}18`, color }}>
          {icon}
        </div>
        {trend && (
          <div className={`kpi-trend ${trendDir === 'up' ? 'trend-up' : 'trend-down'}`}>
            {trendDir === 'up' ? '↑' : '↓'} {trend}
          </div>
        )}
      </div>
      <div className="kpi-body">
        {loading
          ? <div className="kpi-skeleton" />
          : <div className="kpi-value" style={{ color }}>
              <AnimatedCounter target={value} suffix={suffix} decimals={decimals} />
            </div>
        }
        <div className="kpi-label">{label}</div>
      </div>
      {sparkData && (
        <div className="kpi-spark">
          <Sparkline data={sparkData} color={color} />
        </div>
      )}
    </div>
  );
}

/* ─── DonutRing ──────────────────────────────────────────────────── */
function DonutRing({ pct = 0, size = 130, stroke = 14 }) {
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;
  const color = pct < 30 ? '#22C55E' : pct < 65 ? '#F59E0B' : '#EF4444';
  return (
    <svg width={size} height={size} className="donut-svg">
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={stroke} />
      <circle
        cx={size/2} cy={size/2} r={r} fill="none"
        stroke={color} strokeWidth={stroke}
        strokeDasharray={circ}
        strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transform: 'rotate(-90deg)', transformOrigin: 'center', transition: 'stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)' }}
      />
      <text x="50%" y="50%" dominantBaseline="middle" textAnchor="middle" fill={color} fontSize="22" fontWeight="700" fontFamily="JetBrains Mono, monospace">
        {Math.round(pct)}
      </text>
    </svg>
  );
}

export default function Dashboard() {
  const [data, setData] = useState({ stats: null, history: [], health: null });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const [historyRes, healthRes] = await Promise.all([
          api.history(20).catch(() => []),
          api.health().catch(() => null)
        ]);
        const historyList = historyRes || [];
        const total = historyList.length || 0;
        const approved = historyList.filter(h => {
          const d = (h.decision || '').toLowerCase();
          return d === 'approve' || d === 'approved';
        }).length;
        const rejected = historyList.filter(h => {
          const d = (h.decision || '').toLowerCase();
          return d === 'reject' || d === 'rejected';
        }).length;
        const avgRisk = total > 0 ? (historyList.reduce((a, h) => a + (h.default_probability || 0) * 100, 0) / total) : 0;
        const approvalRate = total > 0 ? (approved / total) * 100 : 0;
        // Sparkline: approval per last N apps (chunked into 8 points)
        const chunk = Math.max(1, Math.floor(total / 8));
        const sparkApproval = Array.from({ length: 8 }, (_, i) => {
          const slice = historyList.slice(i * chunk, (i + 1) * chunk);
          if (!slice.length) return 0;
          return (slice.filter(h => (h.decision || '').toLowerCase().includes('approve')).length / slice.length) * 100;
        });
        const avgLoan = total > 0 ? historyList.reduce((a, h) => a + (h.loan_amount || 0), 0) / total : 0;
        setData({ 
          stats: { total, approved, rejected, avgRisk, approvalRate, avgLoan }, 
          history: historyList, 
          sparkApproval,
          health: healthRes 
        });
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const { stats, history, sparkApproval } = data;

  /* ── Anomaly Detection Engine ──────────────────────── */
  function detectAnomalies(hist) {
    if (!hist || hist.length < 3) return [];
    const anomalies = [];
    const total = hist.length;

    // 1. Approval rate anomaly
    const half = Math.floor(total / 2);
    const recentRate = hist.slice(0, half).filter(h => h.decision?.toLowerCase() === 'approve').length / Math.max(half, 1) * 100;
    const olderRate  = hist.slice(half).filter(h => h.decision?.toLowerCase() === 'approve').length / Math.max(total - half, 1) * 100;
    const rateDelta = recentRate - olderRate;
    if (Math.abs(rateDelta) > 25) {
      anomalies.push({
        severity: Math.abs(rateDelta) > 40 ? 'high' : 'medium',
        title: rateDelta > 0 ? 'Approval Rate Spike' : 'Approval Rate Drop',
        msg: `Approval rate ${rateDelta > 0 ? 'increased' : 'decreased'} by ${Math.abs(rateDelta).toFixed(0)}% in recent applications vs earlier batch.`
      });
    }

    // 2. High-risk clustering
    const highRisk = hist.filter(h => (h.default_probability || 0) > 0.5);
    const highRiskPct = (highRisk.length / total) * 100;
    if (highRiskPct > 40) {
      anomalies.push({
        severity: highRiskPct > 60 ? 'high' : 'medium',
        title: 'High-Risk Clustering',
        msg: `${highRiskPct.toFixed(0)}% of applications are in the high-risk band (>50% default probability) — significantly above expected baseline.`
      });
    }

    // 3. Consecutive same decisions
    let streak = 1;
    for (let i = 1; i < Math.min(hist.length, 10); i++) {
      if (hist[i].decision === hist[0].decision) streak++;
      else break;
    }
    if (streak >= 5) {
      anomalies.push({
        severity: streak >= 8 ? 'high' : 'low',
        title: 'Decision Pattern Detected',
        msg: `Last ${streak} applications all received "${hist[0].decision}" — may indicate a systematic bias or data drift worth investigating.`
      });
    }

    // 4. Credit score uniformity
    const scores = hist.map(h => h.credit_score || 0).filter(s => s > 0);
    if (scores.length > 3) {
      const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
      const variance = scores.reduce((a, s) => a + Math.pow(s - avgScore, 2), 0) / scores.length;
      if (variance < 100) {
        anomalies.push({
          severity: 'low',
          title: 'Low Credit Score Variance',
          msg: `Credit scores cluster tightly around ${Math.round(avgScore)} — limited diversity in applicant profiles may indicate sampling bias.`
        });
      }
    }

    return anomalies;
  }

  const anomalies = detectAnomalies(history);

  return (
    <div className="db-page">
      {error && (
        <div className="db-error-banner">
          <span className="db-error-icon">⚠</span>
          <span>{error}</span>
        </div>
      )}

      {/* ── Anomaly Alerts ────────────────────────────────────── */}
      {!loading && (
        <div className="db-monitoring-grid">
          <div className="db-anomaly-section">
            <div className="db-anomaly-header">
              <span className="db-anomaly-icon">🔍</span>
              <span className="db-anomaly-title">Real-time Anomaly Detection — {anomalies.length} flag{anomalies.length !== 1 ? 's' : ''}</span>
            </div>
            {anomalies.length > 0 ? (
              <div className="db-anomaly-list">
                {anomalies.map((a, i) => (
                  <div key={i} className={`db-anomaly-card severity-${a.severity}`}>
                    <div className="db-anomaly-sev">{a.severity.toUpperCase()}</div>
                    <div className="db-anomaly-content">
                      <div className="db-anomaly-name">{a.title}</div>
                      <div className="db-anomaly-msg">{a.msg}</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="db-no-anomalies">
                <div className="db-check-circle">✓</div>
                <p>No systemic anomalies detected in the current data stream.</p>
              </div>
            )}
          </div>

          <div className="db-sidebar-card db-health-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <h3 className="db-section-title" style={{ margin: 0 }}>System Provenance</h3>
              <span style={{ fontSize: '11px', background: 'rgba(34, 197, 94, 0.15)', color: '#22c55e', padding: '2px 8px', borderRadius: '999px', fontWeight: '700', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
                ACTIVE
              </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '13px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '6px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <span style={{ color: '#94a3b8' }}>Model</span>
                <span style={{ color: '#f8fafc', fontWeight: '600' }}>LightGBM v3.1</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '6px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <span style={{ color: '#94a3b8' }}>Calibration</span>
                <span style={{ color: '#38bdf8', fontFamily: 'monospace' }}>oof-iso-v3.1</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '6px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <span style={{ color: '#94a3b8' }}>Policy Engine</span>
                <span style={{ color: '#fbbf24', fontFamily: 'monospace' }}>v3.1-policy-v1</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '6px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <span style={{ color: '#94a3b8' }}>Database</span>
                <span style={{ color: '#22c55e', fontWeight: '600' }}>SQLite Connected</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Decisions Served</span>
                <span style={{ color: '#f8fafc', fontFamily: 'monospace', fontWeight: '700' }}>
                  {data.health?.predictions_served !== undefined ? data.health.predictions_served : (stats?.total || 0)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── KPI Strip ─────────────────────────────────────────── */}
      <div className="db-kpi-grid">
        <KpiCard
          icon={<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>}
          label="Total Applications" value={stats?.total || 0}
          color="#A78BFA" sparkData={sparkApproval} trend="Live" trendDir="up" loading={loading}
        />
        <KpiCard
          icon={<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>}
          label="Approval Rate" value={stats?.approvalRate || 0} suffix="%" decimals={1}
          color="#22C55E" sparkData={sparkApproval} trend={`${stats?.approved || 0} approved`} trendDir="up" loading={loading}
        />
        <KpiCard
          icon={<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>}
          label="Avg Loan Amount" value={stats?.avgLoan || 0} prefix="$" suffix="" decimals={0}
          color="#E8A020" loading={loading}
        />
        <KpiCard
          icon={<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"/></svg>}
          label="Avg Default Risk" value={stats?.avgRisk || 0} suffix="%" decimals={1}
          color="#EF4444" loading={loading}
        />
      </div>

      {/* ── Main Grid ─────────────────────────────────────────── */}
      <div className="db-main-grid">

        {/* Table */}
        <div className="db-table-card">
          <div className="db-card-header">
            <h2 className="db-section-title">Recent Applications</h2>
            <span className="db-live-dot"><span className="db-live-pulse" />Live</span>
          </div>
          <div className="db-table-wrap">
            <table className="db-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th className="align-right">Loan</th>
                  <th className="align-center">Credit</th>
                  <th>Default Risk</th>
                  <th className="align-center">Decision</th>
                </tr>
              </thead>
              <tbody>
                {loading
                  ? Array.from({ length: 7 }).map((_, i) => (
                      <tr key={i} className="skeleton-row">
                        {[1,2,3,4,5].map(j => <td key={j}><div className="cell-skeleton" /></td>)}
                      </tr>
                    ))
                  : history.map((row, i) => (
                      <tr key={i} className="db-row">
                        <td className="db-time">{new Date(row.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}<span className="db-date">{new Date(row.timestamp).toLocaleDateString()}</span></td>
                        <td className="align-right mono db-loan">${(row.loan_amount || 0).toLocaleString()}</td>
                        <td className="align-center">
                          <span className={`credit-pill ${row.credit_score >= 700 ? 'credit-good' : row.credit_score >= 600 ? 'credit-fair' : 'credit-poor'}`}>
                            {row.credit_score || 'N/A'}
                          </span>
                        </td>
                        <td><RiskBar score={(row.default_probability || 0) * 100} /></td>
                        <td className="align-center"><DecisionBadge decision={row.decision} /></td>
                      </tr>
                    ))
                }
              </tbody>
            </table>
            {!loading && !history.length && (
              <div className="db-empty">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>
                <p>No applications yet — run your first analysis</p>
              </div>
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div className="db-sidebar">

          {/* Portfolio Donut */}
          <div className="db-sidebar-card db-portfolio-card">
            <h3 className="db-section-title">Portfolio Risk</h3>
            <div className="db-donut-wrap" style={{ marginTop: '10px' }}>
              {loading ? (
                <div className="donut-skeleton" />
              ) : (
                <GaugeChart 
                  value={stats?.avgRisk || 0} 
                  size={200} 
                  label="Avg Default Risk"
                  suffix="%"
                />
              )}
            </div>
            <div className="db-split-stats">
              <div className="db-split-item">
                <span className="db-split-dot" style={{ background: '#22C55E' }} />
                <span className="db-split-label">Approved</span>
                <span className="db-split-val text-success">{stats?.approved || 0}</span>
              </div>
              <div className="db-split-item">
                <span className="db-split-dot" style={{ background: '#EF4444' }} />
                <span className="db-split-label">Rejected</span>
                <span className="db-split-val text-danger">{stats?.rejected || 0}</span>
              </div>
            </div>
          </div>

          {/* Approval Trend */}
          <div className="db-sidebar-card">
            <h3 className="db-section-title" style={{ marginBottom: '16px' }}>Approval Trend</h3>
            <div className="db-spark-full">
              {!loading && sparkApproval && <Sparkline data={sparkApproval} color="#22C55E" width={240} height={70} />}
            </div>
            <p className="db-spark-caption">Approval rate over recent batches</p>
          </div>

          {/* Quick Stats */}
          <div className="db-sidebar-card db-quick-stats">
            <h3 className="db-section-title" style={{ marginBottom: '12px' }}>Quick Stats</h3>
            {[
              { label: 'High Risk',   val: `${stats ? Math.round(history.filter(h => (h.default_probability||0)*100 > 50).length / Math.max(stats.total,1) * 100) : 0}%`, color: '#EF4444' },
              { label: 'Low Risk',    val: `${stats ? Math.round(history.filter(h => (h.default_probability||0)*100 < 25).length / Math.max(stats.total,1) * 100) : 0}%`, color: '#22C55E' },
              { label: 'Avg Credit',  val: stats ? Math.round(history.reduce((a,h) => a+(h.credit_score||0), 0)/Math.max(stats.total,1)) : 0, color: '#A78BFA' },
            ].map(({ label, val, color }) => (
              <div key={label} className="qs-row">
                <span className="qs-label">{label}</span>
                <span className="qs-val mono" style={{ color }}>{loading ? '—' : val}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
