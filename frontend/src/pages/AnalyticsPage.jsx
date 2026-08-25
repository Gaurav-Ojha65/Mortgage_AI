import React, { Suspense, lazy } from 'react';
import Card from '../components/ui/Card';
import '../components/analytics.css';
import ErrorBoundary from '../components/ErrorBoundary';
import { getModelComparison } from '../utils/api';

const ModelComparison = lazy(() => import('../components/ModelComparison'));
const MonteCarlo3D = lazy(() => import('../components/MonteCarlo3D'));
const FairnessTab = lazy(() => import('../components/FairnessTab'));

function ComponentLoader({ label = "Loading visualization..." }) {
  return (
    <div style={{
      padding: '40px',
      textAlign: 'center',
      color: '#64748b',
      background: 'rgba(255, 255, 255, 0.02)',
      borderRadius: '12px',
      border: '1px solid rgba(255, 255, 255, 0.05)'
    }}>
      {label}
    </div>
  );
}

export default function AnalyticsPage() {
  const [isMock, setIsMock] = React.useState(false);
  const [activeTab, setActiveTab] = React.useState('models');
  const [importanceData, setImportanceData] = React.useState(null);

  React.useEffect(() => {
    getModelComparison()
      .then(data => {
        if (data.is_mock) setIsMock(true);
        if (data.importance && (data.importance.LightGBM || data.importance.lightgbm)) {
          const lgbImp = data.importance.LightGBM || data.importance.lightgbm;
          const sorted = Object.entries(lgbImp)
            .map(([k, v]) => ({ name: k.replace(/_/g, ' '), val: parseFloat(v) }))
            .sort((a, b) => b.val - a.val)
            .slice(0, 6);
          setImportanceData(sorted);
        }
      })
      .catch(err => console.error("Analytics fetch failed", err));
  }, []);

  const tabs = [
    { id: 'models', label: 'Models & Calibration', icon: '🧠' },
    { id: 'fairness', label: 'Fair Lending Audit', icon: '⚖️' },
  ];

  // Verified canonical metrics from reports/metrics/hpo_vs_baseline.json
  const canonicalMetrics = [
    { label: 'ROC-AUC', value: '0.8615', baseline: '0.8599', delta: '+0.0016', color: '#22c55e' },
    { label: 'PR-AUC', value: '0.3995', baseline: '0.3947', delta: '+0.0048', color: '#22c55e' },
    { label: 'Brier Score', value: '0.0492', baseline: '0.0494', delta: '-0.0002', color: '#22c55e' },
    { label: 'Weighted ECE', value: '0.0012', baseline: '0.0018', delta: '-0.0006', color: '#22c55e' },
    { label: 'Macro ECE', value: '0.0129', baseline: '0.0353', delta: '-0.0224', color: '#22c55e' },
  ];

  const defaultImportance = [
    { name: 'Credit Utilization', val: 10.71 },
    { name: 'Open Credit Lines', val: 9.31 },
    { name: 'Late Payment Severity', val: 8.85 },
    { name: 'Loan Term', val: 8.65 },
    { name: 'Loan Purpose', val: 7.18 },
    { name: 'Employment History', val: 7.08 },
  ];

  const currentImportance = importanceData || defaultImportance;
  const maxImp = Math.max(...currentImportance.map(i => i.val), 1);

  return (
    <div className="analytics-page">
      {isMock && (
        <div style={{
          background: '#FEF9C3',
          border: '1px solid #FEF08A',
          color: '#854D0E',
          padding: '12px 20px',
          borderRadius: '8px',
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          fontSize: '14px',
          fontWeight: '500'
        }}>
          <span style={{ fontSize: '18px' }}>⚠️</span>
          Live model unavailable — showing demo data
        </div>
      )}

      {/* ── Canonical Header ─────────────────────────────── */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.95))',
        border: '1px solid rgba(59, 130, 246, 0.3)',
        borderRadius: '16px',
        padding: '24px',
        marginBottom: '28px',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.25)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <span style={{
                background: 'rgba(34, 197, 94, 0.15)',
                color: '#22c55e',
                border: '1px solid rgba(34, 197, 94, 0.4)',
                padding: '4px 12px',
                borderRadius: '999px',
                fontSize: '12px',
                fontWeight: '700',
                letterSpacing: '0.05em'
              }}>
                CHAMPION / PRODUCTION
              </span>
              <span style={{
                background: 'rgba(59, 130, 246, 0.15)',
                color: '#60a5fa',
                border: '1px solid rgba(59, 130, 246, 0.4)',
                padding: '4px 12px',
                borderRadius: '999px',
                fontSize: '12px',
                fontWeight: '600'
              }}>
                N = 21,398 Test Set
              </span>
            </div>
            <h1 style={{ fontSize: '26px', fontWeight: '800', color: '#f8fafc', margin: '0 0 6px 0' }}>
              LightGBM v3.1 — CANONICAL
            </h1>
            <p style={{ color: '#94a3b8', fontSize: '14px', margin: 0, lineHeight: 1.5 }}>
              Calibrated with <strong>oof-iso-v3.1</strong> (Out-of-Fold Isotonic) · Policy Engine: <strong>v3.1-policy-v1</strong>
            </p>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Cost / Applicant</div>
            <div style={{ fontSize: '24px', fontWeight: '800', fontFamily: 'monospace', color: '#38bdf8' }}>$189.84</div>
            <div style={{ fontSize: '12px', color: '#22c55e' }}>-$0.97 vs v3.0 Baseline</div>
          </div>
        </div>
      </div>

      {/* ── Tab Switcher ─────────────────────────────────── */}
      <div style={{
        display: 'flex', gap: '4px', marginBottom: '24px',
        background: '#1e293b', padding: '4px', borderRadius: '10px', width: 'fit-content',
        border: '1px solid rgba(255, 255, 255, 0.05)'
      }}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '8px 20px',
              borderRadius: '8px',
              border: 'none',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              transition: 'all 0.2s',
              background: activeTab === tab.id ? '#334155' : 'transparent',
              color: activeTab === tab.id ? '#f8fafc' : '#94a3b8',
              boxShadow: activeTab === tab.id ? '0 1px 3px rgba(0,0,0,0.3)' : 'none',
            }}
          >
            <span>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* ── Models Tab ───────────────────────────────────── */}
      {activeTab === 'models' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px' }}>
            {/* Canonical Benchmark Metrics Card */}
            <Card elevated>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#f8fafc' }}>Canonical Benchmark (v3.1)</h3>
                <span style={{ fontSize: '11px', color: '#64748b', fontFamily: 'monospace' }}>test.csv (N=21,398)</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {canonicalMetrics.map((m) => (
                  <div key={m.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '8px', borderBottom: '1px solid rgba(255, 255, 255, 0.05)'}}>
                    <div>
                      <span style={{ color: '#cbd5e1', fontSize: '13px', fontWeight: '500' }}>{m.label}</span>
                      <span style={{ color: '#64748b', fontSize: '11px', marginLeft: '8px' }}>(Base: {m.baseline})</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="mono" style={{ fontSize: '14px', fontWeight: '700', color: '#f8fafc' }}>{m.value}</span>
                      <span style={{ fontSize: '11px', color: m.color, fontFamily: 'monospace', fontWeight: '600' }}>{m.delta}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            {/* Frozen Decision Policy Card */}
            <Card elevated>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#f8fafc' }}>Frozen Decision Policy</h3>
                <span style={{ fontSize: '11px', color: '#38bdf8', fontWeight: '600' }}>v3.1-policy-v1</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <div style={{ background: 'rgba(34, 197, 94, 0.08)', border: '1px solid rgba(34, 197, 94, 0.25)', borderRadius: '8px', padding: '10px 14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: '12px', fontWeight: '700', color: '#4ade80' }}>APPROVE</div>
                    <div style={{ fontSize: '11px', color: '#94a3b8' }}>p ≤ 0.045 (71.1% auto-approved)</div>
                  </div>
                  <span style={{ fontSize: '12px', fontFamily: 'monospace', color: '#4ade80', fontWeight: '700' }}>71.05%</span>
                </div>

                <div style={{ background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.25)', borderRadius: '8px', padding: '10px 14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: '12px', fontWeight: '700', color: '#fbbf24' }}>MANUAL REVIEW</div>
                    <div style={{ fontSize: '11px', color: '#94a3b8' }}>0.045 &lt; p &lt; 0.335 (underwriting triage)</div>
                  </div>
                  <span style={{ fontSize: '12px', fontFamily: 'monospace', color: '#fbbf24', fontWeight: '700' }}>24.09%</span>
                </div>

                <div style={{ background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.25)', borderRadius: '8px', padding: '10px 14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: '12px', fontWeight: '700', color: '#f87171' }}>REJECT</div>
                    <div style={{ fontSize: '11px', color: '#94a3b8' }}>p ≥ 0.335 (high-risk threshold)</div>
                  </div>
                  <span style={{ fontSize: '12px', fontFamily: 'monospace', color: '#f87171', fontWeight: '700' }}>4.86%</span>
                </div>
              </div>
            </Card>

            {/* Feature Importance Card */}
            <Card elevated>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#f8fafc' }}>Feature Attribution</h3>
                <span style={{ fontSize: '11px', color: '#64748b' }}>LightGBM split gain</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {currentImportance.map((feat) => {
                  const pctWidth = (feat.val / maxImp) * 100;
                  return (
                    <div key={feat.name} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                        <span style={{ color: '#cbd5e1' }}>{feat.name}</span>
                        <span style={{ color: '#f59e0b', fontFamily: 'monospace', fontWeight: '600' }}>{feat.val.toFixed(1)}%</span>
                      </div>
                      <div style={{ height: '6px', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${pctWidth}%`, height: '100%', background: 'linear-gradient(90deg, #f59e0b, #d97706)', borderRadius: '3px' }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>
          
          <div style={{ marginTop: '24px' }}>
            <ErrorBoundary>
              <Suspense fallback={<ComponentLoader label="Loading Model Comparison..." />}>
                <ModelComparison />
              </Suspense>
            </ErrorBoundary>
          </div>

          <div style={{ marginTop: '24px' }}>
            <ErrorBoundary>
              <Suspense fallback={<ComponentLoader label="Loading 3D Monte Carlo Stress Simulation..." />}>
                <MonteCarlo3D />
              </Suspense>
            </ErrorBoundary>
          </div>
        </>
      )}

      {/* ── Fairness Tab ──────────────────────────────────── */}
      {activeTab === 'fairness' && (
        <ErrorBoundary>
          <Suspense fallback={<ComponentLoader label="Loading Fair Lending Analytics..." />}>
            <FairnessTab />
          </Suspense>
        </ErrorBoundary>
      )}
    </div>
  );
}
