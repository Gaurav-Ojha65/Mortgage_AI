import React, { useState, useEffect, useCallback } from 'react';
import { api } from '../api';

const RETENTION_OPTIONS = [
  { value: 30, label: '30 Days' },
  { value: 60, label: '60 Days' },
  { value: 90, label: '90 Days' },
  { value: 180, label: '180 Days' },
];

export default function AdminPage() {
  const [retentionDays, setRetentionDays] = useState(90);
  const [oldCount, setOldCount] = useState(null);
  const [loading, setLoading] = useState(false);
  const [purging, setPurging] = useState(false);
  const [purgeResult, setPurgeResult] = useState(null);
  const [error, setError] = useState(null);

  const fetchCount = useCallback(async (days) => {
    setLoading(true);
    setError(null);
    setPurgeResult(null);
    try {
      const res = await api.oldCount(days);
      setOldCount(res);
    } catch (err) {
      const msg = err.message || '';
      if (msg.includes('401') || msg.toLowerCase().includes('authentication') || msg.toLowerCase().includes('unauthorized')) {
        setError('Admin authentication required. Please log in with an admin account.');
      } else if (msg.includes('404') || msg.toLowerCase().includes('not found')) {
        setError('Endpoint not found. Ensure the backend is running with the latest code.');
      } else {
        setError(msg || 'Failed to fetch record count.');
      }
      setOldCount(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCount(retentionDays);
  }, [retentionDays, fetchCount]);

  const handlePurge = async () => {
    if (!window.confirm(
      `⚠️ CRITICAL ACTION: This will permanently delete ${oldCount?.record_count ?? '?'} record(s) older than ${retentionDays} days.\n\nThis action cannot be undone. Continue?`
    )) return;

    setPurging(true);
    setError(null);
    try {
      const res = await api.purge(retentionDays);
      setPurgeResult(res);
      fetchCount(retentionDays);
    } catch (err) {
      setError(err.message);
    } finally {
      setPurging(false);
    }
  };

  const count = oldCount?.record_count ?? 0;
  const hasRecords = oldCount && oldCount.record_count > 0;

  return (
    <div className="page-container">
      <div style={{ marginBottom: '40px' }}>
        <h1 style={{ fontSize: '32px', letterSpacing: '-0.5px', marginBottom: '8px' }}>
          System <span className="text-gold">Administration</span>
        </h1>
        <p className="text-secondary" style={{ fontSize: '16px' }}>
          Governance tools and automated data compliance controls.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '24px' }}>
        {/* ── Data Retention Panel ──────────────────────────── */}
        <section style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-lg)',
          overflow: 'hidden',
          boxShadow: 'var(--shadow-sm)',
        }}>
          <div style={{
            padding: '24px',
            borderBottom: '1px solid var(--border)',
            background: 'rgba(232, 160, 32, 0.03)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <div style={{
                width: '40px', height: '40px', borderRadius: '8px',
                background: 'rgba(232, 160, 32, 0.1)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--gold)'
              }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 8V20a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8"></path><path d="M1 3h22v5H1z"></path><path d="M10 12h4"></path></svg>
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 600 }}>Data Retention Policy</h3>
                <p style={{ margin: 0, fontSize: '13px' }} className="text-muted">Manage decision history cleanup and storage optimization.</p>
              </div>
            </div>
            
            <div style={{
              padding: '6px 12px', borderRadius: '20px',
              background: 'rgba(34, 197, 94, 0.1)', color: 'var(--success)',
              fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px'
            }}>
              System Active
            </div>
          </div>

          <div style={{ padding: '32px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px' }}>
              
              {/* Left Column: Configuration */}
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: 'var(--gold)', textTransform: 'uppercase', marginBottom: '16px', letterSpacing: '1px' }}>
                  Retention Threshold
                </label>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
                  {RETENTION_OPTIONS.map(opt => (
                    <button
                      key={opt.value}
                      onClick={() => setRetentionDays(opt.value)}
                      style={{
                        padding: '12px',
                        borderRadius: 'var(--radius-md)',
                        border: '1px solid',
                        borderColor: retentionDays === opt.value ? 'var(--gold)' : 'var(--border)',
                        background: retentionDays === opt.value ? 'rgba(232, 160, 32, 0.1)' : 'transparent',
                        color: retentionDays === opt.value ? 'var(--gold)' : 'var(--text-secondary)',
                        fontWeight: retentionDays === opt.value ? 700 : 500,
                        fontSize: '14px',
                        transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                        textAlign: 'center'
                      }}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>

                <div style={{
                  marginTop: '32px', padding: '16px', background: 'rgba(0,0,0,0.2)', 
                  borderRadius: 'var(--radius-md)', borderLeft: '3px solid var(--gold)'
                }}>
                  <p className="text-secondary" style={{ fontSize: '12px', lineHeight: 1.6, margin: 0 }}>
                    <span style={{ color: 'var(--gold)', fontWeight: 700 }}>POLICY NOTE:</span> Decision records older than the selected threshold are flagged for permanent removal. Purge operations are non-reversible and audited for compliance.
                  </p>
                </div>
              </div>

              {/* Right Column: Status & Action */}
              <div style={{ 
                background: 'rgba(255,255,255,0.02)', 
                borderRadius: 'var(--radius-lg)', 
                border: '1px dashed var(--border)',
                padding: '24px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '8px' }}>
                  Records exceeding {retentionDays} days
                </div>
                
                <div className="mono" style={{
                  fontSize: '64px', fontWeight: 800,
                  color: loading ? 'var(--text-muted)' : (hasRecords ? 'var(--danger)' : 'var(--success)'),
                  textShadow: hasRecords ? '0 0 20px rgba(239, 68, 68, 0.2)' : 'none',
                  lineHeight: 1
                }}>
                  {loading ? '...' : count}
                </div>

                <div style={{ marginTop: '24px', width: '100%' }}>
                  <button
                    onClick={handlePurge}
                    disabled={purging || !hasRecords}
                    style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px',
                      width: '100%', padding: '16px',
                      background: hasRecords ? 'var(--danger)' : 'rgba(255,255,255,0.05)',
                      color: hasRecords ? '#fff' : 'var(--text-muted)',
                      border: 'none', borderRadius: 'var(--radius-md)',
                      fontWeight: 700, fontSize: '15px',
                      cursor: (purging || !hasRecords) ? 'not-allowed' : 'pointer',
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                      boxShadow: hasRecords ? '0 4px 12px rgba(239, 68, 68, 0.3)' : 'none'
                    }}
                  >
                    {purging ? (
                      <span className="animate-pulse">PROCESS PURGE...</span>
                    ) : (
                      <>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                        EXECUTE PURGE
                      </>
                    )}
                  </button>
                  
                  {!hasRecords && !loading && (
                    <p style={{ marginTop: '12px', fontSize: '11px', color: 'var(--success)', fontWeight: 600 }}>
                      ✓ All records within compliance threshold
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div style={{
                marginTop: '24px', background: 'rgba(239, 68, 68, 0.1)', 
                border: '1px solid var(--danger)', color: 'var(--danger)',
                padding: '16px', borderRadius: 'var(--radius-md)', fontSize: '13px',
                display: 'flex', alignItems: 'center', gap: '12px'
              }}>
                <span style={{ fontSize: '18px' }}>⚠️</span> {error}
              </div>
            )}

            {/* Success Message */}
            {purgeResult && (
              <div style={{
                marginTop: '24px', background: 'rgba(34, 197, 94, 0.1)', 
                border: '1px solid var(--success)', color: 'var(--success)',
                padding: '16px', borderRadius: 'var(--radius-md)', fontSize: '13px',
                display: 'flex', alignItems: 'center', gap: '12px'
              }}>
                <span style={{ fontSize: '18px' }}>✅</span> {purgeResult.message}
              </div>
            )}
          </div>
        </section>

        {/* ── System Information Footer ──────────────────────────── */}
        <footer style={{ 
          display: 'flex', justifyContent: 'space-between', 
          padding: '0 8px', fontSize: '12px', color: 'var(--text-muted)' 
        }}>
          <div>Mortgage AI Core v2.0.0 — Security Tier: High</div>
          <div>Last System Audit: {new Date().toLocaleDateString()}</div>
        </footer>
      </div>
    </div>
  );
}
