import React, { useState, useEffect, useCallback } from "react";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// =============================================================================
// Styles - Matching existing dark theme
// =============================================================================

const S = {
  container: {
    width: "100%",
    maxWidth: 1200,
    margin: "0 auto",
    padding: "1.5rem",
    fontFamily: "'IBM Plex Mono', monospace",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "1.5rem",
    flexWrap: "wrap",
    gap: 12,
  },
  title: {
    fontSize: "1.3rem",
    fontWeight: 700,
    color: "#f1f5f9",
    margin: 0,
  },
  subtitle: {
    fontSize: "0.8rem",
    color: "#64748b",
    margin: "0.25rem 0 0",
  },
  scoreCard: {
    background: "linear-gradient(135deg, #1e293b, #0f172a)",
    border: "1px solid #334155",
    borderRadius: 12,
    padding: "1.5rem",
    marginBottom: "1.5rem",
  },
  scoreGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: "1.5rem",
  },
  scoreItem: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 8,
  },
  scoreValue: {
    fontSize: "2.5rem",
    fontWeight: 800,
    fontFamily: "monospace",
  },
  scoreLabel: {
    fontSize: "0.75rem",
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  gauge: {
    width: "100%",
    height: 8,
    background: "#334155",
    borderRadius: 4,
    overflow: "hidden",
    marginTop: 8,
  },
  gaugeFill: {
    height: "100%",
    transition: "width 0.5s ease",
  },
  warningBanner: {
    background: "rgba(248, 113, 113, 0.1)",
    border: "1px solid #f87171",
    borderRadius: 8,
    padding: "1rem 1.25rem",
    marginBottom: "1.5rem",
  },
  warningTitle: {
    fontSize: "0.85rem",
    fontWeight: 700,
    color: "#f87171",
    marginBottom: "0.5rem",
    display: "flex",
    alignItems: "center",
    gap: 6,
  },
  warningText: {
    fontSize: "0.8rem",
    color: "#fca5a5",
    lineHeight: 1.6,
  },
  card: {
    background: "#1e293b",
    border: "1px solid #334155",
    borderRadius: 12,
    padding: "1.25rem",
    marginBottom: "1.5rem",
  },
  cardTitle: {
    fontSize: "0.95rem",
    fontWeight: 600,
    color: "#f1f5f9",
    marginBottom: "1rem",
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: "0.8rem",
  },
  th: {
    textAlign: "left",
    padding: "0.75rem 0.5rem",
    color: "#64748b",
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    borderBottom: "1px solid #334155",
  },
  td: {
    padding: "0.75rem 0.5rem",
    color: "#e2e8f0",
    borderBottom: "1px solid #1e293b",
  },
  badge: {
    display: "inline-flex",
    alignItems: "center",
    padding: "0.2rem 0.5rem",
    borderRadius: 4,
    fontSize: "0.7rem",
    fontWeight: 700,
    fontFamily: "monospace",
  },
  badgeGreen: {
    background: "rgba(34, 211, 165, 0.15)",
    color: "#22d3a5",
  },
  badgeYellow: {
    background: "rgba(251, 191, 36, 0.15)",
    color: "#fbbf24",
  },
  badgeRed: {
    background: "rgba(248, 113, 113, 0.15)",
    color: "#f87171",
  },
  barChart: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  barRow: {
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  barLabel: {
    width: 120,
    fontSize: "0.75rem",
    color: "#94a3b8",
  },
  barContainer: {
    flex: 1,
    height: 24,
    background: "#334155",
    borderRadius: 4,
    overflow: "hidden",
  },
  bar: {
    height: "100%",
    transition: "width 0.5s ease",
  },
  barValue: {
    width: 60,
    fontSize: "0.75rem",
    color: "#e2e8f0",
    fontFamily: "monospace",
    textAlign: "right",
  },
  button: {
    background: "linear-gradient(135deg, #60a5fa, #3b82f6)",
    border: "none",
    borderRadius: 8,
    padding: "0.75rem 1.5rem",
    color: "#fff",
    fontFamily: "monospace",
    fontSize: "0.85rem",
    fontWeight: 700,
    cursor: "pointer",
    transition: "all 0.2s",
  },
  buttonSecondary: {
    background: "transparent",
    border: "1px solid #334155",
    borderRadius: 8,
    padding: "0.75rem 1.5rem",
    color: "#64748b",
    fontFamily: "monospace",
    fontSize: "0.8rem",
    cursor: "pointer",
  },
  buttonGroup: {
    display: "flex",
    gap: "0.75rem",
    marginTop: "1rem",
  },
  mitigationCard: {
    background: "rgba(96, 165, 250, 0.05)",
    border: "1px solid rgba(96, 165, 250, 0.3)",
    borderRadius: 8,
    padding: "1rem",
    marginTop: "0.75rem",
  },
  mitigationTitle: {
    fontSize: "0.8rem",
    fontWeight: 600,
    color: "#60a5fa",
    marginBottom: "0.5rem",
  },
  mitigationText: {
    fontSize: "0.75rem",
    color: "#94a3b8",
    lineHeight: 1.6,
  },
  spinner: {
    width: 24,
    height: 24,
    border: "3px solid #334155",
    borderTopColor: "#60a5fa",
    borderRadius: "50%",
    animation: "spin 0.7s linear infinite",
  },
  emptyState: {
    textAlign: "center",
    padding: "3rem 1rem",
    color: "#64748b",
  },
};

// =============================================================================
// Helper Functions
// =============================================================================

const getScoreColor = (score) => {
  if (score >= 80) return "#22d3a5";
  if (score >= 60) return "#fbbf24";
  return "#f87171";
};

const getBadgeStyle = (value, threshold = 0.04) => {
  if (value <= threshold) return S.badgeGreen;
  if (value <= threshold * 2) return S.badgeYellow;
  return S.badgeRed;
};

const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;

// =============================================================================
// Sub-Components
// =============================================================================

function ScoreGauge({ score, label }) {
  const color = getScoreColor(score);

  return (
    <div style={S.scoreItem}>
      <div style={{ ...S.scoreValue, color }}>{score.toFixed(0)}</div>
      <div style={S.scoreLabel}>{label}</div>
      <div style={S.gauge}>
        <div
          style={{
            ...S.gaugeFill,
            width: `${score}%`,
            background: color,
          }}
        />
      </div>
    </div>
  );
}

function ViolationBanner({ violations }) {
  const critical = violations.filter((v) => v.severity === "high");
  const warnings = violations.filter((v) => v.severity === "medium");

  if (violations.length === 0) return null;

  return (
    <div style={S.warningBanner}>
      <div style={S.warningTitle}>
        ⚠ {critical.length > 0 ? "ECOA/FAIR HOUSING VIOLATION DETECTED" : "Fairness Concerns"}
      </div>
      <div style={S.warningText}>
        {critical.length > 0 && (
          <div>
            <strong>{critical.length} critical violation(s):</strong> These disparities
            exceed regulatory thresholds and require immediate remediation.
          </div>
        )}
        {warnings.length > 0 && (
          <div style={{ marginTop: "0.5rem" }}>
            <strong>{warnings.length} warning(s):</strong> Monitor these disparities
            closely as they approach regulatory limits.
          </div>
        )}
      </div>
    </div>
  );
}

function GroupMetricsTable({ groups }) {
  if (!groups || groups.length === 0) return null;

  const maxRate = Math.max(...groups.map((g) => g.approval_rate));
  const referenceGroup = groups.find((g) => g.approval_rate === maxRate)?.group_name;

  return (
    <table style={S.table}>
      <thead>
        <tr>
          <th style={S.th}>Group</th>
          <th style={S.th}>Samples</th>
          <th style={S.th}>Approval Rate</th>
          <th style={S.th}>Disparity</th>
          <th style={S.th}>FPR</th>
          <th style={S.th}>FNR</th>
        </tr>
      </thead>
      <tbody>
        {groups.map((group, i) => {
          const disparity = maxRate - group.approval_rate;
          const badgeStyle = getBadgeStyle(disparity);

          return (
            <tr key={i}>
              <td style={S.td}>{group.group_name}</td>
              <td style={S.td}>{group.sample_size.toLocaleString()}</td>
              <td style={S.td}>{formatPercent(group.approval_rate)}</td>
              <td style={S.td}>
                <span style={{ ...S.badge, ...badgeStyle }}>
                  {disparity >= 0 ? "+" : ""}{formatPercent(disparity)}
                </span>
              </td>
              <td style={S.td}>{formatPercent(group.false_positive_rate)}</td>
              <td style={S.td}>{formatPercent(group.false_negative_rate)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function ApprovalRateChart({ groups }) {
  if (!groups || groups.length === 0) return null;

  const maxRate = Math.max(...groups.map((g) => g.approval_rate));

  return (
    <div style={S.barChart}>
      {groups.map((group, i) => (
        <div key={i} style={S.barRow}>
          <div style={S.barLabel}>{group.group_name.split(": ")[1] || group.group_name}</div>
          <div style={S.barContainer}>
            <div
              style={{
                ...S.bar,
                width: `${(group.approval_rate / maxRate) * 100}%`,
                background: group.approval_rate >= maxRate * 0.96
                  ? "#22d3a5"
                  : group.approval_rate >= maxRate * 0.80
                  ? "#fbbf24"
                  : "#f87171",
              }}
            />
          </div>
          <div style={S.barValue}>{formatPercent(group.approval_rate)}</div>
        </div>
      ))}
    </div>
  );
}

function MitigationsPanel({ mitigations }) {
  if (!mitigations || mitigations.length === 0) return null;

  return (
    <div>
      {mitigations.map((m, i) => (
        <div key={i} style={S.mitigationCard}>
          <div style={S.mitigationTitle}>
            {m.type === "none" ? "✓ No Action Required" : `Recommendation: ${m.recommendation}`}
          </div>
          {m.description && (
            <div style={S.mitigationText}>{m.description}</div>
          )}
          {m.implementation && (
            <div style={{ ...S.mitigationText, marginTop: "0.5rem", color: "#60a5fa" }}>
              Implementation: {m.implementation}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export default function FairnessAudit() {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  // Load report on mount
  useEffect(() => {
    loadReport();
  }, []);

  const loadReport = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/api/fairness/report`);

      if (!response.ok) {
        throw new Error(`Failed to load report (${response.status})`);
      }

      const data = await response.json();
      setReport(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Run new audit
  const runAudit = async () => {
    setRunning(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/api/fairness/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_name: "mortgage_model",
          sample_size: 10000,
        }),
      });

      if (!response.ok) {
        throw new Error(`Audit failed (${response.status})`);
      }

      const result = await response.json();
      console.log("Audit started:", result);

      // Poll for completion
      await pollForCompletion();
    } catch (err) {
      setError(err.message);
      setRunning(false);
    }
  };

  // Poll for audit completion
  const pollForCompletion = async () => {
    const maxAttempts = 20;
    const interval = 2000;

    for (let i = 0; i < maxAttempts; i++) {
      await new Promise((resolve) => setTimeout(resolve, interval));

      try {
        const response = await fetch(`${API_URL}/api/fairness/status`);
        const status = await response.json();

        if (status.has_report && status.last_audit) {
          await loadReport();
          setRunning(false);
          return;
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }

    setRunning(false);
    setError("Audit timed out. Please try again.");
  };

  // Export report
  const exportReport = async () => {
    try {
      const response = await fetch(`${API_URL}/api/fairness/export`, {
        method: "POST",
      });

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "fairness_report.json";
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export failed:", err);
    }
  };

  // Render loading state
  if (loading) {
    return (
      <div style={{ ...S.container, display: "flex", alignItems: "center", justifyContent: "center", minHeight: 400 }}>
        <div style={S.spinner} />
        <p style={{ color: "#64748b", marginLeft: 12 }}>Loading fairness report...</p>
      </div>
    );
  }

  // Render empty state
  if (!report || report.violations?.length === 0 && report.group_metrics?.length === 0) {
    return (
      <div style={S.container}>
        <div style={S.header}>
          <div>
            <h1 style={S.title}>Fairness Audit</h1>
            <p style={S.subtitle}>Bias detection and regulatory compliance</p>
          </div>
          <button onClick={runAudit} disabled={running} style={S.button}>
            {running ? "Running..." : "Run Audit"}
          </button>
        </div>

        <div style={S.emptyState}>
          <p style={{ fontSize: "1.2rem", marginBottom: "0.5rem" }}>No audit data available</p>
          <p>Run a fairness audit to evaluate model bias across demographic groups.</p>
          {error && <p style={{ color: "#f87171", marginTop: "1rem" }}>Error: {error}</p>}
        </div>
      </div>
    );
  }

  // Render full report
  return (
    <div style={S.container}>
      {/* Header */}
      <div style={S.header}>
        <div>
          <h1 style={S.title}>Fairness Audit</h1>
          <p style={S.subtitle}>
            Model: {report.model_name} · {new Date(report.audit_timestamp).toLocaleDateString()}
          </p>
        </div>
        <div style={S.buttonGroup}>
          <button onClick={runAudit} disabled={running} style={{ ...S.button, opacity: running ? 0.7 : 1 }}>
            {running ? "Running..." : "Re-run Audit"}
          </button>
          <button onClick={exportReport} style={S.buttonSecondary}>
            Export
          </button>
        </div>
      </div>

      {/* Warning banner if violations */}
      {report.violations && report.violations.length > 0 && (
        <ViolationBanner violations={report.violations} />
      )}

      {/* Score card */}
      <div style={S.scoreCard}>
        <div style={S.scoreGrid}>
          <ScoreGauge score={report.overall_fairness_score} label="Fairness Score" />
          <ScoreGauge score={100 - (report.violations?.length || 0) * 10} label="Compliance" />
          <div style={S.scoreItem}>
            <div style={{ ...S.scoreValue, color: "#e2e8f0" }}>
              {report.total_samples?.toLocaleString()}
            </div>
            <div style={S.scoreLabel}>Samples Audited</div>
          </div>
        </div>
      </div>

      {/* Violations list */}
      {report.violations && report.violations.length > 0 && (
        <div style={S.card}>
          <div style={S.cardTitle}>⚠ Detected Violations</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {report.violations.map((v, i) => (
              <div
                key={i}
                style={{
                  padding: "0.75rem",
                  background: "rgba(248, 113, 113, 0.1)",
                  borderRadius: 6,
                  border: `1px solid ${v.severity === "high" ? "#f87171" : "#fbbf24"}`,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ color: v.severity === "high" ? "#f87171" : "#fbbf24", fontWeight: 700 }}>
                    [{v.severity.toUpperCase()}] {v.type}
                  </span>
                  <span style={{ fontFamily: "monospace", color: "#94a3b8" }}>
                    {v.metric}: {v.value.toFixed(4)}
                  </span>
                </div>
                <div style={{ color: "#fca5a5", fontSize: "0.8rem", marginTop: 4 }}>
                  {v.description}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Group metrics table */}
      <div style={S.card}>
        <div style={S.cardTitle}>📊 Approval Rates by Group</div>
        <GroupMetricsTable groups={report.group_metrics} />
      </div>

      {/* Bar chart */}
      <div style={S.card}>
        <div style={S.cardTitle}>📈 Visual Comparison</div>
        <ApprovalRateChart groups={report.group_metrics} />
      </div>

      {/* Mitigations */}
      <div style={S.card}>
        <div style={S.cardTitle}>💡 Recommendations</div>
        <MitigationsPanel mitigations={report.mitigations} />
      </div>

      {/* Summary */}
      {report.summary && (
        <div style={{ ...S.card, background: "rgba(96, 165, 250, 0.05)" }}>
          <div style={S.cardTitle}>📋 Summary</div>
          <p style={{ color: "#94a3b8", fontSize: "0.85rem", lineHeight: 1.6 }}>
            {report.summary}
          </p>
        </div>
      )}
    </div>
  );
}
