import React, { useState, useEffect, useRef } from "react";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8001/ws/live";
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8001";

const S = {
  container: {
    background: "#0f172a",
    borderRadius: 12,
    padding: "1.5rem",
    border: "1px solid #334155",
    color: "#e2e8f0",
    fontFamily: "'IBM Plex Mono', monospace",
    height: "500px",
    display: "flex",
    flexDirection: "column",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "1rem",
    paddingBottom: "0.75rem",
    borderBottom: "1px solid #334155",
  },
  title: {
    fontSize: "1.1rem",
    fontWeight: 700,
    margin: 0,
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
  },
  liveIndicator: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#22d3a5",
    animation: "pulse 1.5s ease-in-out infinite",
  },
  status: {
    fontSize: "0.75rem",
    padding: "0.25rem 0.5rem",
    borderRadius: 4,
    fontWeight: 600,
  },
  connected: {
    background: "#22d3a520",
    color: "#22d3a5",
    border: "1px solid #22d3a560",
  },
  disconnected: {
    background: "#f8717120",
    color: "#f87171",
    border: "1px solid #f8717160",
  },
  feed: {
    flex: 1,
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
  },
  decision: {
    padding: "0.75rem",
    borderRadius: 8,
    background: "#1e293b",
    border: "1px solid #334155",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    transition: "all 0.3s ease",
    animation: "slideIn 0.3s ease",
  },
  decisionInfo: {
    display: "flex",
    flexDirection: "column",
    gap: "0.25rem",
  },
  decisionTime: {
    fontSize: "0.7rem",
    color: "#64748b",
  },
  decisionAmount: {
    fontSize: "0.85rem",
    fontWeight: 600,
    color: "#e2e8f0",
  },
  decisionMeta: {
    fontSize: "0.7rem",
    color: "#94a3b8",
  },
  decisionBadge: {
    padding: "0.35rem 0.75rem",
    borderRadius: 6,
    fontSize: "0.75rem",
    fontWeight: 700,
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  approved: {
    background: "#22d3a520",
    color: "#22d3a5",
    border: "1px solid #22d3a560",
  },
  rejected: {
    background: "#f8717120",
    color: "#f87171",
    border: "1px solid #f8717160",
  },
  review: {
    background: "#fbbf2420",
    color: "#fbbf24",
    border: "1px solid #fbbf2460",
  },
  empty: {
    textAlign: "center",
    padding: "3rem",
    color: "#64748b",
    fontSize: "0.9rem",
  },
  stats: {
    display: "flex",
    gap: "1rem",
    marginTop: "1rem",
    paddingTop: "1rem",
    borderTop: "1px solid #334155",
  },
  stat: {
    flex: 1,
    textAlign: "center",
  },
  statValue: {
    fontSize: "1.25rem",
    fontWeight: 700,
  },
  statLabel: {
    fontSize: "0.7rem",
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
};

export default function LiveDecisionFeed() {
  const [decisions, setDecisions] = useState([]);
  const [connected, setConnected] = useState(false);
  const [stats, setStats] = useState({ total: 0, approved: 0, rejected: 0 });
  const wsRef = useRef(null);
  const feedRef = useRef(null);
  const reconnectTimeout = useRef(null);

  const connect = () => {
    try {
      wsRef.current = new WebSocket(WS_URL);

      wsRef.current.onopen = () => {
        console.log("WebSocket connected");
        setConnected(true);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "prediction") {
            const newDecision = {
              id: Date.now(),
              timestamp: new Date(),
              decision: data.result.decision || data.result.data?.decision,
              probability: data.result.approval_probability || data.result.data?.approval_probability,
              amount: data.result.loan_amount,
              creditScore: data.result.credit_score,
              user: data.user,
            };

            setDecisions((prev) => [newDecision, ...prev].slice(0, 50));
            setStats((prev) => ({
              total: prev.total + 1,
              approved: prev.approved + (newDecision.decision === "APPROVED" ? 1 : 0),
              rejected: prev.rejected + (newDecision.decision === "REJECTED" ? 1 : 0),
            }));
          }
        } catch (err) {
          console.error("WebSocket message error:", err);
        }
      };

      wsRef.current.onclose = () => {
        console.log("WebSocket disconnected");
        setConnected(false);
        reconnectTimeout.current = setTimeout(connect, 5000);
      };

      wsRef.current.onerror = (error) => {
        console.error("WebSocket error:", error);
        setConnected(false);
      };
    } catch (err) {
      console.error("WebSocket connection error:", err);
      setConnected(false);
    }
  };

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
    };
  }, []);

  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = 0;
    }
  }, [decisions]);

  const getDecisionStyle = (decision) => {
    switch (decision?.toUpperCase()) {
      case "APPROVED":
        return S.approved;
      case "REJECTED":
        return S.rejected;
      default:
        return S.review;
    }
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  };

  return (
    <div style={S.container}>
      <div style={S.header}>
        <h2 style={S.title}>
          {connected && <span style={S.liveIndicator} />}
          Live Decision Feed
        </h2>
        <span
          style={{
            ...S.status,
            ...(connected ? S.connected : S.disconnected),
          }}
        >
          {connected ? "LIVE" : "OFFLINE"}
        </span>
      </div>

      <div style={S.feed} ref={feedRef}>
        {decisions.length === 0 ? (
          <div style={S.empty}>
            {connected
              ? "Waiting for decisions..."
              : "Connecting to server..."}
          </div>
        ) : (
          decisions.map((decision) => (
            <div key={decision.id} style={S.decision}>
              <div style={S.decisionInfo}>
                <span style={S.decisionTime}>
                  {formatTime(decision.timestamp)}
                </span>
                <span style={S.decisionAmount}>
                  ${decision.amount?.toLocaleString() || "N/A"}
                </span>
                <span style={S.decisionMeta}>
                  CS: {decision.creditScore} • User: {decision.user}
                </span>
              </div>
              <span
                style={{
                  ...S.decisionBadge,
                  ...getDecisionStyle(decision.decision),
                }}
              >
                {decision.decision}
              </span>
            </div>
          ))
        )}
      </div>

      <div style={S.stats}>
        <div style={S.stat}>
          <div style={{ ...S.statValue, color: "#e2e8f0" }}>{stats.total}</div>
          <div style={S.statLabel}>Total</div>
        </div>
        <div style={S.stat}>
          <div style={{ ...S.statValue, color: "#22d3a5" }}>{stats.approved}</div>
          <div style={S.statLabel}>Approved</div>
        </div>
        <div style={S.stat}>
          <div style={{ ...S.statValue, color: "#f87171" }}>{stats.rejected}</div>
          <div style={S.statLabel}>Rejected</div>
        </div>
        <div style={S.stat}>
          <div style={{ ...S.statValue, color: "#fbbf24" }}>
            {stats.total > 0 ? ((stats.approved / stats.total) * 100).toFixed(1) : 0}%
          </div>
          <div style={S.statLabel}>Approval Rate</div>
        </div>
      </div>

      <style>{
        `
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `</style>
    </div>
  );
}
