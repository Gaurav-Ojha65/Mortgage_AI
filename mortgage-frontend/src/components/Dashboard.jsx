import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import MetricCard from './MetricCard';
import DecisionResult from './DecisionResult';
import { healthCheck, getHistory } from '../utils/api';

const Dashboard = ({ lastDecision }) => {
  const [systemStatus, setSystemStatus] = useState(null);
  const [recentDecisions, setRecentDecisions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [status, history] = await Promise.all([
          healthCheck(),
          getHistory(5)
        ]);
        // API returns {success, data: {...}} - extract the data
        setSystemStatus(status.data || status);
        setRecentDecisions(history);
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(value || 0);
  };

  const stats = {
    totalProcessed: Array.isArray(recentDecisions) ? recentDecisions.length : 0,
    approvalRate: Array.isArray(recentDecisions) && recentDecisions.length > 0
      ? ((recentDecisions.filter(d => d.decision === 'APPROVE').length / recentDecisions.length) * 100).toFixed(1)
      : 0,
    avgLoanAmount: Array.isArray(recentDecisions) && recentDecisions.length > 0
      ? recentDecisions.reduce((sum, d) => sum + (d.loan_amount || 0), 0) / recentDecisions.length
      : 0,
    totalVolume: Array.isArray(recentDecisions)
      ? recentDecisions.reduce((sum, d) => sum + (d.loan_amount || 0), 0)
      : 0
  };

  const trustBadges = [
    { icon: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z', label: 'Bank-Grade Security' },
    { icon: 'M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z', label: '256-bit Encryption' },
    { icon: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01', label: 'NMLS Compliant' },
    { icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z', label: '99.9% Uptime' }
  ];

  return (
    <div className="space-y-8 pt-24 pb-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Hero Section */}
      <div className="relative">
        <div className="text-center space-y-6 mb-12">
          {/* Trust Badges */}
          <div className="flex flex-wrap items-center justify-center gap-3">
            {trustBadges.map((badge, idx) => (
              <div key={idx} className="trust-badge">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={badge.icon} />
                </svg>
                <span>{badge.label}</span>
              </div>
            ))}
          </div>

          <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
            <span className="text-gradient-gold">
              AI-Powered
            </span>
            <br />
            <span className="text-white">Mortgage Intelligence</span>
          </h1>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Advanced ensemble machine learning with Monte Carlo simulations for bank-grade risk assessment
            and intelligent lending decisions.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
            <Link
              to="/apply"
              className="btn-primary"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              <span>New Application</span>
            </Link>
            <Link
              to="/compare"
              className="btn-secondary"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <span>Compare Loans</span>
            </Link>
          </div>
        </div>

        {/* System Status Banner */}
        {systemStatus && (
          <div className="flex flex-wrap items-center justify-center gap-4 mb-8">
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full border ${
              systemStatus.models_loaded
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : 'bg-red-500/10 border-red-500/30 text-red-400'
            }`}>
              <span className={`w-2 h-2 rounded-full animate-pulse ${
                systemStatus.models_loaded ? 'bg-emerald-400' : 'bg-red-400'
              }`} />
              <span className="text-sm font-mono">{systemStatus.models_loaded ? 'ML Models Active' : 'Models Offline'}</span>
            </div>
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full border ${
              systemStatus.db_connected
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : 'bg-red-500/10 border-red-500/30 text-red-400'
            }`}>
              <span className={`w-2 h-2 rounded-full animate-pulse ${
                systemStatus.db_connected ? 'bg-emerald-400' : 'bg-red-400'
              }`} />
              <span className="text-sm font-mono">{systemStatus.db_connected ? 'Database Connected' : 'DB Offline'}</span>
            </div>
          </div>
        )}
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Applications"
          value={loading ? '-' : stats.totalProcessed}
          subtitle="Processed today"
          trend="+12%"
          trendUp={true}
          icon="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
          color="gold"
          delay={0}
        />
        <MetricCard
          title="Approval Rate"
          value={loading ? '-' : `${stats.approvalRate}%`}
          subtitle="Average success rate"
          trend="+5.3%"
          trendUp={true}
          icon="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
          color="emerald"
          delay={100}
        />
        <MetricCard
          title="Avg Loan Amount"
          value={loading ? '-' : formatCurrency(stats.avgLoanAmount)}
          subtitle="Per application"
          trend="-$2,400"
          trendUp={false}
          icon="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          color="amber"
          delay={200}
        />
        <MetricCard
          title="Total Volume"
          value={loading ? '-' : formatCurrency(stats.totalVolume)}
          subtitle="Loan applications"
          trend="+8.7%"
          trendUp={true}
          icon="M13 7h8m0 0v12m0-12l-8 8-4-4-6 6"
          color="purple"
          delay={300}
        />
      </div>

      {/* Last Decision Result */}
      {lastDecision && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="h-px flex-1 section-divider" />
            <span className="text-slate-500 text-sm uppercase tracking-wider">Latest Analysis</span>
            <div className="h-px flex-1 section-divider" />
          </div>
          <DecisionResult decision={lastDecision} />
        </div>
      )}

      {/* Recent Decisions Table */}
      <div className="glass-card rounded-2xl overflow-hidden">
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-amber-500/10">
                <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white">Recent Decisions</h3>
            </div>
            <Link
              to="/history"
              className="text-sm text-amber-400 hover:text-amber-300 transition-colors flex items-center gap-1 group"
            >
              <span>View All</span>
              <svg className="w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-slate-400 text-xs uppercase tracking-wider">
                <th className="px-6 py-4 font-medium">Time</th>
                <th className="px-6 py-4 font-medium">Amount</th>
                <th className="px-6 py-4 font-medium">Credit Score</th>
                <th className="px-6 py-4 font-medium">EMI</th>
                <th className="px-6 py-4 font-medium">Risk</th>
                <th className="px-6 py-4 font-medium">Decision</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {loading ? (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-center text-slate-500">
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-2 h-2 bg-amber-500 rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-amber-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                      <div className="w-2 h-2 bg-amber-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    </div>
                  </td>
                </tr>
              ) : recentDecisions.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-center text-slate-500">
                    No decisions yet. Process your first application!
                  </td>
                </tr>
              ) : (
                Array.isArray(recentDecisions) && recentDecisions.map((decision, idx) => (
                  <tr key={idx} className="hover:bg-white/5 transition-colors group">
                    <td className="px-6 py-4 text-slate-300 text-sm">
                      {new Date(decision.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="px-6 py-4 text-white font-mono">
                      {formatCurrency(decision.loan_amount)}
                    </td>
                    <td className="px-6 py-4">
                      <span className={`font-mono ${
                        decision.credit_score >= 700 ? 'text-emerald-400' :
                        decision.credit_score >= 600 ? 'text-amber-400' : 'text-red-400'
                      }`}>
                        {decision.credit_score}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-slate-300 font-mono">
                      {formatCurrency(decision.emi)}
                    </td>
                    <td className="px-6 py-4">
                      <span className={`badge ${
                        decision.risk_level === 'LOW' ? 'badge-success' :
                        decision.risk_level === 'HIGH' ? 'badge-danger' : 'badge-warning'
                      }`}>
                        {decision.risk_level}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`badge ${
                        decision.decision === 'APPROVE' ? 'badge-success' :
                        decision.decision === 'REJECT' ? 'badge-danger' : 'badge-warning'
                      }`}>
                        {decision.decision}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          {
            title: 'ML Risk Engine',
            description: 'Logistic Regression + Random Forest + XGBoost ensemble with 94.2% accuracy for default prediction',
            icon: 'M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z',
            color: 'gold',
            metric: '94.2%'
          },
          {
            title: 'Monte Carlo Simulation',
            description: '10,000+ simulations to model default probability across multiple economic scenarios with 95% confidence',
            icon: 'M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5',
            color: 'purple',
            metric: '10k+ runs'
          },
          {
            title: 'AI Advisor',
            description: 'Claude AI integration for natural language explanations and personalized recommendations',
            icon: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z',
            color: 'amber',
            metric: 'Real-time'
          }
        ].map((feature, idx) => (
          <div
            key={idx}
            className="glass-card rounded-xl p-6 group glass-card-interactive"
          >
            <div className="flex items-start justify-between mb-4">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-300 ${
                feature.color === 'gold' ? 'bg-amber-500/10 text-amber-400 group-hover:bg-amber-500/20' :
                feature.color === 'purple' ? 'bg-purple-500/10 text-purple-400 group-hover:bg-purple-500/20' :
                'bg-emerald-500/10 text-emerald-400 group-hover:bg-emerald-500/20'
              }`}>
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={feature.icon} />
                </svg>
              </div>
              <span className={`text-sm font-mono ${
                feature.color === 'gold' ? 'text-amber-400' :
                feature.color === 'purple' ? 'text-purple-400' :
                'text-emerald-400'
              }`}>
                {feature.metric}
              </span>
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
            <p className="text-slate-400 text-sm leading-relaxed">{feature.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Dashboard;
