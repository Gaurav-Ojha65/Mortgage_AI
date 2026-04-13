import React from 'react';
import GaugeChart from './GaugeChart';

const DecisionResult = ({ decision }) => {
  if (!decision) return null;

  const getDecisionConfig = (decisionType) => {
    switch (decisionType) {
      case 'APPROVE':
        return {
          color: 'emerald',
          gradient: 'from-emerald-500/20 to-emerald-600/10',
          border: 'border-emerald-500/50',
          icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
          label: 'APPROVED',
          description: 'Application meets all lending criteria',
          badge: 'badge-success'
        };
      case 'REJECT':
        return {
          color: 'red',
          gradient: 'from-red-500/20 to-red-600/10',
          border: 'border-red-500/50',
          icon: 'M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z',
          label: 'REJECTED',
          description: 'Application does not meet requirements',
          badge: 'badge-danger'
        };
      default:
        return {
          color: 'amber',
          gradient: 'from-amber-500/20 to-amber-600/10',
          border: 'border-amber-500/50',
          icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
          label: 'UNDER REVIEW',
          description: 'Additional review required',
          badge: 'badge-warning'
        };
    }
  };

  const getRiskConfig = (riskLevel) => {
    switch (riskLevel) {
      case 'LOW':
        return { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', score: 25 };
      case 'HIGH':
        return { color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/30', score: 85 };
      default:
        return { color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30', score: 50 };
    }
  };

  const decisionConfig = getDecisionConfig(decision.decision);
  const riskConfig = getRiskConfig(decision.risk_level);

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(value || 0);
  };

  const formatPercent = (value) => {
    return `${((value || 0) * 100).toFixed(1)}%`;
  };

  // Calculate approval confidence for gauge
  const confidenceValue = decision.approval_probability ? (decision.approval_probability * 100) : 50;

  return (
    <div className="space-y-6">
      {/* Main Decision Card */}
      <div className={`relative rounded-2xl p-8 bg-gradient-to-br ${decisionConfig.gradient} border ${decisionConfig.border} overflow-hidden`}>
        {/* Animated Background */}
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_30%,rgba(245,158,11,0.1),transparent_50%)]" />
        </div>

        <div className="relative z-10 flex flex-col lg:flex-row items-center lg:items-start gap-8">
          {/* Icon */}
          <div className={`w-20 h-20 rounded-2xl flex items-center justify-center bg-${decisionConfig.color}-500/20 border border-${decisionConfig.color}-500/50 shrink-0`}>
            <svg className={`w-10 h-10 text-${decisionConfig.color}-400`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={decisionConfig.icon} />
            </svg>
          </div>

          {/* Text Content */}
          <div className="text-center lg:text-left flex-1">
            <h2 className={`text-4xl font-bold text-${decisionConfig.color}-400 mb-2 tracking-tight`}>
              {decisionConfig.label}
            </h2>
            <p className="text-slate-300 text-lg">{decisionConfig.description}</p>
            <div className="mt-4 flex flex-wrap items-center justify-center lg:justify-start gap-3">
              <span className={`badge ${riskConfig.bg} ${riskConfig.color} ${riskConfig.border}`}>
                Risk: {decision.risk_level}
              </span>
              {decision.approval_probability && (
                <span className="badge badge-info">
                  Confidence: {formatPercent(decision.approval_probability)}
                </span>
              )}
              <span className="trust-badge">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
                Bank-Grade Analysis
              </span>
            </div>
          </div>

          {/* EMI Display */}
          <div className="text-center lg:text-right">
            <div className="text-slate-400 text-sm uppercase tracking-wider mb-1">Monthly EMI</div>
            <div className="text-4xl font-bold text-gradient-gold font-mono">{formatCurrency(decision.emi)}</div>
            <div className="text-slate-500 text-sm mt-1">Estimated payment</div>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Default Probability with Gauge */}
        <div className="glass-card rounded-xl p-6 flex flex-col items-center">
          <div className="flex items-center justify-between w-full mb-4">
            <span className="text-slate-400 text-sm">Default Risk</span>
            <svg className="w-5 h-5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <GaugeChart
            value={(decision.default_probability || 0) * 100}
            max={100}
            size={140}
            strokeWidth={12}
            label="Risk Score"
            suffix="%"
          />
        </div>

        {/* Approval Confidence */}
        <div className="glass-card rounded-xl p-6 flex flex-col items-center">
          <div className="flex items-center justify-between w-full mb-4">
            <span className="text-slate-400 text-sm">Approval Confidence</span>
            <svg className="w-5 h-5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <GaugeChart
            value={confidenceValue}
            max={100}
            size={140}
            strokeWidth={12}
            label="Confidence"
            suffix="%"
          />
        </div>

        {/* Safe Income Threshold */}
        <div className="glass-card rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-slate-400 text-sm">Safe Income Threshold</span>
            <svg className="w-5 h-5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="text-2xl font-bold text-gradient-gold font-mono">
            {formatCurrency(decision.monte_carlo?.safe_income_threshold)}
          </div>
          <div className="mt-4">
            <div className="flex justify-between text-xs text-slate-500 mb-1">
              <span>Minimum Recommended</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div className="h-2 rounded-full bg-gradient-to-r from-amber-400 to-amber-600 w-3/4" />
            </div>
          </div>
        </div>

        {/* Worst Case EMI */}
        <div className="glass-card rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-slate-400 text-sm">Worst Case EMI</span>
            <svg className="w-5 h-5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v12m0-12l-8 8-4-4-6 6" />
            </svg>
          </div>
          <div className="text-2xl font-bold text-amber-400 font-mono">
            {formatCurrency(decision.monte_carlo?.worst_case_emi)}
          </div>
          <div className="mt-4">
            <div className="flex justify-between text-xs text-slate-500 mb-1">
              <span>95th Percentile</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div className="h-2 rounded-full bg-gradient-to-r from-red-400 to-red-600 w-4/5" />
            </div>
          </div>
        </div>
      </div>

      {/* Debt to Income Bar */}
      <div className="glass-card rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 5l4 4 4-4" />
            </svg>
            <span className="text-slate-300 font-medium">Debt-to-Income Ratio</span>
          </div>
          <span className="text-2xl font-bold text-gradient-gold font-mono">
            {((decision.feature_values?.debt_to_income_ratio || 0) * 100).toFixed(1)}%
          </span>
        </div>
        <div className="relative">
          <div className="w-full bg-slate-700 rounded-full h-3">
            <div
              className="h-3 rounded-full bg-gradient-to-r from-emerald-500 via-amber-400 to-red-500 transition-all duration-1000"
              style={{ width: `${Math.min((decision.feature_values?.debt_to_income_ratio || 0) * 100, 100)}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-slate-500 mt-2">
            <span>0% (Healthy)</span>
            <span className="text-amber-400">43% (Max Recommended)</span>
            <span>100% (Critical)</span>
          </div>
        </div>
      </div>

      {/* AI Advice Section */}
      {decision.advice && (
        <div className="glass-card rounded-xl p-6 border-l-4 border-l-cta-500">
          <div className="flex items-start space-x-4">
            <div className="p-3 rounded-xl bg-cta-500/10 shrink-0">
              <svg className="w-6 h-6 text-cta-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-3">
                <h3 className="text-lg font-semibold text-white">AI Recommendation</h3>
                <span className="px-2 py-0.5 rounded-full text-xs font-mono bg-cta-500/20 text-cta-400 border border-cta-500/30">
                  Claude AI
                </span>
              </div>
              <div className="text-slate-300 leading-relaxed space-y-2">
                {decision.advice.split(';').map((item, idx) => (
                  <div key={idx} className="flex items-start space-x-3">
                    <span className="w-6 h-6 rounded-full bg-cta-500/20 text-cta-400 flex items-center justify-center text-xs shrink-0 mt-0.5">
                      {idx + 1}
                    </span>
                    <span className="text-slate-300">{item.trim()}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Feature Values Detail */}
      {decision.feature_values && (
        <div className="glass-card rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Calculated Metrics
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {Object.entries(decision.feature_values).map(([key, value]) => (
              <div key={key} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50 hover:border-amber-500/30 transition-colors">
                <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">
                  {key.replace(/_/g, ' ')}
                </div>
                <div className="text-white font-mono text-lg">
                  {typeof value === 'number' ? value.toFixed(3) : value}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DecisionResult;
