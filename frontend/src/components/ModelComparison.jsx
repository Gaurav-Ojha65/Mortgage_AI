import React, { useState, useEffect, useRef } from 'react';
import { toast } from 'react-toastify';
import { getModelComparison, compareAllModels, switchModel } from '../utils/api';
import GaugeChart from './GaugeChart';

const DEFAULT_APPLICANT = {
  credit_score: 650,
  annual_income: 50000,
  loan_amount: 15000,
  loan_term: 36,
  dti_ratio: 0.3,
  employment_years: 3,
  num_credit_lines: 3,
  num_derogatory_marks: 0,
  credit_utilization: 0.3,
  late_payment_severity_score: 0.95,
  home_ownership: 1,
  purpose_encoded: 0,
  num_late_payments: 0,
  savings_balance: 5000,
  monthly_expenses: 2000
};

const ModelComparison = () => {
  const [comparison, setComparison] = useState(null);
  const [applicant, setApplicant] = useState(DEFAULT_APPLICANT);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [metricsLoading, setMetricsLoading] = useState(true);
  const initialized = useRef(false);

  useEffect(() => {
    if (!initialized.current) {
      loadComparison();
      initialized.current = true;
    }
  }, []);


  const loadComparison = async () => {
    setMetricsLoading(true);
    try {
      const data = await getModelComparison();
      setComparison(data);
    } catch (error) {
      console.error('Failed to load comparison:', error);
      toast.error('Failed to load model comparison metrics');
    } finally {
      setMetricsLoading(false);
    }
  };

  const handleApplicantChange = (field, value) => {
    setApplicant(prev => ({ ...prev, [field]: value }));
  };

  const runComparison = async () => {
    setLoading(true);
    try {
      const data = await compareAllModels(applicant);
      setResults(data);
      toast.success('All models scored this applicant');
    } catch (error) {
      console.error('Comparison failed:', error);
      toast.error('Failed to run model comparison');
    } finally {
      setLoading(false);
    }
  };

  const handleSwitchModel = async (modelName) => {
    try {
      await switchModel(modelName);
      toast.success(`${modelName} is now the active model`);
      loadComparison();
    } catch (error) {
      console.error('Failed to switch model:', error);
      toast.error('Failed to switch active model');
    }
  };

  const formatPercent = (val) => ((val || 0) * 100).toFixed(1);
  const formatCurrency = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val || 0);

  const getDecisionBadge = (decision) => {
    const d = (decision || '').toUpperCase().replace(/_/g, ' ');
    if (d === 'APPROVE' || d === 'APPROVED') {
      return <span className="badge badge-success">APPROVE</span>;
    }
    if (d === 'REJECT' || d === 'REJECTED') {
      return <span className="badge badge-danger">REJECT</span>;
    }
    return <span className="badge badge-warning">MANUAL REVIEW</span>;
  };

  const getRiskBadge = (risk) => {
    const r = (risk || '').toUpperCase();
    const colors = {
      VERY_LOW: 'text-emerald-400',
      LOW: 'text-emerald-400',
      MODERATE: 'text-amber-400',
      MEDIUM: 'text-amber-400',
      HIGH: 'text-red-400',
      SEVERE: 'text-red-600',
      CRITICAL: 'text-red-600'
    };
    return <span className={`font-medium ${colors[r] || 'text-amber-400'}`}>{r.replace(/_/g, ' ')}</span>;
  };

  const metrics = comparison?.metrics || {};
  const winner = comparison?.winner;

  if (metricsLoading) {
    return (
      <div className="pt-24 pb-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin h-8 w-8 border-2 border-amber-500 border-t-transparent rounded-full" />
        </div>
      </div>
    );
  }

  return (
    <div className="pt-24 pb-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {comparison?.is_mock && (
        <div style={{
          background: 'rgba(234, 179, 8, 0.1)',
          border: '1px solid rgba(234, 179, 8, 0.3)',
          borderRadius: '12px',
          padding: '16px',
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          color: '#EAB308'
        }}>
          <span style={{ fontSize: '20px' }}>⚠️</span>
          <div>
            <div style={{ fontWeight: '600' }}>Live model unavailable — showing demo data</div>
            <div style={{ fontSize: '13px', opacity: 0.8 }}>No comparison report found in /models directory. Run training pipeline to generate real metrics.</div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="text-center mb-10 space-y-4">
        <div className="inline-flex items-center gap-2 text-slate-400 text-sm mb-2">
          <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <span>Multi-Model Ensemble Analysis</span>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-white">Model Comparison</h1>
        <p className="text-slate-400 max-w-2xl mx-auto">
          Compare Logistic Regression, XGBoost, and LightGBM performance metrics.
          Score any applicant through all three models simultaneously.
        </p>
      </div>

      {/* Model Metrics Table */}
      <div className="glass-card rounded-2xl p-8 mb-8">
        <h2 className="text-xl font-semibold text-white mb-6">Training Metrics</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left text-slate-400 text-sm font-medium pb-3">Model</th>
                <th className="text-right text-slate-400 text-sm font-medium pb-3">AUC</th>
                <th className="text-right text-slate-400 text-sm font-medium pb-3">F1</th>
                <th className="text-right text-slate-400 text-sm font-medium pb-3">Precision</th>
                <th className="text-right text-slate-400 text-sm font-medium pb-3">Recall</th>
                <th className="text-right text-slate-400 text-sm font-medium pb-3">Train Time</th>
                <th className="text-right text-slate-400 text-sm font-medium pb-3">Status</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(metrics).map(([name, m]) => {
                const isLgb = name.toLowerCase() === 'lightgbm';
                const isWinner = isLgb || name === winner;
                const displayName = isLgb ? 'LightGBM v3.1 — CANONICAL' : name;
                return (
                  <tr key={name} className="border-b border-white/5">
                    <td className={`py-4 font-medium ${isWinner ? 'text-emerald-400' : 'text-white'}`}>
                      {displayName} {isLgb && <span className="ml-2 text-xs px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-bold">CHAMPION</span>}
                    </td>
                    <td className="py-4 text-right font-mono text-white">{(m.roc_auc || 0).toFixed(4)}</td>
                    <td className="py-4 text-right font-mono text-white">{(m.f1 || 0).toFixed(4)}</td>
                    <td className="py-4 text-right font-mono text-white">{(m.precision || 0).toFixed(4)}</td>
                    <td className="py-4 text-right font-mono text-white">{(m.recall || 0).toFixed(4)}</td>
                    <td className="py-4 text-right font-mono text-slate-400">{m.train_time_s}s</td>
                    <td className="py-4 text-right">
                      {isWinner ? (
                        <span className="badge badge-success">Active</span>
                      ) : (
                        <button
                          onClick={() => handleSwitchModel(name.toLowerCase().replace(' ', ''))}
                          className="text-xs text-slate-400 hover:text-amber-400 transition-colors"
                        >
                          Set Active
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Applicant Scorer */}
      <div className="glass-card rounded-2xl p-8 mb-8">
        <h2 className="text-xl font-semibold text-white mb-6">Live Applicant Scorer</h2>
        <p className="text-slate-400 mb-6">Adjust applicant parameters and score through all 3 models</p>

        {/* Input Sliders */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <div className="space-y-2">
            <label className="text-slate-300 text-sm">Credit Score: {applicant.credit_score}</label>
            <input type="range" min="300" max="850" value={applicant.credit_score}
              onChange={(e) => handleApplicantChange('credit_score', parseInt(e.target.value))}
              className="w-full accent-amber-500"
            />
          </div>
          <div className="space-y-2">
            <label className="text-slate-300 text-sm">Annual Income: {formatCurrency(applicant.annual_income)}</label>
            <input type="range" min="20000" max="200000" step="1000" value={applicant.annual_income}
              onChange={(e) => handleApplicantChange('annual_income', parseInt(e.target.value))}
              className="w-full accent-amber-500"
            />
          </div>
          <div className="space-y-2">
            <label className="text-slate-300 text-sm">Loan Amount: {formatCurrency(applicant.loan_amount)}</label>
            <input type="range" min="1000" max="500000" step="1000" value={applicant.loan_amount}
              onChange={(e) => handleApplicantChange('loan_amount', parseInt(e.target.value))}
              className="w-full accent-amber-500"
            />
          </div>
          <div className="space-y-2">
            <label className="text-slate-300 text-sm">Loan Term: {applicant.loan_term} months</label>
            <input type="range" min="12" max="360" step="12" value={applicant.loan_term}
              onChange={(e) => handleApplicantChange('loan_term', parseInt(e.target.value))}
              className="w-full accent-amber-500"
            />
          </div>
          <div className="space-y-2">
            <label className="text-slate-300 text-sm">DTI Ratio: {(applicant.dti_ratio * 100).toFixed(0)}%</label>
            <input type="range" min="0" max="100" step="1" value={applicant.dti_ratio * 100}
              onChange={(e) => handleApplicantChange('dti_ratio', parseInt(e.target.value) / 100)}
              className="w-full accent-amber-500"
            />
          </div>
          <div className="space-y-2">
            <label className="text-slate-300 text-sm">Employment Years: {applicant.employment_years}</label>
            <input type="range" min="0" max="40" value={applicant.employment_years}
              onChange={(e) => handleApplicantChange('employment_years', parseFloat(e.target.value))}
              className="w-full accent-amber-500"
            />
          </div>
          <div className="space-y-2">
            <label className="text-slate-300 text-sm">Credit Utilization: {(applicant.credit_utilization * 100).toFixed(0)}%</label>
            <input type="range" min="0" max="100" step="1" value={applicant.credit_utilization * 100}
              onChange={(e) => handleApplicantChange('credit_utilization', parseInt(e.target.value) / 100)}
              className="w-full accent-amber-500"
            />
          </div>
          <div className="space-y-2">
            <label className="text-slate-300 text-sm">Late Payment Severity: {(applicant.late_payment_severity_score * 100).toFixed(0)}%</label>
            <input type="range" min="0" max="100" step="1" value={applicant.late_payment_severity_score * 100}
              onChange={(e) => handleApplicantChange('late_payment_severity_score', parseInt(e.target.value) / 100)}
              className="w-full accent-amber-500"
            />
          </div>
          <div className="space-y-2">
            <label className="text-slate-300 text-sm">Late Payments (12mo): {applicant.num_late_payments}</label>
            <input type="range" min="0" max="10" value={applicant.num_late_payments}
              onChange={(e) => handleApplicantChange('num_late_payments', parseInt(e.target.value))}
              className="w-full accent-amber-500"
            />
          </div>
        </div>

        <button
          onClick={runComparison}
          disabled={loading}
          className="w-full btn-primary disabled:opacity-50"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Running All Models...
            </span>
          ) : 'Score Through All Models'}
        </button>
      </div>

      {/* Results */}
      {results && (
        <div className="space-y-8">
          {/* Consensus Banner */}
          <div className={`glass-card rounded-2xl p-6 ${results.consensus.all_agree ? 'border border-emerald-500/30' : 'border border-amber-500/30'}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-xl ${results.consensus.all_agree ? 'bg-emerald-500/10' : 'bg-amber-500/10'}`}>
                  <svg className={`w-6 h-6 ${results.consensus.all_agree ? 'text-emerald-400' : 'text-amber-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={results.consensus.all_agree ? "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" : "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"} />
                  </svg>
                </div>
                <div>
                  <p className="text-slate-400 text-sm">{results.consensus.all_agree ? 'All models agree' : 'Models disagree'}</p>
                  <p className="text-white font-semibold text-lg">
                    Consensus: {getDecisionBadge(results.consensus.final_decision)} ({formatPercent(results.consensus.avg_probability)}% approval)
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-slate-400 text-sm">Models Approved</p>
                <p className="text-white font-semibold text-2xl">{results.consensus.disagreement_count}/3</p>
              </div>
            </div>
          </div>

          {/* Model Results Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Object.entries(results.models).map(([modelName, modelResult]) => {
              const modelDisplayName = modelName.charAt(0).toUpperCase() + modelName.slice(1);
              const isThisWinner = modelName === winner;
              return (
                <div key={modelName} className={`glass-card rounded-2xl p-6 relative ${isThisWinner ? 'ring-2 ring-amber-500/50' : ''}`}>
                  {isThisWinner && (
                    <div className="absolute top-0 right-0">
                      <div className="bg-amber-500 text-slate-900 text-xs px-3 py-1 rounded-bl-lg font-semibold">Winner</div>
                    </div>
                  )}
                  <h3 className="text-lg font-semibold text-white mb-4">{modelDisplayName}</h3>

                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400 text-sm">Decision</span>
                      {getDecisionBadge(modelResult.decision)}
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400 text-sm">Approval Prob</span>
                      <span className="text-white font-mono">{(modelResult.approval_probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400 text-sm">Risk Level</span>
                      {getRiskBadge(modelResult.risk_level)}
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400 text-sm">Default Prob</span>
                      <span className="text-white font-mono">{(modelResult.default_probability * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  <div className="mt-6 pt-4 border-t border-white/5 flex justify-center">
                    <GaugeChart
                      value={modelResult.approval_probability * 100}
                      max={100}
                      size={100}
                      strokeWidth={6}
                      showValue={false}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelComparison;