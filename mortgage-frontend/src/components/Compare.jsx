import React, { useState } from 'react';
import { toast } from 'react-toastify';
import { compareLoans } from '../utils/api';
import GaugeChart from './GaugeChart';

const DEFAULT_ITEM = {
  loan_amount: 0,
  decision: 'pending',
  emi: 0,
  risk_level: 'unknown',
  default_probability: 0,
  worst_case_emi: 0
};

const Compare = () => {
  const [formData, setFormData] = useState({
    income: '',
    loan_amount: '',
    credit_score: ''
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [errors, setErrors] = useState({});

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: null }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    if (!formData.income || formData.income <= 0) newErrors.income = 'Required';
    if (!formData.loan_amount || formData.loan_amount <= 0) newErrors.loan_amount = 'Required';
    if (!formData.credit_score || formData.credit_score < 300 || formData.credit_score > 850) {
      newErrors.credit_score = 'Invalid credit score';
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    setLoading(true);
    try {
      const response = await compareLoans({
        income: parseFloat(formData.income),
        loan_amount: parseFloat(formData.loan_amount),
        credit_score: parseInt(formData.credit_score)
      });
      setResults(response);
      toast.success('Comparison complete!');
    } catch (error) {
      console.error('Comparison failed:', error);
      toast.error('Failed to compare loans');
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(value || 0);
  };

  const getDecisionConfig = (decision) => {
    switch (decision) {
      case 'approve':
        return { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', badge: 'badge-success' };
      case 'reject':
        return { color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/30', badge: 'badge-danger' };
      default:
        return { color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30', badge: 'badge-warning' };
    }
  };

  const scenarioConfig = {
    low: {
      label: 'Conservative',
      multiplier: '50%',
      color: 'emerald',
      icon: 'M13 7h8m0 0v12m0-12l-8 8-4-4-6 6',
      description: 'Lower loan amount with minimal risk'
    },
    medium: {
      label: 'Standard',
      multiplier: '100%',
      color: 'purple',
      icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
      description: 'Balanced loan amount for optimal terms',
      recommended: true
    },
    high: {
      label: 'Aggressive',
      multiplier: '150%',
      color: 'amber',
      icon: 'M13 7h8m0 0v12m0-12l-8 8-4-4-6 6',
      description: 'Higher loan amount with increased risk'
    }
  };

  const hasResults = results && results.comparison;

  return (
    <div className="pt-24 pb-12 px-4 sm:px-6 lg:px-8 max-w-6xl mx-auto">
      {/* Trust Header */}
      <div className="text-center mb-10 space-y-4">
        <div className="inline-flex items-center gap-2 text-slate-400 text-sm mb-2">
          <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>AI-Powered Scenario Analysis</span>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-white">Compare Loan Scenarios</h1>
        <p className="text-slate-400 max-w-2xl mx-auto">
          Compare three loan amounts side-by-side using our ensemble ML models to find the optimal
          borrowing amount for your financial profile.
        </p>
      </div>

      {/* Input Form */}
      <div className="glass-card rounded-2xl p-8 mb-8 max-w-2xl mx-auto">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-2">
              <label htmlFor="income" className="label label-required">Monthly Income</label>
              <div className="relative">
                <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 font-mono pointer-events-none">$</span>
                <input
                  type="number"
                  id="income"
                  name="income"
                  value={formData.income}
                  onChange={handleChange}
                  className={`input-field pl-10 ${errors.income ? 'error' : ''}`}
                  placeholder="5000"
                  aria-invalid={errors.income ? 'true' : 'false'}
                />
              </div>
              {errors.income && <p className="error-text">{errors.income}</p>}
            </div>

            <div className="space-y-2">
              <label htmlFor="loan_amount" className="label label-required">Base Loan Amount</label>
              <div className="relative">
                <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 font-mono pointer-events-none">$</span>
                <input
                  type="number"
                  id="loan_amount"
                  name="loan_amount"
                  value={formData.loan_amount}
                  onChange={handleChange}
                  className={`input-field pl-10 ${errors.loan_amount ? 'error' : ''}`}
                  placeholder="100000"
                  aria-invalid={errors.loan_amount ? 'true' : 'false'}
                />
              </div>
              {errors.loan_amount && <p className="error-text">{errors.loan_amount}</p>}
            </div>

            <div className="space-y-2">
              <label htmlFor="credit_score" className="label label-required">Credit Score</label>
              <input
                type="number"
                id="credit_score"
                name="credit_score"
                value={formData.credit_score}
                onChange={handleChange}
                min={300}
                max={850}
                className={`input-field ${errors.credit_score ? 'error' : ''}`}
                placeholder="700"
                aria-invalid={errors.credit_score ? 'true' : 'false'}
              />
              {errors.credit_score && <p className="error-text">{errors.credit_score}</p>}
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Comparing...</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span>Compare Scenarios</span>
              </>
            )}
          </button>
        </form>
      </div>

      {/* Results */}
      {hasResults ? (
        <div className="space-y-8">
          <div className="flex items-center gap-3">
            <div className="h-px flex-1 section-divider" />
            <span className="text-slate-500 text-sm uppercase tracking-wider">Comparison Results</span>
            <div className="h-px flex-1 section-divider" />
          </div>

          {/* Scenarios Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {['low', 'medium', 'high'].map((scenario, idx) => {
              const rawData = results.comparison?.[scenario];
              const data = { ...DEFAULT_ITEM, ...rawData };
              const config = getDecisionConfig(data.decision);
              const scenarioInfo = scenarioConfig[scenario] || scenarioConfig.low;

              return (
                <div
                  key={scenario}
                  className={`glass-card rounded-2xl p-6 relative overflow-hidden ${
                    scenarioInfo.recommended ? 'ring-2 ring-amber-500/50' : ''
                  }`}
                >
                  {/* Recommended Badge */}
                  {scenarioInfo.recommended && (
                    <div className="absolute top-0 right-0">
                      <div className="bg-amber-500 text-slate-900 text-xs px-3 py-1 rounded-bl-lg font-semibold flex items-center gap-1">
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        Recommended
                      </div>
                    </div>
                  )}

                  {/* Header */}
                  <div className="mb-6">
                    <div className="flex items-center gap-2 mb-2">
                      <div className={`p-1.5 rounded-lg ${scenarioInfo.color === 'emerald' ? 'bg-emerald-500/10 text-emerald-400' : scenarioInfo.color === 'purple' ? 'bg-purple-500/10 text-purple-400' : 'bg-amber-500/10 text-amber-400'}`}>
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={scenarioInfo.icon} />
                        </svg>
                      </div>
                      <span className="text-slate-400 text-sm uppercase tracking-wider">{scenarioInfo.label}</span>
                    </div>
                    <h3 className="text-3xl font-bold text-gradient-gold">{formatCurrency(data.loan_amount)}</h3>
                    <p className="text-slate-500 text-sm">{scenarioInfo.multiplier} of base amount</p>
                  </div>

                  {/* Decision Badge */}
                  <div className={`badge ${config.badge} mb-6`}>
                    {data.decision}
                  </div>

                  {/* Metrics */}
                  <div className="space-y-4">
                    <div className="flex justify-between items-center py-3 border-b border-white/5">
                      <span className="text-slate-400 text-sm">Monthly EMI</span>
                      <span className="text-white font-mono font-medium">{formatCurrency(data.emi)}</span>
                    </div>

                    <div className="flex justify-between items-center py-3 border-b border-white/5">
                      <span className="text-slate-400 text-sm">Risk Level</span>
                      <span className={`font-medium ${
                        data.risk_level === 'low' ? 'text-emerald-400' :
                        data.risk_level === 'high' ? 'text-red-400' : 'text-amber-400'
                      }`}>
                        {data.risk_level}
                      </span>
                    </div>

                    <div className="flex justify-between items-center py-3 border-b border-white/5">
                      <span className="text-slate-400 text-sm">Default Probability</span>
                      <span className={`font-mono font-medium ${
                        data.default_probability > 0.3 ? 'text-red-400' :
                        data.default_probability > 0.15 ? 'text-amber-400' : 'text-emerald-400'
                      }`}>
                        {((data.default_probability || 0) * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="flex justify-between items-center py-3">
                      <span className="text-slate-400 text-sm">Worst Case EMI</span>
                      <span className="text-amber-400 font-mono font-medium">{formatCurrency(data.worst_case_emi)}</span>
                    </div>
                  </div>

                  {/* Risk Gauge */}
                  <div className="mt-6 pt-6 border-t border-white/5 flex justify-center">
                    <GaugeChart
                      value={((data.default_probability || 0) * 100)}
                      max={100}
                      size={120}
                      strokeWidth={8}
                      showValue={false}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Summary */}
          <div className="glass-card rounded-2xl p-8">
            <div className="flex items-start gap-4">
              <div className="p-3 rounded-xl bg-cta-500/10 shrink-0">
                <svg className="w-6 h-6 text-cta-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-white mb-4">AI Recommendation</h3>
                <div className="text-slate-300 leading-relaxed space-y-3">
                  <p>Based on your financial profile with a credit score of <strong className="text-white">{results.credit_score || 0}</strong> and monthly income of <strong className="text-white">{formatCurrency(results.income)}</strong>:</p>
                  <ul className="space-y-2">
                    <li className="flex items-start gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-2 shrink-0" />
                      <span>The <span className="text-emerald-400 font-medium">Conservative</span> option ({formatCurrency(results.comparison?.low?.loan_amount || 0)}) carries the lowest risk with a {((results.comparison?.low?.default_probability || 0) * 100).toFixed(1)}% default probability.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-purple-400 mt-2 shrink-0" />
                      <span>The <span className="text-purple-400 font-medium">Standard</span> option ({formatCurrency(results.comparison?.medium?.loan_amount || 0)}) represents a balanced approach with manageable risk levels.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-2 shrink-0" />
                      <span>The <span className="text-amber-400 font-medium">Aggressive</span> option ({formatCurrency(results.comparison?.high?.loan_amount || 0)}) may strain your finances with a {((results.comparison?.high?.default_probability || 0) * 100).toFixed(1)}% default probability.</span>
                    </li>
                  </ul>
                  <div className="mt-4 p-4 bg-amber-500/10 rounded-xl border border-amber-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      <span className="text-amber-400 font-semibold">Final Recommendation</span>
                    </div>
                    <p className="text-slate-300 text-sm">
                      Consider the {results.comparison?.medium?.decision === 'approve' ? 'Standard' : 'Conservative'} option
                      for optimal balance between loan amount and financial security.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <p className="text-center text-slate-500">No comparison data</p>
      )}
    </div>
  );
};

export default Compare;