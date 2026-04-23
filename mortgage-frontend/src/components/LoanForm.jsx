import React, { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import { analyzeLoan } from '../utils/api';
import DecisionResult from './DecisionResult';

const STORAGE_KEY = 'mortgage_loan_form_data';

const LoanForm = ({ onDecision }) => {
  const [formData, setFormData] = useState(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? JSON.parse(saved) : {
      income: '',
      loan_amount: '',
      interest_rate: '8.5',
      loan_term: '5',
      credit_score: '',
      existing_loans: '0'
    };
  });

  // Save to localStorage on every change
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(formData));
  }, [formData]);

  const clearSavedData = () => {
    localStorage.removeItem(STORAGE_KEY);
    setFormData({
      income: '',
      loan_amount: '',
      interest_rate: '8.5',
      loan_term: '5',
      credit_score: '',
      existing_loans: '0'
    });
  };
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [errors, setErrors] = useState({});

  const validateForm = () => {
    const newErrors = {};

    if (!formData.income || formData.income <= 0) {
      newErrors.income = 'Monthly income must be greater than 0';
    }
    if (!formData.loan_amount || formData.loan_amount <= 0) {
      newErrors.loan_amount = 'Loan amount must be greater than 0';
    }
    if (!formData.credit_score || formData.credit_score < 300 || formData.credit_score > 850) {
      newErrors.credit_score = 'Credit score must be between 300 and 850';
    }
    if (!formData.interest_rate || formData.interest_rate <= 0) {
      newErrors.interest_rate = 'Interest rate must be greater than 0';
    }
    if (!formData.loan_term || formData.loan_term <= 0) {
      newErrors.loan_term = 'Loan term must be greater than 0';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error when user types
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: null }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      toast.error('Please fix the errors in the form');
      return;
    }

    setLoading(true);

    try {
      const response = await analyzeLoan({
        income: parseFloat(formData.income),
        loan_amount: parseFloat(formData.loan_amount),
        interest_rate: parseFloat(formData.interest_rate),
        loan_term: parseInt(formData.loan_term),
        credit_score: parseInt(formData.credit_score),
        existing_loans: parseInt(formData.existing_loans)
      });

      setResult(response);
      onDecision && onDecision(response);
      toast.success('Analysis complete!');
    } catch (error) {
      console.error('Analysis failed:', error);
      toast.error(error.response?.data?.detail || 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value) => {
    if (!value) return '';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(value);
  };

  const inputFields = [
    {
      name: 'income',
      label: 'Monthly Income',
      type: 'number',
      placeholder: '5000',
      prefix: '$',
      help: 'Your gross monthly income before taxes',
      min: 0
    },
    {
      name: 'loan_amount',
      label: 'Loan Amount',
      type: 'number',
      placeholder: '100000',
      prefix: '$',
      help: 'Total amount you want to borrow',
      min: 0
    },
    {
      name: 'credit_score',
      label: 'Credit Score',
      type: 'number',
      placeholder: '700',
      min: 300,
      max: 850,
      help: 'Your FICO credit score (300-850)'
    },
    {
      name: 'interest_rate',
      label: 'Interest Rate',
      type: 'number',
      step: '0.1',
      placeholder: '8.5',
      suffix: '%',
      help: 'Annual interest rate',
      min: 0,
      max: 50
    },
    {
      name: 'loan_term',
      label: 'Loan Term',
      type: 'number',
      placeholder: '5',
      suffix: 'years',
      help: 'Duration of the loan',
      min: 1,
      max: 40
    },
    {
      name: 'existing_loans',
      label: 'Existing Loans',
      type: 'number',
      placeholder: '0',
      help: 'Number of other active loans',
      min: 0
    }
  ];

  const trustIndicators = [
    { icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z', text: 'No credit check required' },
    { icon: 'M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z', text: 'Bank-grade encryption' },
    { icon: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01', text: 'No commitment required' }
  ];

  return (
    <div className="pt-24 pb-12 px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto">
      {/* Trust Header */}
      <div className="text-center mb-10 space-y-4">
        <div className="flex flex-wrap items-center justify-center gap-3">
          {trustIndicators.map((item, idx) => (
            <div key={idx} className="flex items-center gap-2 text-slate-400 text-sm">
              <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={item.icon} />
              </svg>
              <span>{item.text}</span>
            </div>
          ))}
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-white">
          Loan Application
        </h1>
        <p className="text-slate-400 max-w-xl mx-auto">
          Enter your financial details below. Our AI will analyze your application using
          ensemble ML models and Monte Carlo simulations for bank-grade accuracy.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Form Section */}
        <div className="glass-card rounded-2xl p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {inputFields.map((field) => (
                <div key={field.name} className="space-y-2">
                  <label htmlFor={field.name} className="label label-required">
                    {field.label}
                  </label>
                  <div className="relative">
                    {field.prefix && (
                      <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 font-mono pointer-events-none">
                        {field.prefix}
                      </span>
                    )}
                    <input
                      type={field.type}
                      name={field.name}
                      id={field.name}
                      value={formData[field.name]}
                      onChange={handleChange}
                      min={field.min}
                      max={field.max}
                      step={field.step}
                      className={`input-field ${field.prefix ? 'pl-10' : 'pl-4'} ${field.suffix ? 'pr-16' : 'pr-4'} ${
                        errors[field.name] ? 'error' : ''
                      }`}
                      placeholder={field.placeholder}
                      aria-invalid={errors[field.name] ? 'true' : 'false'}
                      aria-describedby={errors[field.name] ? `${field.name}-error` : `${field.name}-help`}
                    />

                    {field.suffix && (
                      <span className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 text-sm pointer-events-none">
                        {field.suffix}
                      </span>
                    )}
                  </div>

                  {errors[field.name] ? (
                    <p id={`${field.name}-error`} className="error-text flex items-center gap-1" role="alert">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <span>{errors[field.name]}</span>
                    </p>
                  ) : (
                    <p id={`${field.name}-help`} className="text-slate-500 text-xs">{field.help}</p>
                  )}
                </div>
              ))}
            </div>

            {/* Submit Button */}
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
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                  </svg>
                  <span>Analyze Application</span>
                </>
              )}
            </button>

            {/* Clear saved data */}
            <div className="mt-4 text-center">
              <button
                type="button"
                onClick={clearSavedData}
                className="text-sm text-slate-500 hover:text-slate-400 underline transition-colors"
              >
                Clear saved data
              </button>
            </div>
          </form>

          {/* Quick Info */}
          <div className="mt-8 pt-6 border-t border-white/10">
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-amber-500/10 mt-1">
                <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h4 className="text-sm font-semibold text-white mb-1">What happens next?</h4>
                <ul className="text-slate-400 text-sm space-y-1">
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                    <span>ML ensemble calculates approval probability</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                    <span>Monte Carlo runs 10,000+ scenarios</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                    <span>Risk engine assesses your profile</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                    <span>AI generates personalized recommendations</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Preview Card */}
        <div className="space-y-6">
          <div className="glass-card rounded-2xl p-6">
            <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
              <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
              Application Preview
            </h3>

            <div className="space-y-4">
              <div className="flex justify-between items-center py-3 border-b border-white/5">
                <span className="text-slate-400">Monthly Income</span>
                <span className="text-white font-mono">{formatCurrency(formData.income) || '-'}</span>
              </div>
              <div className="flex justify-between items-center py-3 border-b border-white/5">
                <span className="text-slate-400">Loan Amount</span>
                <span className="text-white font-mono">{formatCurrency(formData.loan_amount) || '-'}</span>
              </div>
              <div className="flex justify-between items-center py-3 border-b border-white/5">
                <span className="text-slate-400">Credit Score</span>
                <span className={`font-mono ${
                  formData.credit_score >= 700 ? 'text-emerald-400' :
                  formData.credit_score >= 600 ? 'text-amber-400' : 'text-red-400'
                }`}>
                  {formData.credit_score || '-'}
                </span>
              </div>
              <div className="flex justify-between items-center py-3 border-b border-white/5">
                <span className="text-slate-400">Interest Rate</span>
                <span className="text-white font-mono">{formData.interest_rate ? `${formData.interest_rate}%` : '-'}</span>
              </div>
              <div className="flex justify-between items-center py-3 border-b border-white/5">
                <span className="text-slate-400">Loan Term</span>
                <span className="text-white font-mono">{formData.loan_term ? `${formData.loan_term} years` : '-'}</span>
              </div>
              <div className="flex justify-between items-center py-3">
                <span className="text-slate-400">Debt-to-Income</span>
                <span className="text-white font-mono">
                  {formData.income && formData.loan_amount
                    ? `${((formData.loan_amount / (formData.income * 12 * formData.loan_term)) * 100).toFixed(1)}%`
                    : '-'}
                </span>
              </div>
            </div>
          </div>

          {/* Credit Score Guide */}
          <div className="glass-card rounded-2xl p-6">
            <h3 className="text-sm font-semibold text-white mb-4 uppercase tracking-wider flex items-center gap-2">
              <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Credit Score Guide
            </h3>
            <div className="space-y-3">
              {[
                { label: 'Excellent', range: '750+', color: 'bg-emerald-400', text: 'text-emerald-400', rate: 'Best Rates' },
                { label: 'Good', range: '700-749', color: 'bg-blue-400', text: 'text-blue-400', rate: 'Good Rates' },
                { label: 'Fair', range: '650-699', color: 'bg-amber-400', text: 'text-amber-400', rate: 'Standard Rates' },
                { label: 'Poor', range: '<650', color: 'bg-red-400', text: 'text-red-400', rate: 'High Rates' }
              ].map((item, idx) => (
                <div key={idx} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${item.color}`} />
                    <span className="text-slate-300 text-sm">{item.label} ({item.range})</span>
                  </div>
                  <span className={`${item.text} text-xs font-medium`}>{item.rate}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Security Badge */}
          <div className="glass-card rounded-2xl p-6 text-center">
            <div className="w-12 h-12 rounded-full bg-amber-500/10 flex items-center justify-center mx-auto mb-3">
              <svg className="w-6 h-6 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <h4 className="text-white font-semibold mb-1">Your Data is Secure</h4>
            <p className="text-slate-400 text-sm">
              All data is encrypted with 256-bit TLS. We never store your personal information.
            </p>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {result && (
        <div className="mt-8 space-y-6">
          <div className="flex items-center gap-3">
            <div className="h-px flex-1 section-divider" />
            <span className="text-slate-500 text-sm uppercase tracking-wider">Analysis Results</span>
            <div className="h-px flex-1 section-divider" />
          </div>
          <DecisionResult decision={result} />
        </div>
      )}
    </div>
  );
};

export default LoanForm;
