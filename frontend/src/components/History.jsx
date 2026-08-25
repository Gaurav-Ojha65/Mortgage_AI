import React, { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import { getHistory } from '../utils/api';

const History = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('ALL');

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const data = await getHistory(100);
      setHistory(data);
    } catch (error) {
      console.error('Failed to fetch history:', error);
      toast.error('Failed to load history');
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

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  const filteredHistory = Array.isArray(history) ? history.filter(item => {
    if (filter === 'ALL') return true;
    return item.decision === filter;
  }) : [];

  const stats = {
    total: Array.isArray(history) ? history.length : 0,
    approved: Array.isArray(history) ? history.filter(h => h.decision === 'APPROVE').length : 0,
    rejected: Array.isArray(history) ? history.filter(h => h.decision === 'REJECT').length : 0,
    conditional: Array.isArray(history) ? history.filter(h => h.decision === 'CONDITIONAL').length : 0,
    avgLoanAmount: Array.isArray(history) && history.length > 0
      ? history.reduce((sum, h) => sum + (h.loan_amount || 0), 0) / history.length
      : 0
  };

  const statCards = [
    { label: 'Total Applications', value: stats.total, color: 'text-white', icon: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2' },
    { label: 'Approved', value: stats.approved, color: 'text-emerald-400', icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z' },
    { label: 'Rejected', value: stats.rejected, color: 'text-red-400', icon: 'M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z' },
    { label: 'Avg Amount', value: formatCurrency(stats.avgLoanAmount), color: 'text-gradient-gold', icon: 'M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z' }
  ];

  const filters = [
    { key: 'ALL', label: 'All Decisions', count: history.length },
    { key: 'APPROVE', label: 'Approved', count: stats.approved },
    { key: 'CONDITIONAL', label: 'Under Review', count: stats.conditional },
    { key: 'REJECT', label: 'Rejected', count: stats.rejected }
  ];

  return (
    <div className="pt-24 pb-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Trust Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-8 gap-4">
        <div>
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
            <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Audit Trail</span>
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">Decision History</h1>
          <p className="text-slate-400">View all previous loan applications and their outcomes</p>
        </div>
        <button
          onClick={fetchHistory}
          disabled={loading}
          className="btn-secondary"
        >
          <svg className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          <span>{loading ? 'Refreshing...' : 'Refresh'}</span>
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {statCards.map((stat, idx) => (
          <div key={idx} className="metric-card">
            <div className="flex items-center gap-2 mb-2">
              <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={stat.icon} />
              </svg>
              <div className="text-slate-400 text-sm">{stat.label}</div>
            </div>
            <div className={`text-2xl font-bold font-mono ${stat.color}`}>{stat.value}</div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2 mb-6">
        {filters.map((f) => (
          <button
            key={f.key}
            onClick={() => setFilter(f.key)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2 ${
              filter === f.key
                ? 'bg-amber-500 text-slate-900'
                : 'bg-slate-800/50 text-slate-400 border border-slate-700/50 hover:text-white hover:bg-slate-700'
            }`}
          >
            {f.label}
            <span className={`text-xs px-2 py-0.5 rounded-full ${
              filter === f.key ? 'bg-slate-900/30 text-slate-900' : 'bg-slate-700 text-slate-300'
            }`}>
              {f.count}
            </span>
          </button>
        ))}
      </div>

      {/* History Table */}
      <div className="glass-card rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-800/50">
              <tr className="text-left text-slate-400 text-xs uppercase tracking-wider">
                <th className="px-6 py-4 font-medium">Date</th>
                <th className="px-6 py-4 font-medium">Loan Amount</th>
                <th className="px-6 py-4 font-medium">Income</th>
                <th className="px-6 py-4 font-medium">Credit Score</th>
                <th className="px-6 py-4 font-medium">EMI</th>
                <th className="px-6 py-4 font-medium">Default Risk</th>
                <th className="px-6 py-4 font-medium">Risk Level</th>
                <th className="px-6 py-4 font-medium">Decision</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {loading ? (
                Array.from({ length: 5 }).map((_, idx) => (
                  <tr key={idx}>
                    {Array.from({ length: 8 }).map((__, colIdx) => (
                      <td key={colIdx} className="px-6 py-4">
                        <div className="skeleton h-4 w-20" />
                      </td>
                    ))}
                  </tr>
                ))
              ) : filteredHistory.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-6 py-12 text-center">
                    <div className="flex flex-col items-center gap-3">
                      <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center">
                        <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </div>
                      <p className="text-slate-500">No decisions found</p>
                    </div>
                  </td>
                </tr>
              ) : (
                Array.isArray(filteredHistory) && filteredHistory.map((item, idx) => (
                  <tr key={idx} className="hover:bg-white/5 transition-colors">
                    <td className="px-6 py-4 text-slate-300 text-sm">
                      {formatDate(item.timestamp)}
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-white font-mono font-medium">{formatCurrency(item.loan_amount)}</span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-slate-300 font-mono">{formatCurrency(item.income)}/mo</span>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`font-mono font-medium ${
                        item.credit_score >= 700 ? 'text-emerald-400' :
                        item.credit_score >= 600 ? 'text-amber-400' : 'text-red-400'
                      }`}>
                        {item.credit_score}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-slate-300 font-mono">{formatCurrency(item.emi)}</span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <span className={`font-mono ${
                          item.default_probability > 0.3 ? 'text-red-400' :
                          item.default_probability > 0.15 ? 'text-amber-400' : 'text-emerald-400'
                        }`}>
                          {((item.default_probability || 0) * 100).toFixed(1)}%
                        </span>
                        <div className="w-16 bg-slate-700 rounded-full h-1.5">
                          <div
                            className={`h-1.5 rounded-full ${
                              item.default_probability > 0.3 ? 'bg-red-500' :
                              item.default_probability > 0.15 ? 'bg-amber-500' : 'bg-emerald-500'
                            }`}
                            style={{ width: `${Math.min((item.default_probability || 0) * 100, 100)}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`badge ${
                        item.risk_level === 'LOW' ? 'badge-success' :
                        item.risk_level === 'HIGH' ? 'badge-danger' : 'badge-warning'
                      }`}>
                        {item.risk_level}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`badge ${
                        item.decision === 'APPROVE' ? 'badge-success' :
                        item.decision === 'REJECT' ? 'badge-danger' : 'badge-warning'
                      }`}>
                        {item.decision}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Summary Footer */}
      {!loading && filteredHistory.length > 0 && (
        <div className="mt-6 flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-slate-500">
          <span>Showing {filteredHistory.length} of {history.length} decisions</span>
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Last updated: {new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default History;
