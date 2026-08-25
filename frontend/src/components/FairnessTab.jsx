import React, { useEffect, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { api } from '../api';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export default function FairnessTab() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.fairness()
      .then(res => setData(res))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '60px 0', color: '#94a3b8' }}>
        <div style={{ fontSize: '32px', marginBottom: '12px', animation: 'pulse 1.5s infinite' }}>📊</div>
        Loading fairness metrics…
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        background: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b',
        padding: '16px 20px', borderRadius: '8px', fontSize: '14px',
      }}>
        ⚠️ {error}
      </div>
    );
  }

  if (!data) return null;

  const chartOptions = (title) => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: true, text: title, color: '#334155', font: { size: 15, weight: 600 } },
      tooltip: {
        callbacks: {
          label: (ctx) => `Approval Rate: ${(ctx.raw * 100).toFixed(1)}%`,
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        ticks: {
          callback: (v) => `${(v * 100).toFixed(0)}%`,
          color: '#64748b',
          font: { size: 12 },
        },
        grid: { color: '#f1f5f9' },
      },
      x: {
        ticks: { color: '#334155', font: { size: 12, weight: 500 } },
        grid: { display: false },
      },
    },
  });

  const ageChartData = {
    labels: (data.by_age || []).map(d => d.age_band),
    datasets: [{
      data: (data.by_age || []).map(d => d.approval_rate),
      backgroundColor: ['#6366f1', '#8b5cf6', '#a855f7', '#c084fc'],
      borderRadius: 6,
      maxBarThickness: 56,
    }],
  };

  const regionChartData = {
    labels: (data.by_region || []).map(d => d.region),
    datasets: [{
      data: (data.by_region || []).map(d => d.approval_rate),
      backgroundColor: ['#0ea5e9', '#06b6d4', '#14b8a6', '#10b981', '#22c55e', '#84cc16'],
      borderRadius: 6,
      maxBarThickness: 56,
    }],
  };

  const hasAgeData = data.by_age && data.by_age.some(d => d.total > 0);
  const hasRegionData = data.by_region && data.by_region.length > 0;

  return (
    <div>
      <div style={{
        display: 'grid',
        gridTemplateColumns: hasRegionData ? '1fr 1fr' : '1fr',
        gap: '24px',
      }}>
        {/* Age Band Chart */}
        <div style={{
          background: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0',
          padding: '24px', minHeight: '320px',
        }}>
          {hasAgeData ? (
            <div style={{ height: '280px' }}>
              <Bar data={ageChartData} options={chartOptions('Approval Rate by Age Band')} />
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '80px 0', color: '#94a3b8' }}>
              <div style={{ fontSize: '28px', marginBottom: '8px' }}>📭</div>
              No age-band data recorded yet.
              <br />
              <span style={{ fontSize: '12px' }}>Submit applications with age_band to populate this chart.</span>
            </div>
          )}
        </div>

        {/* Region Chart */}
        <div style={{
          background: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0',
          padding: '24px', minHeight: '320px',
        }}>
          {hasRegionData ? (
            <div style={{ height: '280px' }}>
              <Bar data={regionChartData} options={chartOptions('Approval Rate by Region')} />
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '80px 0', color: '#94a3b8' }}>
              <div style={{ fontSize: '28px', marginBottom: '8px' }}>📭</div>
              No region data recorded yet.
              <br />
              <span style={{ fontSize: '12px' }}>Submit applications with region to populate this chart.</span>
            </div>
          )}
        </div>
      </div>

      {/* Summary Table */}
      {hasAgeData && (
        <div style={{
          marginTop: '24px', background: '#fff', borderRadius: '12px',
          border: '1px solid #e2e8f0', padding: '20px', overflow: 'auto',
        }}>
          <h4 style={{ margin: '0 0 12px', color: '#334155', fontSize: '14px', fontWeight: 600 }}>
            Detailed Breakdown
          </h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px', color: '#64748b', fontWeight: 600 }}>Group</th>
                <th style={{ textAlign: 'right', padding: '8px', color: '#64748b', fontWeight: 600 }}>Total</th>
                <th style={{ textAlign: 'right', padding: '8px', color: '#64748b', fontWeight: 600 }}>Approved</th>
                <th style={{ textAlign: 'right', padding: '8px', color: '#64748b', fontWeight: 600 }}>Rate</th>
              </tr>
            </thead>
            <tbody>
              {[...(data.by_age || []), ...(data.by_region || [])].map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px', color: '#334155', fontWeight: 500 }}>{row.age_band || row.region}</td>
                  <td style={{ padding: '8px', textAlign: 'right', color: '#475569' }}>{row.total}</td>
                  <td style={{ padding: '8px', textAlign: 'right', color: '#475569' }}>{row.approved}</td>
                  <td style={{ padding: '8px', textAlign: 'right', fontWeight: 600, color: row.approval_rate >= 0.5 ? '#10b981' : '#ef4444' }}>
                    {(row.approval_rate * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div style={{
        marginTop: '16px', padding: '12px 16px', background: '#f0fdf4',
        border: '1px solid #bbf7d0', borderRadius: '8px', fontSize: '12px',
        color: '#166534', display: 'flex', alignItems: 'center', gap: '8px',
      }}>
        ⚖️ Fairness monitoring helps detect potential bias in lending decisions across demographic groups.
      </div>
    </div>
  );
}
