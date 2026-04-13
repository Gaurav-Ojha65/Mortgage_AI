import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { toast } from 'react-toastify';
import { analyzeLoan } from '../utils/api';

const MonteCarlo3D = () => {
  const [formData, setFormData] = useState({
    income: '6000',
    loan_amount: '100000',
    interest_rate: '8.5',
    loan_term: '5',
    credit_score: '720',
    existing_loans: '0'
  });
  const [loading, setLoading] = useState(false);
  const [simulationData, setSimulationData] = useState(null);
  const [activeView, setActiveView] = useState('3d');

  // Generate Monte Carlo simulation data
  const generateSimulationData = () => {
    const nSimulations = 10000;
    const income = parseFloat(formData.income);
    const loanAmount = parseFloat(formData.loan_amount);
    const interestRate = parseFloat(formData.interest_rate);
    const loanTerm = parseInt(formData.loan_term);

    // Calculate base EMI
    const monthlyRate = interestRate / 12 / 100;
    const nMonths = loanTerm * 12;
    const emi = loanAmount * monthlyRate * Math.pow(1 + monthlyRate, nMonths) /
                (Math.pow(1 + monthlyRate, nMonths) - 1);

    const simulations = [];
    const incomeVariations = [];
    const emiRatios = [];
    const defaultProbs = [];

    for (let i = 0; i < nSimulations; i++) {
      // Simulate income variations (-20% to +20%)
      const incomeFactor = 0.8 + Math.random() * 0.4;
      const simulatedIncome = income * incomeFactor;

      // Simulate interest rate variations (-2% to +2%)
      const rateVariation = (Math.random() - 0.5) * 4;
      const simulatedRate = Math.max(0.1, interestRate + rateVariation);
      const simMonthlyRate = simulatedRate / 12 / 100;

      // Calculate EMI with variations
      const simEmi = loanAmount * simMonthlyRate * Math.pow(1 + simMonthlyRate, nMonths) /
                     (Math.pow(1 + simMonthlyRate, nMonths) - 1);

      // EMI to income ratio
      const emiRatio = simEmi / simulatedIncome;

      // Default probability based on ratio
      const defaultProb = Math.min(0.95, Math.max(0.05, emiRatio * 2 - 0.1 + (Math.random() - 0.5) * 0.1));

      // Scenario type
      let scenario;
      if (defaultProb < 0.15) scenario = 'Optimistic';
      else if (defaultProb < 0.35) scenario = 'Base';
      else scenario = 'Pessimistic';

      simulations.push({
        id: i,
        income: simulatedIncome,
        emi: simEmi,
        emiRatio: emiRatio,
        defaultProb: defaultProb,
        scenario: scenario,
        interestRate: simulatedRate
      });

      incomeVariations.push(simulatedIncome);
      emiRatios.push(emiRatio);
      defaultProbs.push(defaultProb);
    }

    return {
      simulations,
      incomeVariations,
      emiRatios,
      defaultProbs,
      baseEmi: emi,
      baseIncome: income,
      avgDefaultProb: defaultProbs.reduce((a, b) => a + b, 0) / defaultProbs.length,
      maxDefaultProb: Math.max(...defaultProbs),
      minDefaultProb: Math.min(...defaultProbs)
    };
  };

  const runSimulation = async () => {
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

      const simData = generateSimulationData();
      setSimulationData({ ...simData, apiResponse: response });
      toast.success('10,000 simulations complete!');
    } catch (error) {
      console.error('Simulation failed:', error);
      toast.warning('Using demo data - API connection failed');
      const simData = generateSimulationData();
      setSimulationData(simData);
    } finally {
      setLoading(false);
    }
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    runSimulation();
  }, []);

  const get3DPlotData = () => {
    if (!simulationData) return [];

    const { simulations } = simulationData;

    // Sample data for performance (take every 10th point)
    const sampledSimulations = simulations.filter((_, i) => i % 10 === 0);
    const optimistic = sampledSimulations.filter(s => s.scenario === 'Optimistic');
    const base = sampledSimulations.filter(s => s.scenario === 'Base');
    const pessimistic = sampledSimulations.filter(s => s.scenario === 'Pessimistic');

    return [
      {
        x: optimistic.map(s => s.income),
        y: optimistic.map(s => s.emiRatio * 100),
        z: optimistic.map(s => s.defaultProb * 100),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Optimistic',
        marker: {
          size: 3,
          color: '#10B981',
          opacity: 0.8,
          line: { color: '#10B981', width: 0.3 }
        },
        hovertemplate: 'Income: $%{x:,.0f}<br>EMI Ratio: %{y:.1f}%<br>Default Risk: %{z:.1f}%<extra></extra>'
      },
      {
        x: base.map(s => s.income),
        y: base.map(s => s.emiRatio * 100),
        z: base.map(s => s.defaultProb * 100),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Base Case',
        marker: {
          size: 3,
          color: '#8B5CF6',
          opacity: 0.8,
          line: { color: '#8B5CF6', width: 0.3 }
        },
        hovertemplate: 'Income: $%{x:,.0f}<br>EMI Ratio: %{y:.1f}%<br>Default Risk: %{z:.1f}%<extra></extra>'
      },
      {
        x: pessimistic.map(s => s.income),
        y: pessimistic.map(s => s.emiRatio * 100),
        z: pessimistic.map(s => s.defaultProb * 100),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Pessimistic',
        marker: {
          size: 3,
          color: '#F59E0B',
          opacity: 0.8,
          line: { color: '#F59E0B', width: 0.3 }
        },
        hovertemplate: 'Income: $%{x:,.0f}<br>EMI Ratio: %{y:.1f}%<br>Default Risk: %{z:.1f}%<extra></extra>'
      }
    ];
  };

  const getHistogramData = () => {
    if (!simulationData) return [];

    return [
      {
        x: simulationData.defaultProbs.map(p => p * 100),
        type: 'histogram',
        name: 'Default Probability Distribution',
        nbinsx: 60,
        marker: {
          color: 'rgba(245, 158, 11, 0.7)',
          line: { color: '#F59E0B', width: 1 }
        },
        hovertemplate: 'Default Probability: %{x:.1f}%<br>Count: %{y}<extra></extra>'
      }
    ];
  };

  const getHeatmapData = () => {
    if (!simulationData) return [];

    // Create 2D histogram data
    const xBins = 25;
    const yBins = 25;
    const xMin = Math.min(...simulationData.incomeVariations);
    const xMax = Math.max(...simulationData.incomeVariations);
    const yMin = Math.min(...simulationData.emiRatios);
    const yMax = Math.max(...simulationData.emiRatios);

    const xStep = (xMax - xMin) / xBins;
    const yStep = (yMax - yMin) / yBins;

    const z = Array(yBins).fill(0).map(() => Array(xBins).fill(0));

    simulationData.simulations.forEach(s => {
      const xIdx = Math.min(Math.floor((s.income - xMin) / xStep), xBins - 1);
      const yIdx = Math.min(Math.floor((s.emiRatio - yMin) / yStep), yBins - 1);
      z[yIdx][xIdx]++;
    });

    const xLabels = Array(xBins).fill(0).map((_, i) => xMin + i * xStep);
    const yLabels = Array(yBins).fill(0).map((_, i) => (yMin + i * yStep) * 100);

    return [
      {
        z: z,
        x: xLabels,
        y: yLabels,
        type: 'heatmap',
        colorscale: [
          [0, '#0F172A'],
          [0.2, '#1E293B'],
          [0.4, '#8B5CF6'],
          [0.6, '#A78BFA'],
          [0.8, '#FBBF24'],
          [1, '#F59E0B']
        ],
        hovertemplate: 'Income: $%{x:,.0f}<br>EMI Ratio: %{y:.1f}%<br>Simulations: %{z}<extra></extra>'
      }
    ];
  };

  const plotLayout3D = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#94a3b8', family: 'IBM Plex Sans, sans-serif' },
    scene: {
      xaxis: {
        title: 'Monthly Income ($)',
        color: '#94a3b8',
        gridcolor: 'rgba(148, 163, 184, 0.1)',
        tickformat: '$,.0f'
      },
      yaxis: {
        title: 'EMI to Income Ratio (%)',
        color: '#94a3b8',
        gridcolor: 'rgba(148, 163, 184, 0.1)'
      },
      zaxis: {
        title: 'Default Probability (%)',
        color: '#94a3b8',
        gridcolor: 'rgba(148, 163, 184, 0.1)'
      },
      bgcolor: 'rgba(15, 23, 42, 0.5)',
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.2 }
      }
    },
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(15, 23, 42, 0.9)',
      bordercolor: 'rgba(245, 158, 11, 0.3)',
      borderwidth: 1,
      font: { color: '#f8fafc' }
    },
    margin: { l: 0, r: 0, t: 30, b: 0 },
    showlegend: true
  };

  const plotLayout2D = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#94a3b8', family: 'IBM Plex Sans, sans-serif' },
    xaxis: {
      title: 'Default Probability (%)',
      color: '#94a3b8',
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      zerolinecolor: 'rgba(148, 163, 184, 0.2)'
    },
    yaxis: {
      title: 'Frequency',
      color: '#94a3b8',
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      zerolinecolor: 'rgba(148, 163, 184, 0.2)'
    },
    margin: { l: 60, r: 20, t: 30, b: 60 },
    bargap: 0.1
  };

  const plotLayoutHeatmap = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#94a3b8', family: 'IBM Plex Sans, sans-serif' },
    xaxis: {
      title: 'Monthly Income ($)',
      color: '#94a3b8',
      tickformat: '$,.0f'
    },
    yaxis: {
      title: 'EMI Ratio (%)',
      color: '#94a3b8'
    },
    margin: { l: 60, r: 20, t: 30, b: 60 }
  };

  const plotConfig = {
    displayModeBar: true,
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
  };

  const viewButtons = [
    { key: '3d', label: '3D Visualization', icon: 'M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5' },
    { key: 'distribution', label: 'Distribution', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
    { key: 'heatmap', label: 'Density Map', icon: 'M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0121 18.382V7.618a1 1 0 00-.553-.894L15 7m0 13V7' }
  ];

  const scenarioData = simulationData ? [
    { name: 'Optimistic', color: 'emerald', bg: 'bg-emerald-500', desc: 'High income stability, favorable rates', count: simulationData.simulations.filter(s => s.scenario === 'Optimistic').length },
    { name: 'Base Case', color: 'purple', bg: 'bg-purple-500', desc: 'Normal market conditions', count: simulationData.simulations.filter(s => s.scenario === 'Base').length },
    { name: 'Pessimistic', color: 'amber', bg: 'bg-amber-500', desc: 'Economic stress scenarios', count: simulationData.simulations.filter(s => s.scenario === 'Pessimistic').length }
  ] : [];

  return (
    <div className="pt-24 pb-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Trust Header */}
      <div className="text-center mb-8 space-y-4">
        <div className="inline-flex items-center gap-2 text-slate-400 text-sm">
          <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>Bank-Grade Risk Modeling</span>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-white">
          Monte Carlo Simulation
        </h1>
        <p className="text-slate-400 max-w-2xl mx-auto">
          10,000+ simulations model different economic scenarios to predict
          default probability across income variations and market conditions.
        </p>
      </div>

      {/* Controls */}
      <div className="glass-card rounded-2xl p-6 mb-8">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
          {[
            { name: 'income', label: 'Income', prefix: '$', type: 'number' },
            { name: 'loan_amount', label: 'Loan', prefix: '$', type: 'number' },
            { name: 'interest_rate', label: 'Rate', suffix: '%', type: 'number', step: '0.1' },
            { name: 'loan_term', label: 'Term', suffix: 'yr', type: 'number' },
            { name: 'credit_score', label: 'Credit', type: 'number' },
            { name: 'existing_loans', label: 'Loans', type: 'number' }
          ].map((field) => (
            <div key={field.name}>
              <label htmlFor={field.name} className="block text-xs text-slate-400 mb-1 uppercase tracking-wider">{field.label}</label>
              <div className="relative">
                {field.prefix && (
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 text-sm pointer-events-none">{field.prefix}</span>
                )}
                <input
                  type={field.type}
                  id={field.name}
                  name={field.name}
                  value={formData[field.name]}
                  onChange={(e) => setFormData(prev => ({ ...prev, [field.name]: e.target.value }))}
                  step={field.step}
                  className={`w-full bg-slate-800/50 border border-slate-700 rounded-lg py-2 text-white text-sm focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500/50 transition-all ${field.prefix ? 'pl-7' : 'pl-3'} ${field.suffix ? 'pr-10' : 'pr-3'}`}
                />
                {field.suffix && (
                  <span className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 text-sm pointer-events-none">{field.suffix}</span>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            {viewButtons.map((view) => (
              <button
                key={view.key}
                onClick={() => setActiveView(view.key)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${
                  activeView === view.key
                    ? 'bg-amber-500 text-slate-900'
                    : 'bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700'
                }`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={view.icon} />
                </svg>
                {view.label}
              </button>
            ))}
          </div>

          <button
            onClick={runSimulation}
            disabled={loading}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Running 10,000 Simulations...</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Run Simulation</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="glass-card rounded-2xl p-6">
        <div className="h-[500px] md:h-[600px]">
          {activeView === '3d' && simulationData && (
            <Plot
              data={get3DPlotData()}
              layout={plotLayout3D}
              config={plotConfig}
              style={{ width: '100%', height: '100%' }}
            />
          )}
          {activeView === 'distribution' && simulationData && (
            <Plot
              data={getHistogramData()}
              layout={plotLayout2D}
              config={plotConfig}
              style={{ width: '100%', height: '100%' }}
            />
          )}
          {activeView === 'heatmap' && simulationData && (
            <Plot
              data={getHeatmapData()}
              layout={plotLayoutHeatmap}
              config={plotConfig}
              style={{ width: '100%', height: '100%' }}
            />
          )}
        </div>
      </div>

      {/* Stats Grid */}
      {simulationData && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <div className="metric-card">
            <div className="text-slate-400 text-sm mb-1">Simulations Run</div>
            <div className="metric-value text-2xl">10,000</div>
          </div>
          <div className="metric-card">
            <div className="text-slate-400 text-sm mb-1">Avg Default Risk</div>
            <div className="text-2xl font-bold text-amber-400 font-mono">{(simulationData.avgDefaultProb * 100).toFixed(1)}%</div>
          </div>
          <div className="metric-card">
            <div className="text-slate-400 text-sm mb-1">Max Risk Scenario</div>
            <div className="text-2xl font-bold text-red-400 font-mono">{(simulationData.maxDefaultProb * 100).toFixed(1)}%</div>
          </div>
          <div className="metric-card">
            <div className="text-slate-400 text-sm mb-1">Base EMI</div>
            <div className="text-2xl font-bold text-emerald-400 font-mono">
              ${simulationData.baseEmi.toFixed(0)}
            </div>
          </div>
        </div>
      )}

      {/* Scenario Breakdown */}
      {simulationData && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
          {scenarioData.map((scenario) => (
            <div key={scenario.name} className="glass-card rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${scenario.bg}`} />
                  <span className="text-white font-medium">{scenario.name}</span>
                </div>
                <span className={`text-lg font-mono ${scenario.color === 'emerald' ? 'text-emerald-400' : scenario.color === 'purple' ? 'text-purple-400' : 'text-amber-400'}`}>
                  {((scenario.count / 10000) * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-slate-500 text-sm mb-3">{scenario.desc}</p>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${scenario.bg} transition-all duration-1000`}
                  style={{ width: `${(scenario.count / 10000) * 100}%` }}
                />
              </div>
              <div className="mt-2 text-xs text-slate-500">
                {scenario.count.toLocaleString()} simulations
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default MonteCarlo3D;
