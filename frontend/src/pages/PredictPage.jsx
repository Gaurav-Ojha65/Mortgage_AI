import React, { useState } from 'react';
import Card from '../components/ui/Card';
import AlertBanner from '../components/ui/AlertBanner';
import StepForm from '../components/layout/StepForm';
import RiskGauge from '../components/data/RiskGauge';
import ProbBar from '../components/data/ProbBar';
import { api } from '../api';
import { useAppStore } from '../store';
import './PredictPage.css';

export default function PredictPage() {
  const { currentPrediction, isPredicting, predictionError, setPredictionState, clearPrediction } = useAppStore();

  const [lastSubmittedData, setLastSubmittedData] = useState(null);
  const [simExpanded, setSimExpanded] = useState(false);
  const [simCreditScore, setSimCreditScore] = useState(700);
  const [simDTI, setSimDTI] = useState(30);
  const [simResult, setSimResult] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simError, setSimError] = useState(null);

  const handlePredict = async (data) => {
    setPredictionState({ isPredicting: true, predictionError: null });
    setLastSubmittedData(data);
    setSimResult(null);
    try {
      const result = await api.predict(data);
      setPredictionState({ currentPrediction: result, isPredicting: false });
      
      setSimCreditScore(data.credit_score || 700);
      const inc = data.income || 5000;
      const exp = data.monthly_expenses || (inc * 0.3);
      const dti = ((exp / inc) * 100) || 30;
      setSimDTI(Math.round(dti));
    } catch (err) {
      setPredictionState({ predictionError: err.message, isPredicting: false });
    }
  };

  const handleSimulate = async () => {
    if (!lastSubmittedData) return;
    setIsSimulating(true);
    setSimError(null);
    try {
      const simData = { ...lastSubmittedData };
      simData.credit_score = simCreditScore;
      simData.monthly_expenses = (simDTI / 100) * (simData.income || 5000);
      const result = await api.predict(simData);
      setSimResult(result);
    } catch (err) {
      setSimError(err.message);
    } finally {
      setIsSimulating(false);
    }
  };

  return (
    <div className="predict-page">
      <div className="predict-grid">
        <div className="predict-left">
          <Card className="form-card">
            <div className="card-header">
              <h2>New Application</h2>
              <p className="text-secondary">Enter the applicant details below to run the AI risk ensemble.</p>
            </div>
            
            {predictionError && (
              <div className="mb-md">
                <AlertBanner type="error" title="Prediction Failed" message={predictionError} />
              </div>
            )}
            
            <StepForm onSubmit={handlePredict} isPredicting={isPredicting} />
          </Card>
        </div>

        <div className="predict-right">
          <Card className="result-card">
            <div className="card-header">
              <h2>Analysis Result</h2>
            </div>
            
            {!currentPrediction && !isPredicting && (
              <div className="empty-result">
                <svg className="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-muted">Awaiting application data...</p>
              </div>
            )}

            {isPredicting && (
              <div className="predicting-state">
                <div className="scanner"></div>
                <p className="mono animate-pulse text-gold">Running LightGBM v3.1 Inference...</p>
              </div>
            )}

            {currentPrediction && !isPredicting && (
              <div className="prediction-results slide-up">
                <div className="result-badge-container">
                  {(() => {
                    const norm = (currentPrediction.decision || '').toUpperCase().replace(/_/g, ' ');
                    const cls = norm === 'APPROVE' || norm === 'APPROVED' ? 'approve' : (norm === 'REJECT' || norm === 'REJECTED' ? 'reject' : 'manual-review');
                    const label = norm === 'APPROVE' || norm === 'APPROVED' ? 'APPROVE' : (norm === 'REJECT' || norm === 'REJECTED' ? 'REJECT' : 'MANUAL REVIEW');
                    return (
                      <span className={`decision-badge ${cls}`}>
                        {label}
                      </span>
                    );
                  })()}
                </div>
                
                <div className="gauge-section">
                  <RiskGauge score={currentPrediction.risk_score || ((currentPrediction.calibrated_default_probability !== undefined ? currentPrediction.calibrated_default_probability : (currentPrediction.default_probability || 0)) * 100)} />
                </div>
                
                <div className="probs-section">
                  <ProbBar probability={currentPrediction.approval_probability || (1 - (currentPrediction.calibrated_default_probability || 0))} label="Approval Probability" />
                </div>
                
                <div className="factors-section">
                  <h4 className="factors-title">Model Insights</h4>
                  <ul className="factors-list">
                    {(currentPrediction.plain_english || currentPrediction.top_factors || ['Credit Score', 'Debt-to-Income Ratio', 'Loan Amount']).map((f, i) => (
                      <li key={i} className="factor-item">
                        <span className="factor-dot"></span>
                        {f}
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="simulator-panel" style={{ marginTop: '20px', borderTop: '1px solid #e2e8f0', paddingTop: '15px' }}>
                  <div 
                    onClick={() => setSimExpanded(!simExpanded)} 
                    style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
                  >
                    <h4 style={{ margin: 0, color: '#334155', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px' }}>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20v-6M6 20V10M18 20V4"></path></svg>
                      Simulate Changes
                    </h4>
                    <span style={{ transform: simExpanded ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s', color: '#64748b' }}>▼</span>
                  </div>
                  
                  {simExpanded && (
                    <div style={{ marginTop: '15px', padding: '15px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                      <div style={{ marginBottom: '15px' }}>
                        <label style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: 600, color: '#475569', marginBottom: '8px' }}>
                          Credit Score <span>{simCreditScore}</span>
                        </label>
                        <input 
                          type="range" min="300" max="850" 
                          value={simCreditScore} 
                          onChange={(e) => setSimCreditScore(parseInt(e.target.value))}
                          style={{ width: '100%', cursor: 'pointer', accentColor: '#3b82f6' }}
                        />
                      </div>
                      
                      <div style={{ marginBottom: '15px' }}>
                        <label style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: 600, color: '#475569', marginBottom: '8px' }}>
                          DTI Ratio <span>{simDTI}%</span>
                        </label>
                        <input 
                          type="range" min="0" max="100" 
                          value={simDTI} 
                          onChange={(e) => setSimDTI(parseInt(e.target.value))}
                          style={{ width: '100%', cursor: 'pointer', accentColor: '#3b82f6' }}
                        />
                      </div>
                      
                      <button 
                        onClick={handleSimulate}
                        disabled={isSimulating}
                        style={{ width: '100%', padding: '8px', background: '#1e293b', color: 'white', border: 'none', borderRadius: '6px', fontWeight: 600, cursor: isSimulating ? 'not-allowed' : 'pointer', opacity: isSimulating ? 0.7 : 1, transition: 'all 0.2s' }}
                      >
                        {isSimulating ? 'Running Simulation...' : 'Simulate'}
                      </button>
                      
                      {simError && <p style={{ color: '#ef4444', fontSize: '12px', marginTop: '10px' }}>{simError}</p>}
                      
                      {simResult && (
                        <div style={{ marginTop: '15px', padding: '12px', background: 'white', borderRadius: '6px', border: '1px solid #cbd5e1', animation: 'fadeIn 0.3s ease-in' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                            <span style={{ fontSize: '13px', fontWeight: 600, color: '#334155' }}>New Decision:</span>
                            <span className={`decision-badge ${simResult.decision?.toLowerCase() || 'review'}`} style={{ transform: 'scale(0.85)', margin: 0 }}>
                              {simResult.decision || 'REVIEW'}
                            </span>
                          </div>
                          
                          {(() => {
                            const origScore = Math.round((currentPrediction.default_probability || 0) * 100);
                            const newScore = Math.round((simResult.default_probability || 0) * 100);
                            const delta = newScore - origScore;
                            const isImprovement = delta < 0;
                            const color = isImprovement ? '#10b981' : (delta > 0 ? '#ef4444' : '#64748b');
                            const dirText = isImprovement ? 'improved' : (delta > 0 ? 'worsened' : 'remained');
                            
                            return (
                              <div style={{ fontSize: '13px', color: '#475569', background: `${color}15`, padding: '8px', borderRadius: '4px', borderLeft: `3px solid ${color}` }}>
                                Default Risk <strong style={{ color }}>{dirText}</strong> from <strong>{origScore}%</strong> to <strong>{newScore}%</strong>
                              </div>
                            );
                          })()}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
