import React, { useEffect, useState } from 'react';
import './RiskGauge.css';

export default function RiskGauge({ score, maxScore = 100, label = 'Risk Score' }) {
  const [currentScore, setCurrentScore] = useState(0);
  
  // Smooth animation effect
  useEffect(() => {
    const timer = setTimeout(() => {
      setCurrentScore(score);
    }, 100);
    return () => clearTimeout(timer);
  }, [score]);

  // Gauge calculations
  const radius = 60;
  const circumference = Math.PI * radius; // Half-circle
  const strokeDashoffset = circumference - (currentScore / maxScore) * circumference;

  const getRiskColor = (s) => {
    if (s > 75) return 'var(--danger)'; // High risk
    if (s > 40) return 'var(--warning)'; // Medium risk
    return 'var(--success)'; // Low risk
  };

  const color = getRiskColor(currentScore);

  return (
    <div className="risk-gauge-wrapper">
      <div className="gauge-container">
        <svg viewBox="0 0 160 100" className="gauge-svg">
          {/* Background Arc */}
          <path
            className="gauge-bg"
            d="M 20 80 A 60 60 0 0 1 140 80"
            fill="none"
            strokeWidth="16"
            strokeLinecap="round"
          />
          {/* Progress Arc */}
          <path
            className="gauge-progress"
            d="M 20 80 A 60 60 0 0 1 140 80"
            fill="none"
            strokeWidth="16"
            strokeLinecap="round"
            stroke={color}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
          />
        </svg>
        <div className="gauge-content">
          <span className="gauge-score mono" style={{ color }}>{Math.round(currentScore)}</span>
          <span className="gauge-label">{label}</span>
        </div>
      </div>
    </div>
  );
}
