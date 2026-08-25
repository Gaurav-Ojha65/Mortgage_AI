import React, { useEffect, useState } from 'react';
import './ProbBar.css';

export default function ProbBar({ probability, label }) {
  const [currentProb, setCurrentProb] = useState(0);

  useEffect(() => {
    // Small delay to ensure the mount animation happens
    const raf = requestAnimationFrame(() => {
      setCurrentProb(probability);
    });
    return () => cancelAnimationFrame(raf);
  }, [probability]);

  // Color logic
  const getColor = (p) => {
    if (p > 0.8) return 'var(--success)';
    if (p > 0.5) return 'var(--warning)';
    return 'var(--danger)';
  };

  const color = getColor(currentProb);
  const percentage = (currentProb * 100).toFixed(1);

  return (
    <div className="prob-bar-wrapper">
      <div className="prob-bar-header">
        <span className="prob-label">{label}</span>
        <span className="prob-value mono">{percentage}%</span>
      </div>
      <div className="prob-track">
        <div 
          className="prob-fill" 
          style={{ 
            width: `${percentage}%`,
            backgroundColor: color,
            boxShadow: `0 0 10px ${color}`
          }} 
        />
      </div>
    </div>
  );
}
