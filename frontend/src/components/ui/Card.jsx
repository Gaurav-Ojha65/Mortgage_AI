import React from 'react';
import './Card.css';

export default function Card({ children, className = '', elevated = false, interactive = false, ...props }) {
  const classes = ['ui-card', elevated ? 'elevated' : '', interactive ? 'interactive' : '', className].filter(Boolean).join(' ');
  
  return (
    <div className={classes} {...props}>
      {children}
    </div>
  );
}
