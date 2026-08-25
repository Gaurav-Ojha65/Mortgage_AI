import React, { forwardRef } from 'react';
import './Input.css';

const Input = forwardRef(({ label, error, helperText, className = '', ...props }, ref) => {
  const hasError = !!error;

  return (
    <div className={`ui-input-wrapper ${className}`}>
      {label && <label className="ui-label">{label}</label>}
      <input 
        ref={ref}
        className={`ui-input ${hasError ? 'error' : ''}`}
        {...props} 
      />
      {hasError && <span className="ui-error-text">{error}</span>}
      {!hasError && helperText && <span className="ui-helper-text">{helperText}</span>}
    </div>
  );
});

Input.displayName = 'Input';
export default Input;
