import React from 'react';
import { useLocation } from 'react-router-dom';
import { useAppStore } from '../../store';
import './TopNavbar.css';

export default function TopNavbar() {
  const location = useLocation();
  const { toggleSidebar } = useAppStore();

  const getPageTitle = () => {
    switch(location.pathname) {
      case '/dashboard': return 'Dashboard';
      case '/predict': return 'Risk Prediction';
      case '/analytics': return 'Analytics';
      case '/history': return 'History';
      default: return 'Mortgage Risk Analytics';
    }
  };

  return (
    <header className="top-navbar">
      <div className="navbar-left">
        <button className="mobile-toggle" onClick={toggleSidebar}>
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        <h1 className="page-title">{getPageTitle()}</h1>
      </div>
      <div className="navbar-right">
        <div className="system-status">
          <span className="status-indicator"></span>
          <span className="status-text mono">SYSTEM ONLINE</span>
        </div>
        <div className="user-profile">
          <div className="avatar">A</div>
        </div>
      </div>
    </header>
  );
}
