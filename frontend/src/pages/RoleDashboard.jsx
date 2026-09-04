import React from 'react';
import { Link } from 'react-router-dom';
import { useAppStore } from '../store';

const Card = ({to, title, text}) => (
  <Link to={to} style={{display:'block',padding:24,border:'1px solid rgba(148,163,184,.2)',borderRadius:16,textDecoration:'none',color:'inherit',background:'rgba(15,23,42,.35)'}}>
    <h3 style={{margin:'0 0 8px'}}>{title}</h3><p style={{margin:0,color:'#94a3b8'}}>{text}</p>
  </Link>
);

export default function RoleDashboard(){
  const user = useAppStore(s=>s.user) || {};
  const role = user.role;
  if(role === 'admin') return <div><h1>Admin Dashboard</h1><p>Platform governance, users, auditability and model oversight.</p><div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(220px,1fr))',gap:16,marginTop:24}}><Card to="/admin" title="User Management" text="Create and manage platform accounts and roles."/><Card to="/analytics" title="Model Analytics" text="Monitor model performance and risk analytics."/><Card to="/history" title="Audit & History" text="Review decision history and governance records."/></div></div>;
  if(role === 'loan_officer' || role === 'underwriter') return <div><h1>Loan Officer Dashboard</h1><p>Review applications, understand AI risk, and manage underwriting decisions.</p><div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(220px,1fr))',gap:16,marginTop:24}}><Card to="/predict" title="Review Application" text="Run the canonical calibrated risk assessment."/><Card to="/history" title="Application Queue" text="Review recent applications and decision history."/><Card to="/analytics" title="Risk Analytics" text="Inspect model and portfolio analytics."/></div></div>;
  return <div><h1>Applicant Dashboard</h1><p>Apply for a mortgage, check eligibility, and understand your application.</p><div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(220px,1fr))',gap:16,marginTop:24}}><Card to="/eligibility" title="Check Eligibility" text="Start with a quick affordability and eligibility pre-check."/><Card to="/predict" title="Apply / Assess Risk" text="Submit your financial information for AI risk assessment."/><Card to="/history" title="My Applications" text="Track your own application and decision history."/><Card to="/emi" title="EMI Calculator" text="Estimate your monthly mortgage payment."/></div></div>;
}
