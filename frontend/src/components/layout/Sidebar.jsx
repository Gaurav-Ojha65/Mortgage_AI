import React from 'react';
import { NavLink } from 'react-router-dom';
import { useAppStore } from '../../store';
import './Sidebar.css';

const items = {
  common: [
    {id:'dashboard',label:'Dashboard',path:'/dashboard'},
    {id:'history',label:'My Applications',path:'/history'},
    {id:'emi',label:'EMI Calculator',path:'/emi'},
  ],
  applicant: [
    {id:'eligibility',label:'Eligibility Check',path:'/eligibility'},
    {id:'predict',label:'Apply / Risk Check',path:'/predict'},
  ],
  officer: [
    {id:'predict',label:'Review Application',path:'/predict'},
    {id:'analytics',label:'Risk Analytics',path:'/analytics'},
  ],
  admin: [
    {id:'analytics',label:'Analytics',path:'/analytics'},
    {id:'admin',label:'Admin Console',path:'/admin'},
  ],
};

export default function Sidebar() {
  const { isSidebarCollapsed, toggleSidebar, user, logout } = useAppStore();
  const role = user?.role;
  const roleItems = role === 'admin' ? items.admin : (role === 'loan_officer' || role === 'underwriter') ? items.officer : items.applicant;
  const navItems = [...items.common, ...roleItems];
  return (
    <aside className={`sidebar ${isSidebarCollapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header"><div className="logo-container"><span className="logo-text">{isSidebarCollapsed ? 'M' : 'Mortgage AI'}</span></div></div>
      <nav className="sidebar-nav">
        {navItems.map(item => <NavLink key={`${item.id}-${item.path}`} to={item.path} className={({isActive})=>`nav-item ${isActive?'active':''}`} title={isSidebarCollapsed?item.label:''}><span className="nav-label">{!isSidebarCollapsed && item.label}</span></NavLink>)}
      </nav>
      <div className="sidebar-footer">
        {!isSidebarCollapsed && <div style={{padding:'10px 20px',color:'#94a3b8',fontSize:12,textTransform:'uppercase'}}>{role === 'loan_officer' || role === 'underwriter' ? 'Loan Officer' : role === 'admin' ? 'Administrator' : 'Applicant'}</div>}
        <button onClick={logout} className="nav-item" style={{width:'100%',border:'none',background:'transparent',cursor:'pointer'}}><span className="nav-label" style={{color:'#f87171'}}>{!isSidebarCollapsed && 'Log Out'}</span></button>
        <button onClick={toggleSidebar} className="collapse-btn">{isSidebarCollapsed ? '→' : '←'}</button>
      </div>
    </aside>
  );
}
