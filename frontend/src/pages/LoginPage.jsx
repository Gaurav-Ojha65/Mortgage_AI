import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import { useAppStore } from '../store';
import { api } from '../api';
import './LoginPage.css';

export default function LoginPage() {
  const [username,setUsername]=useState(''); const [password,setPassword]=useState(''); const [isLoading,setIsLoading]=useState(false);
  const navigate=useNavigate(); const login=useAppStore(s=>s.login);
  const handleSubmit=async e=>{e.preventDefault(); if(!username||!password){toast.warning('Enter username and password');return;} setIsLoading(true); try {const response=await api.login(username,password); const token=response.token||response.access_token; const user=response.user||{username}; if(!token) throw new Error('Invalid authentication response'); login(user,token); toast.success(`Welcome, ${user.full_name||username}`); navigate('/dashboard');} catch(error){toast.error(error.message||'Invalid username or password');} finally{setIsLoading(false);}};
  return <div className="login-page"><div className="login-card"><div className="login-header"><div className="login-logo">⌂</div><h1 className="login-title">Mortgage AI</h1><p className="login-subtitle">Secure mortgage application & underwriting platform</p></div><form className="login-form" onSubmit={handleSubmit}><div className="login-field"><label htmlFor="username">Username</label><input id="username" className="login-input" value={username} onChange={e=>setUsername(e.target.value)} autoComplete="username" placeholder="Enter username"/></div><div className="login-field"><label htmlFor="password">Password</label><input id="password" type="password" className="login-input" value={password} onChange={e=>setPassword(e.target.value)} autoComplete="current-password" placeholder="Enter password"/></div><button type="submit" className="login-btn" disabled={isLoading}>{isLoading?'Signing in…':'Sign In'}</button></form><div className="login-footer"><p>Three-role access: <strong>Applicant</strong> · <strong>Loan Officer</strong> · <strong>Admin</strong></p><p style={{fontSize:'12px',color:'#94a3b8',marginTop:8}}>Demo: user / user2024 · officer / lo2024 · admin / admin123</p></div></div></div>;
}
