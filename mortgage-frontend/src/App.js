import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Navigation from './components/Navigation';
import Dashboard from './components/Dashboard';
import LoanForm from './components/LoanForm';
import History from './components/History';
import Compare from './components/Compare';
import MonteCarlo3D from './components/MonteCarlo3D';
import ModelComparison from './components/ModelComparison';
import DecisionExplainer from './components/DecisionExplainer';

function App() {
  const [lastDecision, setLastDecision] = useState(null);

  return (
    <Router>
      <div className="min-h-screen bg-grid relative">
        {/* Trust & Authority Background */}
        <div className="fixed inset-0 pointer-events-none">
          {/* Base gradient */}
          <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950" />

          {/* Gold accent glows - top right */}
          <div className="absolute top-0 right-1/4 w-[600px] h-[600px] bg-amber-500/5 rounded-full blur-[120px]" />

          {/* Purple accent glows - bottom left */}
          <div className="absolute bottom-0 left-1/4 w-[500px] h-[500px] bg-purple-500/5 rounded-full blur-[100px]" />

          {/* Subtle gradient overlay */}
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(245,158,11,0.03),transparent_50%)]" />
        </div>

        {/* Navigation */}
        <Navigation />

        {/* Main Content */}
        <main className="relative z-10">
          <Routes>
            <Route path="/" element={<Dashboard lastDecision={lastDecision} />} />
            <Route path="/apply" element={<LoanForm onDecision={setLastDecision} />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/monte-carlo" element={<MonteCarlo3D />} />
            <Route path="/models" element={<ModelComparison />} />
            <Route path="/explain" element={<DecisionExplainer />} />
            <Route path="/history" element={<History />} />
          </Routes>
        </main>

        {/* Toast Container */}
        <ToastContainer
          position="top-right"
          autoClose={5000}
          hideProgressBar={false}
          newestOnTop={false}
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
          theme="dark"
          toastStyle={{
            background: 'rgba(15, 23, 42, 0.95)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            backdropFilter: 'blur(12px)',
            borderRadius: '12px',
            color: '#f8fafc'
          }}
          progressStyle={{
            background: 'linear-gradient(90deg, #F59E0B, #D97706)'
          }}
        />
      </div>
    </Router>
  );
}

export default App;
