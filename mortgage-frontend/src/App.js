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
import Signup from './components/Signup';
import Login from './components/Login';

function App() {
  const [lastDecision, setLastDecision] = useState(null);

  return (
    <Router>
      <div className="min-h-screen bg-grid relative">
        {/* Toast Container - Global */}
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

        <Routes>
          {/* Auth Routes - Full Screen, No Navigation */}
          <Route path="/signup" element={<Signup />} />
          <Route path="/login" element={<Login />} />

          {/* App Routes - With Navigation */}
          <Route path="/*" element={
            <>
              <Navigation />
              <main className="relative z-10 pt-20">
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
            </>
          } />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
