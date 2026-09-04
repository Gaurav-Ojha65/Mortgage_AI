import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { useAppStore } from './store';
import Layout from './components/layout/Layout';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const PredictPage = lazy(() => import('./pages/PredictPage'));
const AnalyticsPage = lazy(() => import('./pages/AnalyticsPage'));
const HistoryPage = lazy(() => import('./pages/HistoryPage'));
const EmiCalculator = lazy(() => import('./pages/EmiCalculator'));
const EligibilityCheck = lazy(() => import('./pages/EligibilityCheck'));
const LoginPage = lazy(() => import('./pages/LoginPage'));
const AdminPage = lazy(() => import('./pages/AdminPage'));
const RoleDashboard = lazy(() => import('./pages/RoleDashboard'));

function PageLoader() {
  return <div style={{display:'flex',alignItems:'center',justifyContent:'center',minHeight:'200px',color:'#64748b'}}>Loading view...</div>;
}

function ProtectedRoute({ children, roles }) {
  const { token, user } = useAppStore();
  if (!token) return <Navigate to="/login" replace />;
  if (roles && (!user || !roles.includes(user.role))) return <Navigate to="/dashboard" replace />;
  return children;
}

function PublicRoute({ children }) {
  const token = useAppStore((state) => state.token);
  if (token) return <Navigate to="/dashboard" replace />;
  return children;
}

function App() {
  return (
    <>
      <ToastContainer position="top-right" autoClose={3000} theme="dark" />
      <BrowserRouter>
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/login" element={<PublicRoute><LoginPage /></PublicRoute>} />
            <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
              <Route index element={<Navigate to="/dashboard" replace />} />
              <Route path="dashboard" element={<RoleDashboard />} />
              <Route path="predict" element={<ProtectedRoute roles={['user','loan_officer','admin']}><PredictPage /></ProtectedRoute>} />
              <Route path="analytics" element={<ProtectedRoute roles={['loan_officer','admin']}><AnalyticsPage /></ProtectedRoute>} />
              <Route path="history" element={<HistoryPage />} />
              <Route path="emi" element={<EmiCalculator />} />
              <Route path="eligibility" element={<ProtectedRoute roles={['user','loan_officer','admin']}><EligibilityCheck /></ProtectedRoute>} />
              <Route path="admin" element={<ProtectedRoute roles={['admin']}><AdminPage /></ProtectedRoute>} />
            </Route>
          </Routes>
        </Suspense>
      </BrowserRouter>
    </>
  );
}

export default App;
