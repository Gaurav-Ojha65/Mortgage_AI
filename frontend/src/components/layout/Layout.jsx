import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import TopNavbar from './TopNavbar';

export default function Layout() {
  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <TopNavbar />
        <main className="page-container">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
