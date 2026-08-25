import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:8001', changeOrigin: true },
      '/health': { target: 'http://localhost:8001', changeOrigin: true },
      '/analyze': { target: 'http://localhost:8001', changeOrigin: true },
      '/compare': { target: 'http://localhost:8001', changeOrigin: true },
      '/auth': { target: 'http://localhost:8001', changeOrigin: true },
      '/audit': { target: 'http://localhost:8001', changeOrigin: true },
      '/whatif': { target: 'http://localhost:8001', changeOrigin: true },
      '/history': { target: 'http://localhost:8001', changeOrigin: true },
      '/policy': { target: 'http://localhost:8001', changeOrigin: true },
      '/documents': { target: 'http://localhost:8001', changeOrigin: true },
      '/errors': { target: 'http://localhost:8001', changeOrigin: true },
      '/metrics': { target: 'http://localhost:8001', changeOrigin: true },
    },
  },
  build: {
    outDir: 'build',
    sourcemap: false,
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      output: {
        manualChunks: {
          plotly: ['plotly.js', 'react-plotly.js'],
          chart: ['chart.js', 'react-chartjs-2'],
          vendor: ['react', 'react-dom', 'react-router-dom'],
          state: ['zustand', 'react-hook-form', 'zod', '@hookform/resolvers'],
        },
      },
    },
  },
});
