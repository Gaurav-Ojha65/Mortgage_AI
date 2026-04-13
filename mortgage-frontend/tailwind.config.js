/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        // Trust & Authority Design System
        primary: {
          DEFAULT: '#F59E0B',
          50: '#FEF3C7',
          100: '#FDE68A',
          200: '#FCD34D',
          300: '#FBBF24',
          400: '#F59E0B',
          500: '#D97706',
          600: '#B45309',
          700: '#92400E',
          800: '#78350F',
          900: '#451A03',
        },
        cta: {
          DEFAULT: '#8B5CF6',
          50: '#F5F3FF',
          100: '#EDE9FE',
          200: '#DDD6FE',
          300: '#C4B5FD',
          400: '#A78BFA',
          500: '#8B5CF6',
          600: '#7C3AED',
          700: '#6D28D9',
          800: '#5B21B6',
          900: '#4C1D95',
        },
        slate: {
          850: '#172033',
          900: '#0F172A',
          950: '#020617',
        },
        // Risk levels
        risk: {
          low: '#10B981',
          medium: '#F59E0B',
          high: '#EF4444',
        },
        // Decision states
        decision: {
          approve: '#10B981',
          reject: '#EF4444',
          review: '#F59E0B',
        }
      },
      fontFamily: {
        sans: ['IBM Plex Sans', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite',
        'badge-pulse': 'badgePulse 2s ease-in-out infinite',
        'metric-reveal': 'metricReveal 0.5s ease-out forwards',
        'gauge-fill': 'gaugeFill 1s ease-out forwards',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        glow: {
          '0%, 100%': { opacity: 0.5 },
          '50%': { opacity: 1 },
        },
        badgePulse: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(245, 158, 11, 0.4)' },
          '50%': { boxShadow: '0 0 0 8px rgba(245, 158, 11, 0)' },
        },
        metricReveal: {
          '0%': { opacity: 0, transform: 'translateY(20px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        },
        gaugeFill: {
          '0%': { strokeDashoffset: '283' },
          '100%': { strokeDashoffset: 'var(--gauge-offset, 0)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
      boxShadow: {
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        'gold': '0 0 20px rgba(245, 158, 11, 0.3)',
        'purple': '0 0 20px rgba(139, 92, 246, 0.3)',
      },
    },
  },
  plugins: [
    function({ addComponents }) {
      addComponents({
        '.glass-card': {
          background: 'rgba(15, 23, 42, 0.7)',
          backdropFilter: 'blur(12px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '1rem',
        },
        '.glass-nav': {
          background: 'rgba(15, 23, 42, 0.85)',
          backdropFilter: 'blur(16px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        },
        '.btn-primary': {
          background: 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)',
          color: '#0F172A',
          fontWeight: '600',
          padding: '0.75rem 1.5rem',
          borderRadius: '0.75rem',
          transition: 'all 150ms ease-in-out',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 10px 25px rgba(245, 158, 11, 0.4)',
          },
          '&:active': {
            transform: 'translateY(0)',
          },
          '&:focus-visible': {
            outline: '2px solid #F59E0B',
            outlineOffset: '2px',
          },
        },
        '.btn-cta': {
          background: 'linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%)',
          color: '#fff',
          fontWeight: '600',
          padding: '0.75rem 1.5rem',
          borderRadius: '0.75rem',
          transition: 'all 150ms ease-in-out',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 10px 25px rgba(139, 92, 246, 0.4)',
          },
          '&:focus-visible': {
            outline: '2px solid #8B5CF6',
            outlineOffset: '2px',
          },
        },
        '.btn-secondary': {
          background: 'rgba(255, 255, 255, 0.1)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          color: '#F8FAFC',
          fontWeight: '500',
          padding: '0.75rem 1.5rem',
          borderRadius: '0.75rem',
          transition: 'all 150ms ease-in-out',
          '&:hover': {
            background: 'rgba(255, 255, 255, 0.15)',
            borderColor: 'rgba(255, 255, 255, 0.3)',
          },
          '&:focus-visible': {
            outline: '2px solid #94A3B8',
            outlineOffset: '2px',
          },
        },
        '.badge': {
          display: 'inline-flex',
          alignItems: 'center',
          padding: '0.25rem 0.75rem',
          borderRadius: '9999px',
          fontSize: '0.75rem',
          fontWeight: '600',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        },
        '.badge-success': {
          background: 'rgba(16, 185, 129, 0.15)',
          color: '#10B981',
          border: '1px solid rgba(16, 185, 129, 0.3)',
        },
        '.badge-warning': {
          background: 'rgba(245, 158, 11, 0.15)',
          color: '#F59E0B',
          border: '1px solid rgba(245, 158, 11, 0.3)',
        },
        '.badge-danger': {
          background: 'rgba(239, 68, 68, 0.15)',
          color: '#EF4444',
          border: '1px solid rgba(239, 68, 68, 0.3)',
        },
        '.badge-info': {
          background: 'rgba(59, 130, 246, 0.15)',
          color: '#3B82F6',
          border: '1px solid rgba(59, 130, 246, 0.3)',
        },
        '.trust-badge': {
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0.5rem 1rem',
          background: 'rgba(245, 158, 11, 0.1)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: '9999px',
          fontSize: '0.875rem',
          fontWeight: '500',
          color: '#FBBF24',
        },
        '.input-field': {
          width: '100%',
          background: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '0.75rem',
          padding: '0.75rem 1rem',
          color: '#F8FAFC',
          transition: 'all 150ms ease-in-out',
          '&:focus': {
            outline: 'none',
            borderColor: '#F59E0B',
            boxShadow: '0 0 0 3px rgba(245, 158, 11, 0.15)',
          },
          '&::placeholder': {
            color: '#64748B',
          },
        },
        '.label': {
          display: 'block',
          fontSize: '0.875rem',
          fontWeight: '500',
          color: '#94A3B8',
          marginBottom: '0.5rem',
        },
        '.section-title': {
          fontSize: '1.25rem',
          fontWeight: '600',
          color: '#F8FAFC',
          marginBottom: '1rem',
        },
        '.text-gradient-gold': {
          background: 'linear-gradient(135deg, #FBBF24 0%, #F59E0B 50%, #D97706 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
        },
        '.text-gradient-purple': {
          background: 'linear-gradient(135deg, #A78BFA 0%, #8B5CF6 50%, #7C3AED 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
        },
      });
    },
  ],
};
