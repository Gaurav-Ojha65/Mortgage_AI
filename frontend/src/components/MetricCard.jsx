import React, { useEffect, useRef, useState } from 'react';

const MetricCard = ({ title, value, subtitle, trend, trendUp, icon, color = 'gold', delay = 0 }) => {
  const [isVisible, setIsVisible] = useState(false);
  const cardRef = useRef(null);

  // Intersection Observer for reveal animation
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );

    if (cardRef.current) {
      observer.observe(cardRef.current);
    }

    return () => observer.disconnect();
  }, []);

  // Trust & Authority color schemes
  const colorSchemes = {
    gold: {
      gradient: 'from-amber-500/15 to-amber-600/10',
      border: 'border-amber-500/30',
      borderHover: 'group-hover:border-amber-500/50',
      text: 'text-amber-400',
      bg: 'bg-amber-500/10',
      accent: 'via-amber-500',
      glow: 'group-hover:shadow-amber-500/20',
      iconBg: 'bg-amber-500/15',
    },
    purple: {
      gradient: 'from-purple-500/15 to-purple-600/10',
      border: 'border-purple-500/30',
      borderHover: 'group-hover:border-purple-500/50',
      text: 'text-purple-400',
      bg: 'bg-purple-500/10',
      accent: 'via-purple-500',
      glow: 'group-hover:shadow-purple-500/20',
      iconBg: 'bg-purple-500/15',
    },
    emerald: {
      gradient: 'from-emerald-500/15 to-emerald-600/10',
      border: 'border-emerald-500/30',
      borderHover: 'group-hover:border-emerald-500/50',
      text: 'text-emerald-400',
      bg: 'bg-emerald-500/10',
      accent: 'via-emerald-500',
      glow: 'group-hover:shadow-emerald-500/20',
      iconBg: 'bg-emerald-500/15',
    },
    blue: {
      gradient: 'from-blue-500/15 to-blue-600/10',
      border: 'border-blue-500/30',
      borderHover: 'group-hover:border-blue-500/50',
      text: 'text-blue-400',
      bg: 'bg-blue-500/10',
      accent: 'via-blue-500',
      glow: 'group-hover:shadow-blue-500/20',
      iconBg: 'bg-blue-500/15',
    },
    red: {
      gradient: 'from-red-500/15 to-red-600/10',
      border: 'border-red-500/30',
      borderHover: 'group-hover:border-red-500/50',
      text: 'text-red-400',
      bg: 'bg-red-500/10',
      accent: 'via-red-500',
      glow: 'group-hover:shadow-red-500/20',
      iconBg: 'bg-red-500/15',
    },
  };

  const scheme = colorSchemes[color] || colorSchemes.gold;

  return (
    <div
      ref={cardRef}
      className={`relative group rounded-2xl p-6 bg-gradient-to-br ${scheme.gradient} border ${scheme.border} ${scheme.borderHover} backdrop-blur-sm overflow-hidden transition-all duration-300 hover:-translate-y-1 hover:shadow-lg ${scheme.glow} ${isVisible ? 'animate-metric-reveal' : 'opacity-0'}`}
      style={{ animationDelay: `${delay}ms` }}
    >
      {/* Top Accent Line with Gold Gradient */}
      <div className={`absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent ${scheme.accent} to-transparent opacity-50 group-hover:opacity-80 transition-opacity duration-300`} />

      {/* Subtle glow effect on hover */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(245,158,11,0.08),transparent_70%)]" />
      </div>

      <div className="relative z-10">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <span className="text-slate-400 text-sm font-medium tracking-wide uppercase">{title}</span>
          {icon && (
            <div className={`p-2.5 rounded-xl ${scheme.iconBg} ${scheme.text} group-hover:scale-110 transition-transform duration-300`}>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={icon} />
              </svg>
            </div>
          )}
        </div>

        {/* Value */}
        <div className="mb-3">
          <span className="text-3xl font-bold tracking-tight font-mono text-gradient-gold">
            {value}
          </span>
        </div>

        {/* Subtitle & Trend */}
        <div className="flex items-center justify-between">
          <span className="text-slate-500 text-sm">{subtitle}</span>
          {trend && (
            <div className={`flex items-center gap-1 text-sm font-medium ${
              trendUp ? 'text-emerald-400' : 'text-red-400'
            }`}>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={
                  trendUp ? 'M5 10l7-7m0 0l7 7m-7-7v18' : 'M19 14l-7 7m0 0l-7-7m7 7V3'
                } />
              </svg>
              <span>{trend}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MetricCard;
