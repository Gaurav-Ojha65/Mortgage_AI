import React, { useEffect, useRef, useState } from 'react';

const GaugeChart = ({
  value = 0,
  min = 0,
  max = 100,
  size = 200,
  strokeWidth = 20,
  showValue = true,
  label,
  suffix = '%'
}) => {
  const [animatedValue, setAnimatedValue] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const containerRef = useRef(null);

  // Intersection Observer for animation trigger
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

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, []);

  // Animate the value
  useEffect(() => {
    if (!isVisible) return;

    const duration = 1000;
    const startTime = Date.now();
    const startValue = 0;

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Ease out cubic
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const current = startValue + (value - startValue) * easeOut;

      setAnimatedValue(current);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [value, isVisible]);

  // Calculate color based on value percentage
  const percentage = ((value - min) / (max - min)) * 100;
  const getColor = (pct) => {
    if (pct <= 4.5) return '#10B981'; // Green - Low risk
    if (pct <= 33.5) return '#F59E0B'; // Amber - Medium risk
    return '#EF4444'; // Red - High risk
  };

  const color = getColor(percentage);

  // SVG calculations
  const radius = (size - strokeWidth) / 2;
  const center = size / 2;

  // Background arc path
  const describeArc = (x, y, r, startAngle, endAngle) => {
    const start = polarToCartesian(x, y, r, endAngle);
    const end = polarToCartesian(x, y, r, startAngle);
    const largeArcFlag = endAngle - startAngle <= 180 ? '0' : '1';

    return [
      'M', start.x, start.y,
      'A', r, r, 0, largeArcFlag, 0, end.x, end.y
    ].join(' ');
  };

  const polarToCartesian = (centerX, centerY, radius, angleInDegrees) => {
    const angleInRadians = (angleInDegrees - 180) * Math.PI / 180.0;
    return {
      x: centerX + radius * Math.cos(angleInRadians),
      y: centerY + radius * Math.sin(angleInRadians)
    };
  };

  // Create gradient for the arc
  const gradientId = `gaugeGradient-${Math.random().toString(36).substr(2, 9)}`;

  return (
    <div
      ref={containerRef}
      className="gauge-container"
      style={{ width: size, height: size * 0.75 }}
    >
      <svg width={size} height={size * 0.75} viewBox={`0 0 ${size} ${size * 0.75}`}>
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#10B981" />
            <stop offset="50%" stopColor="#F59E0B" />
            <stop offset="100%" stopColor="#EF4444" />
          </linearGradient>
          <filter id={`glow-${gradientId}`}>
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background track */}
        <path
          d={describeArc(center, center, radius, 135, 405)}
          fill="none"
          stroke="rgba(148, 163, 184, 0.15)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Value arc */}
        <path
          d={describeArc(center, center, radius, 135, 135 + (270 * (animatedValue / max)))}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          filter={`url(#glow-${gradientId})`}
          style={{
            transition: 'stroke-dashoffset 0.1s ease-out'
          }}
        />

        {/* Tick marks */}
        {[0, 25, 50, 75, 100].map((tick) => {
          const angle = 135 + (270 * (tick / 100));
          const tickInner = polarToCartesian(center, center, radius - strokeWidth / 2 - 5, angle);
          const tickOuter = polarToCartesian(center, center, radius - strokeWidth / 2 - 15, angle);

          return (
            <g key={tick}>
              <line
                x1={tickInner.x}
                y1={tickInner.y}
                x2={tickOuter.x}
                y2={tickOuter.y}
                stroke="rgba(148, 163, 184, 0.5)"
                strokeWidth={2}
              />
              <text
                x={polarToCartesian(center, center, radius - strokeWidth / 2 - 30, angle).x}
                y={polarToCartesian(center, center, radius - strokeWidth / 2 - 30, angle).y}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="#94A3B8"
                fontSize={size * 0.06}
                fontFamily="IBM Plex Sans, sans-serif"
              >
                {tick}
              </text>
            </g>
          );
        })}

        {/* Needle */}
        {isVisible && (
          <g
            style={{
              transformOrigin: `${center}px ${center}px`,
              transform: `rotate(${(animatedValue / max) * 270 - 135}deg)`,
              transition: 'transform 0.1s ease-out'
            }}
          >
            <line
              x1={center}
              y1={center}
              x2={center}
              y2={center - radius + strokeWidth / 2 + 10}
              stroke={color}
              strokeWidth={4}
              strokeLinecap="round"
              filter={`url(#glow-${gradientId})`}
            />
            <circle cx={center} cy={center} r={8} fill={color} />
          </g>
        )}
      </svg>

      {/* Center value */}
      {showValue && (
        <div className="gauge-value flex flex-col items-center" style={{ top: size * 0.55 }}>
          <span className="text-3xl font-bold text-white" style={{ fontSize: size * 0.2 }}>
            {Math.round(animatedValue)}{suffix}
          </span>
          {label && (
            <span className="text-sm text-slate-400 mt-1" style={{ fontSize: size * 0.08 }}>
              {label}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default GaugeChart;
