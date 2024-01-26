import React from 'react';

export function scoreColor(score) {
  const steps = [
    [10, 'var(--neg-5)', 'var(--on-strong)'],
    [25, 'var(--neg-4)', 'var(--on-strong)'],
    [40, 'var(--neg-3)', 'var(--on-strong)'],
    [47, 'var(--neg-1)', 'var(--on-weak)'],
    [53, 'var(--neutral)', 'var(--text-primary)'],
    [60, 'var(--pos-1)', 'var(--on-weak)'],
    [75, 'var(--pos-3)', 'var(--on-strong)'],
    [90, 'var(--pos-4)', 'var(--on-strong)'],
    [101, 'var(--pos-5)', 'var(--on-strong)'],
  ];
  const found = steps.find(([limit]) => score < limit) || steps[steps.length - 1];
  return { background: found[1], color: found[2] };
}

export function StyleBars({ series }) {
  const axes = Object.keys(series[0].style);
  return (
    <div className="style-bars">
      {axes.map((axis) => (
        <div className="style-row" key={axis}>
          <span className="style-label">{axis.replace(/_/g, ' ')}</span>
          <div className="style-track">
            {series.map((entry, index) => (
              <div
                key={entry.riot_id}
                className="style-fill"
                style={{
                  width: `${entry.style[axis]}%`,
                  background: `var(--series-${index + 1})`,
                }}
                title={`${entry.riot_id}: ${entry.style[axis]} percentile ${axis.replace(/_/g, ' ')}`}
              />
            ))}
          </div>
          <span className="style-value">{series.map((entry) => entry.style[axis]).join(' / ')}</span>
        </div>
      ))}
    </div>
  );
}

export function DriverBars({ drivers }) {
  const span = Math.max(...drivers.map((driver) => Math.abs(driver.impact)), 1e-6);
  return (
    <div className="drivers">
      {drivers.map((driver) => {
        const ratio = (Math.abs(driver.impact) / span) * 50;
        const positive = driver.impact >= 0;
        return (
          <div className="driver" key={driver.axis}>
            <span className="driver-label">{driver.label}</span>
            <div className="driver-track">
              <span className="driver-mid" />
              <span
                className="driver-fill"
                style={{
                  left: positive ? '50%' : `${50 - ratio}%`,
                  width: `${ratio}%`,
                  background: positive ? 'var(--series-1)' : 'var(--critical)',
                }}
              />
            </div>
            <span className="driver-value">
              {positive ? '+' : '−'}
              {Math.abs(driver.impact).toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function Donut({ value, label, caption }) {
  const radius = 34;
  const circumference = 2 * Math.PI * radius;
  const filled = Math.max(0, Math.min(1, value)) * circumference;
  return (
    <div className="donut">
      <svg viewBox="0 0 88 88" width="88" height="88" role="img">
        <circle cx="44" cy="44" r={radius} fill="none" stroke="var(--grid)" strokeWidth="8" />
        <circle
          cx="44"
          cy="44"
          r={radius}
          fill="none"
          stroke="var(--series-1)"
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference - filled}`}
          transform="rotate(-90 44 44)"
        />
        <text x="44" y="49" textAnchor="middle" className="donut-value">
          {label}
        </text>
      </svg>
      <span className="caption">{caption}</span>
    </div>
  );
}
