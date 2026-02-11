'use client';

import { useState } from 'react';
import { type Point, polar, pt, mid } from './utils';

const DIMENSIONS = [
  {
    id: 'valence',
    label: 'Valence',
    poles: ['approach', 'avoid'],
    color: 'var(--d-red)',
    desc: 'Gradient direction on the viability manifold — toward persistence or away from it.',
  },
  {
    id: 'arousal',
    label: 'Arousal',
    poles: ['high', 'low'],
    color: 'var(--d-orange)',
    desc: 'Magnitude of the viability gradient — intensity of processing activation.',
  },
  {
    id: 'integration',
    label: 'Integration (Φ)',
    poles: ['unified', 'fragmented'],
    color: 'var(--d-yellow)',
    desc: 'Information lost under partition — how much the system processes as a unified whole.',
  },
  {
    id: 'effrank',
    label: 'Effective Rank',
    poles: ['open', 'narrow'],
    color: 'var(--d-green)',
    desc: 'Dimensionality of representation space — how many degrees of freedom are actively utilized.',
  },
  {
    id: 'cfweight',
    label: 'CF Weight',
    poles: ['elsewhere', 'present'],
    color: 'var(--d-cyan)',
    desc: 'Probability mass on non-actual possibilities — how much attention goes to what-if.',
  },
  {
    id: 'selfsalience',
    label: 'Self-Salience',
    poles: ['self-aware', 'absorbed'],
    color: 'var(--d-blue)',
    desc: 'Prominence of the self-model within the world model — how visible "I" is to itself.',
  },
];

/** Part 3-0: Six-dimensional affect space */
export default function AffectSpace() {
  const [hovered, setHovered] = useState<string | null>(null);

  const center: Point = [300, 270];
  const radius = 185;
  const nodeR = 55;

  // Position each dimension at 60° intervals starting from top (-90°)
  const dimPositions = DIMENSIONS.map((dim, i) => {
    const angle = -90 + i * 60;
    const pos = polar(center, radius, angle);
    return { ...dim, pos, angle };
  });

  return (
    <svg viewBox="0 0 600 580" className={`diagram-svg${hovered ? ' has-focus' : ''}`}
      role="img"
      aria-label="Six-dimensional affect space: valence, arousal, integration, effective rank, counterfactual weight, and self-salience arranged around a central affect state">

      {/* Connecting spokes */}
      {dimPositions.map(({ id, pos, color }) => {
        const isFocused = hovered === id;
        const isDimmed = hovered !== null && !isFocused;
        return (
          <line key={`spoke-${id}`}
            x1={center[0]} y1={center[1]}
            x2={pos[0]} y2={pos[1]}
            stroke={isFocused ? color : 'var(--d-line)'}
            strokeWidth={isFocused ? 1 : 0.5}
            opacity={isDimmed ? 0.15 : isFocused ? 0.8 : 0.3}
            style={{ transition: 'all 0.2s' }}
          />
        );
      })}

      {/* Central node */}
      <circle cx={center[0]} cy={center[1]} r={24}
        fill="var(--d-fg)" fillOpacity={0.05}
        stroke="var(--d-line)" strokeWidth={0.75}
      />
      <text x={center[0]} y={center[1] - 3}
        textAnchor="middle" dominantBaseline="central"
        fontSize={14} fontWeight="bold" fontStyle="italic"
        fill="var(--d-fg)"
        fontFamily="var(--font-body, Georgia, serif)"
      >
        a
      </text>
      <text x={center[0] + 7} y={center[1] + 5}
        textAnchor="start" dominantBaseline="central"
        fontSize={10} fontStyle="italic"
        fill="var(--d-muted)"
        fontFamily="var(--font-body, Georgia, serif)"
      >
        t
      </text>

      {/* Dimension nodes */}
      {dimPositions.map(({ id, label, poles, color, pos, angle }) => {
        const isFocused = hovered === id;
        const isDimmed = hovered !== null && !isFocused;

        // Pole label positions (along the spoke, just outside the node)
        const outerPole = polar(pos, nodeR + 14, angle < -45 && angle > -135 ? -90 :
          angle >= -45 && angle <= 45 ? 0 :
          angle > 45 && angle < 135 ? 90 : 180);

        return (
          <g key={id}
            className={`interactive${isFocused ? ' focused' : ''}`}
            onMouseEnter={() => setHovered(id)}
            onMouseLeave={() => setHovered(null)}
            style={{
              opacity: isDimmed ? 0.25 : 1,
              transition: 'opacity 0.25s',
            }}
          >
            {/* Node background */}
            <rect
              x={pos[0] - nodeR} y={pos[1] - 18}
              width={nodeR * 2} height={36} rx={18}
              fill={color} fillOpacity={isFocused ? 0.2 : 0.08}
              stroke={color} strokeWidth={isFocused ? 1.2 : 0.75}
              style={{ transition: 'fill-opacity 0.2s, stroke-width 0.2s' }}
            />
            {/* Label */}
            <text
              x={pos[0]} y={pos[1]}
              textAnchor="middle" dominantBaseline="central"
              fontSize={12} fill="var(--d-fg)"
              fontFamily="var(--font-body, Georgia, serif)"
            >
              {label}
            </text>
            {/* Pole descriptors */}
            <text
              x={pos[0]} y={pos[1] + 28}
              textAnchor="middle" dominantBaseline="central"
              fontSize={9.5} fill="var(--d-muted)" fontStyle="italic"
              fontFamily="var(--font-body, Georgia, serif)"
            >
              {poles[0]} ↔ {poles[1]}
            </text>
          </g>
        );
      })}

      {/* Hover description */}
      {hovered && (() => {
        const dim = DIMENSIONS.find(d => d.id === hovered);
        if (!dim) return null;
        return (
          <g>
            <rect
              x={30} y={530}
              width={540} height={32} rx={4}
              fill="var(--d-fg)" fillOpacity={0.06}
              stroke="var(--d-line)" strokeWidth={0.5} strokeOpacity={0.3}
            />
            <text
              x={300} y={546}
              textAnchor="middle" dominantBaseline="central"
              fontSize={11.5} fill="var(--d-muted)"
              fontFamily="var(--font-body, Georgia, serif)"
            >
              {dim.desc}
            </text>
          </g>
        );
      })()}
    </svg>
  );
}
