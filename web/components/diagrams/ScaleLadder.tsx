'use client';

import { useState } from 'react';
import { type Point, arrowPath } from './utils';

const LEVELS = [
  {
    label: 'Unstable Microdynamics',
    color: 'var(--d-red)',
    domain: 'physics',
    desc: 'Thermal fluctuations, random molecular motion — the raw material from which order emerges.',
  },
  {
    label: 'Metastable Attractors',
    color: 'var(--d-orange)',
    domain: 'chemistry',
    desc: 'Spontaneous order above critical thresholds — convection cells, oscillating reactions.',
  },
  {
    label: 'Emergent Boundaries',
    color: 'var(--d-yellow)',
    domain: '',
    desc: 'Cell membranes, skins — structures that resist dissolution and separate inside from outside.',
  },
  {
    label: 'Active Regulation',
    color: 'var(--d-green)',
    domain: 'biology',
    desc: 'Homeostasis, immune response — active maintenance of boundaries against perturbation.',
  },
  {
    label: 'World Model',
    color: 'var(--d-cyan)',
    domain: '',
    desc: 'Internal representation of external regularities — compressed prediction of environment. (V20: C_wm = 0.10–0.15)',
  },
  {
    label: 'Self-Model',
    color: 'var(--d-blue)',
    domain: 'psychology',
    desc: "The system's representation of itself within its own world model — recursive fold. (V20: SM salience > 1.0 in 2/3 seeds)",
  },
  {
    label: 'Metacognitive Dimensionality',
    color: 'var(--d-violet)',
    domain: '',
    desc: 'Awareness of awareness — the system monitors its own monitoring, adding representational depth. (V20: nascent; likely requires bottleneck selection to fully develop)',
  },
];

const TRANSITIONS = [
  'bifurcation',
  'selection',
  'maintenance',
  'POMDP structure',
  'ρ > ρ_c',
  'recursion',
];

/** Part 1-4: Scale ladder from physics to metacognition */
export default function ScaleLadder() {
  const [hovered, setHovered] = useState<number | null>(null);

  const cx = 280;
  const startY = 45;
  const gap = 76;
  const boxW = 220, boxH = 42, boxR = 6;

  const levelY = (i: number) => startY + i * gap;

  return (
    <svg viewBox="0 0 560 610" className={`diagram-svg${hovered !== null ? ' has-focus' : ''}`}
      role="img"
      aria-label="Scale ladder from physics to metacognition: seven levels of organization from unstable microdynamics through metastable attractors, emergent boundaries, active regulation, world models, self-models, to metacognitive dimensionality">

      {/* Domain labels on the left */}
      {LEVELS.map((level, i) => level.domain ? (
        <text key={`domain-${i}`}
          x={cx - boxW / 2 - 16} y={levelY(i)}
          textAnchor="end" dominantBaseline="central"
          fontSize={10.5} fill="var(--d-muted)" fontStyle="italic"
          fontFamily="var(--font-body, Georgia, serif)"
          opacity={hovered !== null && hovered !== i ? 0.3 : 1}
          style={{ transition: 'opacity 0.2s' }}
        >
          {level.domain}
        </text>
      ) : null)}

      {/* Level boxes */}
      {LEVELS.map((level, i) => {
        const y = levelY(i);
        const isFocused = hovered === i;
        const isDimmed = hovered !== null && !isFocused;
        return (
          <g key={level.label}
            className={`interactive${isFocused ? ' focused' : ''}`}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            style={{ opacity: isDimmed ? 0.3 : 1, transition: 'opacity 0.2s' }}
          >
            <rect
              x={cx - boxW / 2} y={y - boxH / 2}
              width={boxW} height={boxH} rx={boxR}
              fill={level.color} fillOpacity={isFocused ? 0.18 : 0.08}
              stroke={level.color} strokeWidth={isFocused ? 1.2 : 0.75}
              style={{ transition: 'fill-opacity 0.2s, stroke-width 0.2s' }}
            />
            <text
              x={cx} y={y}
              textAnchor="middle" dominantBaseline="central"
              fontSize={13} fill="var(--d-fg)"
              fontFamily="var(--font-body, Georgia, serif)"
            >
              {level.label}
            </text>
          </g>
        );
      })}

      {/* Transition arrows with labels */}
      {TRANSITIONS.map((label, i) => {
        const fromY = levelY(i) + boxH / 2 + 3;
        const toY = levelY(i + 1) - boxH / 2 - 3;
        const isDimmed = hovered !== null && hovered !== i && hovered !== i + 1;
        return (
          <g key={`trans-${i}`}
            style={{ opacity: isDimmed ? 0.2 : 1, transition: 'opacity 0.2s' }}
          >
            <line
              x1={cx} y1={fromY} x2={cx} y2={toY}
              stroke="var(--d-line)" strokeWidth={0.75}
            />
            <path
              d={arrowPath([cx, toY], 90, 5)}
              stroke="var(--d-line)" strokeWidth={0.75} fill="none"
            />
            <text
              x={cx + boxW / 2 + 14} y={(fromY + toY) / 2}
              textAnchor="start" dominantBaseline="central"
              fontSize={10.5} fontStyle="italic" fill="var(--d-muted)"
              fontFamily="var(--font-body, Georgia, serif)"
            >
              {label}
            </text>
          </g>
        );
      })}

      {/* Hover description tooltip */}
      {hovered !== null && (
        <g>
          <rect
            x={20} y={levelY(6) + boxH / 2 + 20}
            width={520} height={36} rx={4}
            fill="var(--d-fg)" fillOpacity={0.06}
            stroke="var(--d-line)" strokeWidth={0.5} strokeOpacity={0.3}
          />
          <text
            x={280} y={levelY(6) + boxH / 2 + 38}
            textAnchor="middle" dominantBaseline="central"
            fontSize={11.5} fill="var(--d-muted)"
            fontFamily="var(--font-body, Georgia, serif)"
          >
            {LEVELS[hovered].desc}
          </text>
        </g>
      )}
    </svg>
  );
}
