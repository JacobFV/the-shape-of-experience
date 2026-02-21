'use client';

import { useState } from 'react';

/**
 * PracticeMap — Epilogue diagram
 * Maps contemplative/therapeutic practices to the affect dimensions they target.
 * Interactive: hover a practice to highlight which dimensions it modulates.
 */

interface Practice {
  name: string;
  description: string;
  color: string;
  targets: { dim: string; effect: 'increase' | 'decrease' | 'stabilize' }[];
}

const DIMS = ['V', 'A', 'Φ', 'r_eff', 'CF', 'SM', 'ι'];
const DIM_LABELS: Record<string, string> = {
  V: 'Valence',
  A: 'Arousal',
  'Φ': 'Integration',
  r_eff: 'Eff. Rank',
  CF: 'Counterfactual',
  SM: 'Self-Model',
  'ι': 'Inhibition',
};

const PRACTICES: Practice[] = [
  {
    name: 'Meditation',
    description: 'Sustained attention, integration maintenance',
    color: 'var(--d-blue)',
    targets: [
      { dim: 'A', effect: 'decrease' },
      { dim: 'Φ', effect: 'increase' },
      { dim: 'SM', effect: 'decrease' },
      { dim: 'ι', effect: 'decrease' },
    ],
  },
  {
    name: 'Deep Work',
    description: 'Extended focused engagement without interruption',
    color: 'var(--d-violet)',
    targets: [
      { dim: 'Φ', effect: 'increase' },
      { dim: 'r_eff', effect: 'increase' },
      { dim: 'SM', effect: 'decrease' },
    ],
  },
  {
    name: 'Physical Exercise',
    description: 'Embodied presence, arousal regulation',
    color: 'var(--d-green)',
    targets: [
      { dim: 'V', effect: 'increase' },
      { dim: 'A', effect: 'stabilize' },
      { dim: 'SM', effect: 'decrease' },
      { dim: 'ι', effect: 'decrease' },
    ],
  },
  {
    name: 'Manifold Hygiene',
    description: 'Keeping relationship types clean and uncontaminated',
    color: 'var(--d-pink)',
    targets: [
      { dim: 'Φ', effect: 'increase' },
      { dim: 'V', effect: 'increase' },
    ],
  },
  {
    name: 'ι Calibration',
    description: 'Voluntary modulation of participatory perception',
    color: 'var(--d-yellow)',
    targets: [
      { dim: 'ι', effect: 'stabilize' },
      { dim: 'r_eff', effect: 'increase' },
      { dim: 'CF', effect: 'stabilize' },
    ],
  },
  {
    name: 'Grief Work',
    description: 'Updating self-model after loss',
    color: 'var(--d-orange)',
    targets: [
      { dim: 'SM', effect: 'stabilize' },
      { dim: 'r_eff', effect: 'increase' },
      { dim: 'CF', effect: 'decrease' },
    ],
  },
  {
    name: 'Communion',
    description: 'Genuine resonance with another mind',
    color: 'var(--d-cyan)',
    targets: [
      { dim: 'Φ', effect: 'increase' },
      { dim: 'SM', effect: 'decrease' },
      { dim: 'ι', effect: 'decrease' },
      { dim: 'V', effect: 'increase' },
    ],
  },
];

const W = 700;
const H = 460;

// Layout: practices on left, dimensions on right, connection lines between
const PRAC_X = 140;
const DIM_X = 560;
const TOP_Y = 60;
const PRAC_SPACING = (H - TOP_Y - 40) / PRACTICES.length;
const DIM_SPACING = (H - TOP_Y - 40) / DIMS.length;

export default function PracticeMap() {
  const [hoveredPrac, setHoveredPrac] = useState<number | null>(null);
  const [hoveredDim, setHoveredDim] = useState<string | null>(null);

  const pracY = (i: number) => TOP_Y + 20 + i * PRAC_SPACING;
  const dimY = (i: number) => TOP_Y + 20 + i * DIM_SPACING;

  // Get all connections for highlighting
  const activeConnections = new Set<string>();
  if (hoveredPrac !== null) {
    PRACTICES[hoveredPrac].targets.forEach((t) => {
      activeConnections.add(`${hoveredPrac}-${t.dim}`);
    });
  }
  if (hoveredDim !== null) {
    PRACTICES.forEach((p, pi) => {
      p.targets.forEach((t) => {
        if (t.dim === hoveredDim) {
          activeConnections.add(`${pi}-${t.dim}`);
        }
      });
    });
  }

  const isAnyHover = hoveredPrac !== null || hoveredDim !== null;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: '100%', maxWidth: 740 }}
      role="img"
      aria-label="Practice to affect dimension mapping"
    >
      {/* Title */}
      <text x={W / 2} y={22} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}>
        Navigation Training
      </text>
      <text x={W / 2} y={40} textAnchor="middle" fill="var(--d-muted)" fontSize={10}>
        Practices mapped to the affect dimensions they modulate
      </text>

      {/* Column headers */}
      <text x={PRAC_X} y={TOP_Y} textAnchor="middle" fill="var(--d-muted)" fontSize={11} fontWeight={600}>
        PRACTICE
      </text>
      <text x={DIM_X} y={TOP_Y} textAnchor="middle" fill="var(--d-muted)" fontSize={11} fontWeight={600}>
        DIMENSION
      </text>

      {/* Connection lines */}
      {PRACTICES.map((prac, pi) =>
        prac.targets.map((target) => {
          const di = DIMS.indexOf(target.dim);
          if (di === -1) return null;

          const key = `${pi}-${target.dim}`;
          const isActive = activeConnections.has(key);
          const isDimmed = isAnyHover && !isActive;

          const y1 = pracY(pi);
          const y2 = dimY(di);

          // Color based on effect
          const lineColor =
            target.effect === 'increase'
              ? 'var(--d-green)'
              : target.effect === 'decrease'
              ? 'var(--d-red)'
              : 'var(--d-yellow)';

          return (
            <path
              key={key}
              d={`M ${PRAC_X + 80} ${y1} C ${(PRAC_X + DIM_X) / 2} ${y1}, ${(PRAC_X + DIM_X) / 2} ${y2}, ${DIM_X - 80} ${y2}`}
              fill="none"
              stroke={isActive ? lineColor : 'var(--d-line)'}
              strokeWidth={isActive ? 2.5 : 1}
              opacity={isDimmed ? 0.08 : isActive ? 0.7 : 0.15}
              style={{ transition: 'opacity 0.3s, stroke-width 0.3s' }}
            />
          );
        })
      )}

      {/* Practice nodes */}
      {PRACTICES.map((prac, pi) => {
        const y = pracY(pi);
        const isActive = hoveredPrac === pi || (hoveredDim !== null && prac.targets.some((t) => t.dim === hoveredDim));
        const isDimmed = isAnyHover && !isActive;

        return (
          <g
            key={pi}
            style={{
              cursor: 'pointer',
              opacity: isDimmed ? 0.3 : 1,
              transition: 'opacity 0.3s',
            }}
            onMouseEnter={() => setHoveredPrac(pi)}
            onMouseLeave={() => setHoveredPrac(null)}
          >
            <circle cx={PRAC_X + 74} cy={y} r={5} fill={prac.color} />
            <text
              x={PRAC_X + 64}
              y={y + 4}
              textAnchor="end"
              fill={isActive ? prac.color : 'var(--d-fg)'}
              fontSize={12}
              fontWeight={isActive ? 700 : 500}
            >
              {prac.name}
            </text>
            {isActive && (
              <text
                x={PRAC_X - 68}
                y={y + 16}
                textAnchor="start"
                fill="var(--d-muted)"
                fontSize={8}
              >
                {prac.description}
              </text>
            )}
          </g>
        );
      })}

      {/* Dimension nodes */}
      {DIMS.map((dim, di) => {
        const y = dimY(di);
        const isActive = hoveredDim === dim || (hoveredPrac !== null && PRACTICES[hoveredPrac].targets.some((t) => t.dim === dim));
        const isDimmed = isAnyHover && !isActive;

        return (
          <g
            key={di}
            style={{
              cursor: 'pointer',
              opacity: isDimmed ? 0.3 : 1,
              transition: 'opacity 0.3s',
            }}
            onMouseEnter={() => setHoveredDim(dim)}
            onMouseLeave={() => setHoveredDim(null)}
          >
            <circle cx={DIM_X - 74} cy={y} r={5} fill={isActive ? 'var(--d-fg)' : 'var(--d-line)'} />
            <text
              x={DIM_X - 64}
              y={y + 4}
              fill={isActive ? 'var(--d-fg)' : 'var(--d-muted)'}
              fontSize={12}
              fontWeight={isActive ? 700 : 500}
            >
              {dim}
            </text>
            <text
              x={DIM_X - 64}
              y={y + 16}
              fill="var(--d-muted)"
              fontSize={8}
            >
              {DIM_LABELS[dim]}
            </text>
          </g>
        );
      })}

      {/* Legend */}
      <g transform={`translate(${W / 2 - 100}, ${H - 18})`}>
        <line x1={0} y1={0} x2={20} y2={0} stroke="var(--d-green)" strokeWidth={2} />
        <text x={24} y={4} fill="var(--d-muted)" fontSize={9}>increase</text>
        <line x1={80} y1={0} x2={100} y2={0} stroke="var(--d-red)" strokeWidth={2} />
        <text x={104} y={4} fill="var(--d-muted)" fontSize={9}>decrease</text>
        <line x1={155} y1={0} x2={175} y2={0} stroke="var(--d-yellow)" strokeWidth={2} />
        <text x={179} y={4} fill="var(--d-muted)" fontSize={9}>stabilize</text>
      </g>
    </svg>
  );
}
