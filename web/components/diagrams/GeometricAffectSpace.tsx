'use client';

import { useState } from 'react';
import { type Point, polar, pt } from './utils';

/**
 * GeometricAffectSpace — Part II diagram
 * 6-spoke radar chart with labeled axes (the six affect dimensions)
 * 3-4 preloaded emotion profiles as colored polygons
 * Click/hover to toggle each emotion on/off
 */

const AXES = [
  { id: 'V', label: 'Valence', short: 'V' },
  { id: 'A', label: 'Arousal', short: 'A' },
  { id: 'Phi', label: 'Integration', short: 'Φ' },
  { id: 'reff', label: 'Eff. Rank', short: 'r' },
  { id: 'CF', label: 'CF Weight', short: 'CF' },
  { id: 'SM', label: 'Self-Model', short: 'SM' },
];

interface EmotionProfile {
  name: string;
  color: string;
  // Values 0-1 for each of the 6 axes (V, A, Φ, r_eff, CF, SM)
  values: number[];
}

const EMOTIONS: EmotionProfile[] = [
  {
    name: 'Joy',
    color: 'var(--d-green)',
    values: [0.9, 0.7, 0.7, 0.6, 0.2, 0.4],
  },
  {
    name: 'Grief',
    color: 'var(--d-blue)',
    values: [0.1, 0.3, 0.8, 0.3, 0.7, 0.9],
  },
  {
    name: 'Curiosity',
    color: 'var(--d-orange)',
    values: [0.65, 0.6, 0.5, 0.9, 0.6, 0.3],
  },
  {
    name: 'Fear',
    color: 'var(--d-red)',
    values: [0.1, 0.9, 0.4, 0.2, 0.8, 0.7],
  },
];

const W = 600;
const H = 520;
const CX = 300;
const CY = 250;
const RADIUS = 160;
const LEVELS = [0.25, 0.5, 0.75, 1.0];

export default function GeometricAffectSpace() {
  const [active, setActive] = useState<Set<string>>(new Set(['Joy', 'Grief']));
  const [hovered, setHovered] = useState<string | null>(null);

  const toggle = (name: string) => {
    setActive(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const axisAngle = (i: number) => -90 + i * 60;
  const valueToPoint = (axisIdx: number, value: number): Point => {
    return polar([CX, CY], value * RADIUS, axisAngle(axisIdx));
  };

  const profilePath = (values: number[]): string => {
    return values.map((v, i) => {
      const p = valueToPoint(i, v);
      return `${i === 0 ? 'M' : 'L'} ${pt(p)}`;
    }).join(' ') + ' Z';
  };

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="diagram-svg"
      role="img"
      aria-label="Radar chart of six affect dimensions showing profiles for joy, grief, curiosity, and fear"
    >
      {/* Title */}
      <text x={CX} y={28} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}
        fontFamily="var(--font-body, Georgia, serif)">
        Geometric Affect Space
      </text>
      <text x={CX} y={46} textAnchor="middle" fill="var(--d-muted)" fontSize={10.5}
        fontFamily="var(--font-body, Georgia, serif)">
        each emotion as a shape in six-dimensional coordinates
      </text>

      {/* Grid levels */}
      {LEVELS.map(level => {
        const points = AXES.map((_, i) => valueToPoint(i, level));
        const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${pt(p)}`).join(' ') + ' Z';
        return (
          <path key={level} d={d} fill="none" stroke="var(--d-line)"
            strokeWidth={level === 1 ? 0.75 : 0.4} opacity={0.3} />
        );
      })}

      {/* Axis spokes */}
      {AXES.map((axis, i) => {
        const end = valueToPoint(i, 1.0);
        const labelPos = valueToPoint(i, 1.18);
        const shortPos = valueToPoint(i, 1.06);
        return (
          <g key={axis.id}>
            <line x1={CX} y1={CY} x2={end[0]} y2={end[1]}
              stroke="var(--d-line)" strokeWidth={0.5} opacity={0.4} />
            <text x={labelPos[0]} y={labelPos[1] - 6} textAnchor="middle"
              dominantBaseline="central" fill="var(--d-fg)" fontSize={10}
              fontFamily="var(--font-body, Georgia, serif)">
              {axis.label}
            </text>
            <text x={shortPos[0]} y={shortPos[1] + 8} textAnchor="middle"
              dominantBaseline="central" fill="var(--d-muted)" fontSize={9}
              fontStyle="italic" fontFamily="var(--font-body, Georgia, serif)">
              {axis.short}
            </text>
          </g>
        );
      })}

      {/* Emotion profiles */}
      {EMOTIONS.filter(e => active.has(e.name)).map(emotion => {
        const isHov = hovered === emotion.name;
        return (
          <g key={emotion.name} opacity={hovered && !isHov ? 0.25 : 1}
            style={{ transition: 'opacity 0.2s' }}>
            <path d={profilePath(emotion.values)}
              fill={emotion.color} fillOpacity={isHov ? 0.2 : 0.1}
              stroke={emotion.color} strokeWidth={isHov ? 2 : 1.5}
              style={{ transition: 'fill-opacity 0.2s, stroke-width 0.2s' }}
            />
            {/* Dots at vertices */}
            {emotion.values.map((v, i) => {
              const p = valueToPoint(i, v);
              return (
                <circle key={i} cx={p[0]} cy={p[1]}
                  r={isHov ? 4 : 3} fill={emotion.color}
                  style={{ transition: 'r 0.15s' }}
                />
              );
            })}
          </g>
        );
      })}

      {/* Legend / toggles */}
      {EMOTIONS.map((emotion, i) => {
        const lx = 80 + i * 130;
        const ly = H - 38;
        const isActive = active.has(emotion.name);
        const isHov = hovered === emotion.name;
        return (
          <g key={emotion.name} style={{ cursor: 'pointer' }}
            onClick={() => toggle(emotion.name)}
            onMouseEnter={() => setHovered(emotion.name)}
            onMouseLeave={() => setHovered(null)}
          >
            <rect x={lx - 40} y={ly - 14} width={80} height={28} rx={14}
              fill={emotion.color} fillOpacity={isActive ? (isHov ? 0.25 : 0.12) : 0.03}
              stroke={emotion.color} strokeWidth={isActive ? 1.2 : 0.5}
              opacity={isActive ? 1 : 0.4}
              style={{ transition: 'all 0.2s' }}
            />
            <text x={lx} y={ly} textAnchor="middle" dominantBaseline="central"
              fill={isActive ? emotion.color : 'var(--d-muted)'} fontSize={12} fontWeight={600}
              fontFamily="var(--font-body, Georgia, serif)"
              style={{ transition: 'fill 0.2s' }}>
              {emotion.name}
            </text>
          </g>
        );
      })}

      <text x={CX} y={H - 8} textAnchor="middle" fill="var(--d-muted)" fontSize={9}
        fontFamily="var(--font-body, Georgia, serif)">
        click to toggle emotions on/off
      </text>
    </svg>
  );
}
