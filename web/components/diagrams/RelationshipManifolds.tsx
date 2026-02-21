'use client';

import { useState } from 'react';
import { type Point, polar, pt, smoothClosed } from './utils';

/**
 * RelationshipManifolds — Part IV diagram
 * Six relationship types as overlapping viability regions
 * Hover to highlight each type's properties
 */

interface RelType {
  id: string;
  label: string;
  info: string;
  reciprocity: string;
  color: string;
  center: Point;
  rx: number;
  ry: number;
  rotation: number;
}

const TYPES: RelType[] = [
  {
    id: 'friendship',
    label: 'Friendship',
    info: 'mutual flourishing, open-ended',
    reciprocity: 'long-horizon',
    color: 'var(--d-green)',
    center: [200, 200],
    rx: 95, ry: 70,
    rotation: -15,
  },
  {
    id: 'transaction',
    label: 'Transaction',
    info: 'bounded exchange, measurable',
    reciprocity: 'immediate',
    color: 'var(--d-orange)',
    center: [430, 195],
    rx: 85, ry: 65,
    rotation: 10,
  },
  {
    id: 'therapy',
    label: 'Therapy',
    info: 'asymmetric care, structured termination',
    reciprocity: 'asymmetric',
    color: 'var(--d-blue)',
    center: [310, 140],
    rx: 80, ry: 55,
    rotation: 5,
  },
  {
    id: 'employment',
    label: 'Employment',
    info: 'hierarchical authority + exchange',
    reciprocity: 'contractual',
    color: 'var(--d-yellow)',
    center: [370, 290],
    rx: 90, ry: 60,
    rotation: 20,
  },
  {
    id: 'romance',
    label: 'Romance',
    info: 'exclusive bonding, high Φ coupling',
    reciprocity: 'infinite-horizon',
    color: 'var(--d-red)',
    center: [180, 310],
    rx: 90, ry: 65,
    rotation: -10,
  },
  {
    id: 'parenthood',
    label: 'Parenthood',
    info: 'unconditional, developmental arc',
    reciprocity: 'non-reciprocal',
    color: 'var(--d-violet)',
    center: [270, 260],
    rx: 85, ry: 60,
    rotation: -5,
  },
];

const W = 620;
const H = 460;

export default function RelationshipManifolds() {
  const [hovered, setHovered] = useState<string | null>(null);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={`diagram-svg${hovered ? ' has-focus' : ''}`}
      role="img"
      aria-label="Six relationship types as overlapping viability manifolds: friendship, transaction, therapy, employment, romance, parenthood"
    >
      {/* Title */}
      <text x={W / 2} y={28} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}
        fontFamily="var(--font-body, Georgia, serif)">
        Relationship Manifolds
      </text>
      <text x={W / 2} y={46} textAnchor="middle" fill="var(--d-muted)" fontSize={10.5}
        fontFamily="var(--font-body, Georgia, serif)">
        each type defines its own viability region — overlap creates contamination
      </text>

      {/* Relationship ellipses */}
      {TYPES.map(t => {
        const isFocused = hovered === t.id;
        const isDimmed = hovered !== null && !isFocused;

        return (
          <g key={t.id}
            className={`interactive${isFocused ? ' focused' : ''}`}
            onMouseEnter={() => setHovered(t.id)}
            onMouseLeave={() => setHovered(null)}
            style={{
              opacity: isDimmed ? 0.15 : 1,
              transition: 'opacity 0.25s',
              cursor: 'pointer',
            }}
          >
            <ellipse
              cx={t.center[0]} cy={t.center[1]}
              rx={t.rx} ry={t.ry}
              transform={`rotate(${t.rotation} ${t.center[0]} ${t.center[1]})`}
              fill={t.color} fillOpacity={isFocused ? 0.18 : 0.07}
              stroke={t.color} strokeWidth={isFocused ? 2 : 1}
              strokeDasharray={isFocused ? 'none' : '6 4'}
              style={{ transition: 'fill-opacity 0.2s, stroke-width 0.2s' }}
            />
            {/* Label */}
            <text x={t.center[0]} y={t.center[1] - 8} textAnchor="middle"
              dominantBaseline="central" fill={t.color}
              fontSize={isFocused ? 13 : 11.5} fontWeight={700}
              fontFamily="var(--font-body, Georgia, serif)"
              style={{ transition: 'font-size 0.2s' }}>
              {t.label}
            </text>
            {/* Reciprocity tag */}
            <text x={t.center[0]} y={t.center[1] + 10} textAnchor="middle"
              dominantBaseline="central" fill="var(--d-muted)" fontSize={9}
              fontFamily="var(--font-body, Georgia, serif)">
              {t.reciprocity}
            </text>
          </g>
        );
      })}

      {/* Hover info panel */}
      {hovered && (() => {
        const t = TYPES.find(r => r.id === hovered);
        if (!t) return null;
        return (
          <g>
            <rect x={40} y={H - 55} width={W - 80} height={32} rx={4}
              fill="var(--d-fg)" fillOpacity={0.06}
              stroke="var(--d-line)" strokeWidth={0.5} strokeOpacity={0.3} />
            <text x={W / 2} y={H - 35} textAnchor="middle" dominantBaseline="central"
              fill="var(--d-muted)" fontSize={11}
              fontFamily="var(--font-body, Georgia, serif)">
              {t.label}: {t.info}
            </text>
          </g>
        );
      })()}

      {/* Bottom note */}
      <text x={W / 2} y={H - 10} textAnchor="middle" fill="var(--d-muted)" fontSize={9.5}
        fontStyle="italic" fontFamily="var(--font-body, Georgia, serif)">
        hover to highlight each relationship type
      </text>
    </svg>
  );
}
