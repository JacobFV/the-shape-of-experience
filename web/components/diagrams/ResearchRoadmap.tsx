'use client';

import { useState } from 'react';

/**
 * ResearchRoadmap — Part VII diagram
 * The 5 research priorities as a visual roadmap
 * Each priority links to specific experiments (V-numbers)
 * Status indicators (complete, in progress, planned)
 */

type Status = 'complete' | 'active' | 'planned';

interface Priority {
  id: string;
  num: number;
  label: string;
  detail: string;
  experiments: string;
  status: Status;
  color: string;
}

const PRIORITIES: Priority[] = [
  {
    id: 'geometry',
    num: 1,
    label: 'Universal Geometry',
    detail: 'Confirm affect structure in uncontaminated systems',
    experiments: 'V10, V13-V18, VLM',
    status: 'complete',
    color: 'var(--d-green)',
  },
  {
    id: 'dynamics',
    num: 2,
    label: 'Integration Dynamics',
    detail: 'What architectural features produce Φ increase under stress?',
    experiments: 'V20-V27, V31-V34',
    status: 'complete',
    color: 'var(--d-blue)',
  },
  {
    id: 'plasticity',
    num: 3,
    label: 'Individual Plasticity',
    detail: 'Within-lifetime adaptation bridging the attention bottleneck',
    experiments: 'V35, V36',
    status: 'active',
    color: 'var(--d-orange)',
  },
  {
    id: 'social',
    num: 4,
    label: 'Social-Scale Φ',
    detail: 'Measure collective integration in multi-agent systems',
    experiments: 'planned',
    status: 'planned',
    color: 'var(--d-violet)',
  },
  {
    id: 'developmental',
    num: 5,
    label: 'Developmental Ordering',
    detail: 'Test emergence ladder predictions in human development',
    experiments: 'proposed',
    status: 'planned',
    color: 'var(--d-cyan)',
  },
];

const W = 640;
const H = 440;
const START_X = 80;
const CARD_W = 480;
const CARD_H = 58;
const GAP = 14;
const START_Y = 70;

const STATUS_ICON: Record<Status, { label: string; fill: string }> = {
  complete: { label: '●', fill: 'var(--d-green)' },
  active: { label: '◐', fill: 'var(--d-orange)' },
  planned: { label: '○', fill: 'var(--d-muted)' },
};

export default function ResearchRoadmap() {
  const [hovered, setHovered] = useState<string | null>(null);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={`diagram-svg${hovered ? ' has-focus' : ''}`}
      role="img"
      aria-label="Research roadmap showing five priorities from universal geometry through developmental ordering"
    >
      {/* Title */}
      <text x={W / 2} y={28} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}
        fontFamily="var(--font-body, Georgia, serif)">
        Research Roadmap
      </text>
      <text x={W / 2} y={46} textAnchor="middle" fill="var(--d-muted)" fontSize={10.5}
        fontFamily="var(--font-body, Georgia, serif)">
        five priorities for the empirical program
      </text>

      {/* Timeline line */}
      <line
        x1={START_X - 20} y1={START_Y + CARD_H / 2}
        x2={START_X - 20} y2={START_Y + (PRIORITIES.length - 1) * (CARD_H + GAP) + CARD_H / 2}
        stroke="var(--d-line)" strokeWidth={2} opacity={0.2}
      />

      {/* Priority cards */}
      {PRIORITIES.map((p, i) => {
        const y = START_Y + i * (CARD_H + GAP);
        const isFocused = hovered === p.id;
        const isDimmed = hovered !== null && !isFocused;
        const statusInfo = STATUS_ICON[p.status];

        return (
          <g key={p.id}
            className={`interactive${isFocused ? ' focused' : ''}`}
            onMouseEnter={() => setHovered(p.id)}
            onMouseLeave={() => setHovered(null)}
            style={{
              opacity: isDimmed ? 0.25 : 1,
              transition: 'opacity 0.25s',
              cursor: 'pointer',
            }}
          >
            {/* Timeline dot */}
            <circle cx={START_X - 20} cy={y + CARD_H / 2} r={6}
              fill={statusInfo.fill} opacity={0.8} />

            {/* Connector line */}
            <line x1={START_X - 14} y1={y + CARD_H / 2}
              x2={START_X + 10} y2={y + CARD_H / 2}
              stroke={p.color} strokeWidth={1} opacity={0.4} />

            {/* Card */}
            <rect x={START_X + 10} y={y} width={CARD_W} height={CARD_H} rx={6}
              fill={p.color} fillOpacity={isFocused ? 0.12 : 0.04}
              stroke={p.color} strokeWidth={isFocused ? 1.5 : 0.75}
              style={{ transition: 'fill-opacity 0.2s, stroke-width 0.2s' }}
            />

            {/* Priority number */}
            <text x={START_X + 30} y={y + CARD_H / 2} textAnchor="middle"
              dominantBaseline="central" fill={p.color} fontSize={18} fontWeight={700}
              fontFamily="var(--font-body, Georgia, serif)" opacity={0.4}>
              {p.num}
            </text>

            {/* Label */}
            <text x={START_X + 55} y={y + 20} textAnchor="start"
              fill="var(--d-fg)" fontSize={13} fontWeight={700}
              fontFamily="var(--font-body, Georgia, serif)">
              {p.label}
            </text>

            {/* Detail */}
            <text x={START_X + 55} y={y + 38} textAnchor="start"
              fill="var(--d-muted)" fontSize={10}
              fontFamily="var(--font-body, Georgia, serif)">
              {p.detail}
            </text>

            {/* Experiments badge */}
            <text x={START_X + CARD_W - 5} y={y + CARD_H / 2} textAnchor="end"
              dominantBaseline="central" fill={p.color} fontSize={9} fontStyle="italic"
              fontFamily="var(--font-body, Georgia, serif)">
              {p.experiments}
            </text>

            {/* Status indicator */}
            <text x={START_X + CARD_W + 20} y={y + CARD_H / 2} textAnchor="start"
              dominantBaseline="central" fill={statusInfo.fill} fontSize={11}>
              {statusInfo.label}
            </text>
          </g>
        );
      })}

      {/* Legend */}
      {(['complete', 'active', 'planned'] as Status[]).map((s, i) => {
        const lx = 160 + i * 150;
        const ly = H - 20;
        const info = STATUS_ICON[s];
        return (
          <g key={s}>
            <text x={lx} y={ly} fill={info.fill} fontSize={11}
              textAnchor="end" dominantBaseline="central">
              {info.label}
            </text>
            <text x={lx + 6} y={ly} fill="var(--d-muted)" fontSize={10}
              textAnchor="start" dominantBaseline="central"
              fontFamily="var(--font-body, Georgia, serif)">
              {s}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
