'use client';

import { useState } from 'react';
import { type Point, arrowPath } from './utils';

const LEVELS = [
  { label: 'Neural', color: 'var(--d-red)', time: 'ms — s', desc: 'Neurotransmitter modulation, optogenetics, pharmacology' },
  { label: 'Individual', color: 'var(--d-orange)', time: 'min — yrs', desc: 'Therapy, meditation, psychedelics, habit formation' },
  { label: 'Dyadic', color: 'var(--d-yellow)', time: 'hrs — decades', desc: 'Attachment repair, couples therapy, mentorship' },
  { label: 'Small Group', color: 'var(--d-green)', time: 'days — yrs', desc: 'Group therapy, team design, ritual, communitas' },
  { label: 'Organizational', color: 'var(--d-cyan)', time: 'months — decades', desc: 'Institutional design, incentive structures, governance' },
  { label: 'Cultural', color: 'var(--d-blue)', time: 'yrs — centuries', desc: 'Education systems, media, mythology, legal frameworks' },
  { label: 'Superorganism', color: 'var(--d-violet)', time: 'decades — millennia', desc: 'Civilizational patterns, ecological constraints, evolutionary pressures' },
];

/** Part 4-0: Scale-specific interventions with characteristic timescales */
export default function ScaleInterventions() {
  const [hovered, setHovered] = useState<number | null>(null);

  const cx = 240;
  const startY = 45;
  const gap = 72;
  const boxW = 200, boxH = 40, boxR = 6;

  const levelY = (i: number) => startY + i * gap;

  return (
    <svg viewBox="0 0 560 590" className={`diagram-svg${hovered !== null ? ' has-focus' : ''}`}
      role="img"
      aria-label="Scale-specific interventions from neural to superorganism, each with characteristic timescales">

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
            {/* Timescale label */}
            <text
              x={cx + boxW / 2 + 18} y={y}
              textAnchor="start" dominantBaseline="central"
              fontSize={11} fill="var(--d-muted)" fontStyle="italic"
              fontFamily="var(--font-body, Georgia, serif)"
            >
              {level.time}
            </text>
          </g>
        );
      })}

      {/* Connecting arrows */}
      {LEVELS.slice(0, -1).map((_, i) => {
        const fromY = levelY(i) + boxH / 2 + 3;
        const toY = levelY(i + 1) - boxH / 2 - 3;
        const isDimmed = hovered !== null && hovered !== i && hovered !== i + 1;
        return (
          <g key={`arrow-${i}`}
            style={{ opacity: isDimmed ? 0.2 : 1, transition: 'opacity 0.2s' }}
          >
            <line x1={cx} y1={fromY} x2={cx} y2={toY}
              stroke="var(--d-line)" strokeWidth={0.75} />
            <path d={arrowPath([cx, toY], 90, 5)}
              stroke="var(--d-line)" strokeWidth={0.75} fill="none" />
          </g>
        );
      })}

      {/* Hover description */}
      {hovered !== null && (
        <g>
          <rect
            x={20} y={levelY(6) + boxH / 2 + 18}
            width={520} height={32} rx={4}
            fill="var(--d-fg)" fillOpacity={0.06}
            stroke="var(--d-line)" strokeWidth={0.5} strokeOpacity={0.3}
          />
          <text
            x={280} y={levelY(6) + boxH / 2 + 34}
            textAnchor="middle" dominantBaseline="central"
            fontSize={11} fill="var(--d-muted)"
            fontFamily="var(--font-body, Georgia, serif)"
          >
            {LEVELS[hovered].desc}
          </text>
        </g>
      )}
    </svg>
  );
}
