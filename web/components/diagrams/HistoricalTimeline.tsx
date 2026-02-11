'use client';

import { useState } from 'react';
import { type Point, arrowPath } from './utils';

const ERAS = [
  {
    label: 'Pre-Axial',
    date: '~50k BCE',
    color: 'var(--d-red)',
    innovation: 'ritual, myth',
    desc: 'Shared attention coordinates group consciousness through ritual and oral tradition.',
  },
  {
    label: 'Axial Age',
    date: '800 BCE',
    color: 'var(--d-orange)',
    innovation: 'self-model manipulation',
    desc: 'Discovery that the self-model can be deliberately modified — birth of philosophy and religion.',
  },
  {
    label: 'Renaissance',
    date: '1400 CE',
    color: 'var(--d-purple)',
    innovation: 'perspectivity',
    desc: 'Recognition of inherent perspectivity — multiple valid viewpoints coexist.',
  },
  {
    label: 'Scientific Rev.',
    date: '1600 CE',
    color: 'var(--d-yellow)',
    innovation: 'world-model expansion',
    desc: 'Systematic expansion of world models through controlled observation and experiment.',
  },
  {
    label: 'Philosophical',
    date: '1900 CE',
    color: 'var(--d-teal)',
    innovation: 'subject deepening',
    desc: 'Deep examination of the experiencing subject itself — phenomenology, existentialism.',
  },
  {
    label: 'Psych. Turn',
    date: '1950 CE',
    color: 'var(--d-green)',
    innovation: 'inner space mapping',
    desc: 'Systematic mapping of inner space — psychometrics, therapy, neuroscience.',
  },
  {
    label: 'Digital / AI',
    date: '2000 CE',
    color: 'var(--d-blue)',
    innovation: 'cognitive extension',
    desc: 'Cognitive extension and externalization — attention becomes a contested resource.',
  },
];

/** Part 5-0: Historical eras of consciousness technology */
export default function HistoricalTimeline() {
  const [hovered, setHovered] = useState<number | null>(null);

  const timelineY = 260;
  const boxY = 100;
  const boxW = 105, boxH = 56, boxR = 6;

  // Evenly space eras across the width
  const eraX = (i: number) => 75 + i * 120;

  return (
    <svg viewBox="0 0 930 420" className={`diagram-svg${hovered !== null ? ' has-focus' : ''}`}
      role="img"
      aria-label="Historical timeline of consciousness technology: seven eras from pre-axial ritual through digital/AI cognitive extension">

      {/* Timeline arrow */}
      <line x1={40} y1={timelineY} x2={890} y2={timelineY}
        stroke="var(--d-line)" strokeWidth={1} />
      <path d={arrowPath([890, timelineY], 0, 7)}
        stroke="var(--d-line)" strokeWidth={1} fill="none" />

      {/* Timeline tick marks and date labels */}
      {ERAS.map((era, i) => {
        const x = eraX(i);
        const isDimmed = hovered !== null && hovered !== i;
        return (
          <g key={`tick-${i}`}
            style={{ opacity: isDimmed ? 0.25 : 1, transition: 'opacity 0.2s' }}
          >
            <line x1={x} y1={timelineY - 4} x2={x} y2={timelineY + 4}
              stroke="var(--d-line)" strokeWidth={0.75} />
            <text x={x} y={timelineY + 20}
              textAnchor="middle" dominantBaseline="central"
              fontSize={10} fill="var(--d-muted)"
              fontFamily="var(--font-body, Georgia, serif)"
            >
              {era.date}
            </text>
          </g>
        );
      })}

      {/* Era boxes + connectors */}
      {ERAS.map((era, i) => {
        const x = eraX(i);
        const isFocused = hovered === i;
        const isDimmed = hovered !== null && !isFocused;

        return (
          <g key={era.label}
            className={`interactive${isFocused ? ' focused' : ''}`}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            style={{ opacity: isDimmed ? 0.25 : 1, transition: 'opacity 0.2s' }}
          >
            {/* Connector line */}
            <line x1={x} y1={boxY + boxH / 2 + boxH / 2 + 4} x2={x} y2={timelineY - 5}
              stroke={era.color} strokeWidth={0.5} opacity={0.5}
              strokeDasharray="3,3"
            />

            {/* Box */}
            <rect
              x={x - boxW / 2} y={boxY - boxH / 2}
              width={boxW} height={boxH} rx={boxR}
              fill={era.color} fillOpacity={isFocused ? 0.2 : 0.08}
              stroke={era.color} strokeWidth={isFocused ? 1.2 : 0.75}
              style={{ transition: 'fill-opacity 0.2s, stroke-width 0.2s' }}
            />

            {/* Era name */}
            <text x={x} y={boxY - 5}
              textAnchor="middle" dominantBaseline="central"
              fontSize={12} fontWeight="500" fill="var(--d-fg)"
              fontFamily="var(--font-body, Georgia, serif)"
            >
              {era.label}
            </text>

            {/* Innovation label */}
            <text x={x} y={boxY + 14}
              textAnchor="middle" dominantBaseline="central"
              fontSize={9.5} fontStyle="italic" fill="var(--d-muted)"
              fontFamily="var(--font-body, Georgia, serif)"
            >
              {era.innovation}
            </text>

            {/* Dot on timeline */}
            <circle cx={x} cy={timelineY} r={isFocused ? 4 : 2.5}
              fill={era.color}
              style={{ transition: 'r 0.2s' }}
            />
          </g>
        );
      })}

      {/* Hover description */}
      {hovered !== null && (
        <g>
          <rect
            x={40} y={350}
            width={850} height={36} rx={4}
            fill="var(--d-fg)" fillOpacity={0.06}
            stroke="var(--d-line)" strokeWidth={0.5} strokeOpacity={0.3}
          />
          <text
            x={465} y={368}
            textAnchor="middle" dominantBaseline="central"
            fontSize={12} fill="var(--d-muted)"
            fontFamily="var(--font-body, Georgia, serif)"
          >
            {ERAS[hovered].desc}
          </text>
        </g>
      )}
    </svg>
  );
}
