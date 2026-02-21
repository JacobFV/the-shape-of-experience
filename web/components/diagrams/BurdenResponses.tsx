'use client';

import { useState } from 'react';
import { type Point, polar, pt, arrowPath, angleBetween } from './utils';

/**
 * BurdenResponses — Part III diagram
 * Central "burden" node (self-awareness + mortality) radiating to four response strategies
 * Each response maps to affect signature (V, A, Φ, ι values)
 * Hover to highlight response and its affect coordinates
 */

interface Response {
  id: string;
  label: string;
  mechanism: string;
  affects: { V: string; A: string; Phi: string; iota: string };
  color: string;
  angle: number;
}

const RESPONSES: Response[] = [
  {
    id: 'terror',
    label: 'Terror Management',
    mechanism: 'symbolic immortality projects',
    affects: { V: '+', A: 'high', Phi: 'low', iota: '0.4-0.6' },
    color: 'var(--d-red)',
    angle: -45,
  },
  {
    id: 'meaning',
    label: 'Meaning Maintenance',
    mechanism: 'coherence-restoring narratives',
    affects: { V: '+', A: 'med', Phi: 'high', iota: '0.3-0.7' },
    color: 'var(--d-blue)',
    angle: 45,
  },
  {
    id: 'attachment',
    label: 'Attachment',
    mechanism: 'co-regulation via bonded other',
    affects: { V: '+', A: 'low', Phi: 'high', iota: '0.1-0.3' },
    color: 'var(--d-green)',
    angle: 135,
  },
  {
    id: 'flow',
    label: 'Flow',
    mechanism: 'absorption dissolves self-model',
    affects: { V: '+', A: 'high', Phi: 'high', iota: '0.0-0.2' },
    color: 'var(--d-orange)',
    angle: -135,
  },
];

const W = 620;
const H = 500;
const CX = W / 2;
const CY = 230;
const INNER_R = 45;
const OUTER_R = 160;
const NODE_W = 110;
const NODE_H = 80;

export default function BurdenResponses() {
  const [hovered, setHovered] = useState<string | null>(null);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={`diagram-svg${hovered ? ' has-focus' : ''}`}
      role="img"
      aria-label="Four existential burden responses radiating from central self-awareness node: terror management, meaning maintenance, attachment, and flow"
    >
      {/* Title */}
      <text x={CX} y={24} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}
        fontFamily="var(--font-body, Georgia, serif)">
        Responses to the Existential Burden
      </text>
      <text x={CX} y={42} textAnchor="middle" fill="var(--d-muted)" fontSize={10.5}
        fontFamily="var(--font-body, Georgia, serif)">
        four strategies for a system that knows it will end
      </text>

      {/* Radiating arms and response nodes */}
      {RESPONSES.map(resp => {
        const pos = polar([CX, CY], OUTER_R, resp.angle);
        const armEnd = polar([CX, CY], OUTER_R - NODE_W / 2 - 10, resp.angle);
        const armStart = polar([CX, CY], INNER_R + 6, resp.angle);
        const isFocused = hovered === resp.id;
        const isDimmed = hovered !== null && !isFocused;

        return (
          <g key={resp.id}
            className={`interactive${isFocused ? ' focused' : ''}`}
            onMouseEnter={() => setHovered(resp.id)}
            onMouseLeave={() => setHovered(null)}
            style={{
              opacity: isDimmed ? 0.2 : 1,
              transition: 'opacity 0.25s',
              cursor: 'pointer',
            }}
          >
            {/* Connecting arm */}
            <line x1={armStart[0]} y1={armStart[1]} x2={armEnd[0]} y2={armEnd[1]}
              stroke={resp.color} strokeWidth={isFocused ? 2 : 1.2}
              opacity={isFocused ? 0.8 : 0.4}
              style={{ transition: 'stroke-width 0.2s, opacity 0.2s' }}
            />
            {/* Arrow at end */}
            <path d={arrowPath(armEnd, resp.angle, 6)}
              stroke={resp.color} strokeWidth={1.2} fill="none"
              opacity={isFocused ? 0.8 : 0.4} />

            {/* Response node */}
            <rect
              x={pos[0] - NODE_W / 2} y={pos[1] - NODE_H / 2}
              width={NODE_W} height={NODE_H} rx={8}
              fill={resp.color} fillOpacity={isFocused ? 0.15 : 0.06}
              stroke={resp.color} strokeWidth={isFocused ? 1.5 : 0.75}
              style={{ transition: 'fill-opacity 0.2s, stroke-width 0.2s' }}
            />
            <text x={pos[0]} y={pos[1] - 18} textAnchor="middle"
              dominantBaseline="central" fill={resp.color} fontSize={11.5} fontWeight={700}
              fontFamily="var(--font-body, Georgia, serif)">
              {resp.label}
            </text>
            <text x={pos[0]} y={pos[1] + 2} textAnchor="middle"
              dominantBaseline="central" fill="var(--d-muted)" fontSize={8.5}
              fontFamily="var(--font-body, Georgia, serif)">
              {resp.mechanism}
            </text>

            {/* Affect signature on hover */}
            {isFocused && (
              <g>
                <text x={pos[0]} y={pos[1] + 22} textAnchor="middle"
                  fill={resp.color} fontSize={8.5}
                  fontFamily="var(--font-body, Georgia, serif)">
                  V={resp.affects.V}  A={resp.affects.A}  Φ={resp.affects.Phi}  ι={resp.affects.iota}
                </text>
              </g>
            )}
          </g>
        );
      })}

      {/* Central burden node */}
      <circle cx={CX} cy={CY} r={INNER_R}
        fill="var(--d-fg)" fillOpacity={0.06}
        stroke="var(--d-fg)" strokeWidth={1.5} />
      <text x={CX} y={CY - 12} textAnchor="middle" dominantBaseline="central"
        fill="var(--d-fg)" fontSize={12} fontWeight={700}
        fontFamily="var(--font-body, Georgia, serif)">
        The Burden
      </text>
      <text x={CX} y={CY + 6} textAnchor="middle" dominantBaseline="central"
        fill="var(--d-muted)" fontSize={9.5}
        fontFamily="var(--font-body, Georgia, serif)">
        self-awareness
      </text>
      <text x={CX} y={CY + 20} textAnchor="middle" dominantBaseline="central"
        fill="var(--d-muted)" fontSize={9.5}
        fontFamily="var(--font-body, Georgia, serif)">
        + mortality
      </text>

      {/* Bottom insight */}
      <text x={CX} y={H - 30} textAnchor="middle" fill="var(--d-muted)" fontSize={10.5}
        fontStyle="italic" fontFamily="var(--font-body, Georgia, serif)">
        each response modulates a different region of affect space
      </text>
      <text x={CX} y={H - 14} textAnchor="middle" fill="var(--d-muted)" fontSize={9.5}
        fontFamily="var(--font-body, Georgia, serif)">
        hover to see affect signatures
      </text>
    </svg>
  );
}
