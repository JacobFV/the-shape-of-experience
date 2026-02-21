'use client';

import { useState } from 'react';
import { type Point, polar, pt, arrowPath } from './utils';

/**
 * IotaFeedback — Part V diagram
 * Two self-reinforcing loops:
 * Low ι path: participatory perception → gods visible → rituals strengthen → ι stays low
 * High ι path: mechanistic perception → gods invisible → rituals weaken → further disenchantment
 */

interface LoopNode {
  label: string;
  angle: number;
}

const LOW_NODES: LoopNode[] = [
  { label: 'low ι', angle: -90 },
  { label: 'participatory\nperception', angle: 0 },
  { label: 'gods visible', angle: 90 },
  { label: 'rituals\nstrengthen', angle: 180 },
];

const HIGH_NODES: LoopNode[] = [
  { label: 'high ι', angle: -90 },
  { label: 'mechanistic\nperception', angle: 0 },
  { label: 'gods invisible', angle: 90 },
  { label: 'rituals\nweaken', angle: 180 },
];

const W = 660;
const H = 400;
const LOOP_R = 100;
const NODE_R = 38;
const LEFT_CX = 175;
const RIGHT_CX = 485;
const CY = 210;

function LoopDiagram({ nodes, cx, cy, color, label, isFocused, isDimmed, onEnter, onLeave }: {
  nodes: LoopNode[];
  cx: number; cy: number;
  color: string;
  label: string;
  isFocused: boolean;
  isDimmed: boolean;
  onEnter: () => void;
  onLeave: () => void;
}) {
  const positions = nodes.map(n => ({
    ...n,
    pos: polar([cx, cy], LOOP_R, n.angle),
  }));

  return (
    <g
      className={`interactive${isFocused ? ' focused' : ''}`}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      style={{
        opacity: isDimmed ? 0.2 : 1,
        transition: 'opacity 0.3s',
        cursor: 'pointer',
      }}
    >
      {/* Loop label */}
      <text x={cx} y={cy - LOOP_R - NODE_R - 18} textAnchor="middle"
        fill={color} fontSize={13} fontWeight={700}
        fontFamily="var(--font-body, Georgia, serif)">
        {label}
      </text>

      {/* Curved arrows between nodes */}
      {positions.map((node, i) => {
        const next = positions[(i + 1) % positions.length];
        const midAngle = (node.angle + next.angle) / 2 + (next.angle < node.angle ? 180 : 0);
        const curveR = LOOP_R + 20;
        const midPt = polar([cx, cy], curveR, midAngle);

        const startPt = polar(node.pos, NODE_R + 2, node.angle + 90);
        const endPt = polar(next.pos, NODE_R + 2, next.angle - 90);

        return (
          <g key={i}>
            <path
              d={`M ${pt(startPt)} Q ${pt(midPt)} ${pt(endPt)}`}
              fill="none" stroke={color} strokeWidth={isFocused ? 1.8 : 1.2}
              opacity={0.5} strokeDasharray={isFocused ? 'none' : '4 3'}
              style={{ transition: 'stroke-width 0.2s' }}
            />
            {/* Arrowhead at end */}
            <path d={arrowPath(endPt, next.angle - 90 + 20, 5)}
              stroke={color} strokeWidth={1.2} fill="none" opacity={0.6} />
          </g>
        );
      })}

      {/* Nodes */}
      {positions.map((node, i) => (
        <g key={i}>
          <circle cx={node.pos[0]} cy={node.pos[1]} r={NODE_R}
            fill={color} fillOpacity={isFocused ? 0.15 : 0.06}
            stroke={color} strokeWidth={isFocused ? 1.5 : 0.75}
            style={{ transition: 'fill-opacity 0.2s, stroke-width 0.2s' }}
          />
          {/* Handle multiline labels */}
          {node.label.split('\n').map((line, li) => (
            <text key={li} x={node.pos[0]} y={node.pos[1] + (li - (node.label.split('\n').length - 1) / 2) * 13}
              textAnchor="middle" dominantBaseline="central"
              fill="var(--d-fg)" fontSize={10}
              fontFamily="var(--font-body, Georgia, serif)">
              {line}
            </text>
          ))}
        </g>
      ))}

      {/* Center loop arrow indicator */}
      <text x={cx} y={cy} textAnchor="middle" dominantBaseline="central"
        fill={color} fontSize={18} opacity={0.3}>
        ↻
      </text>
    </g>
  );
}

export default function IotaFeedback() {
  const [hovered, setHovered] = useState<'low' | 'high' | null>(null);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={`diagram-svg${hovered ? ' has-focus' : ''}`}
      role="img"
      aria-label="Two self-reinforcing feedback loops: low iota maintains participatory perception while high iota deepens mechanistic disenchantment"
    >
      {/* Title */}
      <text x={W / 2} y={24} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}
        fontFamily="var(--font-body, Georgia, serif)">
        The ι Feedback Loop
      </text>
      <text x={W / 2} y={42} textAnchor="middle" fill="var(--d-muted)" fontSize={10.5}
        fontFamily="var(--font-body, Georgia, serif)">
        each mode reinforces its own perceptual conditions
      </text>

      {/* Low ι loop */}
      <LoopDiagram
        nodes={LOW_NODES} cx={LEFT_CX} cy={CY}
        color="var(--d-green)" label="Enchantment Cycle"
        isFocused={hovered === 'low'}
        isDimmed={hovered !== null && hovered !== 'low'}
        onEnter={() => setHovered('low')}
        onLeave={() => setHovered(null)}
      />

      {/* High ι loop */}
      <LoopDiagram
        nodes={HIGH_NODES} cx={RIGHT_CX} cy={CY}
        color="var(--d-red)" label="Disenchantment Cycle"
        isFocused={hovered === 'high'}
        isDimmed={hovered !== null && hovered !== 'high'}
        onEnter={() => setHovered('high')}
        onLeave={() => setHovered(null)}
      />

      {/* Center divider with bidirectional arrow */}
      <line x1={W / 2 - 20} y1={CY - 40} x2={W / 2 + 20} y2={CY - 40}
        stroke="var(--d-line)" strokeWidth={0.5} opacity={0.3} />
      <line x1={W / 2 - 20} y1={CY + 40} x2={W / 2 + 20} y2={CY + 40}
        stroke="var(--d-line)" strokeWidth={0.5} opacity={0.3} />
      <text x={W / 2} y={CY} textAnchor="middle" dominantBaseline="central"
        fill="var(--d-muted)" fontSize={9} fontStyle="italic"
        fontFamily="var(--font-body, Georgia, serif)">
        basin boundary
      </text>

      {/* Bottom insight */}
      <text x={W / 2} y={H - 20} textAnchor="middle" fill="var(--d-muted)" fontSize={10.5}
        fontStyle="italic" fontFamily="var(--font-body, Georgia, serif)">
        ι ≈ 0.30 is the evolutionary default — high ι is the departure
      </text>
    </svg>
  );
}
