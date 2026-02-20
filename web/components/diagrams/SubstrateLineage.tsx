'use client';

import { useState } from 'react';

/**
 * The Lenia substrate lineage tree (V11 → V18).
 *
 * Shows how each substrate variant built on the previous one,
 * what it added, and whether it helped or hurt.
 */

interface SubNode {
  id: string;
  label: string;
  added: string;
  result: string;
  status: 'positive' | 'mixed' | 'negative';
  col: number;
  row: number;
}

const NODES: SubNode[] = [
  { id: 'V11', label: 'V11: Lenia Evolution', added: 'Baseline CA', result: 'Curriculum > complexity',
    status: 'mixed', col: 0, row: 0 },
  { id: 'V12', label: 'V12: Attention', added: '+ evolvable attention', result: '+2.0pp, 42% Φ↑ cycles',
    status: 'mixed', col: 1, row: 0 },
  { id: 'V13', label: 'V13: Content Coupling', added: '+ content-based kernels', result: 'Foundation substrate',
    status: 'positive', col: 2, row: 0 },
  { id: 'V14', label: 'V14: Chemotaxis', added: '+ directed motion', result: 'Motility selectable',
    status: 'mixed', col: 0, row: 1 },
  { id: 'V15', label: 'V15: Temporal Memory', added: '+ memory channels', result: 'Memory selectable, Φ-stress 2×',
    status: 'positive', col: 0, row: 2 },
  { id: 'V16', label: 'V16: Hebbian', added: '+ plasticity', result: 'HURTS robustness (0.892)',
    status: 'negative', col: 0, row: 3 },
  { id: 'V17', label: 'V17: Signaling', added: '+ quorum signals', result: 'Suppressed 2/3 seeds',
    status: 'mixed', col: 1, row: 3 },
  { id: 'V18', label: 'V18: Boundary', added: '+ boundary sensing', result: 'BEST: rob 0.969, max 1.651',
    status: 'positive', col: 2, row: 3 },
];

const EDGES: { from: string; to: string }[] = [
  { from: 'V11', to: 'V12' },
  { from: 'V12', to: 'V13' },
  { from: 'V13', to: 'V14' },
  { from: 'V14', to: 'V15' },
  { from: 'V15', to: 'V16' },
  { from: 'V15', to: 'V17' },
  { from: 'V15', to: 'V18' },
];

const statusColor = (s: SubNode['status']) => {
  switch (s) {
    case 'positive': return 'var(--d-green)';
    case 'mixed': return 'var(--d-yellow)';
    case 'negative': return 'var(--d-red)';
  }
};

const statusIcon = (s: SubNode['status']) => {
  switch (s) {
    case 'positive': return '✓';
    case 'mixed': return '~';
    case 'negative': return '✗';
  }
};

export default function SubstrateLineage() {
  const [hovered, setHovered] = useState<string | null>(null);

  const colW = 175;
  const rowH = 65;
  const boxW = 155;
  const boxH = 50;
  const startX = 40;
  const startY = 50;

  const nodePos = (node: SubNode) => ({
    x: startX + node.col * colW,
    y: startY + node.row * rowH,
  });

  const nodeMap = new Map(NODES.map(n => [n.id, n]));
  const totalW = startX * 2 + 3 * colW;
  const totalH = startY + 4 * rowH + 50;

  return (
    <svg viewBox={`0 0 ${totalW} ${totalH}`} className="diagram-svg" role="img"
      aria-label="Substrate lineage: how each Lenia variant from V11 to V18 built on previous work">

      {/* Title */}
      <text x={totalW / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        The Substrate Ladder (V11–V18)
      </text>
      <text x={totalW / 2} y={35} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        what each variant added, and whether it helped
      </text>

      {/* Arrow defs */}
      <defs>
        <marker id="sl-arrow" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
          <polygon points="0 0, 7 2.5, 0 5" fill="var(--d-line)" />
        </marker>
      </defs>

      {/* Edges */}
      {EDGES.map((edge, i) => {
        const from = nodeMap.get(edge.from)!;
        const to = nodeMap.get(edge.to)!;
        const fp = nodePos(from);
        const tp = nodePos(to);
        const isHL = hovered === edge.from || hovered === edge.to;
        const isDimmed = hovered && !isHL;

        // Determine connection direction
        let x1: number, y1: number, x2: number, y2: number;
        if (from.row === to.row) {
          // Horizontal
          x1 = fp.x + boxW; y1 = fp.y + boxH / 2;
          x2 = tp.x; y2 = tp.y + boxH / 2;
        } else if (from.col === to.col) {
          // Vertical
          x1 = fp.x + boxW / 2; y1 = fp.y + boxH;
          x2 = tp.x + boxW / 2; y2 = tp.y;
        } else {
          // Diagonal
          x1 = fp.x + boxW / 2; y1 = fp.y + boxH;
          x2 = tp.x + boxW / 2; y2 = tp.y;
        }

        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        const path = from.col === to.col && from.row !== to.row
          ? `M ${x1} ${y1} L ${x2} ${y2}`
          : `M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`;

        return (
          <g key={i} style={{ opacity: isDimmed ? 0.15 : 1, transition: 'opacity 0.2s' }}>
            <path d={path} fill="none"
              stroke={isHL ? 'var(--d-fg)' : 'var(--d-line)'}
              strokeWidth={isHL ? 1.5 : 1} opacity={isHL ? 0.7 : 0.4}
              markerEnd="url(#sl-arrow)" />
            {/* "added" label */}
            <text x={mx} y={my - 5} textAnchor="middle"
              fontSize={7.5} fill="var(--d-muted)" fontStyle="italic"
              fontFamily="var(--font-body, Georgia, serif)"
              style={{ pointerEvents: 'none' }}>
              {nodeMap.get(edge.to)!.added}
            </text>
          </g>
        );
      })}

      {/* Nodes */}
      {NODES.map(node => {
        const { x, y } = nodePos(node);
        const color = statusColor(node.status);
        const isHL = hovered === node.id;
        const isConnected = hovered ? EDGES.some(
          e => (e.from === hovered && e.to === node.id) ||
               (e.to === hovered && e.from === node.id)
        ) : false;
        const isDimmed = hovered && !isHL && !isConnected;

        return (
          <g key={node.id}
            onMouseEnter={() => setHovered(node.id)}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: 'pointer', opacity: isDimmed ? 0.2 : 1, transition: 'opacity 0.2s' }}>
            <rect x={x} y={y} width={boxW} height={boxH} rx={5}
              fill={color} fillOpacity={isHL ? 0.18 : 0.07}
              stroke={color} strokeWidth={isHL ? 2 : 1}
              style={{ transition: 'fill-opacity 0.2s, stroke-width 0.2s' }} />
            {/* Title */}
            <text x={x + boxW / 2} y={y + 18}
              textAnchor="middle" fontSize={11} fontWeight={600} fill={color}
              fontFamily="var(--font-body, Georgia, serif)">
              {node.label}
            </text>
            {/* Result */}
            <text x={x + boxW / 2} y={y + 34}
              textAnchor="middle" fontSize={8.5} fill="var(--d-muted)"
              fontFamily="var(--font-body, Georgia, serif)">
              {node.result}
            </text>
            {/* Status icon */}
            <text x={x + boxW - 12} y={y + 12}
              textAnchor="middle" fontSize={10} fontWeight={700} fill={color}
              fontFamily="var(--font-body, Georgia, serif)">
              {statusIcon(node.status)}
            </text>
          </g>
        );
      })}

      {/* Bottom annotation */}
      <text x={totalW / 2} y={totalH - 12} textAnchor="middle"
        fontSize={9} fill="var(--d-muted)" fontStyle="italic"
        fontFamily="var(--font-body, Georgia, serif)">
        V15 branches three ways: plasticity hurts, signaling is suppressed, boundary sensing wins
      </text>
    </svg>
  );
}
