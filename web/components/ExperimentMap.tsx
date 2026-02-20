'use client';

import React, { useState } from 'react';
import Link from 'next/link';

// ─── Experiment data ──────────────────────────────────────────────
type Status = 'positive' | 'mixed' | 'negative' | 'null' | 'planned' | 'running' | 'foundation';

interface ExpNode {
  id: string;
  title: string;
  status: Status;
  x: number;
  y: number;
  width?: number;
  section?: string; // slug for linking
  subtitle?: string;
}

interface ExpEdge {
  from: string;
  to: string;
  label?: string;
  dashed?: boolean;
}

const COLORS: Record<Status, { fill: string; stroke: string; text: string }> = {
  positive:   { fill: '#1a3a2a', stroke: '#4ade80', text: '#4ade80' },
  mixed:      { fill: '#3a3a1a', stroke: '#facc15', text: '#facc15' },
  negative:   { fill: '#3a1a1a', stroke: '#f87171', text: '#f87171' },
  null:       { fill: '#2a2a2a', stroke: '#6b7280', text: '#9ca3af' },
  planned:    { fill: '#1a2a3a', stroke: '#60a5fa', text: '#60a5fa' },
  running:    { fill: '#1a2a3a', stroke: '#60a5fa', text: '#60a5fa' },
  foundation: { fill: '#1a2a3a', stroke: '#a78bfa', text: '#a78bfa' },
};

const STATUS_LABELS: Record<Status, string> = {
  positive:   'Confirmed',
  mixed:      'Mixed',
  negative:   'Negative',
  null:       'Null',
  planned:    'Planned',
  running:    'Running',
  foundation: 'Foundation',
};

// ─── Layout constants ─────────────────────────────────────────────
const NODE_W = 170;
const NODE_H = 52;
const PHASE_LABEL_W = 28;

// Grid spacings
const COL = 195;  // horizontal spacing between nodes
const ROW = 72;   // vertical spacing between rows

// ─── Define all experiment nodes ──────────────────────────────────
const nodes: ExpNode[] = [
  // Phase 1: LLM (row 0)
  { id: 'V2-V9', title: 'V2-V9: LLM Affect', status: 'mixed', x: 50, y: 30, width: 180, section: 'v2-v9-llm-affect-signatures', subtitle: 'Opposite dynamics to bio' },

  // Phase 2: MARL (row 0, right)
  { id: 'V10', title: 'V10: MARL Ablation', status: 'positive', x: 450, y: 30, section: 'v10-marl-forcing-function-ablation', subtitle: 'Geometry is baseline' },

  // Phase 3: Lenia substrate (rows 1-4)
  { id: 'V11', title: 'V11: Lenia Evolution', status: 'mixed', x: 50, y: 30 + ROW, section: 'v11-lenia-ca-evolution', subtitle: 'Curriculum > complexity' },
  { id: 'V12', title: 'V12: Attention Lenia', status: 'mixed', x: 50 + COL, y: 30 + ROW, section: 'v12-attention-based-lenia', subtitle: 'Necessary not sufficient' },
  { id: 'V13', title: 'V13: Content Coupling', status: 'foundation', x: 50 + COL * 2, y: 30 + ROW, width: 180, section: 'v13-content-based-coupling', subtitle: 'Foundation substrate' },

  // V13 measurement experiments (row 2)
  { id: 'Exp0-12', title: 'Exp 0-12: Measurements', status: 'positive', x: 50 + COL * 2, y: 30 + ROW * 2, width: 190, section: 'the-emergence-experiment-program', subtitle: '11 experiments on V13' },

  // Substrate evolution (rows 2-3)
  { id: 'V14', title: 'V14: Chemotaxis', status: 'mixed', x: 50, y: 30 + ROW * 2, section: 'v14-chemotactic-lenia', subtitle: 'Directed motion' },
  { id: 'V15', title: 'V15: Temporal Memory', status: 'positive', x: 50, y: 30 + ROW * 3, section: 'v15-temporal-memory', subtitle: 'Memory selectable' },
  { id: 'V16', title: 'V16: Hebbian Plasticity', status: 'negative', x: 50, y: 30 + ROW * 4, section: 'v16-hebbian-plasticity', subtitle: 'Hurts robustness' },
  { id: 'V17', title: 'V17: Quorum Signaling', status: 'mixed', x: 50 + COL, y: 30 + ROW * 4, section: 'v17-quorum-signaling', subtitle: 'Suppressed 2/3 seeds' },
  { id: 'V18', title: 'V18: Boundary Lenia', status: 'positive', x: 50 + COL * 2, y: 30 + ROW * 4, section: 'v18-boundary-dependent-lenia', subtitle: 'Best robustness: 0.969' },

  // V19 Mechanism (row 5)
  { id: 'V19', title: 'V19: Bottleneck Furnace', status: 'positive', x: 50 + COL * 2, y: 30 + ROW * 5, width: 190, section: 'v19-bottleneck-furnace', subtitle: 'Creation confirmed 2/3' },

  // Phase 6: Protocell Agency (rows 6-11)
  { id: 'V20', title: 'V20: Protocell Agency', status: 'positive', x: 50, y: 30 + ROW * 6, width: 190, section: 'v20-protocell-agency', subtitle: 'Wall broken: \u03C1=0.21' },
  { id: 'V21', title: 'V21: CTM Inner Ticks', status: 'mixed', x: 50 + COL, y: 30 + ROW * 7, section: 'v21-ctm-inner-ticks', subtitle: 'Ticks survive, no adapt.' },
  { id: 'V22', title: 'V22: Predictive Gradient', status: 'mixed', x: 50, y: 30 + ROW * 8, width: 190, section: 'v22-intrinsic-predictive-gradient', subtitle: 'Prediction \u2260 integration' },

  // Prediction variants (rows 9-10)
  { id: 'V23', title: 'V23: Multi-Target', status: 'negative', x: 50, y: 30 + ROW * 9, section: 'v23-world-model-gradient', subtitle: 'Specialization \u2260 integ.' },
  { id: 'V24', title: 'V24: TD Value', status: 'mixed', x: 50 + COL, y: 30 + ROW * 9, section: 'v24-td-value-learning', subtitle: 'Best survival, \u03A6 mixed' },
  { id: 'V25', title: 'V25: Predator-Prey', status: 'negative', x: 50 + COL * 2, y: 30 + ROW * 9, section: 'v25-predator-prey', subtitle: 'Rich obs \u2192 reactive' },
  { id: 'V26', title: 'V26: POMDP', status: 'mixed', x: 50 + COL * 2, y: 30 + ROW * 10, section: 'v26-pomdp', subtitle: 'Type encoding, 100% mort' },

  // MLP breakthrough (rows 9-11)
  { id: 'V27', title: 'V27: MLP Head', status: 'positive', x: 50 + COL * 3, y: 30 + ROW * 9, section: 'v27-nonlinear-mlp-head', subtitle: '\u03A6=0.245 record!' },
  { id: 'V28', title: 'V28: Width Sweep', status: 'mixed', x: 50 + COL * 3, y: 30 + ROW * 10, section: 'v28-bottleneck-width-sweep', subtitle: 'Gradient coupling mech.' },
  { id: 'V29', title: 'V29: Social Prediction', status: 'mixed', x: 50 + COL * 2, y: 30 + ROW * 11, section: 'v29-social-prediction', subtitle: '3-seed fluke' },
  { id: 'V30', title: 'V30: Dual Prediction', status: 'negative', x: 50, y: 30 + ROW * 11, section: 'v30-dual-prediction', subtitle: 'Gradient imbalance' },
  { id: 'V31', title: 'V31: 10-Seed Validation', status: 'positive', x: 50 + COL * 3, y: 30 + ROW * 11, width: 190, section: 'v31-10-seed-validation', subtitle: '30/30/40 split, r=0.997' },

  // Next gen (row 12)
  { id: 'V32', title: 'V32: Drought Autopsy', status: 'running', x: 50, y: 30 + ROW * 12.5, section: 'v32-drought-autopsy', subtitle: '50 seeds' },
  { id: 'V33', title: 'V33: Contrastive Pred.', status: 'planned', x: 50 + COL, y: 30 + ROW * 12.5, section: 'v33-contrastive-self-prediction', subtitle: 'Counterfactual rep.' },
  { id: 'V34', title: 'V34: \u03A6-Fitness', status: 'planned', x: 50 + COL * 2, y: 30 + ROW * 12.5, section: 'v34-phi-inclusive-fitness', subtitle: 'Direct selection' },
  { id: 'V35', title: 'V35: Language', status: 'planned', x: 50 + COL * 3, y: 30 + ROW * 12.5, section: 'v35-language-emergence', subtitle: 'Cooperative POMDP' },
];

const edges: ExpEdge[] = [
  // Lenia substrate lineage
  { from: 'V11', to: 'V12', label: 'add attention' },
  { from: 'V12', to: 'V13', label: 'content coupling' },
  { from: 'V13', to: 'Exp0-12', label: 'measure' },
  { from: 'V13', to: 'V14', label: 'add motion' },
  { from: 'V14', to: 'V15', label: 'add memory' },
  { from: 'V15', to: 'V16', label: 'add plasticity' },
  { from: 'V15', to: 'V17', label: 'add signaling' },
  { from: 'V15', to: 'V18', label: 'add boundary' },
  { from: 'V18', to: 'V19', label: 'test mechanism' },

  // Protocell agency lineage (new paradigm)
  { from: 'V18', to: 'V20', label: 'wall \u2192 new substrate', dashed: true },
  { from: 'V20', to: 'V21', label: 'add inner ticks' },
  { from: 'V21', to: 'V22', label: 'add gradient' },

  // Prediction target variations
  { from: 'V22', to: 'V23', label: 'multi-target' },
  { from: 'V22', to: 'V24', label: 'TD value' },
  { from: 'V22', to: 'V25', label: 'richer env' },
  { from: 'V25', to: 'V26', label: 'partial obs' },
  { from: 'V22', to: 'V27', label: 'MLP head' },
  { from: 'V27', to: 'V28', label: 'sweep width' },
  { from: 'V27', to: 'V29', label: 'social target' },
  { from: 'V29', to: 'V30', label: 'dual target' },
  { from: 'V29', to: 'V31', label: '10-seed validation' },

  // Next gen
  { from: 'V31', to: 'V32', label: '50-seed autopsy', dashed: true },
  { from: 'V27', to: 'V33', label: 'contrastive', dashed: true },
  { from: 'V27', to: 'V34', label: '\u03A6 selection', dashed: true },
  { from: 'V27', to: 'V35', label: 'language', dashed: true },
];

// ─── Phase labels (vertical left sidebar) ─────────────────────────
interface PhaseLabel {
  label: string;
  yStart: number;
  yEnd: number;
  color: string;
}

// ─── SVG helpers ──────────────────────────────────────────────────

function getNodeCenter(node: ExpNode): { cx: number; cy: number } {
  const w = node.width || NODE_W;
  return { cx: node.x + w / 2, cy: node.y + NODE_H / 2 };
}

function getNodePort(node: ExpNode, side: 'top' | 'bottom' | 'left' | 'right'): { x: number; y: number } {
  const w = node.width || NODE_W;
  switch (side) {
    case 'top':    return { x: node.x + w / 2, y: node.y };
    case 'bottom': return { x: node.x + w / 2, y: node.y + NODE_H };
    case 'left':   return { x: node.x, y: node.y + NODE_H / 2 };
    case 'right':  return { x: node.x + w, y: node.y + NODE_H / 2 };
  }
}

function computeEdgePath(from: ExpNode, to: ExpNode): string {
  const fc = getNodeCenter(from);
  const tc = getNodeCenter(to);
  const dx = tc.cx - fc.cx;
  const dy = tc.cy - fc.cy;

  let fromSide: 'top' | 'bottom' | 'left' | 'right';
  let toSide: 'top' | 'bottom' | 'left' | 'right';

  if (Math.abs(dy) > Math.abs(dx) * 0.5) {
    fromSide = dy > 0 ? 'bottom' : 'top';
    toSide = dy > 0 ? 'top' : 'bottom';
  } else {
    fromSide = dx > 0 ? 'right' : 'left';
    toSide = dx > 0 ? 'left' : 'right';
  }

  const p1 = getNodePort(from, fromSide);
  const p2 = getNodePort(to, toSide);

  // Cubic bezier with control points for smooth curves
  const cpOffset = Math.min(Math.abs(p2.y - p1.y), Math.abs(p2.x - p1.x)) * 0.4 + 15;
  let c1x: number, c1y: number, c2x: number, c2y: number;

  if (fromSide === 'bottom' || fromSide === 'top') {
    const dir = fromSide === 'bottom' ? 1 : -1;
    c1x = p1.x; c1y = p1.y + dir * cpOffset;
    c2x = p2.x; c2y = p2.y - dir * cpOffset;
  } else {
    const dir = fromSide === 'right' ? 1 : -1;
    c1x = p1.x + dir * cpOffset; c1y = p1.y;
    c2x = p2.x - dir * cpOffset; c2y = p2.y;
  }

  return `M ${p1.x} ${p1.y} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${p2.x} ${p2.y}`;
}

function edgeLabelPos(from: ExpNode, to: ExpNode): { x: number; y: number } {
  const fc = getNodeCenter(from);
  const tc = getNodeCenter(to);
  return { x: (fc.cx + tc.cx) / 2, y: (fc.cy + tc.cy) / 2 - 6 };
}

// ─── Main component ───────────────────────────────────────────────

export default function ExperimentMap() {
  const [hovered, setHovered] = useState<string | null>(null);

  const nodeMap = new Map(nodes.map(n => [n.id, n]));

  // Calculate SVG dimensions
  const maxX = Math.max(...nodes.map(n => n.x + (n.width || NODE_W))) + 40;
  const maxY = Math.max(...nodes.map(n => n.y + NODE_H)) + 40;

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      <svg
        viewBox={`0 0 ${maxX} ${maxY}`}
        width="100%"
        style={{ maxWidth: maxX, minWidth: 600, fontFamily: 'var(--font-mono, monospace)' }}
      >
        <defs>
          <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#6b7280" />
          </marker>
          <marker id="arrowhead-hl" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#e5e7eb" />
          </marker>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        {/* Background */}
        <rect x="0" y="0" width={maxX} height={maxY} fill="#0a0a0a" rx="8" />

        {/* Phase separator lines */}
        {[
          { y: 30 + ROW * 0.5, label: 'Independent baselines' },
          { y: 30 + ROW * 5.5, label: 'Lenia CA substrate' },
          { y: 30 + ROW * 8 - 15, label: 'Protocell agency' },
          { y: 30 + ROW * 12, label: 'Next generation' },
        ].map((sep, i) => (
          <g key={i}>
            <line x1="10" y1={sep.y} x2={maxX - 10} y2={sep.y}
                  stroke="#1f2937" strokeWidth="1" strokeDasharray="4,4" />
            <text x={maxX - 15} y={sep.y - 4} fill="#374151" fontSize="9"
                  textAnchor="end" fontStyle="italic">{sep.label}</text>
          </g>
        ))}

        {/* Edges */}
        {edges.map((edge, i) => {
          const from = nodeMap.get(edge.from);
          const to = nodeMap.get(edge.to);
          if (!from || !to) return null;

          const isHighlighted = hovered === edge.from || hovered === edge.to;
          const path = computeEdgePath(from, to);
          const labelPos = edgeLabelPos(from, to);

          return (
            <g key={i} opacity={hovered && !isHighlighted ? 0.15 : 1}
               style={{ transition: 'opacity 0.2s' }}>
              <path d={path} fill="none"
                    stroke={isHighlighted ? '#e5e7eb' : '#4b5563'}
                    strokeWidth={isHighlighted ? 1.5 : 1}
                    strokeDasharray={edge.dashed ? '4,3' : undefined}
                    markerEnd={`url(#arrowhead${isHighlighted ? '-hl' : ''})`} />
              {edge.label && (
                <text x={labelPos.x} y={labelPos.y}
                      fill={isHighlighted ? '#d1d5db' : '#6b7280'}
                      fontSize="8" textAnchor="middle"
                      style={{ pointerEvents: 'none' }}>
                  {edge.label}
                </text>
              )}
            </g>
          );
        })}

        {/* Nodes */}
        {nodes.map((node) => {
          const w = node.width || NODE_W;
          const c = COLORS[node.status];
          const isHighlighted = hovered === node.id;
          const isConnected = hovered ? edges.some(
            e => (e.from === hovered && e.to === node.id) ||
                 (e.to === hovered && e.from === node.id)
          ) : false;
          const dimmed = hovered && !isHighlighted && !isConnected;

          return (
            <g key={node.id}
               opacity={dimmed ? 0.2 : 1}
               style={{ transition: 'opacity 0.2s', cursor: 'pointer' }}
               onMouseEnter={() => setHovered(node.id)}
               onMouseLeave={() => setHovered(null)}>
              {node.section ? (
                <a href={`/appendix-experiments/${node.section}`}>
                  <rect x={node.x} y={node.y} width={w} height={NODE_H}
                        rx="6" fill={c.fill} stroke={c.stroke}
                        strokeWidth={isHighlighted ? 2 : 1}
                        filter={isHighlighted ? 'url(#glow)' : undefined} />
                  <text x={node.x + 8} y={node.y + 18} fill={c.text}
                        fontSize="11" fontWeight="600">
                    {node.title}
                  </text>
                  {node.subtitle && (
                    <text x={node.x + 8} y={node.y + 34} fill="#9ca3af"
                          fontSize="9">
                      {node.subtitle}
                    </text>
                  )}
                  {/* Status badge */}
                  <text x={node.x + w - 8} y={node.y + 44} fill={c.text}
                        fontSize="7" textAnchor="end" opacity={0.7}>
                    {STATUS_LABELS[node.status]}
                  </text>
                </a>
              ) : (
                <>
                  <rect x={node.x} y={node.y} width={w} height={NODE_H}
                        rx="6" fill={c.fill} stroke={c.stroke}
                        strokeWidth={isHighlighted ? 2 : 1} />
                  <text x={node.x + 8} y={node.y + 18} fill={c.text}
                        fontSize="11" fontWeight="600">
                    {node.title}
                  </text>
                  {node.subtitle && (
                    <text x={node.x + 8} y={node.y + 34} fill="#9ca3af"
                          fontSize="9">
                      {node.subtitle}
                    </text>
                  )}
                </>
              )}
            </g>
          );
        })}

        {/* Legend */}
        <g transform={`translate(${maxX - 190}, ${maxY - 70})`}>
          <rect x="-5" y="-5" width="185" height="65" rx="4"
                fill="#111111" stroke="#1f2937" strokeWidth="1" />
          {[
            { status: 'positive' as Status, label: 'Confirmed / Positive' },
            { status: 'mixed' as Status, label: 'Mixed' },
            { status: 'negative' as Status, label: 'Negative' },
            { status: 'planned' as Status, label: 'Planned / Running' },
          ].map((item, i) => (
            <g key={i} transform={`translate(5, ${i * 14 + 5})`}>
              <rect x="0" y="0" width="10" height="10" rx="2"
                    fill={COLORS[item.status].fill}
                    stroke={COLORS[item.status].stroke}
                    strokeWidth="1" />
              <text x="16" y="9" fill="#9ca3af" fontSize="9">{item.label}</text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}
