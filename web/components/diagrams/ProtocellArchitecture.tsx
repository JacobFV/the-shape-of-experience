'use client';

import { useState } from 'react';

/**
 * Block diagram of the V20+ protocell agent architecture.
 *
 * Shows: observation window → GRU core → hidden state → prediction head / action head
 * with the action–observation feedback loop.
 */

interface Block {
  id: string;
  label: string;
  sublabel?: string;
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
  desc: string;
}

const BLOCKS: Block[] = [
  { id: 'env', label: 'Grid World', sublabel: '32×32, N agents, R resources',
    x: 40, y: 30, w: 140, h: 50, color: 'var(--d-muted)',
    desc: 'Discrete toroidal grid. Agents move, eat, and (optionally) communicate. Resources regenerate stochastically.' },
  { id: 'obs', label: 'Observation', sublabel: 'local window + compass',
    x: 240, y: 30, w: 140, h: 50, color: 'var(--d-blue)',
    desc: 'Each agent sees a local window (5×5 or 1×1) encoding cell type. V26+: noisy compass (σ=0.5) gives approximate resource direction.' },
  { id: 'gru', label: 'GRU Core', sublabel: 'H=16 or H=32',
    x: 240, y: 130, w: 140, h: 60, color: 'var(--d-green)',
    desc: 'Gated Recurrent Unit maintains hidden state across timesteps. ~3,400 parameters per agent. Weights are heritable (evolved).' },
  { id: 'hidden', label: 'Hidden State h', sublabel: 'eff. rank 5–11',
    x: 240, y: 240, w: 140, h: 45, color: 'var(--d-cyan)',
    desc: 'The hidden state carries the agent\'s "internal model." Effective rank 5–11 depending on architecture. This is where integration lives.' },
  { id: 'pred', label: 'Prediction Head', sublabel: 'linear or 2-layer MLP',
    x: 80, y: 330, w: 150, h: 50, color: 'var(--d-orange)',
    desc: 'Predicts own next energy (self-prediction). Linear head = decomposable (V22). 2-layer MLP = gradient coupling across all hidden units (V27). THE architectural switch.' },
  { id: 'act', label: 'Action Head', sublabel: '4 dirs + eat + (comm)',
    x: 350, y: 330, w: 150, h: 50, color: 'var(--d-red)',
    desc: 'Selects action: move in 4 directions, eat resource, or (V35) emit one of K=8 symbols. Actions close the sensory-motor loop that breaks the ρ wall.' },
  { id: 'loss', label: 'SGD Step', sublabel: 'within-lifetime learning',
    x: 80, y: 420, w: 150, h: 42, color: 'var(--d-yellow)',
    desc: 'MSE between predicted and actual next energy drives within-lifetime gradient descent on the prediction head. Learning rate is heritable.' },
];

const ARROWS: { from: string; to: string; label?: string; fromSide?: string; toSide?: string }[] = [
  { from: 'env', to: 'obs', label: 'extract' },
  { from: 'obs', to: 'gru', label: 'input' },
  { from: 'gru', to: 'hidden' },
  { from: 'hidden', to: 'pred', label: 'W₁·tanh(W₂·h)' },
  { from: 'hidden', to: 'act', label: 'softmax(W_a·h)' },
  { from: 'act', to: 'env', label: 'feedback loop', fromSide: 'top', toSide: 'bottom' },
  { from: 'pred', to: 'loss', label: 'MSE(ŷ, y)' },
];

export default function ProtocellArchitecture() {
  const [hovered, setHovered] = useState<string | null>(null);

  const blockMap = new Map(BLOCKS.map(b => [b.id, b]));

  // Arrow path computation
  const arrowPath = (from: Block, to: Block, fromSide?: string, toSide?: string): string => {
    let x1: number, y1: number, x2: number, y2: number;

    if (fromSide === 'top') {
      x1 = from.x + from.w / 2; y1 = from.y;
    } else {
      // Default: bottom of from
      x1 = from.x + from.w / 2; y1 = from.y + from.h;
    }

    if (toSide === 'bottom') {
      x2 = to.x + to.w / 2; y2 = to.y + to.h;
    } else if (toSide === 'left') {
      x2 = to.x; y2 = to.y + to.h / 2;
    } else {
      // Default: top of to
      x2 = to.x + to.w / 2; y2 = to.y;
    }

    // For the feedback loop (act → env), route around the right side
    if (from.id === 'act' && to.id === 'env') {
      const rx = 520;
      return `M ${from.x + from.w / 2} ${from.y} L ${from.x + from.w / 2} ${from.y - 15} L ${rx} ${from.y - 15} L ${rx} ${to.y + to.h / 2} L ${to.x + to.w} ${to.y + to.h / 2}`;
    }

    // Straight or curved
    const dx = Math.abs(x2 - x1);
    const dy = Math.abs(y2 - y1);
    if (dx < 10) {
      return `M ${x1} ${y1} L ${x2} ${y2}`;
    }
    const cp = dy * 0.4;
    return `M ${x1} ${y1} C ${x1} ${y1 + cp}, ${x2} ${y2 - cp}, ${x2} ${y2}`;
  };

  const hoveredBlock = hovered ? blockMap.get(hovered) : null;

  return (
    <svg viewBox="0 0 560 510" className="diagram-svg" role="img"
      aria-label="Protocell agent architecture: observation window feeds GRU core, which outputs to prediction and action heads, with action feeding back to the environment">

      {/* Title */}
      <text x={280} y={18} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Protocell Agent Architecture (V20–V35)
      </text>

      {/* Arrows (behind blocks) */}
      <defs>
        <marker id="pa-arrow" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
          <polygon points="0 0, 7 2.5, 0 5" fill="var(--d-line)" />
        </marker>
      </defs>
      {ARROWS.map((arrow, i) => {
        const from = blockMap.get(arrow.from)!;
        const to = blockMap.get(arrow.to)!;
        const path = arrowPath(from, to, arrow.fromSide, arrow.toSide);
        const isHL = hovered === arrow.from || hovered === arrow.to;
        const isDimmed = hovered && !isHL;

        // Label position (midpoint)
        const fc = { x: from.x + from.w / 2, y: from.y + from.h };
        const tc = { x: to.x + to.w / 2, y: to.y };

        return (
          <g key={i} style={{ opacity: isDimmed ? 0.15 : 1, transition: 'opacity 0.2s' }}>
            <path d={path} fill="none"
              stroke={isHL ? 'var(--d-fg)' : 'var(--d-line)'}
              strokeWidth={isHL ? 1.5 : 1}
              opacity={isHL ? 0.7 : 0.4}
              markerEnd="url(#pa-arrow)" />
            {arrow.label && arrow.from !== 'act' && (
              <text x={(fc.x + tc.x) / 2 + (fc.x < tc.x ? 8 : -8)} y={(fc.y + tc.y) / 2}
                textAnchor="middle" fontSize={8} fill="var(--d-muted)"
                fontFamily="var(--font-body, Georgia, serif)"
                style={{ pointerEvents: 'none' }}>
                {arrow.label}
              </text>
            )}
            {/* Feedback loop label */}
            {arrow.from === 'act' && (
              <text x={530} y={(from.y + (blockMap.get('env')!.y + blockMap.get('env')!.h / 2)) / 2}
                textAnchor="middle" fontSize={8} fill="var(--d-muted)"
                fontFamily="var(--font-body, Georgia, serif)"
                transform={`rotate(90, 530, ${(from.y + (blockMap.get('env')!.y + blockMap.get('env')!.h / 2)) / 2})`}
                style={{ pointerEvents: 'none' }}>
                action→observation loop (ρ wall)
              </text>
            )}
          </g>
        );
      })}

      {/* Blocks */}
      {BLOCKS.map(block => {
        const isHL = hovered === block.id;
        const isDimmed = hovered && !isHL;
        const connectedIds = ARROWS
          .filter(a => a.from === block.id || a.to === block.id)
          .flatMap(a => [a.from, a.to]);
        const isConnected = hovered ? connectedIds.includes(hovered) : false;
        const dim = hovered && !isHL && !isConnected;

        return (
          <g key={block.id}
            onMouseEnter={() => setHovered(block.id)}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: 'pointer', opacity: dim ? 0.2 : 1, transition: 'opacity 0.2s' }}>
            <rect x={block.x} y={block.y} width={block.w} height={block.h} rx={6}
              fill={block.color} fillOpacity={isHL ? 0.18 : 0.08}
              stroke={block.color} strokeWidth={isHL ? 2 : 1}
              style={{ transition: 'fill-opacity 0.2s, stroke-width 0.2s' }} />
            <text x={block.x + block.w / 2} y={block.y + (block.sublabel ? block.h / 2 - 6 : block.h / 2)}
              textAnchor="middle" dominantBaseline="central"
              fontSize={11} fontWeight={600} fill={block.color}
              fontFamily="var(--font-body, Georgia, serif)">
              {block.label}
            </text>
            {block.sublabel && (
              <text x={block.x + block.w / 2} y={block.y + block.h / 2 + 9}
                textAnchor="middle" dominantBaseline="central"
                fontSize={8.5} fill="var(--d-muted)"
                fontFamily="var(--font-body, Georgia, serif)">
                {block.sublabel}
              </text>
            )}
          </g>
        );
      })}

      {/* Tooltip */}
      {hoveredBlock && (
        <g>
          <rect x={20} y={468} width={520} height={32} rx={4}
            fill="var(--d-fg)" fillOpacity={0.05}
            stroke="var(--d-line)" strokeWidth={0.5} strokeOpacity={0.3} />
          <text x={280} y={487} textAnchor="middle" dominantBaseline="central"
            fontSize={10} fill="var(--d-fg)"
            fontFamily="var(--font-body, Georgia, serif)">
            {hoveredBlock.desc}
          </text>
        </g>
      )}
    </svg>
  );
}
