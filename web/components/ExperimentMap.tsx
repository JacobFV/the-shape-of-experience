'use client';

import React, { useState, useEffect, useCallback } from 'react';

// ─── Experiment data ──────────────────────────────────────────────
type Status = 'positive' | 'mixed' | 'negative' | 'null' | 'planned' | 'running' | 'foundation';

interface ExpNode {
  id: string;
  title: string;
  status: Status;
  x: number;
  y: number;
  width?: number;
  section?: string;
  subtitle?: string;
  description?: string;
}

interface ExpEdge {
  from: string;
  to: string;
  label?: string;
  dashed?: boolean;
}

// CSS-var-based colors — light and dark modes handled by diagram-svg vars
const STATUS_CSS: Record<Status, { var: string; label: string }> = {
  positive:   { var: '--d-green',  label: 'Confirmed' },
  mixed:      { var: '--d-yellow', label: 'Mixed' },
  negative:   { var: '--d-red',    label: 'Negative' },
  null:       { var: '--d-muted',  label: 'Null' },
  planned:    { var: '--d-blue',   label: 'Planned' },
  running:    { var: '--d-blue',   label: 'Running' },
  foundation: { var: '--d-violet', label: 'Foundation' },
};

// ─── Layout constants ─────────────────────────────────────────────
const NODE_W = 170;
const NODE_H = 52;
const COL = 195;
const ROW = 72;

// ─── Define all experiment nodes ──────────────────────────────────
const nodes: ExpNode[] = [
  // Phase 1: LLM (row 0)
  { id: 'V2-V9', title: 'V2-V9: LLM Affect', status: 'mixed', x: 50, y: 30, width: 180,
    section: 'v2-v9-llm-affect-signatures', subtitle: 'Opposite dynamics to bio',
    description: 'LLM agents show coherent affect geometry but opposite dynamics to biological systems — integration, self-model salience, and arousal all decrease under threat. Root cause: no survival-shaped learning history.' },

  // Phase 2: MARL (row 0, right)
  { id: 'V10', title: 'V10: MARL Ablation', status: 'positive', x: 450, y: 30,
    section: 'v10-marl-forcing-function-ablation', subtitle: 'Geometry is baseline',
    description: 'All 7 conditions show significant geometric alignment (RSA ρ > 0.21, p < 0.0001). Removing forcing functions slightly increases alignment. Affect geometry is a baseline property of multi-agent survival.' },

  // Phase 3: Lenia substrate (rows 1-4)
  { id: 'V11', title: 'V11: Lenia Evolution', status: 'mixed', x: 50, y: 30 + ROW,
    section: 'v11-lenia-ca-evolution', subtitle: 'Curriculum > complexity',
    description: 'Training regime matters more than substrate complexity. Curriculum training is the only intervention that improves novel-stress generalization.' },
  { id: 'V12', title: 'V12: Attention Lenia', status: 'mixed', x: 50 + COL, y: 30 + ROW,
    section: 'v12-attention-based-lenia', subtitle: 'Necessary not sufficient',
    description: 'Evolvable attention: 42% of cycles show Φ increase under stress (vs 3% convolution). +2.0pp shift — largest single-intervention effect. But system reaches integration threshold without crossing it.' },
  { id: 'V13', title: 'V13: Content Coupling', status: 'foundation', x: 50 + COL * 2, y: 30 + ROW, width: 180,
    section: 'v13-content-based-coupling', subtitle: 'Foundation substrate',
    description: 'Content-based coupling Lenia. 3 seeds, mean robustness 0.923. Foundation substrate for the emergence measurement program (Exp 0-12).' },

  // V13 measurement experiments (row 2)
  { id: 'Exp0-12', title: 'Exp 0-12: Measurements', status: 'positive', x: 50 + COL * 2, y: 30 + ROW * 2, width: 190,
    section: 'the-emergence-experiment-program', subtitle: '12 experiments on V13',
    description: 'Twelve measurement experiments on the V13 substrate. Geometry develops over evolution (RSA 0.01→0.38). Computational animism universal (ι ≈ 0.30). VLM convergence: RSA ρ = 0.54–0.78.' },

  // Substrate evolution (rows 2-3)
  { id: 'V14', title: 'V14: Chemotaxis', status: 'mixed', x: 50, y: 30 + ROW * 2,
    section: 'v14-chemotactic-lenia', subtitle: 'Directed motion',
    description: 'V13 + directed motion via chemical gradient following.' },
  { id: 'V15', title: 'V15: Temporal Memory', status: 'positive', x: 50, y: 30 + ROW * 3,
    section: 'v15-temporal-memory', subtitle: 'Memory selectable',
    description: 'Memory channels selected by evolution. Phi-stress response doubles during bottleneck events.' },
  { id: 'V16', title: 'V16: Hebbian Plasticity', status: 'negative', x: 50, y: 30 + ROW * 4,
    section: 'v16-hebbian-plasticity', subtitle: 'Hurts robustness',
    description: 'NEGATIVE. Hebbian plasticity hurts robustness (0.892, lowest of all substrates). Within-lifetime learning interferes with evolved topology.' },
  { id: 'V17', title: 'V17: Quorum Signaling', status: 'mixed', x: 50 + COL, y: 30 + ROW * 4,
    section: 'v17-quorum-signaling', subtitle: 'Suppressed 2/3 seeds',
    description: 'Highest peak robustness (1.125) but signaling suppressed in 2/3 seeds. Evolution discovers that silence is cheaper than coordination.' },
  { id: 'V18', title: 'V18: Boundary Lenia', status: 'positive', x: 50 + COL * 2, y: 30 + ROW * 4,
    section: 'v18-boundary-dependent-lenia', subtitle: 'Best robustness: 0.969',
    description: 'BEST Lenia substrate. Mean robustness 0.969, max 1.651. Internal gain evolved DOWN — boundary sensing creates robustness by distinguishing self from environment.' },

  // V19 Mechanism (row 5)
  { id: 'V19', title: 'V19: Bottleneck Furnace', status: 'positive', x: 50 + COL * 2, y: 30 + ROW * 5, width: 190,
    section: 'v19-bottleneck-furnace', subtitle: 'Creation confirmed 2/3',
    description: 'The Bottleneck Furnace is generative, not selective. Patterns that survived near-extinction show higher novel-stress robustness than controls. The furnace forges integration — it does not merely reveal it.' },

  // Phase 6: Protocell Agency (rows 6-11)
  { id: 'V20', title: 'V20: Protocell Agency', status: 'positive', x: 50, y: 30 + ROW * 6, width: 190,
    section: 'v20-protocell-agency', subtitle: 'Wall broken: ρ=0.21',
    description: 'SENSORY-MOTOR WALL BROKEN. ρ_sync = 0.21 from initialization. World model capacity C_wm = 0.10–0.15. Self-model salience > 1.0 in 2/3 seeds. The wall is architectural, not evolutionary.' },
  { id: 'V21', title: 'V21: CTM Inner Ticks', status: 'mixed', x: 50 + COL, y: 30 + ROW * 7,
    section: 'v21-ctm-inner-ticks', subtitle: 'Ticks survive, no adapt.',
    description: 'Inner ticks (K=8) don\'t collapse — but no adaptive deliberation emerges. Evolution too slow to discover tick-dependent strategies.' },
  { id: 'V22', title: 'V22: Predictive Gradient', status: 'mixed', x: 50, y: 30 + ROW * 8, width: 190,
    section: 'v22-intrinsic-predictive-gradient', subtitle: 'Prediction ≠ integration',
    description: 'Gradient works (100–15,000× MSE reduction). Learning rate not suppressed by evolution. But robustness not improved. Prediction ≠ integration — the linear readout is the bottleneck.' },

  // Prediction variants (rows 9-10)
  { id: 'V23', title: 'V23: Multi-Target', status: 'negative', x: 50, y: 30 + ROW * 9,
    section: 'v23-world-model-gradient', subtitle: 'Specialization ≠ integ.',
    description: 'NEGATIVE. Weight columns specialize (cos ≈ 0, rank ≈ 2.9). Phi DECREASES (0.079 vs 0.097). Multi-target prediction creates specialization, not integration.' },
  { id: 'V24', title: 'V24: TD Value', status: 'mixed', x: 50 + COL, y: 30 + ROW * 9,
    section: 'v24-td-value-learning', subtitle: 'Best survival, Φ mixed',
    description: 'Mean robustness 1.012 (best prediction experiment). Phi mixed. TD value learning improves survival but the linear readout bottleneck remains.' },
  { id: 'V25', title: 'V25: Predator-Prey', status: 'negative', x: 50 + COL * 2, y: 30 + ROW * 9,
    section: 'v25-predator-prey', subtitle: 'Rich obs → reactive',
    description: 'NEGATIVE. 5×5 observation window too rich — reactive strategies suffice. Rich observations eliminate the need for internal models.' },
  { id: 'V26', title: 'V26: POMDP', status: 'mixed', x: 50 + COL * 2, y: 30 + ROW * 10,
    section: 'v26-pomdp', subtitle: 'Type encoding, 100% mort',
    description: '1×1 obs + noisy compass. Effective rank 3.6–5.7, type accuracy 0.95–0.97. But 100% drought mortality — agents too fragile to survive bottlenecks.' },

  // MLP breakthrough (rows 9-11)
  { id: 'V27', title: 'V27: MLP Head', status: 'positive', x: 50 + COL * 3, y: 30 + ROW * 9,
    section: 'v27-nonlinear-mlp-head', subtitle: 'Φ=0.245 record!',
    description: 'DECOMPOSABILITY WALL BROKEN. Seed 7: Φ = 0.245 (highest ever), eff_rank = 11.34. 2-layer MLP creates gradient coupling across all hidden units. The key architectural ingredient for integration.' },
  { id: 'V28', title: 'V28: Width Sweep', status: 'mixed', x: 50 + COL * 3, y: 30 + ROW * 10,
    section: 'v28-bottleneck-width-sweep', subtitle: 'Gradient coupling mech.',
    description: 'Mechanism is 2-layer gradient coupling (W₂ᵀW₁ᵀ in the gradient), not bottleneck width or nonlinearity. Any 2-layer head couples all hidden units during SGD.' },
  { id: 'V29', title: 'V29: Social Prediction', status: 'mixed', x: 50 + COL * 2, y: 30 + ROW * 11,
    section: 'v29-social-prediction', subtitle: '3-seed preliminary',
    description: 'Social prediction (neighbor energy target) vs self prediction. 3-seed preliminary suggested difference; V31 at 10-seed scale showed no difference (p = 0.93).' },
  { id: 'V30', title: 'V30: Dual Prediction', status: 'negative', x: 50, y: 30 + ROW * 11,
    section: 'v30-dual-prediction', subtitle: 'Gradient imbalance',
    description: 'NEGATIVE. Self MSE 100–150× smaller than social MSE — colonizes representation. Multi-objective gradient interference: any second gradient signal disrupts the primary integration-producing gradient.' },
  { id: 'V31', title: 'V31: 10-Seed Validation', status: 'positive', x: 50 + COL * 3, y: 30 + ROW * 11, width: 190,
    section: 'v31-10-seed-validation', subtitle: '30/30/40 split, r=0.997',
    description: 'Integration is trajectory-dependent. 30% HIGH / 30% MOD / 40% LOW. Post-drought bounce predicts final Φ (r = 0.997, p < 0.0001). Target doesn\'t matter (self vs social, p = 0.93). Biography, not architecture.' },

  // Next gen (row 12)
  { id: 'V32', title: 'V32: Drought Autopsy', status: 'positive', x: 50, y: 30 + ROW * 12.5,
    section: 'v32-drought-autopsy', subtitle: '50 seeds, trajectory',
    description: '50 seeds × 30 cycles × 5 droughts. 22% HIGH / 46% MOD / 32% LOW. Mean bounce across 5 droughts predicts category (ρ = 0.60, p < 10⁻⁵). First bounce alone does not. Integration is built by sustained recovery, not a single crisis.' },
  { id: 'V33', title: 'V33: Contrastive Pred.', status: 'negative', x: 50 + COL, y: 30 + ROW * 12.5,
    section: 'v33-contrastive-self-prediction', subtitle: 'Hurts integration',
    description: 'NEGATIVE. Contrastive loss destabilizes gradient learning. Mean Φ = 0.054 (significantly below baseline). 0% HIGH, 70% LOW. Prediction MSE increases over evolution — the contrastive signal decouples the gradient from the viability signal.' },
  { id: 'V34', title: 'V34: Φ-Fitness', status: 'negative', x: 50 + COL * 2, y: 30 + ROW * 12.5,
    section: 'v34-phi-inclusive-fitness', subtitle: 'Mixed negative, Goodhart',
    description: 'MIXED NEGATIVE. Direct Φ selection does not increase HIGH fraction (20%, within noise). 2/10 seeds show Goodharting. Integration cannot be selected for directly — it must emerge as a byproduct of architecture and forging.' },
  { id: 'V35', title: 'V35: Language', status: 'positive', x: 50 + COL * 3, y: 30 + ROW * 12.5,
    section: 'v35-language-emergence', subtitle: 'Cheap but orthogonal',
    description: 'Referential communication emerges 10/10 seeds (100%). But does NOT lift Φ — Phi-MI ρ = 0.07 (null). Language is cheap, like geometry. It sits at rung 4–5, not rung 8. Communication and integration are orthogonal.' },
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
  { from: 'V18', to: 'V20', label: 'wall → new substrate', dashed: true },
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
  { from: 'V31', to: 'V32', label: '50-seed autopsy' },
  { from: 'V27', to: 'V33', label: 'contrastive' },
  { from: 'V27', to: 'V34', label: 'Φ selection' },
  { from: 'V27', to: 'V35', label: 'language' },
];

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
  const [selected, setSelected] = useState<string | null>(null);

  const nodeMap = new Map(nodes.map(n => [n.id, n]));

  const closeModal = useCallback(() => {
    setSelected(null);
    history.replaceState(null, '', '/appendix-experiments');
  }, []);

  // Close modal on Escape or browser back
  useEffect(() => {
    if (!selected) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeModal(); };
    const onPop = () => setSelected(null);
    window.addEventListener('keydown', onKey);
    window.addEventListener('popstate', onPop);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('popstate', onPop);
    };
  }, [selected, closeModal]);

  const handleNodeClick = (node: ExpNode) => {
    if (!node.section) return;
    setSelected(node.id);
    history.pushState({ experiment: node.id }, '', `/appendix-experiments/${node.section}`);
  };

  const maxX = Math.max(...nodes.map(n => n.x + (n.width || NODE_W))) + 40;
  const maxY = Math.max(...nodes.map(n => n.y + NODE_H)) + 40;

  const selectedNode = selected ? nodeMap.get(selected) : null;

  return (
    <div style={{ width: '100%', overflowX: 'auto', position: 'relative' }}>
      <svg
        viewBox={`0 0 ${maxX} ${maxY}`}
        width="100%"
        className="diagram-svg experiment-map"
        style={{ maxWidth: maxX, minWidth: 600 }}
      >
        <defs>
          <marker id="em-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--d-muted)" />
          </marker>
          <marker id="em-arrow-hl" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--d-fg)" />
          </marker>
          <filter id="em-glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        {/* Phase separator lines */}
        {[
          { y: 30 + ROW * 0.5, label: 'Independent baselines' },
          { y: 30 + ROW * 5.5, label: 'Lenia CA substrate' },
          { y: 30 + ROW * 8 - 15, label: 'Protocell agency' },
          { y: 30 + ROW * 12, label: 'Completed program' },
        ].map((sep, i) => (
          <g key={i}>
            <line x1="10" y1={sep.y} x2={maxX - 10} y2={sep.y}
                  stroke="var(--d-line)" strokeWidth="1" strokeDasharray="4,4" opacity={0.25} />
            <text x={maxX - 15} y={sep.y - 4} fill="var(--d-muted)" fontSize="9"
                  textAnchor="end" fontStyle="italic" opacity={0.6}>{sep.label}</text>
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
                    stroke={isHighlighted ? 'var(--d-fg)' : 'var(--d-line)'}
                    strokeWidth={isHighlighted ? 1.5 : 1}
                    strokeDasharray={edge.dashed ? '4,3' : undefined}
                    opacity={isHighlighted ? 0.8 : 0.35}
                    markerEnd={`url(#em-arrow${isHighlighted ? '-hl' : ''})`} />
              {edge.label && (
                <text x={labelPos.x} y={labelPos.y}
                      fill={isHighlighted ? 'var(--d-fg)' : 'var(--d-muted)'}
                      fontSize="8" textAnchor="middle"
                      opacity={isHighlighted ? 1 : 0.7}
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
          const cssVar = STATUS_CSS[node.status].var;
          const isHighlighted = hovered === node.id;
          const isConnected = hovered ? edges.some(
            e => (e.from === hovered && e.to === node.id) ||
                 (e.to === hovered && e.from === node.id)
          ) : false;
          const dimmed = hovered && !isHighlighted && !isConnected;

          return (
            <g key={node.id}
               opacity={dimmed ? 0.2 : 1}
               style={{ transition: 'opacity 0.2s', cursor: node.section ? 'pointer' : 'default' }}
               onMouseEnter={() => setHovered(node.id)}
               onMouseLeave={() => setHovered(null)}
               onClick={() => handleNodeClick(node)}>
              <rect x={node.x} y={node.y} width={w} height={NODE_H}
                    rx="6"
                    fill={`var(${cssVar})`} fillOpacity={isHighlighted ? 0.18 : 0.08}
                    stroke={`var(${cssVar})`}
                    strokeWidth={isHighlighted ? 2 : 1}
                    filter={isHighlighted ? 'url(#em-glow)' : undefined} />
              <text x={node.x + 8} y={node.y + 18}
                    fill={`var(${cssVar})`}
                    fontSize="11" fontWeight="600">
                {node.title}
              </text>
              {node.subtitle && (
                <text x={node.x + 8} y={node.y + 34}
                      fill="var(--d-muted)" fontSize="9">
                  {node.subtitle}
                </text>
              )}
              {/* Status badge */}
              <text x={node.x + w - 8} y={node.y + 44}
                    fill={`var(${cssVar})`}
                    fontSize="7" textAnchor="end" opacity={0.7}>
                {STATUS_CSS[node.status].label}
              </text>
            </g>
          );
        })}

        {/* Legend */}
        <g transform={`translate(${maxX - 190}, ${maxY - 70})`}>
          <rect x="-5" y="-5" width="185" height="65" rx="4"
                fill="var(--d-fg)" fillOpacity={0.04}
                stroke="var(--d-line)" strokeWidth="0.5" strokeOpacity={0.3} />
          {([
            { status: 'positive' as Status, label: 'Confirmed / Positive' },
            { status: 'mixed' as Status, label: 'Mixed' },
            { status: 'negative' as Status, label: 'Negative' },
            { status: 'foundation' as Status, label: 'Foundation' },
          ]).map((item, i) => (
            <g key={i} transform={`translate(5, ${i * 14 + 5})`}>
              <rect x="0" y="0" width="10" height="10" rx="2"
                    fill={`var(${STATUS_CSS[item.status].var})`} fillOpacity={0.12}
                    stroke={`var(${STATUS_CSS[item.status].var})`}
                    strokeWidth="1" />
              <text x="16" y="9" fill="var(--d-muted)" fontSize="9">{item.label}</text>
            </g>
          ))}
        </g>
      </svg>

      {/* ─── Modal overlay ─────────────────────────────────────────── */}
      {selectedNode && (
        <div
          className="experiment-map-modal-backdrop"
          onClick={(e) => { if (e.target === e.currentTarget) closeModal(); }}
        >
          <div className="experiment-map-modal">
            <button className="experiment-map-modal-close" onClick={closeModal} aria-label="Close">
              ×
            </button>
            <div className="experiment-map-modal-header">
              <span
                className="experiment-map-modal-status"
                data-status={selectedNode.status}
              >
                {STATUS_CSS[selectedNode.status].label}
              </span>
              <h3>{selectedNode.title}</h3>
              {selectedNode.subtitle && (
                <p className="experiment-map-modal-subtitle">{selectedNode.subtitle}</p>
              )}
            </div>
            {selectedNode.description && (
              <p className="experiment-map-modal-desc">{selectedNode.description}</p>
            )}
            {selectedNode.section && (
              <a
                href={`/appendix-experiments/${selectedNode.section}`}
                className="experiment-map-modal-link"
              >
                Read full section →
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
