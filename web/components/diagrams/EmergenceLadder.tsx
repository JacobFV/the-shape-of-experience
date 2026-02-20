'use client';

import { useState } from 'react';
import { type Point, arrowPath } from './utils';

type Status = 'confirmed' | 'mixed' | 'blocked' | 'untested';

interface Rung {
  n: number;
  label: string;
  color: string;
  status: Status;
  experiment: string;
  desc: string;
}

const RUNGS: Rung[] = [
  {
    n: 1, label: 'Affect Dimensions', color: 'var(--d-green)',
    status: 'confirmed', experiment: 'V10, Exp 12',
    desc: 'Valence, arousal, integration, effective rank, CF, SM all measurable. Geometry is cheap — baseline of multi-agent survival.',
  },
  {
    n: 2, label: 'Arousal Modulation', color: 'var(--d-green)',
    status: 'confirmed', experiment: 'V11–V12',
    desc: 'State update rate varies with threat level. Yerkes-Dodson pattern: mild stress increases arousal, severe stress collapses it.',
  },
  {
    n: 3, label: 'Somatic Valence', color: 'var(--d-green)',
    status: 'confirmed', experiment: 'V13–V18',
    desc: 'Gradient on viability manifold — patterns move toward persistence, away from dissolution. No counterfactual needed.',
  },
  {
    n: 4, label: 'Participatory Default', color: 'var(--d-green)',
    status: 'confirmed', experiment: 'Exp 8',
    desc: 'ι ≈ 0.30 universally. Patterns model resources using agent-model templates. Computational animism is the evolutionary default.',
  },
  {
    n: 5, label: 'Referential Communication', color: 'var(--d-green)',
    status: 'confirmed', experiment: 'V35',
    desc: '10/10 seeds develop referential signaling under cooperative POMDP pressure. Language is cheap, like geometry.',
  },
  {
    n: 6, label: 'Temporal Integration', color: 'var(--d-green)',
    status: 'confirmed', experiment: 'V15, V22',
    desc: 'Memory channels selected by evolution; within-lifetime prediction learning works (100–15,000× MSE improvement).',
  },
  {
    n: 7, label: 'Affect Coherence', color: 'var(--d-green)',
    status: 'confirmed', experiment: 'Exp 7, V27',
    desc: 'Structure→behavior alignment develops over evolution (RSA 0.01→0.38). Requires gradient coupling + bottleneck forging.',
  },
  {
    n: 8, label: 'Counterfactual Sensitivity', color: 'var(--d-yellow)',
    status: 'mixed', experiment: 'V20, V33',
    desc: 'ρ_sync = 0.21 breaks the wall (V20). But contrastive prediction does not force CF representation (V33 negative). Requires embodied agency.',
  },
  {
    n: 9, label: 'Self-Model', color: 'var(--d-yellow)',
    status: 'mixed', experiment: 'V20',
    desc: 'SM_sal > 1.0 in 2/3 seeds — agents encode own states more accurately than environment. Emergent but fragile.',
  },
  {
    n: 10, label: 'Normativity', color: 'var(--d-muted)',
    status: 'untested', experiment: 'Exp 9 (null)',
    desc: 'No ΔV asymmetry between cooperative/competitive. Requires agency — the capacity to act otherwise. Not yet within reach.',
  },
];

const WALLS = [
  { afterRung: 7, label: 'Sensory-Motor Wall', sublabel: 'ρ_sync ≈ 0 → 0.21', color: 'var(--d-red)', experiment: 'Broken by V20' },
  { afterRung: 7, label: 'Decomposability Wall', sublabel: 'linear → MLP coupling', color: 'var(--d-orange)', experiment: 'Broken by V27' },
];

const statusColor = (s: Status) => {
  switch (s) {
    case 'confirmed': return 'var(--d-green)';
    case 'mixed': return 'var(--d-yellow)';
    case 'blocked': return 'var(--d-red)';
    case 'untested': return 'var(--d-muted)';
  }
};

const statusIcon = (s: Status) => {
  switch (s) {
    case 'confirmed': return '✓';
    case 'mixed': return '~';
    case 'blocked': return '✗';
    case 'untested': return '?';
  }
};

export default function EmergenceLadder() {
  const [hovered, setHovered] = useState<number | null>(null);

  const cx = 260;
  const startY = 40;
  const rungH = 38;
  const rungGap = 6;
  const step = rungH + rungGap;
  const boxW = 240;
  const wallGap = 62; // extra space for wall annotation between rung 7 and 8

  const rungY = (i: number) => {
    // Rungs 0-6 (ladder rungs 1-7) are packed; after 7, add wall gap
    if (i <= 6) return startY + i * step;
    return startY + 7 * step + wallGap + (i - 7) * step;
  };

  const wallY = startY + 7 * step + wallGap / 2;
  const totalH = rungY(9) + rungH + 80; // room for tooltip

  return (
    <svg viewBox={`0 0 560 ${totalH}`}
      className={`diagram-svg${hovered !== null ? ' has-focus' : ''}`}
      role="img"
      aria-label="Emergence ladder: 10 rungs from affect dimensions to normativity, with two architectural walls between rungs 7 and 8">

      {/* Title */}
      <text x={cx} y={16} textAnchor="middle" fontSize={14} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        The Emergence Ladder
      </text>

      {/* Connection lines between rungs */}
      {RUNGS.slice(0, -1).map((_, i) => {
        const fromY = rungY(i) + rungH;
        const toY = rungY(i + 1);
        const isDimmed = hovered !== null && hovered !== i && hovered !== i + 1;
        // Don't draw line through wall region
        if (i === 6) return null;
        return (
          <line key={`line-${i}`}
            x1={cx} y1={fromY + 2} x2={cx} y2={toY - 2}
            stroke="var(--d-line)" strokeWidth={0.75} strokeDasharray="3,3"
            style={{ opacity: isDimmed ? 0.15 : 0.5, transition: 'opacity 0.2s' }}
          />
        );
      })}

      {/* Wall annotation */}
      <g style={{ opacity: hovered !== null && hovered < 7 ? 0.2 : 1, transition: 'opacity 0.2s' }}>
        {/* Jagged break line */}
        <path
          d={`M ${cx - boxW / 2 - 10} ${wallY - 8} l 8 -6 l -8 -6 l 8 -6 M ${cx + boxW / 2 + 10} ${wallY - 8} l -8 -6 l 8 -6 l -8 -6`}
          stroke="var(--d-red)" strokeWidth={1.5} fill="none" opacity={0.6}
        />
        <path
          d={`M ${cx - boxW / 2 - 10} ${wallY + 8} l 8 6 l -8 6 l 8 6 M ${cx + boxW / 2 + 10} ${wallY + 8} l -8 6 l 8 6 l -8 6`}
          stroke="var(--d-orange)" strokeWidth={1.5} fill="none" opacity={0.6}
        />
        {/* Wall labels */}
        <text x={cx} y={wallY - 12} textAnchor="middle" fontSize={10}
          fill="var(--d-red)" fontWeight={600}
          fontFamily="var(--font-body, Georgia, serif)">
          ρ wall — broken by V20 (agency)
        </text>
        <text x={cx} y={wallY + 4} textAnchor="middle" fontSize={10}
          fill="var(--d-orange)" fontWeight={600}
          fontFamily="var(--font-body, Georgia, serif)">
          decomposability wall — broken by V27 (MLP gradient coupling)
        </text>
      </g>

      {/* Rung boxes */}
      {RUNGS.map((rung, i) => {
        const y = rungY(i);
        const isFocused = hovered === i;
        const isDimmed = hovered !== null && !isFocused;
        const sc = statusColor(rung.status);

        return (
          <g key={rung.n}
            className={`interactive${isFocused ? ' focused' : ''}`}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            style={{ opacity: isDimmed ? 0.25 : 1, transition: 'opacity 0.2s' }}
          >
            <rect
              x={cx - boxW / 2} y={y}
              width={boxW} height={rungH} rx={5}
              fill={sc} fillOpacity={isFocused ? 0.18 : 0.07}
              stroke={sc} strokeWidth={isFocused ? 1.5 : 0.75}
              style={{ transition: 'fill-opacity 0.2s, stroke-width 0.2s' }}
            />
            {/* Rung number */}
            <text
              x={cx - boxW / 2 + 14} y={y + rungH / 2}
              textAnchor="middle" dominantBaseline="central"
              fontSize={11} fill={sc} fontWeight={700}
              fontFamily="var(--font-body, Georgia, serif)"
            >
              {rung.n}
            </text>
            {/* Label */}
            <text
              x={cx} y={y + rungH / 2}
              textAnchor="middle" dominantBaseline="central"
              fontSize={12.5} fill="var(--d-fg)"
              fontFamily="var(--font-body, Georgia, serif)"
            >
              {rung.label}
            </text>
            {/* Status icon */}
            <text
              x={cx + boxW / 2 - 14} y={y + rungH / 2}
              textAnchor="middle" dominantBaseline="central"
              fontSize={12} fill={sc} fontWeight={700}
              fontFamily="var(--font-body, Georgia, serif)"
            >
              {statusIcon(rung.status)}
            </text>
            {/* Experiment label on right */}
            <text
              x={cx + boxW / 2 + 12} y={y + rungH / 2}
              textAnchor="start" dominantBaseline="central"
              fontSize={9.5} fill="var(--d-muted)" fontStyle="italic"
              fontFamily="var(--font-body, Georgia, serif)"
            >
              {rung.experiment}
            </text>
          </g>
        );
      })}

      {/* Left annotation: "cheap" and "expensive" */}
      <text x={cx - boxW / 2 - 12} y={rungY(3)}
        textAnchor="end" dominantBaseline="central"
        fontSize={11} fill="var(--d-green)" fontStyle="italic"
        fontFamily="var(--font-body, Georgia, serif)"
        style={{ opacity: hovered !== null && hovered > 6 ? 0.2 : 0.8, transition: 'opacity 0.2s' }}>
        cheap
      </text>
      <text x={cx - boxW / 2 - 12} y={rungY(8)}
        textAnchor="end" dominantBaseline="central"
        fontSize={11} fill="var(--d-yellow)" fontStyle="italic"
        fontFamily="var(--font-body, Georgia, serif)"
        style={{ opacity: hovered !== null && hovered < 7 ? 0.2 : 0.8, transition: 'opacity 0.2s' }}>
        expensive
      </text>

      {/* Hover tooltip */}
      {hovered !== null && (
        <g>
          <rect
            x={20} y={totalH - 60}
            width={520} height={46} rx={4}
            fill="var(--d-fg)" fillOpacity={0.06}
            stroke="var(--d-line)" strokeWidth={0.5} strokeOpacity={0.3}
          />
          <text
            x={cx} y={totalH - 37}
            textAnchor="middle" dominantBaseline="central"
            fontSize={11} fill="var(--d-fg)"
            fontFamily="var(--font-body, Georgia, serif)"
          >
            {RUNGS[hovered].desc}
          </text>
        </g>
      )}
    </svg>
  );
}
