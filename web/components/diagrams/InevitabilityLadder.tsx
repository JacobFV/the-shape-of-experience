'use client';

import { useState } from 'react';

/**
 * InevitabilityLadder — Introduction diagram
 * Simplified 3-rung version of the emergence ladder:
 * Thermodynamic → Computational → Structural
 * "What this book argues" framing
 */

const RUNGS = [
  {
    id: 'thermo',
    label: 'Thermodynamic',
    subtitle: 'Far-from-equilibrium systems must maintain boundaries',
    evidence: 'established physics',
    color: 'var(--d-green)',
    confidence: 1.0,
  },
  {
    id: 'compute',
    label: 'Computational',
    subtitle: 'Boundary maintenance requires state tracking under uncertainty',
    evidence: 'information theory',
    color: 'var(--d-blue)',
    confidence: 0.85,
  },
  {
    id: 'structural',
    label: 'Structural',
    subtitle: 'State tracking under uncertainty produces affect geometry',
    evidence: 'this book\'s argument',
    color: 'var(--d-violet)',
    confidence: 0.6,
  },
];

const W = 600;
const H = 420;
const RUNG_W = 380;
const RUNG_H = 70;
const GAP = 24;
const START_Y = 60;
const CX = W / 2;

export default function InevitabilityLadder() {
  const [hovered, setHovered] = useState<string | null>(null);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={`diagram-svg${hovered ? ' has-focus' : ''}`}
      role="img"
      aria-label="The inevitability ladder: thermodynamic necessity leads to computational necessity leads to structural necessity"
    >
      {/* Title */}
      <text x={CX} y={28} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}
        fontFamily="var(--font-body, Georgia, serif)">
        The Inevitability Ladder
      </text>
      <text x={CX} y={46} textAnchor="middle" fill="var(--d-muted)" fontSize={10.5}
        fontFamily="var(--font-body, Georgia, serif)">
        each rung follows from the one below it
      </text>

      {/* Rungs bottom-to-top */}
      {RUNGS.slice().reverse().map((rung, revIdx) => {
        const idx = RUNGS.length - 1 - revIdx;
        const y = START_Y + revIdx * (RUNG_H + GAP);
        const isFocused = hovered === rung.id;
        const isDimmed = hovered !== null && !isFocused;
        const rx = CX - RUNG_W / 2;

        return (
          <g
            key={rung.id}
            className={`interactive${isFocused ? ' focused' : ''}`}
            onMouseEnter={() => setHovered(rung.id)}
            onMouseLeave={() => setHovered(null)}
            style={{
              opacity: isDimmed ? 0.25 : 1,
              transition: 'opacity 0.25s',
              cursor: 'pointer',
            }}
          >
            {/* Rung box */}
            <rect
              x={rx} y={y}
              width={RUNG_W} height={RUNG_H}
              rx={8}
              fill={rung.color} fillOpacity={isFocused ? 0.15 : 0.06}
              stroke={rung.color} strokeWidth={isFocused ? 1.5 : 1}
              style={{ transition: 'fill-opacity 0.2s, stroke-width 0.2s' }}
            />

            {/* Rung number */}
            <text x={rx + 24} y={y + RUNG_H / 2 - 6} textAnchor="middle"
              dominantBaseline="central" fill={rung.color} fontSize={20} fontWeight={700}
              fontFamily="var(--font-body, Georgia, serif)" opacity={0.4}>
              {idx + 1}
            </text>

            {/* Label */}
            <text x={rx + 52} y={y + 24} textAnchor="start"
              fill="var(--d-fg)" fontSize={14} fontWeight={700}
              fontFamily="var(--font-body, Georgia, serif)">
              {rung.label} Necessity
            </text>

            {/* Subtitle */}
            <text x={rx + 52} y={y + 43} textAnchor="start"
              fill="var(--d-muted)" fontSize={10.5}
              fontFamily="var(--font-body, Georgia, serif)">
              {rung.subtitle}
            </text>

            {/* Evidence badge */}
            <text x={rx + RUNG_W - 16} y={y + RUNG_H / 2} textAnchor="end"
              dominantBaseline="central" fill={rung.color} fontSize={9} fontStyle="italic"
              fontFamily="var(--font-body, Georgia, serif)">
              {rung.evidence}
            </text>

            {/* Confidence bar */}
            <rect x={rx + 52} y={y + 54} width={(RUNG_W - 80) * rung.confidence} height={3}
              rx={1.5} fill={rung.color} opacity={0.4} />
            <rect x={rx + 52} y={y + 54} width={RUNG_W - 80} height={3}
              rx={1.5} fill="var(--d-line)" opacity={0.1} />

            {/* Upward arrow between rungs (not on top rung) */}
            {revIdx > 0 && (
              <g opacity={isDimmed ? 0.15 : 0.5}>
                <line x1={CX} y1={y - GAP + 4} x2={CX} y2={y - 4}
                  stroke="var(--d-line)" strokeWidth={1} strokeDasharray="4 3" />
                <text x={CX} y={y - GAP / 2 + 1} textAnchor="middle" dominantBaseline="central"
                  fill="var(--d-muted)" fontSize={10}
                  fontFamily="var(--font-body, Georgia, serif)">
                  therefore
                </text>
              </g>
            )}
          </g>
        );
      })}

      {/* Bottom framing text */}
      <text x={CX} y={H - 30} textAnchor="middle" fill="var(--d-muted)" fontSize={11}
        fontStyle="italic" fontFamily="var(--font-body, Georgia, serif)">
        affect is not an epiphenomenon but a geometric inevitability
      </text>
    </svg>
  );
}
