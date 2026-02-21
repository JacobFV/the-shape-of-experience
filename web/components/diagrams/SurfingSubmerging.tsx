'use client';

import { useState } from 'react';

/**
 * SurfingSubmerging — Part VI diagram
 * Two-panel comparison: surfing (integrated human-AI) vs submerging (fragmented).
 * Shows the key diagnostic dimensions for each state.
 */

const W = 680;
const H = 380;
const PANEL_W = 290;
const PANEL_H = 280;
const GAP = 20;
const LEFT_X = (W - 2 * PANEL_W - GAP) / 2;
const RIGHT_X = LEFT_X + PANEL_W + GAP;
const TOP_Y = 60;

interface DiagnosticBar {
  label: string;
  surfValue: number;
  submergeValue: number;
}

const DIAGNOSTICS: DiagnosticBar[] = [
  { label: 'Φ (integration)', surfValue: 0.85, submergeValue: 0.25 },
  { label: 'Self-model coherence', surfValue: 0.8, submergeValue: 0.2 },
  { label: 'Value clarity', surfValue: 0.75, submergeValue: 0.15 },
  { label: 'ι calibration', surfValue: 0.7, submergeValue: 0.1 },
  { label: 'Agency (ρ)', surfValue: 0.8, submergeValue: 0.3 },
  { label: 'Attention sovereignty', surfValue: 0.7, submergeValue: 0.1 },
];

export default function SurfingSubmerging() {
  const [hoveredSide, setHoveredSide] = useState<'surf' | 'submerge' | null>(null);

  const barStartY = TOP_Y + 60;
  const barSpacing = 34;
  const barMaxW = PANEL_W - 80;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: '100%', maxWidth: 720 }}
      role="img"
      aria-label="Surfing versus submerging: two modes of human-AI integration"
    >
      {/* Title */}
      <text x={W / 2} y={22} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}>
        Surfing vs. Submerging
      </text>
      <text x={W / 2} y={40} textAnchor="middle" fill="var(--d-muted)" fontSize={10}>
        Maintaining integration while incorporating AI capabilities
      </text>

      {/* SURFING panel */}
      <g
        onMouseEnter={() => setHoveredSide('surf')}
        onMouseLeave={() => setHoveredSide(null)}
        style={{ cursor: 'pointer' }}
      >
        <rect
          x={LEFT_X}
          y={TOP_Y}
          width={PANEL_W}
          height={PANEL_H}
          rx={10}
          fill="var(--d-green)"
          opacity={hoveredSide === 'submerge' ? 0.03 : 0.06}
          style={{ transition: 'opacity 0.3s' }}
        />
        <rect
          x={LEFT_X}
          y={TOP_Y}
          width={PANEL_W}
          height={PANEL_H}
          rx={10}
          fill="none"
          stroke="var(--d-green)"
          strokeWidth={hoveredSide === 'surf' ? 2 : 1}
          opacity={hoveredSide === 'submerge' ? 0.2 : 0.5}
          style={{ transition: 'opacity 0.3s' }}
        />

        {/* Header */}
        <text
          x={LEFT_X + PANEL_W / 2}
          y={TOP_Y + 24}
          textAnchor="middle"
          fill="var(--d-green)"
          fontSize={16}
          fontWeight={700}
          opacity={hoveredSide === 'submerge' ? 0.4 : 1}
          style={{ transition: 'opacity 0.3s' }}
        >
          SURFING
        </text>
        <text
          x={LEFT_X + PANEL_W / 2}
          y={TOP_Y + 40}
          textAnchor="middle"
          fill="var(--d-muted)"
          fontSize={9}
          opacity={hoveredSide === 'submerge' ? 0.3 : 0.7}
          style={{ transition: 'opacity 0.3s' }}
        >
          integrated, coherent, sovereign
        </text>

        {/* Bars */}
        {DIAGNOSTICS.map((d, i) => {
          const y = barStartY + i * barSpacing;
          const isDimmed = hoveredSide === 'submerge';
          return (
            <g key={i} opacity={isDimmed ? 0.35 : 1} style={{ transition: 'opacity 0.3s' }}>
              <text x={LEFT_X + 8} y={y + 4} fill="var(--d-muted)" fontSize={9}>
                {d.label}
              </text>
              <rect
                x={LEFT_X + 8}
                y={y + 8}
                width={barMaxW}
                height={6}
                rx={3}
                fill="var(--d-line)"
                opacity={0.15}
              />
              <rect
                x={LEFT_X + 8}
                y={y + 8}
                width={barMaxW * d.surfValue}
                height={6}
                rx={3}
                fill="var(--d-green)"
                opacity={0.7}
              />
            </g>
          );
        })}
      </g>

      {/* SUBMERGING panel */}
      <g
        onMouseEnter={() => setHoveredSide('submerge')}
        onMouseLeave={() => setHoveredSide(null)}
        style={{ cursor: 'pointer' }}
      >
        <rect
          x={RIGHT_X}
          y={TOP_Y}
          width={PANEL_W}
          height={PANEL_H}
          rx={10}
          fill="var(--d-red)"
          opacity={hoveredSide === 'surf' ? 0.03 : 0.06}
          style={{ transition: 'opacity 0.3s' }}
        />
        <rect
          x={RIGHT_X}
          y={TOP_Y}
          width={PANEL_W}
          height={PANEL_H}
          rx={10}
          fill="none"
          stroke="var(--d-red)"
          strokeWidth={hoveredSide === 'submerge' ? 2 : 1}
          opacity={hoveredSide === 'surf' ? 0.2 : 0.5}
          style={{ transition: 'opacity 0.3s' }}
        />

        {/* Header */}
        <text
          x={RIGHT_X + PANEL_W / 2}
          y={TOP_Y + 24}
          textAnchor="middle"
          fill="var(--d-red)"
          fontSize={16}
          fontWeight={700}
          opacity={hoveredSide === 'surf' ? 0.4 : 1}
          style={{ transition: 'opacity 0.3s' }}
        >
          SUBMERGING
        </text>
        <text
          x={RIGHT_X + PANEL_W / 2}
          y={TOP_Y + 40}
          textAnchor="middle"
          fill="var(--d-muted)"
          fontSize={9}
          opacity={hoveredSide === 'surf' ? 0.3 : 0.7}
          style={{ transition: 'opacity 0.3s' }}
        >
          fragmented, captured, displaced
        </text>

        {/* Bars */}
        {DIAGNOSTICS.map((d, i) => {
          const y = barStartY + i * barSpacing;
          const isDimmed = hoveredSide === 'surf';
          return (
            <g key={i} opacity={isDimmed ? 0.35 : 1} style={{ transition: 'opacity 0.3s' }}>
              <text x={RIGHT_X + 8} y={y + 4} fill="var(--d-muted)" fontSize={9}>
                {d.label}
              </text>
              <rect
                x={RIGHT_X + 8}
                y={y + 8}
                width={barMaxW}
                height={6}
                rx={3}
                fill="var(--d-line)"
                opacity={0.15}
              />
              <rect
                x={RIGHT_X + 8}
                y={y + 8}
                width={barMaxW * d.submergeValue}
                height={6}
                rx={3}
                fill="var(--d-red)"
                opacity={0.7}
              />
            </g>
          );
        })}
      </g>

      {/* VS divider */}
      <text
        x={W / 2}
        y={TOP_Y + PANEL_H / 2 + 4}
        textAnchor="middle"
        fill="var(--d-muted)"
        fontSize={13}
        fontWeight={700}
        opacity={0.4}
      >
        vs.
      </text>

      {/* Bottom note */}
      <text x={W / 2} y={H - 10} textAnchor="middle" fill="var(--d-muted)" fontSize={9}>
        The diagnostic: Φ_H+A &gt; θ + human retains causal dominance (ρ &gt; 0.5)
      </text>
    </svg>
  );
}
