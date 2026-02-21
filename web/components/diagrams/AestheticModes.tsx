'use client';

import { useState } from 'react';

/**
 * AestheticModes — Part III diagram
 * Interactive comparison of aesthetic forms' affect signatures.
 * Each mode shows a mini bar chart of its dimension values.
 */

interface AestheticForm {
  name: string;
  color: string;
  iotaNote: string;
  dims: { label: string; value: number; direction: 'up' | 'down' | 'neutral' }[];
}

const FORMS: AestheticForm[] = [
  {
    name: 'Tragedy',
    color: 'var(--d-violet)',
    iotaNote: 'sustained low ι',
    dims: [
      { label: 'V', value: 0.2, direction: 'down' },
      { label: 'A', value: 0.7, direction: 'up' },
      { label: 'Φ', value: 0.95, direction: 'up' },
      { label: 'r', value: 0.3, direction: 'down' },
      { label: 'CF', value: 0.8, direction: 'up' },
      { label: 'SM', value: 0.5, direction: 'neutral' },
    ],
  },
  {
    name: 'Comedy',
    color: 'var(--d-yellow)',
    iotaNote: 'ι destabilized briefly',
    dims: [
      { label: 'V', value: 0.85, direction: 'up' },
      { label: 'A', value: 0.8, direction: 'up' },
      { label: 'Φ', value: 0.5, direction: 'neutral' },
      { label: 'r', value: 0.85, direction: 'up' },
      { label: 'CF', value: 0.4, direction: 'neutral' },
      { label: 'SM', value: 0.3, direction: 'down' },
    ],
  },
  {
    name: 'Sublime',
    color: 'var(--d-blue)',
    iotaNote: 'forced ι collapse',
    dims: [
      { label: 'V', value: 0.5, direction: 'neutral' },
      { label: 'A', value: 0.95, direction: 'up' },
      { label: 'Φ', value: 0.8, direction: 'up' },
      { label: 'r', value: 0.95, direction: 'up' },
      { label: 'CF', value: 0.6, direction: 'neutral' },
      { label: 'SM', value: 0.1, direction: 'down' },
    ],
  },
  {
    name: 'Horror',
    color: 'var(--d-red)',
    iotaNote: 'uncontrolled low ι',
    dims: [
      { label: 'V', value: 0.1, direction: 'down' },
      { label: 'A', value: 0.95, direction: 'up' },
      { label: 'Φ', value: 0.6, direction: 'neutral' },
      { label: 'r', value: 0.4, direction: 'neutral' },
      { label: 'CF', value: 0.9, direction: 'up' },
      { label: 'SM', value: 0.9, direction: 'up' },
    ],
  },
  {
    name: 'Blues',
    color: 'var(--d-violet)',
    iotaNote: 'low ι, witnessed',
    dims: [
      { label: 'V', value: 0.25, direction: 'down' },
      { label: 'A', value: 0.5, direction: 'neutral' },
      { label: 'Φ', value: 0.8, direction: 'up' },
      { label: 'r', value: 0.5, direction: 'neutral' },
      { label: 'CF', value: 0.5, direction: 'neutral' },
      { label: 'SM', value: 0.8, direction: 'up' },
    ],
  },
  {
    name: 'Ambient',
    color: 'var(--d-cyan)',
    iotaNote: 'gentle low ι',
    dims: [
      { label: 'V', value: 0.6, direction: 'neutral' },
      { label: 'A', value: 0.15, direction: 'down' },
      { label: 'Φ', value: 0.75, direction: 'up' },
      { label: 'r', value: 0.6, direction: 'neutral' },
      { label: 'CF', value: 0.2, direction: 'down' },
      { label: 'SM', value: 0.15, direction: 'down' },
    ],
  },
];

const W = 720;
const H = 420;
const CARD_W = 105;
const CARD_H = 240;
const GAP = 12;
const START_X = (W - (FORMS.length * CARD_W + (FORMS.length - 1) * GAP)) / 2;
const BARS_TOP = 150;
const BAR_H = 160;
const BAR_W = 10;

export default function AestheticModes() {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: '100%', maxWidth: 780 }}
      role="img"
      aria-label="Aesthetic modes affect signatures comparison"
    >
      {/* Title */}
      <text
        x={W / 2}
        y={28}
        textAnchor="middle"
        fill="var(--d-fg)"
        fontSize={16}
        fontWeight={700}
      >
        Affect Signatures of Aesthetic Forms
      </text>
      <text
        x={W / 2}
        y={46}
        textAnchor="middle"
        fill="var(--d-muted)"
        fontSize={10}
      >
        Each form reliably induces a characteristic configuration in affect space
      </text>

      {/* Dimension labels on left */}
      {FORMS[0].dims.map((dim, di) => {
        const y = BARS_TOP + (di / (FORMS[0].dims.length - 1)) * BAR_H;
        return (
          <text
            key={di}
            x={START_X - 14}
            y={y + 4}
            textAnchor="end"
            fill="var(--d-muted)"
            fontSize={11}
            fontWeight={600}
          >
            {dim.label}
          </text>
        );
      })}

      {/* Cards */}
      {FORMS.map((form, fi) => {
        const x = START_X + fi * (CARD_W + GAP);
        const isHovered = hoveredIdx === fi;
        const isDimmed = hoveredIdx !== null && !isHovered;

        return (
          <g
            key={fi}
            style={{
              opacity: isDimmed ? 0.3 : 1,
              transition: 'opacity 0.3s',
              cursor: 'pointer',
            }}
            onMouseEnter={() => setHoveredIdx(fi)}
            onMouseLeave={() => setHoveredIdx(null)}
          >
            {/* Card background */}
            <rect
              x={x}
              y={70}
              width={CARD_W}
              height={CARD_H + 100}
              rx={8}
              fill={isHovered ? form.color : 'var(--d-line)'}
              opacity={isHovered ? 0.12 : 0.06}
            />

            {/* Form name */}
            <text
              x={x + CARD_W / 2}
              y={90}
              textAnchor="middle"
              fill={form.color}
              fontSize={13}
              fontWeight={700}
            >
              {form.name}
            </text>

            {/* ι note */}
            <text
              x={x + CARD_W / 2}
              y={106}
              textAnchor="middle"
              fill="var(--d-muted)"
              fontSize={8}
              fontStyle="italic"
            >
              {form.iotaNote}
            </text>

            {/* Mini horizontal bars for each dimension */}
            {form.dims.map((dim, di) => {
              const y = BARS_TOP + (di / (form.dims.length - 1)) * BAR_H;
              const barMaxW = CARD_W - 20;
              const barX = x + 10;

              return (
                <g key={di}>
                  {/* Background track */}
                  <rect
                    x={barX}
                    y={y - BAR_W / 2}
                    width={barMaxW}
                    height={BAR_W}
                    rx={3}
                    fill="var(--d-line)"
                    opacity={0.15}
                  />
                  {/* Value bar */}
                  <rect
                    x={barX}
                    y={y - BAR_W / 2}
                    width={barMaxW * dim.value}
                    height={BAR_W}
                    rx={3}
                    fill={form.color}
                    opacity={
                      dim.direction === 'up'
                        ? 0.8
                        : dim.direction === 'down'
                        ? 0.5
                        : 0.35
                    }
                  />
                  {/* Value label on hover */}
                  {isHovered && (
                    <text
                      x={barX + barMaxW * dim.value + 4}
                      y={y + 3}
                      fill={form.color}
                      fontSize={8}
                      fontWeight={600}
                    >
                      {dim.value.toFixed(1)}
                    </text>
                  )}
                </g>
              );
            })}

            {/* Direction arrows summary */}
            <text
              x={x + CARD_W / 2}
              y={BARS_TOP + BAR_H + 30}
              textAnchor="middle"
              fill="var(--d-muted)"
              fontSize={9}
            >
              {form.dims.filter((d) => d.direction === 'up').length}↑{' '}
              {form.dims.filter((d) => d.direction === 'down').length}↓
            </text>
          </g>
        );
      })}

      {/* Legend */}
      <text x={W / 2} y={H - 10} textAnchor="middle" fill="var(--d-muted)" fontSize={9}>
        V = valence | A = arousal | Φ = integration | r = effective rank | CF = counterfactual | SM = self-model
      </text>
    </svg>
  );
}
