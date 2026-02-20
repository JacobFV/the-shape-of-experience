'use client';

import { useState } from 'react';

/**
 * ExistentialStrategies — Part III diagram
 * Maps religious/wisdom traditions by their ι strategy and primary affect target.
 * Two axes: ι range (horizontal) and primary target dimension (vertical).
 */

interface Tradition {
  name: string;
  x: number; // ι center (0-1)
  iotaRange: [number, number]; // min, max ι
  y: number; // vertical position (0=top, 1=bottom) — grouped by target
  target: string; // what dimension they primarily modulate
  color: string;
  example: string;
}

const TRADITIONS: Tradition[] = [
  {
    name: 'Contemplative',
    x: 0.3,
    iotaRange: [0.05, 0.55],
    y: 0.12,
    target: 'SM → 0',
    color: '#60a5fa',
    example: 'Buddhist meditation, Sufi whirling',
  },
  {
    name: 'Devotional',
    x: 0.2,
    iotaRange: [0.05, 0.35],
    y: 0.32,
    target: 'V → positive',
    color: '#f472b6',
    example: 'Bhakti, Evangelical worship',
  },
  {
    name: 'Shamanic',
    x: 0.1,
    iotaRange: [0.0, 0.25],
    y: 0.52,
    target: 'r_eff → max',
    color: '#4ade80',
    example: 'Ayahuasca, vision quest',
  },
  {
    name: 'Legalistic',
    x: 0.5,
    iotaRange: [0.35, 0.65],
    y: 0.72,
    target: 'A → stable',
    color: '#facc15',
    example: 'Orthodox Judaism, traditional Islam',
  },
  {
    name: 'Philosophical',
    x: 0.65,
    iotaRange: [0.35, 0.90],
    y: 0.25,
    target: 'Φ → high',
    color: '#a78bfa',
    example: 'Stoicism, Greek rationalism',
  },
  {
    name: 'Psychedelic',
    x: 0.08,
    iotaRange: [0.0, 0.15],
    y: 0.88,
    target: 'ι → 0 (forced)',
    color: '#2dd4bf',
    example: 'Psilocybin, LSD, DMT',
  },
];

const W = 680;
const H = 440;
const PAD_L = 70;
const PAD_R = 30;
const PAD_T = 65;
const PAD_B = 55;
const CHART_W = W - PAD_L - PAD_R;
const CHART_H = H - PAD_T - PAD_B;

export default function ExistentialStrategies() {
  const [hovered, setHovered] = useState<number | null>(null);

  const toX = (iota: number) => PAD_L + iota * CHART_W;
  const toY = (y: number) => PAD_T + y * CHART_H;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: '100%', maxWidth: 720 }}
      role="img"
      aria-label="Religious and wisdom traditions mapped by ι strategy"
    >
      {/* Title */}
      <text x={W / 2} y={22} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}>
        Technologies of Transcendence
      </text>
      <text x={W / 2} y={40} textAnchor="middle" fill="var(--d-muted)" fontSize={10}>
        Wisdom traditions mapped by ι operating range and primary affect target
      </text>

      {/* X axis — ι */}
      <line x1={PAD_L} y1={PAD_T + CHART_H} x2={PAD_L + CHART_W} y2={PAD_T + CHART_H} stroke="var(--d-line)" strokeWidth={1} />
      {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
        <g key={v}>
          <line x1={toX(v)} y1={PAD_T + CHART_H} x2={toX(v)} y2={PAD_T + CHART_H + 5} stroke="var(--d-line)" />
          <text x={toX(v)} y={PAD_T + CHART_H + 18} textAnchor="middle" fill="var(--d-muted)" fontSize={10}>
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      <text x={PAD_L + CHART_W / 2} y={H - 8} textAnchor="middle" fill="var(--d-muted)" fontSize={11} fontWeight={600}>
        ι operating range
      </text>

      {/* ι zone labels */}
      <text x={toX(0.12)} y={PAD_T - 8} textAnchor="middle" fill="#4ade80" fontSize={9} opacity={0.6}>
        participatory
      </text>
      <text x={toX(0.88)} y={PAD_T - 8} textAnchor="middle" fill="#f87171" fontSize={9} opacity={0.6}>
        mechanistic
      </text>

      {/* Gradient background */}
      <defs>
        <linearGradient id="iota-bg" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stopColor="#4ade80" stopOpacity={0.04} />
          <stop offset="100%" stopColor="#f87171" stopOpacity={0.04} />
        </linearGradient>
      </defs>
      <rect x={PAD_L} y={PAD_T} width={CHART_W} height={CHART_H} fill="url(#iota-bg)" />

      {/* Tradition bubbles */}
      {TRADITIONS.map((t, i) => {
        const cx = toX(t.x);
        const cy = toY(t.y);
        const isHovered = hovered === i;
        const isDimmed = hovered !== null && !isHovered;

        // ι range bar
        const rangeX1 = toX(t.iotaRange[0]);
        const rangeX2 = toX(t.iotaRange[1]);

        return (
          <g
            key={i}
            style={{
              opacity: isDimmed ? 0.25 : 1,
              transition: 'opacity 0.3s',
              cursor: 'pointer',
            }}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
          >
            {/* ι range bar */}
            <line
              x1={rangeX1}
              y1={cy}
              x2={rangeX2}
              y2={cy}
              stroke={t.color}
              strokeWidth={isHovered ? 4 : 2.5}
              strokeLinecap="round"
              opacity={0.5}
            />
            {/* Range endpoints */}
            <circle cx={rangeX1} cy={cy} r={3} fill={t.color} opacity={0.7} />
            <circle cx={rangeX2} cy={cy} r={3} fill={t.color} opacity={0.7} />

            {/* Center dot */}
            <circle cx={cx} cy={cy} r={isHovered ? 10 : 7} fill={t.color} opacity={0.8} />

            {/* Label */}
            <text
              x={cx}
              y={cy - (isHovered ? 16 : 13)}
              textAnchor="middle"
              fill={t.color}
              fontSize={isHovered ? 13 : 11}
              fontWeight={700}
            >
              {t.name}
            </text>

            {/* Target */}
            <text
              x={cx}
              y={cy + 18}
              textAnchor="middle"
              fill="var(--d-muted)"
              fontSize={9}
              fontWeight={600}
            >
              target: {t.target}
            </text>

            {/* Example on hover */}
            {isHovered && (
              <text
                x={cx}
                y={cy + 32}
                textAnchor="middle"
                fill="var(--d-muted)"
                fontSize={8}
                fontStyle="italic"
              >
                {t.example}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
