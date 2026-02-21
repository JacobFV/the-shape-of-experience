'use client';

import { useState } from 'react';

/**
 * NarrowBroadQualia — Part II diagram
 * Split view: narrow qualia (extractable, decomposable) vs broad qualia (unified, non-decomposable)
 * Center bridge: Φ connects them
 * Hover left: features light up independently. Hover right: everything lights up together.
 */

const NARROW_FEATURES = [
  { label: 'redness', y: 0 },
  { label: 'loudness', y: 1 },
  { label: 'sharpness', y: 2 },
  { label: 'warmth', y: 3 },
  { label: 'sweetness', y: 4 },
];

const W = 660;
const H = 380;
const PANEL_W = 230;
const PANEL_H = 260;
const GAP = 80;
const LEFT_X = 40;
const RIGHT_X = LEFT_X + PANEL_W + GAP;
const PANEL_Y = 70;

export default function NarrowBroadQualia() {
  const [hovered, setHovered] = useState<'narrow' | 'broad' | null>(null);
  const [litFeature, setLitFeature] = useState<number | null>(null);

  const bridgeCx = LEFT_X + PANEL_W + GAP / 2;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="diagram-svg"
      role="img"
      aria-label="Narrow qualia as independent extractable features versus broad qualia as unified non-decomposable experience, bridged by integration Phi"
    >
      {/* Title */}
      <text x={W / 2} y={24} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}
        fontFamily="var(--font-body, Georgia, serif)">
        Two Senses of Qualia
      </text>
      <text x={W / 2} y={42} textAnchor="middle" fill="var(--d-muted)" fontSize={10.5}
        fontFamily="var(--font-body, Georgia, serif)">
        narrow (extractable) vs. broad (unified moment)
      </text>

      {/* === LEFT PANEL: Narrow Qualia === */}
      <g
        style={{
          opacity: hovered === 'broad' ? 0.3 : 1,
          transition: 'opacity 0.3s',
          cursor: 'pointer',
        }}
        onMouseEnter={() => setHovered('narrow')}
        onMouseLeave={() => { setHovered(null); setLitFeature(null); }}
      >
        <rect x={LEFT_X} y={PANEL_Y} width={PANEL_W} height={PANEL_H} rx={8}
          fill="var(--d-blue)" fillOpacity={0.04} stroke="var(--d-blue)" strokeWidth={1} />
        <text x={LEFT_X + PANEL_W / 2} y={PANEL_Y + 24} textAnchor="middle"
          fill="var(--d-blue)" fontSize={13} fontWeight={700}
          fontFamily="var(--font-body, Georgia, serif)">
          Narrow Qualia
        </text>
        <text x={LEFT_X + PANEL_W / 2} y={PANEL_Y + 40} textAnchor="middle"
          fill="var(--d-muted)" fontSize={9.5}
          fontFamily="var(--font-body, Georgia, serif)">
          extractable features — testable
        </text>

        {/* Individual feature bars */}
        {NARROW_FEATURES.map((f, i) => {
          const fy = PANEL_Y + 60 + i * 36;
          const isLit = hovered === 'narrow' && (litFeature === i || litFeature === null);
          return (
            <g key={f.label}
              onMouseEnter={() => setLitFeature(i)}
              onMouseLeave={() => setLitFeature(null)}
            >
              <rect x={LEFT_X + 20} y={fy} width={PANEL_W - 40} height={24} rx={4}
                fill="var(--d-blue)"
                fillOpacity={isLit ? 0.2 : 0.06}
                stroke="var(--d-blue)"
                strokeWidth={isLit ? 1.2 : 0.5}
                style={{ transition: 'fill-opacity 0.15s, stroke-width 0.15s' }}
              />
              <text x={LEFT_X + PANEL_W / 2} y={fy + 13} textAnchor="middle"
                dominantBaseline="central" fill={isLit ? 'var(--d-blue)' : 'var(--d-muted)'}
                fontSize={11} fontFamily="var(--font-body, Georgia, serif)"
                style={{ transition: 'fill 0.15s' }}>
                {f.label}
              </text>
            </g>
          );
        })}
      </g>

      {/* === RIGHT PANEL: Broad Qualia === */}
      <g
        style={{
          opacity: hovered === 'narrow' ? 0.3 : 1,
          transition: 'opacity 0.3s',
          cursor: 'pointer',
        }}
        onMouseEnter={() => setHovered('broad')}
        onMouseLeave={() => setHovered(null)}
      >
        <rect x={RIGHT_X} y={PANEL_Y} width={PANEL_W} height={PANEL_H} rx={8}
          fill="var(--d-red)" fillOpacity={0.04} stroke="var(--d-red)" strokeWidth={1} />
        <text x={RIGHT_X + PANEL_W / 2} y={PANEL_Y + 24} textAnchor="middle"
          fill="var(--d-red)" fontSize={13} fontWeight={700}
          fontFamily="var(--font-body, Georgia, serif)">
          Broad Qualia
        </text>
        <text x={RIGHT_X + PANEL_W / 2} y={PANEL_Y + 40} textAnchor="middle"
          fill="var(--d-muted)" fontSize={9.5}
          fontFamily="var(--font-body, Georgia, serif)">
          unified moment — identity thesis
        </text>

        {/* Single unified glow */}
        <ellipse
          cx={RIGHT_X + PANEL_W / 2}
          cy={PANEL_Y + PANEL_H / 2 + 20}
          rx={80} ry={70}
          fill="var(--d-red)"
          fillOpacity={hovered === 'broad' ? 0.15 : 0.06}
          stroke="var(--d-red)"
          strokeWidth={hovered === 'broad' ? 1.5 : 0.75}
          style={{ transition: 'fill-opacity 0.3s, stroke-width 0.3s' }}
        />
        <text x={RIGHT_X + PANEL_W / 2} y={PANEL_Y + PANEL_H / 2 + 10}
          textAnchor="middle" dominantBaseline="central"
          fill="var(--d-red)" fontSize={12} fontStyle="italic"
          fontFamily="var(--font-body, Georgia, serif)"
          opacity={hovered === 'broad' ? 1 : 0.6}>
          the whole moment
        </text>
        <text x={RIGHT_X + PANEL_W / 2} y={PANEL_Y + PANEL_H / 2 + 30}
          textAnchor="middle" dominantBaseline="central"
          fill="var(--d-muted)" fontSize={10}
          fontFamily="var(--font-body, Georgia, serif)">
          non-decomposable
        </text>
      </g>

      {/* === CENTER BRIDGE: Φ === */}
      <g opacity={0.8}>
        {/* Connection lines */}
        <line x1={LEFT_X + PANEL_W} y1={PANEL_Y + PANEL_H / 2}
          x2={bridgeCx - 20} y2={PANEL_Y + PANEL_H / 2}
          stroke="var(--d-green)" strokeWidth={1} strokeDasharray="4 3" />
        <line x1={bridgeCx + 20} y1={PANEL_Y + PANEL_H / 2}
          x2={RIGHT_X} y2={PANEL_Y + PANEL_H / 2}
          stroke="var(--d-green)" strokeWidth={1} strokeDasharray="4 3" />

        {/* Phi circle */}
        <circle cx={bridgeCx} cy={PANEL_Y + PANEL_H / 2} r={18}
          fill="var(--d-green)" fillOpacity={0.12}
          stroke="var(--d-green)" strokeWidth={1.5} />
        <text x={bridgeCx} y={PANEL_Y + PANEL_H / 2}
          textAnchor="middle" dominantBaseline="central"
          fill="var(--d-green)" fontSize={16} fontWeight={700} fontStyle="italic"
          fontFamily="var(--font-body, Georgia, serif)">
          Φ
        </text>

        {/* Bridge label */}
        <text x={bridgeCx} y={PANEL_Y + PANEL_H / 2 + 30}
          textAnchor="middle" fill="var(--d-green)" fontSize={9}
          fontFamily="var(--font-body, Georgia, serif)">
          bridges both senses
        </text>
      </g>

      {/* Bottom insight */}
      <text x={W / 2} y={H - 20} textAnchor="middle" fill="var(--d-muted)" fontSize={10.5}
        fontStyle="italic" fontFamily="var(--font-body, Georgia, serif)">
        every seed develops geometry (narrow) — only ~30% develop high Φ (broad)
      </text>
    </svg>
  );
}
