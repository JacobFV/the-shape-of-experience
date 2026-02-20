'use client';

import { useEffect, useRef, useState } from 'react';

/**
 * Visualization of the V32 integration distribution across 50 seeds.
 *
 * A strip plot / dot plot showing each seed as a dot colored by category
 * (HIGH / MOD / LOW), with density indicated by stacking. Shows the
 * 22% / 46% / 32% split and key statistics.
 */

// Representative Φ values for 50 seeds (inspired by V32 distribution)
// HIGH: 22% = 11 seeds (Φ > 0.12), MOD: 46% = 23 seeds, LOW: 32% = 16 seeds
function mulberry32(seed: number) {
  return () => {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(2032);

interface Seed {
  phi: number;
  category: 'HIGH' | 'MOD' | 'LOW';
  slope: number; // positive for HIGH, negative for LOW
}

// Generate 50 seeds matching V32 distribution
const seeds: Seed[] = [];
// HIGH seeds (11)
for (let i = 0; i < 11; i++) {
  const phi = 0.12 + rand() * 0.35;
  seeds.push({ phi, category: 'HIGH', slope: 0.002 + rand() * 0.008 });
}
// MOD seeds (23)
for (let i = 0; i < 23; i++) {
  const phi = 0.06 + rand() * 0.06;
  seeds.push({ phi, category: 'MOD', slope: -0.001 + rand() * 0.002 });
}
// LOW seeds (16)
for (let i = 0; i < 16; i++) {
  const phi = 0.02 + rand() * 0.04;
  seeds.push({ phi, category: 'LOW', slope: -0.008 + rand() * 0.005 });
}
seeds.sort((a, b) => a.phi - b.phi);

const CAT_COLORS = {
  HIGH: 'var(--d-green)',
  MOD: 'var(--d-yellow)',
  LOW: 'var(--d-red)',
};

export default function IntegrationDistribution() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [revealed, setRevealed] = useState(false);
  const [hovered, setHovered] = useState<number | null>(null);

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setRevealed(true); observer.disconnect(); } },
      { threshold: 0.25 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Layout
  const margin = { left: 50, right: 30, top: 55, bottom: 90 };
  const w = 540;
  const plotH = 160;
  const totalH = margin.top + plotH + margin.bottom;
  const plotW = w - margin.left - margin.right;

  const maxPhi = 0.50;
  const xScale = (phi: number) => margin.left + (phi / maxPhi) * plotW;

  // Bin seeds for stacking (strip plot)
  const binWidth = 0.015;
  const binned = seeds.map((seed, i) => {
    const bin = Math.floor(seed.phi / binWidth);
    const samebin = seeds.filter((s, j) => j < i && Math.floor(s.phi / binWidth) === bin);
    return { ...seed, idx: i, stack: samebin.length };
  });

  const dotR = 5;
  const dotGap = dotR * 2.2;
  const baseY = margin.top + plotH;

  return (
    <svg ref={svgRef} viewBox={`0 0 ${w} ${totalH}`} className="diagram-svg" role="img"
      aria-label="Integration distribution across 50 seeds: 22% HIGH, 46% MODERATE, 32% LOW, showing each seed as a colored dot">

      {/* Title */}
      <text x={w / 2} y={18} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Integration Distribution (V32, 50 seeds)
      </text>
      <text x={w / 2} y={33} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        same architecture, same environment — only biography differs
      </text>

      {/* Category background bands */}
      <rect x={xScale(0)} y={margin.top - 5} width={xScale(0.06) - xScale(0)} height={plotH + 10}
        fill="var(--d-red)" fillOpacity={0.04} rx={2} />
      <rect x={xScale(0.06)} y={margin.top - 5} width={xScale(0.12) - xScale(0.06)} height={plotH + 10}
        fill="var(--d-yellow)" fillOpacity={0.04} rx={2} />
      <rect x={xScale(0.12)} y={margin.top - 5} width={xScale(maxPhi) - xScale(0.12)} height={plotH + 10}
        fill="var(--d-green)" fillOpacity={0.04} rx={2} />

      {/* Category boundary lines */}
      {[0.06, 0.12].map(v => (
        <line key={v} x1={xScale(v)} y1={margin.top - 5} x2={xScale(v)} y2={baseY + 5}
          stroke="var(--d-line)" strokeWidth={0.5} strokeDasharray="3,3" opacity={0.4} />
      ))}

      {/* Axis */}
      <line x1={margin.left} y1={baseY} x2={margin.left + plotW} y2={baseY}
        stroke="var(--d-line)" strokeWidth={0.75} />
      {[0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50].map(v => (
        <g key={v}>
          <line x1={xScale(v)} y1={baseY} x2={xScale(v)} y2={baseY + 4}
            stroke="var(--d-line)" strokeWidth={0.5} />
          <text x={xScale(v)} y={baseY + 14} textAnchor="middle"
            fontSize={8} fill="var(--d-muted)"
            fontFamily="var(--font-body, Georgia, serif)">
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      <text x={margin.left + plotW / 2} y={baseY + 28} textAnchor="middle"
        fontSize={10} fill="var(--d-fg)" fontStyle="italic"
        fontFamily="var(--font-body, Georgia, serif)">
        Φ (late-phase integration)
      </text>

      {/* Dots */}
      {binned.map((seed, i) => {
        const x = xScale(seed.phi);
        const y = baseY - dotR - seed.stack * dotGap;
        const isHL = hovered === i;
        const color = CAT_COLORS[seed.category];

        return (
          <circle key={i}
            cx={x} cy={y} r={isHL ? dotR + 1.5 : dotR}
            fill={color} fillOpacity={isHL ? 0.9 : 0.6}
            stroke={color} strokeWidth={isHL ? 1.5 : 0}
            style={{
              opacity: revealed ? 1 : 0,
              transition: `opacity 0.3s ease ${i * 0.02}s, r 0.15s`,
              cursor: 'pointer',
            }}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
          />
        );
      })}

      {/* Max Φ annotation (seed 23) */}
      {revealed && (
        <g style={{ opacity: 1, transition: 'opacity 0.5s ease 1.2s' }}>
          <line x1={xScale(0.473)} y1={margin.top} x2={xScale(0.473)} y2={baseY - 8}
            stroke="var(--d-green)" strokeWidth={0.75} strokeDasharray="2,2" opacity={0.5} />
          <text x={xScale(0.473)} y={margin.top - 4} textAnchor="middle"
            fontSize={8} fill="var(--d-green)" fontWeight={600}
            fontFamily="var(--font-body, Georgia, serif)">
            seed 23: Φ = 0.473
          </text>
        </g>
      )}

      {/* Category labels + percentages */}
      <g transform={`translate(0, ${baseY + 42})`}>
        {([
          { cat: 'LOW', pct: '32%', n: 16, color: 'var(--d-red)', x: xScale(0.03) },
          { cat: 'MOD', pct: '46%', n: 23, color: 'var(--d-yellow)', x: xScale(0.09) },
          { cat: 'HIGH', pct: '22%', n: 11, color: 'var(--d-green)', x: xScale(0.25) },
        ]).map(({ cat, pct, n, color, x }) => (
          <g key={cat}>
            <text x={x} y={0} textAnchor="middle" fontSize={12} fontWeight={700}
              fill={color} fontFamily="var(--font-body, Georgia, serif)">
              {pct} {cat}
            </text>
            <text x={x} y={14} textAnchor="middle" fontSize={9}
              fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
              ({n} seeds)
            </text>
          </g>
        ))}
      </g>

      {/* Hover tooltip */}
      {hovered !== null && (() => {
        const seed = binned[hovered];
        return (
          <g>
            <rect x={20} y={totalH - 22} width={w - 40} height={18} rx={3}
              fill="var(--d-fg)" fillOpacity={0.05}
              stroke="var(--d-line)" strokeWidth={0.5} strokeOpacity={0.3} />
            <text x={w / 2} y={totalH - 10} textAnchor="middle"
              fontSize={9.5} fill="var(--d-fg)"
              fontFamily="var(--font-body, Georgia, serif)">
              Seed {seed.idx + 1}: Φ = {seed.phi.toFixed(3)} ({seed.category}) — Φ slope {seed.slope > 0 ? '+' : ''}{seed.slope.toFixed(4)}/cycle
            </text>
          </g>
        );
      })()}
    </svg>
  );
}
