'use client';

import { useEffect, useRef, useState } from 'react';

/**
 * Animated causal chain diagram for the Protocell Necessity Chain (V20).
 *
 * Membrane → free energy gradient → world model → self-model → affect geometry
 * Each link is necessary, not contingent. Animation reveals the chain step by step.
 */

interface ChainLink {
  label: string;
  sublabel: string;
  color: string;
  metric?: string;
}

const CHAIN: ChainLink[] = [
  { label: 'Membrane', sublabel: 'Boundary separating self from non-self',
    color: 'var(--d-muted)', metric: 'discrete grid body' },
  { label: 'Free Energy Gradient', sublabel: 'Viability requires energy harvesting',
    color: 'var(--d-red)', metric: 'resource depletion → death' },
  { label: 'World Model', sublabel: 'Prediction of own future energy',
    color: 'var(--d-blue)', metric: 'C_wm = 0.10–0.15' },
  { label: 'Self-Model', sublabel: 'Agent encodes own state more than environment',
    color: 'var(--d-cyan)', metric: 'SM_sal > 1.0 in 2/3 seeds' },
  { label: 'Affect Geometry', sublabel: 'Similarity structure on internal states',
    color: 'var(--d-green)', metric: 'RSA ρ > 0.21, eff. rank 5–11' },
];

export default function NecessityChain() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [revealed, setRevealed] = useState(0);

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          observer.disconnect();
          // Reveal chain links one by one
          let step = 0;
          const interval = setInterval(() => {
            step++;
            setRevealed(step);
            if (step >= CHAIN.length) clearInterval(interval);
          }, 600);
        }
      },
      { threshold: 0.3 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const cx = 280;
  const startY = 50;
  const boxW = 260;
  const boxH = 55;
  const gap = 22;
  const step = boxH + gap;

  const totalH = startY + CHAIN.length * step + 30;

  return (
    <svg ref={svgRef} viewBox={`0 0 560 ${totalH}`} className="diagram-svg" role="img"
      aria-label="The Necessity Chain: each step from membrane to affect geometry is necessary, not contingent">

      {/* Title */}
      <text x={cx} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        The Necessity Chain
      </text>
      <text x={cx} y={37} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        each step necessary, not contingent (V20)
      </text>

      {/* Arrow defs */}
      <defs>
        <marker id="nc-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="var(--d-line)" />
        </marker>
      </defs>

      {CHAIN.map((link, i) => {
        const y = startY + i * step;
        const isRevealed = i < revealed;
        const isLatest = i === revealed - 1;

        return (
          <g key={i} style={{
            opacity: isRevealed ? 1 : 0.08,
            transition: 'opacity 0.5s ease',
          }}>
            {/* Connecting arrow from previous */}
            {i > 0 && (
              <g style={{ opacity: isRevealed ? 1 : 0, transition: 'opacity 0.4s ease' }}>
                <line x1={cx} y1={y - gap + 2} x2={cx} y2={y - 2}
                  stroke="var(--d-line)" strokeWidth={1} opacity={0.5}
                  markerEnd="url(#nc-arrow)" />
                <text x={cx + boxW / 2 + 8} y={y - gap / 2 + 1}
                  textAnchor="start" fontSize={8} fill="var(--d-muted)" fontStyle="italic"
                  fontFamily="var(--font-body, Georgia, serif)">
                  necessitates
                </text>
              </g>
            )}

            {/* Box */}
            <rect x={cx - boxW / 2} y={y} width={boxW} height={boxH} rx={6}
              fill={link.color} fillOpacity={isLatest ? 0.18 : isRevealed ? 0.1 : 0.03}
              stroke={link.color} strokeWidth={isLatest ? 2 : 1}
              style={{ transition: 'fill-opacity 0.3s, stroke-width 0.3s' }} />

            {/* Step number */}
            <text x={cx - boxW / 2 + 16} y={y + boxH / 2}
              textAnchor="middle" dominantBaseline="central"
              fontSize={14} fontWeight={700} fill={link.color} opacity={0.5}
              fontFamily="var(--font-body, Georgia, serif)">
              {i + 1}
            </text>

            {/* Label */}
            <text x={cx} y={y + boxH / 2 - 8}
              textAnchor="middle" dominantBaseline="central"
              fontSize={12} fontWeight={600} fill={link.color}
              fontFamily="var(--font-body, Georgia, serif)">
              {link.label}
            </text>

            {/* Sublabel */}
            <text x={cx} y={y + boxH / 2 + 8}
              textAnchor="middle" dominantBaseline="central"
              fontSize={9} fill="var(--d-muted)"
              fontFamily="var(--font-body, Georgia, serif)">
              {link.sublabel}
            </text>

            {/* Metric on the right */}
            {link.metric && (
              <text x={cx + boxW / 2 + 10} y={y + boxH / 2}
                textAnchor="start" dominantBaseline="central"
                fontSize={8.5} fill="var(--d-muted)" fontStyle="italic"
                fontFamily="var(--font-body, Georgia, serif)">
                {link.metric}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
