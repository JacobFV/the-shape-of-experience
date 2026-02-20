'use client';

import { useEffect, useRef, useState } from 'react';

/**
 * VLM Convergence diagram.
 *
 * Shows that vision-language models trained on human data
 * independently recognize affect in protocell simulations —
 * the strongest universality evidence.
 *
 * Visual: two columns (VLM training source / protocell source)
 * converging on the same affect space, with RSA correlations.
 */

interface VLM {
  name: string;
  rsa: number;
  pVal: string;
}

const VLMS: VLM[] = [
  { name: 'GPT-4o', rsa: 0.72, pVal: '< 0.0001' },
  { name: 'Claude 3.5', rsa: 0.54, pVal: '< 0.0001' },
  { name: 'Gemini 1.5', rsa: 0.78, pVal: '< 0.0001' },
];

const PREDICTIONS = [
  { label: 'Drought → desperation/anxiety', pass: true },
  { label: 'Recovery → relief/optimism', pass: true },
  { label: 'Stable → contentment', pass: true },
  { label: 'Narrative removal → correlation holds', pass: true },
];

export default function VLMConvergence() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          observer.disconnect();
          setTimeout(() => setPhase(1), 300);
          setTimeout(() => setPhase(2), 1200);
          setTimeout(() => setPhase(3), 2000);
        }
      },
      { threshold: 0.3 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const cx = 280;
  const leftX = 90;
  const rightX = 470;

  return (
    <svg ref={svgRef} viewBox="0 0 560 380" className="diagram-svg" role="img"
      aria-label="VLM convergence: vision-language models trained on human data recognize the same affect signatures in uncontaminated protocell simulations">

      {/* Title */}
      <text x={cx} y={20} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Cross-Substrate Convergence
      </text>
      <text x={cx} y={35} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        VLMs trained on human data recognize affect in protocells
      </text>

      {/* Left column: Human training */}
      <g style={{ opacity: phase >= 1 ? 1 : 0.1, transition: 'opacity 0.6s ease' }}>
        <rect x={leftX - 60} y={55} width={120} height={80} rx={8}
          fill="var(--d-blue)" fillOpacity={0.08} stroke="var(--d-blue)" strokeWidth={1} />
        <text x={leftX} y={75} textAnchor="middle" fontSize={11} fontWeight={600}
          fill="var(--d-blue)" fontFamily="var(--font-body, Georgia, serif)">
          Human Data
        </text>
        <text x={leftX} y={92} textAnchor="middle" fontSize={8.5} fill="var(--d-muted)"
          fontFamily="var(--font-body, Georgia, serif)">
          images, text, faces,
        </text>
        <text x={leftX} y={104} textAnchor="middle" fontSize={8.5} fill="var(--d-muted)"
          fontFamily="var(--font-body, Georgia, serif)">
          emotion labels
        </text>
        <text x={leftX} y={120} textAnchor="middle" fontSize={8} fill="var(--d-blue)" fontStyle="italic"
          fontFamily="var(--font-body, Georgia, serif)">
          ↓ train
        </text>
      </g>

      {/* Right column: Protocell simulation */}
      <g style={{ opacity: phase >= 1 ? 1 : 0.1, transition: 'opacity 0.6s ease 0.3s' }}>
        <rect x={rightX - 60} y={55} width={120} height={80} rx={8}
          fill="var(--d-green)" fillOpacity={0.08} stroke="var(--d-green)" strokeWidth={1} />
        <text x={rightX} y={75} textAnchor="middle" fontSize={11} fontWeight={600}
          fill="var(--d-green)" fontFamily="var(--font-body, Georgia, serif)">
          Protocell Sim
        </text>
        <text x={rightX} y={92} textAnchor="middle" fontSize={8.5} fill="var(--d-muted)"
          fontFamily="var(--font-body, Georgia, serif)">
          grid world snapshots,
        </text>
        <text x={rightX} y={104} textAnchor="middle" fontSize={8.5} fill="var(--d-muted)"
          fontFamily="var(--font-body, Georgia, serif)">
          affect metrics, no labels
        </text>
        <text x={rightX} y={120} textAnchor="middle" fontSize={8} fill="var(--d-green)" fontStyle="italic"
          fontFamily="var(--font-body, Georgia, serif)">
          ↓ measure
        </text>
      </g>

      {/* VLM boxes */}
      <g style={{ opacity: phase >= 1 ? 1 : 0.1, transition: 'opacity 0.6s ease 0.2s' }}>
        {VLMS.map((vlm, i) => {
          const y = 148 + i * 38;
          return (
            <g key={vlm.name}>
              <rect x={leftX - 50} y={y} width={100} height={28} rx={5}
                fill="var(--d-blue)" fillOpacity={0.06} stroke="var(--d-blue)" strokeWidth={0.75} />
              <text x={leftX} y={y + 15} textAnchor="middle" dominantBaseline="central"
                fontSize={10} fontWeight={600} fill="var(--d-blue)"
                fontFamily="var(--font-body, Georgia, serif)">
                {vlm.name}
              </text>
            </g>
          );
        })}
      </g>

      {/* Protocell affect space */}
      <g style={{ opacity: phase >= 1 ? 1 : 0.1, transition: 'opacity 0.6s ease 0.4s' }}>
        <rect x={rightX - 50} y={148} width={100} height={104} rx={8}
          fill="var(--d-green)" fillOpacity={0.05} stroke="var(--d-green)" strokeWidth={0.75} />
        <text x={rightX} y={170} textAnchor="middle" fontSize={10} fontWeight={600}
          fill="var(--d-green)" fontFamily="var(--font-body, Georgia, serif)">
          Affect Space
        </text>
        <text x={rightX} y={186} textAnchor="middle" fontSize={8} fill="var(--d-muted)"
          fontFamily="var(--font-body, Georgia, serif)">
          V, A, Φ, r_eff, CF, SM
        </text>
        <text x={rightX} y={200} textAnchor="middle" fontSize={8} fill="var(--d-muted)"
          fontFamily="var(--font-body, Georgia, serif)">
          (uncontaminated)
        </text>
      </g>

      {/* Convergence arrows + RSA values */}
      <defs>
        <marker id="vlm-arrow" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
          <polygon points="0 0, 7 2.5, 0 5" fill="var(--d-orange)" />
        </marker>
      </defs>
      {VLMS.map((vlm, i) => {
        const y = 148 + i * 38 + 14;
        return (
          <g key={`conv-${i}`} style={{
            opacity: phase >= 2 ? 1 : 0,
            transition: `opacity 0.4s ease ${i * 0.2}s`,
          }}>
            <line x1={leftX + 55} y1={y} x2={rightX - 55} y2={185}
              stroke="var(--d-orange)" strokeWidth={1.5} opacity={0.5}
              markerEnd="url(#vlm-arrow)" />
            {/* RSA correlation label */}
            <rect x={(leftX + rightX) / 2 - 38} y={y - 10 + (185 - y) * 0.45}
              width={76} height={16} rx={3}
              fill="var(--d-orange)" fillOpacity={0.1} />
            <text x={(leftX + rightX) / 2} y={y - 1 + (185 - y) * 0.45}
              textAnchor="middle" dominantBaseline="central"
              fontSize={9} fontWeight={600} fill="var(--d-orange)"
              fontFamily="var(--font-body, Georgia, serif)">
              RSA ρ = {vlm.rsa.toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* Predictions checklist */}
      <g style={{
        opacity: phase >= 3 ? 1 : 0,
        transition: 'opacity 0.6s ease',
      }}>
        <text x={cx} y={275} textAnchor="middle" fontSize={11} fontWeight={600}
          fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
          Pre-Registered Predictions: 4/4 PASS
        </text>
        {PREDICTIONS.map((pred, i) => (
          <g key={i} transform={`translate(${cx - 140}, ${288 + i * 18})`}>
            <text x={0} y={0} fontSize={11} fontWeight={700}
              fill="var(--d-green)" fontFamily="var(--font-body, Georgia, serif)">
              ✓
            </text>
            <text x={16} y={0} fontSize={9.5} fill="var(--d-fg)"
              fontFamily="var(--font-body, Georgia, serif)">
              {pred.label}
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
}
