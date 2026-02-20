'use client';

import { useEffect, useRef, useState } from 'react';
import { type Point, smoothOpen, pt } from './utils';

/**
 * Animated diagram showing the bottleneck furnace mechanism.
 *
 * A population bar and Φ line evolve through 5 drought cycles.
 * The animation shows: population crashes → survivors recover → Φ climbs.
 * HIGH seeds show positive Φ slope; LOW seeds show negative.
 */

interface TimePoint {
  t: number;
  pop: number;
  phi: number;
  drought: boolean;
}

// Simplified trajectory inspired by V32 HIGH seed (positive slope)
const HIGH_SEED: TimePoint[] = [
  { t: 0, pop: 256, phi: 0.06, drought: false },
  { t: 1, pop: 256, phi: 0.07, drought: false },
  { t: 2, pop: 256, phi: 0.065, drought: false },
  // Drought 1
  { t: 3, pop: 18, phi: 0.04, drought: true },
  { t: 4, pop: 256, phi: 0.08, drought: false },
  { t: 5, pop: 256, phi: 0.075, drought: false },
  // Drought 2
  { t: 6, pop: 12, phi: 0.035, drought: true },
  { t: 7, pop: 256, phi: 0.095, drought: false },
  { t: 8, pop: 256, phi: 0.09, drought: false },
  // Drought 3
  { t: 9, pop: 8, phi: 0.03, drought: true },
  { t: 10, pop: 256, phi: 0.11, drought: false },
  { t: 11, pop: 256, phi: 0.105, drought: false },
  // Drought 4
  { t: 12, pop: 15, phi: 0.04, drought: true },
  { t: 13, pop: 256, phi: 0.13, drought: false },
  { t: 14, pop: 256, phi: 0.125, drought: false },
  // Drought 5
  { t: 15, pop: 6, phi: 0.03, drought: true },
  { t: 16, pop: 256, phi: 0.15, drought: false },
  { t: 17, pop: 256, phi: 0.145, drought: false },
];

// LOW seed trajectory (negative Φ slope — fails to benefit from forging)
const LOW_SEED: TimePoint[] = [
  { t: 0, pop: 256, phi: 0.065, drought: false },
  { t: 1, pop: 256, phi: 0.06, drought: false },
  { t: 2, pop: 256, phi: 0.058, drought: false },
  { t: 3, pop: 20, phi: 0.035, drought: true },
  { t: 4, pop: 256, phi: 0.055, drought: false },
  { t: 5, pop: 256, phi: 0.05, drought: false },
  { t: 6, pop: 14, phi: 0.03, drought: true },
  { t: 7, pop: 256, phi: 0.048, drought: false },
  { t: 8, pop: 256, phi: 0.045, drought: false },
  { t: 9, pop: 10, phi: 0.025, drought: true },
  { t: 10, pop: 256, phi: 0.042, drought: false },
  { t: 11, pop: 256, phi: 0.04, drought: false },
  { t: 12, pop: 16, phi: 0.028, drought: true },
  { t: 13, pop: 256, phi: 0.038, drought: false },
  { t: 14, pop: 256, phi: 0.035, drought: false },
  { t: 15, pop: 8, phi: 0.022, drought: true },
  { t: 16, pop: 256, phi: 0.033, drought: false },
  { t: 17, pop: 256, phi: 0.03, drought: false },
];

export default function BottleneckFurnace() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [progress, setProgress] = useState(0); // 0 to data.length - 1
  const [animate, setAnimate] = useState(false);
  const rafRef = useRef<number>(0);
  const startRef = useRef<number>(0);

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setAnimate(true); observer.disconnect(); } },
      { threshold: 0.25 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!animate) return;
    const duration = 6000; // total animation time in ms
    const maxIdx = HIGH_SEED.length - 1;

    startRef.current = performance.now();
    const tick = (now: number) => {
      const elapsed = now - startRef.current;
      const t = Math.min(elapsed / duration, 1);
      // Ease-in-out
      const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      setProgress(eased * maxIdx);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      }
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [animate]);

  // Layout
  const margin = { left: 60, right: 20, top: 50, bottom: 55 };
  const w = 540;
  const phiH = 140; // height of Φ chart
  const popH = 60;  // height of population bars
  const gapH = 15;
  const totalH = margin.top + phiH + gapH + popH + margin.bottom;
  const plotW = w - margin.left - margin.right;

  const maxT = HIGH_SEED[HIGH_SEED.length - 1].t;
  const maxPhi = 0.18;

  const xScale = (t: number) => margin.left + (t / maxT) * plotW;
  const phiScale = (phi: number) => margin.top + phiH - (phi / maxPhi) * phiH;
  const popScale = (pop: number) => margin.top + phiH + gapH + popH - (pop / 256) * popH;

  // Build visible path up to current progress
  const buildPhiPath = (data: TimePoint[], maxVisible: number): string => {
    const visible = data.filter((_, i) => i <= Math.ceil(maxVisible));
    if (visible.length < 2) return '';
    const points: Point[] = visible.map((d, i) => {
      const frac = i === Math.ceil(maxVisible) ? maxVisible - Math.floor(maxVisible) : 1;
      return [xScale(d.t), phiScale(d.phi)];
    });
    return smoothOpen(points, 0.2);
  };

  // Population bar current state
  const currentIdx = Math.floor(progress);
  const frac = progress - currentIdx;
  const interpPop = (data: TimePoint[]) => {
    const curr = data[Math.min(currentIdx, data.length - 1)];
    const next = data[Math.min(currentIdx + 1, data.length - 1)];
    return curr.pop + (next.pop - curr.pop) * frac;
  };

  return (
    <svg ref={svgRef} viewBox={`0 0 ${w} ${totalH}`} className="diagram-svg" role="img"
      aria-label="Bottleneck furnace: animated diagram showing how repeated drought-recovery cycles forge integration in HIGH seeds while LOW seeds decline">

      {/* Title */}
      <text x={w / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        The Bottleneck Furnace
      </text>
      <text x={w / 2} y={35} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Repeated near-extinction forges integration — or fails to
      </text>

      {/* Drought bands */}
      {HIGH_SEED.filter(d => d.drought).map(d => (
        <rect key={`dr-${d.t}`}
          x={xScale(d.t) - plotW / maxT * 0.45} y={margin.top - 5}
          width={plotW / maxT * 0.9} height={phiH + gapH + popH + 10}
          fill="var(--d-red)" fillOpacity={0.08} rx={2}
        />
      ))}

      {/* Φ axis */}
      <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + phiH}
        stroke="var(--d-line)" strokeWidth={0.75} />
      <text x={margin.left - 8} y={margin.top + phiH / 2}
        textAnchor="middle" dominantBaseline="central"
        fontSize={12} fill="var(--d-fg)" fontStyle="italic"
        fontFamily="var(--font-body, Georgia, serif)"
        transform={`rotate(-90, ${margin.left - 8}, ${margin.top + phiH / 2})`}>
        Φ (integration)
      </text>
      {/* Φ tick marks */}
      {[0, 0.05, 0.10, 0.15].map(v => (
        <g key={`phitick-${v}`}>
          <line x1={margin.left - 4} y1={phiScale(v)} x2={margin.left} y2={phiScale(v)}
            stroke="var(--d-line)" strokeWidth={0.5} />
          <text x={margin.left - 7} y={phiScale(v)}
            textAnchor="end" dominantBaseline="central"
            fontSize={8} fill="var(--d-muted)"
            fontFamily="var(--font-body, Georgia, serif)">
            {v.toFixed(2)}
          </text>
        </g>
      ))}

      {/* HIGH seed Φ path */}
      <path d={buildPhiPath(HIGH_SEED, progress)}
        fill="none" stroke="var(--d-green)" strokeWidth={2}
        strokeLinecap="round" />
      {/* HIGH current dot */}
      {currentIdx < HIGH_SEED.length && (
        <circle
          cx={xScale(HIGH_SEED[Math.min(currentIdx, HIGH_SEED.length - 1)].t)}
          cy={phiScale(HIGH_SEED[Math.min(currentIdx, HIGH_SEED.length - 1)].phi)}
          r={4} fill="var(--d-green)"
        />
      )}

      {/* LOW seed Φ path */}
      <path d={buildPhiPath(LOW_SEED, progress)}
        fill="none" stroke="var(--d-red)" strokeWidth={2}
        strokeLinecap="round" strokeDasharray="6,3" />
      {currentIdx < LOW_SEED.length && (
        <circle
          cx={xScale(LOW_SEED[Math.min(currentIdx, LOW_SEED.length - 1)].t)}
          cy={phiScale(LOW_SEED[Math.min(currentIdx, LOW_SEED.length - 1)].phi)}
          r={4} fill="var(--d-red)"
        />
      )}

      {/* Legend */}
      <g transform={`translate(${margin.left + plotW - 120}, ${margin.top + 5})`}>
        <line x1={0} y1={0} x2={20} y2={0} stroke="var(--d-green)" strokeWidth={2} />
        <text x={24} y={0} dominantBaseline="central" fontSize={9}
          fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
          HIGH seed (forged)
        </text>
        <line x1={0} y1={14} x2={20} y2={14} stroke="var(--d-red)" strokeWidth={2} strokeDasharray="6,3" />
        <text x={24} y={14} dominantBaseline="central" fontSize={9}
          fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
          LOW seed (eroded)
        </text>
      </g>

      {/* Population axis */}
      <text x={margin.left - 8} y={margin.top + phiH + gapH + popH / 2}
        textAnchor="middle" dominantBaseline="central"
        fontSize={10} fill="var(--d-fg)" fontStyle="italic"
        fontFamily="var(--font-body, Georgia, serif)"
        transform={`rotate(-90, ${margin.left - 8}, ${margin.top + phiH + gapH + popH / 2})`}>
        pop
      </text>

      {/* Population bars (HIGH seed) */}
      {HIGH_SEED.map((d, i) => {
        const visible = i <= Math.ceil(progress);
        const barH = (d.pop / 256) * popH;
        return visible ? (
          <rect key={`pop-${i}`}
            x={xScale(d.t) - 6} y={margin.top + phiH + gapH + popH - barH}
            width={12} height={barH} rx={1}
            fill={d.drought ? 'var(--d-red)' : 'var(--d-green)'}
            fillOpacity={d.drought ? 0.6 : 0.3}
            style={{ transition: 'height 0.1s' }}
          />
        ) : null;
      })}

      {/* Time axis */}
      <line x1={margin.left} y1={margin.top + phiH + gapH + popH}
        x2={margin.left + plotW} y2={margin.top + phiH + gapH + popH}
        stroke="var(--d-line)" strokeWidth={0.75} />
      <text x={margin.left + plotW / 2} y={totalH - 18}
        textAnchor="middle" fontSize={10} fill="var(--d-muted)"
        fontFamily="var(--font-body, Georgia, serif)">
        evolution cycles → (red bands = drought, 90–97% mortality)
      </text>

      {/* Annotation: bounce arrows on HIGH seed */}
      {progress > 10 && (
        <g style={{ opacity: Math.min((progress - 10) / 3, 1), transition: 'opacity 0.5s' }}>
          <text x={xScale(13)} y={phiScale(0.16) - 8}
            textAnchor="middle" fontSize={9} fill="var(--d-green)" fontWeight={600}
            fontFamily="var(--font-body, Georgia, serif)">
            each bounce higher
          </text>
          <text x={xScale(13)} y={phiScale(0.16) + 4}
            textAnchor="middle" fontSize={8} fill="var(--d-green)"
            fontFamily="var(--font-body, Georgia, serif)">
            mean bounce predicts final Φ (ρ = 0.60)
          </text>
        </g>
      )}

      {/* Annotation: LOW eroding */}
      {progress > 12 && (
        <g style={{ opacity: Math.min((progress - 12) / 3, 1), transition: 'opacity 0.5s' }}>
          <text x={xScale(14)} y={phiScale(0.02) + 12}
            textAnchor="middle" fontSize={9} fill="var(--d-red)" fontWeight={600}
            fontFamily="var(--font-body, Georgia, serif)">
            same droughts, declining Φ
          </text>
        </g>
      )}
    </svg>
  );
}
