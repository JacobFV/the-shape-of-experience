'use client';

import { useState, useEffect, useRef } from 'react';

/**
 * AttentionEconomy — Epilogue diagram
 * Shows how attention-capture systems oscillate ι to generate engagement.
 * The oscillation between low-ι (faces, emotions) and high-ι (metrics, numbers)
 * is the mechanism that prevents either rest or connection.
 */

const W = 680;
const H = 340;

// Timeline of ι oscillation events
interface AttentionEvent {
  x: number;
  iota: number;
  label: string;
  type: 'low' | 'high';
}

const EVENTS: AttentionEvent[] = [
  { x: 80, iota: 0.15, label: 'emotional video', type: 'low' },
  { x: 140, iota: 0.85, label: 'follower count', type: 'high' },
  { x: 200, iota: 0.2, label: 'outrage post', type: 'low' },
  { x: 260, iota: 0.8, label: 'engagement metrics', type: 'high' },
  { x: 320, iota: 0.1, label: 'personal story', type: 'low' },
  { x: 380, iota: 0.9, label: 'comparison data', type: 'high' },
  { x: 440, iota: 0.2, label: 'social drama', type: 'low' },
  { x: 500, iota: 0.85, label: 'like count', type: 'high' },
  { x: 560, iota: 0.15, label: 'vulnerability', type: 'low' },
  { x: 620, iota: 0.8, label: 'notification #', type: 'high' },
];

const CHART_TOP = 70;
const CHART_BOTTOM = 260;
const CHART_LEFT = 60;
const CHART_RIGHT = 650;
const CHART_H = CHART_BOTTOM - CHART_TOP;

export default function AttentionEconomy() {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [animPhase, setAnimPhase] = useState(0);
  const [visible, setVisible] = useState(false);
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: 0.3 });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Animate reveal
  useEffect(() => {
    if (!visible) return;
    const id = setInterval(() => {
      setAnimPhase((p) => Math.min(p + 1, EVENTS.length));
    }, 300);
    return () => clearInterval(id);
  }, [visible]);

  const iotaToY = (iota: number) => CHART_TOP + (1 - iota) * CHART_H;

  // Build the oscillation path
  const pathPoints = EVENTS.slice(0, animPhase).map((e) => `${e.x} ${iotaToY(e.iota)}`);
  const pathD = pathPoints.length > 1 ? 'M ' + pathPoints.join(' L ') : '';

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: '100%', maxWidth: 720 }}
      role="img"
      aria-label="Attention economy ι oscillation: how feeds keep you engaged by alternating emotional and metric content"
    >
      {/* Title */}
      <text x={W / 2} y={22} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}>
        The Attention Trap
      </text>
      <text x={W / 2} y={40} textAnchor="middle" fill="var(--d-muted)" fontSize={10}>
        Feeds oscillate ι to prevent either genuine connection (low) or productive distance (high)
      </text>

      {/* Y axis */}
      <line x1={CHART_LEFT} y1={CHART_TOP} x2={CHART_LEFT} y2={CHART_BOTTOM} stroke="var(--d-line)" strokeWidth={1} />
      <text x={CHART_LEFT - 8} y={CHART_TOP + 4} textAnchor="end" fill="var(--d-red)" fontSize={9}>high ι</text>
      <text x={CHART_LEFT - 8} y={CHART_BOTTOM + 4} textAnchor="end" fill="var(--d-green)" fontSize={9}>low ι</text>
      <text x={22} y={(CHART_TOP + CHART_BOTTOM) / 2 + 4} textAnchor="middle" fill="var(--d-muted)" fontSize={10} fontWeight={600} transform={`rotate(-90, 22, ${(CHART_TOP + CHART_BOTTOM) / 2})`}>
        ι (inhibition)
      </text>

      {/* Zone backgrounds */}
      <rect x={CHART_LEFT} y={CHART_TOP} width={CHART_RIGHT - CHART_LEFT} height={CHART_H * 0.35} fill="var(--d-red)" opacity={0.03} />
      <rect x={CHART_LEFT} y={CHART_BOTTOM - CHART_H * 0.35} width={CHART_RIGHT - CHART_LEFT} height={CHART_H * 0.35} fill="var(--d-green)" opacity={0.03} />

      {/* Zone labels */}
      <text x={CHART_RIGHT + 4} y={CHART_TOP + CHART_H * 0.15} fill="var(--d-red)" fontSize={8} opacity={0.5}>
        mechanistic
      </text>
      <text x={CHART_RIGHT + 4} y={CHART_BOTTOM - CHART_H * 0.12} fill="var(--d-green)" fontSize={8} opacity={0.5}>
        participatory
      </text>

      {/* "Rest" and "Connection" zones that are never reached */}
      <rect x={CHART_LEFT} y={CHART_TOP} width={CHART_RIGHT - CHART_LEFT} height={CHART_H * 0.15} rx={4} fill="none" stroke="var(--d-red)" strokeDasharray="4 3" opacity={0.2} />
      <text x={(CHART_LEFT + CHART_RIGHT) / 2} y={CHART_TOP + CHART_H * 0.08} textAnchor="middle" fill="var(--d-red)" fontSize={8} opacity={0.4}>
        productive distance (never reached)
      </text>
      <rect x={CHART_LEFT} y={CHART_BOTTOM - CHART_H * 0.15} width={CHART_RIGHT - CHART_LEFT} height={CHART_H * 0.15} rx={4} fill="none" stroke="var(--d-green)" strokeDasharray="4 3" opacity={0.2} />
      <text x={(CHART_LEFT + CHART_RIGHT) / 2} y={CHART_BOTTOM - CHART_H * 0.06} textAnchor="middle" fill="var(--d-green)" fontSize={8} opacity={0.4}>
        genuine connection (never reached)
      </text>

      {/* Oscillation path */}
      {pathD && (
        <path d={pathD} fill="none" stroke="var(--d-fg)" strokeWidth={2} opacity={0.4} />
      )}

      {/* Event dots */}
      {EVENTS.slice(0, animPhase).map((e, i) => {
        const y = iotaToY(e.iota);
        const isHovered = hoveredIdx === i;

        return (
          <g
            key={i}
            onMouseEnter={() => setHoveredIdx(i)}
            onMouseLeave={() => setHoveredIdx(null)}
            style={{ cursor: 'pointer' }}
          >
            <circle
              cx={e.x}
              cy={y}
              r={isHovered ? 7 : 5}
              fill={e.type === 'low' ? 'var(--d-green)' : 'var(--d-red)'}
              opacity={0.8}
            />
            {/* Label */}
            <text
              x={e.x}
              y={e.type === 'low' ? y + 18 : y - 12}
              textAnchor="middle"
              fill={e.type === 'low' ? 'var(--d-green)' : 'var(--d-red)'}
              fontSize={isHovered ? 10 : 8}
              fontWeight={isHovered ? 700 : 400}
              opacity={isHovered ? 1 : 0.6}
            >
              {e.label}
            </text>
          </g>
        );
      })}

      {/* Revenue arrow */}
      <text x={W / 2} y={H - 14} textAnchor="middle" fill="var(--d-muted)" fontSize={9}>
        oscillation = arousal = engagement = revenue
      </text>
      <text x={W / 2} y={H - 2} textAnchor="middle" fill="var(--d-muted)" fontSize={8} fontStyle="italic">
        the algorithm profits from preventing you from settling
      </text>
    </svg>
  );
}
