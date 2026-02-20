'use client';

import { useState, useEffect, useRef } from 'react';

/**
 * BasinGeometry — Part II diagram
 * Shows affect space as a landscape with basins (stable states),
 * ridges (transitions), and a trajectory moving between them.
 * The landscape has 3 labeled basins: flourishing, depression, anxiety.
 */

const W = 680;
const H = 360;

// Landscape profile points (x, y pairs defining the "terrain")
// Y is inverted — lower Y = higher on screen = higher energy
const LANDSCAPE: [number, number][] = [
  [30, 180],
  [60, 200],
  [100, 240], // basin 1: depression
  [140, 200],
  [170, 160], // ridge
  [200, 180],
  [250, 220],
  [290, 250], // basin 2: neutral/stable
  [330, 220],
  [360, 180],
  [390, 150], // ridge
  [420, 170],
  [460, 210],
  [500, 240], // basin 3: flourishing (deeper)
  [540, 210],
  [570, 170],
  [600, 150], // ridge
  [630, 180],
  [650, 200],
];

// Basins
const BASINS = [
  { x: 100, y: 240, label: 'Depression', color: '#f87171', desc: 'low r_eff, high SM, neg V' },
  { x: 290, y: 250, label: 'Stable Neutral', color: '#facc15', desc: 'moderate all dimensions' },
  { x: 500, y: 240, label: 'Flourishing', color: '#4ade80', desc: 'high Φ, high r_eff, pos V' },
];

// Ridges
const RIDGES = [
  { x: 170, y: 160, label: 'crisis threshold' },
  { x: 390, y: 150, label: 'growth edge' },
];

export default function BasinGeometry() {
  const [ballX, setBallX] = useState(290);
  const [visible, setVisible] = useState(false);
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: 0.3 });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Animate the ball slowly drifting
  useEffect(() => {
    if (!visible) return;
    let frame = 0;
    const id = setInterval(() => {
      frame++;
      // Oscillate between basins
      const t = (Math.sin(frame * 0.015) + 1) / 2; // 0-1
      setBallX(100 + t * 400); // traverse from depression to flourishing
    }, 50);
    return () => clearInterval(id);
  }, [visible]);

  // Get Y position on the landscape for a given X
  const getY = (x: number): number => {
    for (let i = 0; i < LANDSCAPE.length - 1; i++) {
      const [x0, y0] = LANDSCAPE[i];
      const [x1, y1] = LANDSCAPE[i + 1];
      if (x >= x0 && x <= x1) {
        const t = (x - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
      }
    }
    return 200;
  };

  const pathD =
    'M ' + LANDSCAPE.map(([x, y]) => `${x} ${y}`).join(' L ');

  // Fill path (landscape + bottom)
  const fillD =
    pathD + ` L ${LANDSCAPE[LANDSCAPE.length - 1][0]} ${H} L ${LANDSCAPE[0][0]} ${H} Z`;

  const ballY = getY(ballX);

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: '100%', maxWidth: 720 }}
      role="img"
      aria-label="Affect space basin geometry: landscape with depression, neutral, and flourishing basins"
    >
      {/* Title */}
      <text x={W / 2} y={22} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}>
        Basin Geometry of Affect Space
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fill="var(--d-muted)" fontSize={10}>
        Stable states are basins; transitions require crossing ridges
      </text>

      {/* Gradient fill for landscape */}
      <defs>
        <linearGradient id="basin-fill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="var(--d-line)" stopOpacity={0.3} />
          <stop offset="100%" stopColor="var(--d-line)" stopOpacity={0.08} />
        </linearGradient>
      </defs>

      {/* Landscape fill */}
      <path d={fillD} fill="url(#basin-fill)" />

      {/* Landscape line */}
      <path d={pathD} fill="none" stroke="var(--d-fg)" strokeWidth={2} opacity={0.6} />

      {/* Basin labels */}
      {BASINS.map((b, i) => (
        <g key={i}>
          {/* Basin highlight */}
          <circle cx={b.x} cy={b.y - 6} r={28} fill={b.color} opacity={0.08} />
          {/* Arrow pointing down */}
          <line x1={b.x} y1={b.y + 8} x2={b.x} y2={b.y + 20} stroke={b.color} strokeWidth={1.5} markerEnd="none" opacity={0.5} />
          {/* Label */}
          <text x={b.x} y={b.y + 36} textAnchor="middle" fill={b.color} fontSize={12} fontWeight={700}>
            {b.label}
          </text>
          <text x={b.x} y={b.y + 50} textAnchor="middle" fill="var(--d-muted)" fontSize={8}>
            {b.desc}
          </text>
        </g>
      ))}

      {/* Ridge labels */}
      {RIDGES.map((r, i) => (
        <g key={i}>
          <line x1={r.x} y1={r.y - 8} x2={r.x} y2={r.y - 20} stroke="var(--d-muted)" strokeWidth={1} strokeDasharray="3 2" opacity={0.5} />
          <text x={r.x} y={r.y - 24} textAnchor="middle" fill="var(--d-muted)" fontSize={9} fontStyle="italic">
            {r.label}
          </text>
        </g>
      ))}

      {/* Ball (current state) */}
      <circle
        cx={ballX}
        cy={ballY - 8}
        r={7}
        fill="#e2e8f0"
        stroke="var(--d-fg)"
        strokeWidth={1.5}
        style={{ transition: 'cx 0.05s linear, cy 0.05s linear' }}
      />
      {/* Ball label */}
      <text
        x={ballX}
        y={ballY - 22}
        textAnchor="middle"
        fill="var(--d-fg)"
        fontSize={9}
        fontWeight={600}
        style={{ transition: 'x 0.05s linear' }}
      >
        current state
      </text>

      {/* Energy axis label */}
      <text x={18} y={140} fill="var(--d-muted)" fontSize={10} fontWeight={600} transform="rotate(-90, 18, 140)">
        energy
      </text>
      <line x1={26} y1={80} x2={26} y2={200} stroke="var(--d-muted)" strokeWidth={0.5} />
      <polygon points="26,78 23,86 29,86" fill="var(--d-muted)" />

      {/* Annotation: depth = stability */}
      <text x={W - 20} y={H - 12} textAnchor="end" fill="var(--d-muted)" fontSize={8}>
        deeper basin = more stable attractor
      </text>
    </svg>
  );
}
