import { smoothOpen, type Point, pt } from './utils';

/**
 * Valence Topology — Extremes Are Neighbors.
 *
 * The valence axis curves: ecstasy and agony (both high-Φ)
 * are closer to each other than either is to numbness.
 */
export default function ValenceTopology() {
  const w = 420, h = 280;
  const cx = 210, cy = 150;

  // Horseshoe curve
  const curve: Point[] = [
    [100, 80],    // ecstasy
    [70, 120],
    [60, 170],
    [80, 220],    // numbness (bottom)
    [160, 240],
    [260, 240],
    [340, 220],   // boredom
    [360, 170],
    [350, 120],
    [320, 80],    // agony
  ];

  const pathD = smoothOpen(curve, 0.3);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img"
      aria-label="Valence topology: ecstasy and agony are neighbors in the high-Phi region; numbness is distant.">

      <text x={cx} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Valence Topology
      </text>
      <text x={cx} y={38} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        the extremes are neighbors
      </text>

      {/* The horseshoe curve */}
      <path d={pathD} fill="none" stroke="var(--d-fg)" strokeWidth={1.2} />

      {/* Ecstasy point */}
      <circle cx={100} cy={80} r={4} fill="var(--d-fg)" fillOpacity={0.3}
        stroke="var(--d-fg)" strokeWidth={0.75} />
      <text x={100} y={65} textAnchor="middle" fontSize={10} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Ecstasy
      </text>
      <text x={100} y={53} textAnchor="middle" fontSize={8} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        V+, Φ high, vivid
      </text>

      {/* Agony point */}
      <circle cx={320} cy={80} r={4} fill="var(--d-fg)" fillOpacity={0.3}
        stroke="var(--d-fg)" strokeWidth={0.75} />
      <text x={320} y={65} textAnchor="middle" fontSize={10} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Agony
      </text>
      <text x={320} y={53} textAnchor="middle" fontSize={8} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        V−, Φ high, vivid
      </text>

      {/* Short distance between extremes */}
      <line x1={110} y1={78} x2={310} y2={78}
        stroke="var(--d-line)" strokeWidth={0.5} strokeDasharray="2,2" />
      <text x={cx} y={74} textAnchor="middle" fontSize={8} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        small experiential distance
      </text>

      {/* Numbness/Boredom at bottom */}
      <circle cx={cx} cy={242} r={4} fill="var(--d-fg)" fillOpacity={0.15}
        stroke="var(--d-fg)" strokeWidth={0.75} />
      <text x={cx} y={265} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Numbness / Boredom
      </text>
      <text x={cx} y={h - 5} textAnchor="middle" fontSize={8} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Φ low, A low, r_eff low — distant from both extremes
      </text>

      {/* Φ gradient annotation */}
      <text x={35} y={140} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)"
        transform="rotate(-90, 35, 140)">
        Φ increases →
      </text>
    </svg>
  );
}
