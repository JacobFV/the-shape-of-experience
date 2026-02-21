import { smoothOpen, type Point, pt } from './utils';

/**
 * Affect Landscape — basin depths and transition paths.
 *
 * A potential-energy-style landscape showing contentment (deep basin),
 * depression (deep narrow), anxiety (shallow), and transition arrows.
 */
export default function AffectLandscape() {
  const w = 500, h = 280;

  // Landscape curve points (potential energy surface)
  const curve: Point[] = [
    [30, 130],   // left edge
    [70, 140],   // slight rise
    [105, 190],  // depression basin bottom (deep, narrow)
    [135, 140],  // rise
    [170, 148],  // anxiety basin bottom (shallow)
    [195, 140],  // slight rise
    [240, 110],  // saddle
    [310, 200],  // contentment basin bottom (deep, wide)
    [380, 110],  // rise out
    [410, 125],  // anger region
    [440, 115],  // rise
    [470, 120],  // right edge
  ];

  const pathD = smoothOpen(curve, 0.3);

  // Ball position (in contentment basin)
  const ballX = 310, ballY = 196;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img"
      aria-label="Affect landscape showing basins of different depths: deep contentment, deep narrow depression, shallow anxiety.">

      <text x={w / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Affect Landscape
      </text>
      <text x={w / 2} y={38} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        basin depth determines stability; transitions are causal
      </text>

      {/* Landscape curve */}
      <path d={pathD} fill="none" stroke="var(--d-fg)" strokeWidth={1.2} />

      {/* Shade under the curve */}
      <path d={`${pathD} L 470,${h - 40} L 30,${h - 40} Z`}
        fill="var(--d-fg)" fillOpacity={0.03} />

      {/* Ball */}
      <circle cx={ballX} cy={ballY} r={5}
        fill="var(--d-fg)" fillOpacity={0.3} stroke="var(--d-fg)" strokeWidth={0.75} />

      {/* Basin labels */}
      {/* Depression */}
      <text x={105} y={208} textAnchor="middle" fontSize={9} fontWeight={500}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        depression
      </text>
      <text x={105} y={220} textAnchor="middle" fontSize={7.5} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        deep, narrow
      </text>

      {/* Anxiety */}
      <text x={172} y={164} textAnchor="middle" fontSize={9} fontWeight={500}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        anxiety
      </text>
      <text x={172} y={176} textAnchor="middle" fontSize={7.5} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        shallow
      </text>

      {/* Contentment */}
      <text x={310} y={218} textAnchor="middle" fontSize={9} fontWeight={500}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        contentment
      </text>
      <text x={310} y={230} textAnchor="middle" fontSize={7.5} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        deep, wide
      </text>

      {/* Anger */}
      <text x={415} y={142} textAnchor="middle" fontSize={9} fontWeight={500}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        anger
      </text>

      {/* Transition arrows */}
      {/* Fear → Anger */}
      <path d="M 195,138 Q 300,80 400,123" fill="none"
        stroke="var(--d-line)" strokeWidth={0.6} strokeDasharray="3,2" />
      <text x={300} y={92} textAnchor="middle" fontSize={7} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        causal attribution externalizes
      </text>

      {/* Depression → Contentment (requires energy input) */}
      <path d="M 130,138 Q 220,65 290,108" fill="none"
        stroke="var(--d-line)" strokeWidth={0.6} strokeDasharray="3,2" />
      <text x={210} y={80} textAnchor="middle" fontSize={7} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        activation energy required
      </text>

      {/* Axes */}
      <text x={15} y={165} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)"
        transform="rotate(-90, 15, 165)">
        stability →
      </text>

      {/* Bottom note */}
      <text x={w / 2} y={h - 12} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Affects are trajectories on this landscape, not static points
      </text>
    </svg>
  );
}
