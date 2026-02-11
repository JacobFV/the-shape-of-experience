import { arrowPath, pt, type Point } from './utils';

/** Part 1-0: Supercritical pitchfork bifurcation diagram */
export default function PitchforkBifurcation() {
  // Plot area keypoints
  const left = 70, right = 560, top = 50, bottom = 320;
  const zero = 185; // y-coordinate for x*=0 line
  const jc = 280;   // x-coordinate for critical point

  // Branch curves: x*(J) = ±sqrt(J-Jc) mapped to plot coords
  const upperBranch: Point[] = [];
  const lowerBranch: Point[] = [];
  for (let j = jc; j <= right; j += 10) {
    const t = (j - jc) / (right - jc);
    const amp = Math.sqrt(t) * 120;
    upperBranch.push([j, zero - amp]);
    lowerBranch.push([j, zero + amp]);
  }

  const branchPath = (pts: Point[]) =>
    `M ${pt(pts[0])} ` + pts.slice(1).map(p => `L ${pt(p)}`).join(' ');

  // Small trajectory arrows near branches
  const trajArrow = (from: Point, to: Point, color: string) => {
    const dx = to[0] - from[0];
    const dy = to[1] - from[1];
    const len = Math.sqrt(dx * dx + dy * dy);
    const dir = Math.atan2(dy, dx) * 180 / Math.PI;
    return (
      <g>
        <line x1={from[0]} y1={from[1]} x2={to[0]} y2={to[1]}
          stroke={color} strokeWidth={0.6} opacity={0.5} />
        <path d={arrowPath(to, dir, 5)} stroke={color} strokeWidth={0.6}
          fill="none" opacity={0.5} />
      </g>
    );
  };

  return (
    <svg viewBox="0 0 620 380" className="diagram-svg" role="img"
      aria-label="Supercritical pitchfork bifurcation: a single stable equilibrium splits into two stable branches at a critical parameter value">
      {/* Axes */}
      <line x1={left} y1={bottom} x2={right + 15} y2={bottom}
        stroke="var(--d-line)" strokeWidth={0.75} />
      <path d={arrowPath([right + 15, bottom], 0, 6)}
        stroke="var(--d-line)" strokeWidth={0.75} fill="none" />
      <line x1={left} y1={bottom} x2={left} y2={top - 10}
        stroke="var(--d-line)" strokeWidth={0.75} />
      <path d={arrowPath([left, top - 10], -90, 6)}
        stroke="var(--d-line)" strokeWidth={0.75} fill="none" />

      {/* Axis labels */}
      <text x={right + 10} y={bottom + 22} textAnchor="end"
        fontSize={15} fontStyle="italic" fill="var(--d-fg)"
        fontFamily="var(--font-body, Georgia, serif)">J</text>
      <text x={left - 14} y={top - 5} textAnchor="middle"
        fontSize={15} fontStyle="italic" fill="var(--d-fg)"
        fontFamily="var(--font-body, Georgia, serif)">x*</text>

      {/* Zero reference line (thin) */}
      <line x1={left} y1={zero} x2={right} y2={zero}
        stroke="var(--d-line)" strokeWidth={0.3} strokeDasharray="4,4" opacity={0.3} />

      {/* Pre-critical stable branch (blue solid) */}
      <line x1={left} y1={zero} x2={jc} y2={zero}
        stroke="var(--d-blue)" strokeWidth={1.5} />

      {/* Post-critical unstable branch (red dashed) */}
      <line x1={jc} y1={zero} x2={right} y2={zero}
        stroke="var(--d-red)" strokeWidth={1} strokeDasharray="6,4" />

      {/* Upper stable branch (blue) */}
      <path d={branchPath(upperBranch)}
        fill="none" stroke="var(--d-blue)" strokeWidth={1.5} />

      {/* Lower stable branch (blue) */}
      <path d={branchPath(lowerBranch)}
        fill="none" stroke="var(--d-blue)" strokeWidth={1.5} />

      {/* Critical point */}
      <circle cx={jc} cy={zero} r={3.5} fill="var(--d-fg)" />

      {/* Jc tick and label */}
      <line x1={jc} y1={bottom - 3} x2={jc} y2={bottom + 3}
        stroke="var(--d-line)" strokeWidth={0.75} />
      <text x={jc} y={bottom + 20} textAnchor="middle"
        fontSize={13} fontStyle="italic" fill="var(--d-fg)"
        fontFamily="var(--font-body, Georgia, serif)">
        J<tspan dy={3} fontSize={10}>c</tspan>
      </text>

      {/* Trajectory arrows showing flow */}
      {/* Arrows toward upper branch */}
      {trajArrow([420, zero - 20], [420, zero - 55], 'var(--d-blue)')}
      {trajArrow([480, zero - 20], [480, zero - 70], 'var(--d-blue)')}
      {/* Arrows toward lower branch */}
      {trajArrow([420, zero + 20], [420, zero + 55], 'var(--d-blue)')}
      {trajArrow([480, zero + 20], [480, zero + 70], 'var(--d-blue)')}
      {/* Arrows away from unstable */}
      {trajArrow([350, zero - 3], [350, zero - 20], 'var(--d-red)')}
      {trajArrow([350, zero + 3], [350, zero + 20], 'var(--d-red)')}

      {/* Branch labels */}
      <text x={170} y={zero - 14} textAnchor="middle" fontSize={11.5}
        fill="var(--d-blue)" fontFamily="var(--font-body, Georgia, serif)">
        stable
      </text>
      <text x={450} y={zero - 10} textAnchor="middle" fontSize={11.5}
        fill="var(--d-red)" fontFamily="var(--font-body, Georgia, serif)">
        unstable
      </text>

      {/* Attractor labels */}
      <text x={right - 15} y={upperBranch[upperBranch.length - 1][1] - 12}
        textAnchor="end" fontSize={11} fill="var(--d-muted)"
        fontFamily="var(--font-body, Georgia, serif)">
        structured attractor 1
      </text>
      <text x={right - 15} y={lowerBranch[lowerBranch.length - 1][1] + 16}
        textAnchor="end" fontSize={11} fill="var(--d-muted)"
        fontFamily="var(--font-body, Georgia, serif)">
        structured attractor 2
      </text>
    </svg>
  );
}
