import { smoothOpen, type Point } from './utils';

/**
 * Meaning Cost Curve — M(ι) = M₀ · e^(α·ι)
 *
 * Exponential cost of meaning generation as inhibition increases.
 * At low ι, meaning is cheap (participatory default).
 * At high ι, meaning is expensive (must be constructed).
 */
export default function MeaningCost() {
  const w = 400, h = 250;

  // Axis bounds
  const axisL = 60, axisR = 370, axisB = 190, axisT = 55;
  const axisW = axisR - axisL, axisH = axisB - axisT;

  // Generate exponential curve points
  const nPoints = 30;
  const curve: Point[] = Array.from({ length: nPoints + 1 }, (_, i) => {
    const t = i / nPoints; // 0 to 1
    const x = axisL + t * axisW;
    const y = axisB - (Math.exp(3 * t) - 1) / (Math.E ** 3 - 1) * axisH;
    return [x, y] as Point;
  });

  const pathD = smoothOpen(curve, 0.2);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img"
      aria-label="Meaning cost increases exponentially with inhibition coefficient iota.">

      <text x={w / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        The Cost of Meaning
      </text>
      <text x={w / 2} y={38} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        M(ι) = M₀ · e^(α·ι)
      </text>

      {/* Axes */}
      <line x1={axisL} y1={axisB} x2={axisR} y2={axisB}
        stroke="var(--d-fg)" strokeWidth={0.75} />
      <line x1={axisL} y1={axisB} x2={axisL} y2={axisT}
        stroke="var(--d-fg)" strokeWidth={0.75} />

      {/* X axis labels */}
      <text x={axisL} y={axisB + 16} textAnchor="middle" fontSize={9}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        0
      </text>
      <text x={axisR} y={axisB + 16} textAnchor="middle" fontSize={9}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        1
      </text>
      <text x={axisL} y={axisB + 30} fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        participatory
      </text>
      <text x={axisR} y={axisB + 30} textAnchor="end" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        mechanistic
      </text>
      <text x={(axisL + axisR) / 2} y={axisB + 16} textAnchor="middle" fontSize={10}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        ι →
      </text>

      {/* Y axis label */}
      <text x={axisL - 8} y={(axisT + axisB) / 2} textAnchor="middle" fontSize={9}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)"
        transform={`rotate(-90, ${axisL - 8}, ${(axisT + axisB) / 2})`}>
        M(ι) →
      </text>

      {/* Curve */}
      <path d={pathD} fill="none" stroke="var(--d-fg)" strokeWidth={1.2} />

      {/* Shaded region at high ι */}
      <rect x={axisL + axisW * 0.7} y={axisT} width={axisW * 0.3} height={axisH}
        fill="var(--d-fg)" fillOpacity={0.04} />

      {/* Annotations */}
      {/* Low ι */}
      <text x={axisL + axisW * 0.15} y={axisB - 30} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)" fontStyle="italic">
        meaning is cheap
      </text>
      <text x={axisL + axisW * 0.15} y={axisB - 18} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)" fontStyle="italic">
        (default)
      </text>

      {/* High ι */}
      <text x={axisL + axisW * 0.85} y={axisT + 25} textAnchor="middle" fontSize={8}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)" fontStyle="italic">
        meaning is expensive
      </text>
      <text x={axisL + axisW * 0.85} y={axisT + 37} textAnchor="middle" fontSize={8}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)" fontStyle="italic">
        (must be constructed)
      </text>

      {/* ι ≈ 0.30 marker */}
      {(() => {
        const ix = axisL + 0.3 * axisW;
        const iy = axisB - (Math.exp(3 * 0.3) - 1) / (Math.E ** 3 - 1) * axisH;
        return (
          <g>
            <line x1={ix} y1={axisB} x2={ix} y2={iy}
              stroke="var(--d-line)" strokeWidth={0.5} strokeDasharray="2,2" />
            <circle cx={ix} cy={iy} r={3}
              fill="var(--d-fg)" fillOpacity={0.3} stroke="var(--d-fg)" strokeWidth={0.5} />
            <text x={ix + 5} y={iy - 5} fontSize={8}
              fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              ι ≈ 0.30
            </text>
            <text x={ix + 5} y={iy + 7} fontSize={7} fontStyle="italic"
              fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
              evolutionary default
            </text>
          </g>
        );
      })()}

      {/* Bottom */}
      <text x={w / 2} y={h - 6} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        The meaning crisis is structural, not philosophical
      </text>
    </svg>
  );
}
