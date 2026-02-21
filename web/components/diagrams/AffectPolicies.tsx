import { type Point } from './utils';

/**
 * Affect Policies — Life philosophies as target regions in affect space.
 *
 * 2D projection with labeled target regions for Stoicism, Buddhism,
 * Existentialism, Hedonism, Epicureanism.
 */
export default function AffectPolicies() {
  const w = 440, h = 320;

  // Axes
  const axisL = 60, axisR = 400, axisB = 260, axisT = 60;
  const axisW = axisR - axisL, axisH = axisB - axisT;

  // Map affect-space coordinates to SVG
  const toSvg = (arousal: number, cfWeight: number): Point => [
    axisL + arousal * axisW,
    axisB - cfWeight * axisH,
  ];

  const philosophies = [
    { name: 'Stoicism', a: 0.2, cf: 0.15, rx: 28, ry: 22,
      note: 'fixed moderate ι' },
    { name: 'Buddhism', a: 0.15, cf: 0.3, rx: 25, ry: 30,
      note: 'ι flexibility training' },
    { name: 'Existentialism', a: 0.55, cf: 0.8, rx: 35, ry: 22,
      note: 'high CF, high r_eff' },
    { name: 'Hedonism', a: 0.85, cf: 0.2, rx: 28, ry: 22,
      note: 'V+, high A' },
    { name: 'Epicureanism', a: 0.45, cf: 0.12, rx: 30, ry: 20,
      note: 'moderate V+, low A' },
  ];

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img"
      aria-label="Life philosophies as target regions in affect space: each aims for a distinct region.">

      <text x={w / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Philosophies as Affect Policies
      </text>
      <text x={w / 2} y={38} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        each targets a region of affect space
      </text>

      {/* Axes */}
      <line x1={axisL} y1={axisB} x2={axisR} y2={axisB}
        stroke="var(--d-fg)" strokeWidth={0.75} />
      <line x1={axisL} y1={axisB} x2={axisL} y2={axisT}
        stroke="var(--d-fg)" strokeWidth={0.75} />

      {/* Axis labels */}
      <text x={(axisL + axisR) / 2} y={axisB + 22} textAnchor="middle" fontSize={10}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Arousal →
      </text>
      <text x={axisL - 10} y={(axisT + axisB) / 2} textAnchor="middle" fontSize={10}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)"
        transform={`rotate(-90, ${axisL - 10}, ${(axisT + axisB) / 2})`}>
        Counterfactual Weight →
      </text>

      {/* Philosophy regions */}
      {philosophies.map(({ name, a, cf, rx, ry, note }) => {
        const [x, y] = toSvg(a, cf);
        return (
          <g key={name}>
            <ellipse cx={x} cy={y} rx={rx} ry={ry}
              fill="var(--d-fg)" fillOpacity={0.04}
              stroke="var(--d-fg)" strokeWidth={0.75}
              strokeDasharray={name === 'Buddhism' ? '3,2' : 'none'} />
            <text x={x} y={y - 3} textAnchor="middle" fontSize={9} fontWeight={500}
              fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              {name}
            </text>
            <text x={x} y={y + 10} textAnchor="middle" fontSize={7} fontStyle="italic"
              fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
              {note}
            </text>
          </g>
        );
      })}

      {/* Buddhism bidirectional ι arrow */}
      {(() => {
        const [bx, by] = toSvg(0.15, 0.3);
        return (
          <g>
            <line x1={bx - 30} y1={by + 22} x2={bx + 30} y2={by + 22}
              stroke="var(--d-line)" strokeWidth={0.5} />
            <text x={bx} y={by + 34} textAnchor="middle" fontSize={7}
              fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
              ← ι range →
            </text>
          </g>
        );
      })()}

      {/* Grid lines (light) */}
      {[0.25, 0.5, 0.75].map(t => (
        <g key={`grid-${t}`}>
          <line x1={axisL + t * axisW} y1={axisB} x2={axisL + t * axisW} y2={axisT}
            stroke="var(--d-line)" strokeWidth={0.2} />
          <line x1={axisL} y1={axisB - t * axisH} x2={axisR} y2={axisB - t * axisH}
            stroke="var(--d-line)" strokeWidth={0.2} />
        </g>
      ))}

      {/* Bottom note */}
      <text x={w / 2} y={h - 6} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        A philosophy is testable: does the practice reliably move practitioners toward its target region?
      </text>
    </svg>
  );
}
