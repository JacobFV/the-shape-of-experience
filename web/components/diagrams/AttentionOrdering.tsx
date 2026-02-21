import { arrowPath } from './utils';

/**
 * Attention Ordering — Three substrate conditions from V12.
 *
 * Fixed-local attention (extinction) < Convolution (life without integration)
 * < Evolvable attention (life at threshold).
 */
export default function AttentionOrdering() {
  const w = 480, h = 200;
  const axisL = 50, axisR = 440;
  const axisY = 110;
  const axisW = axisR - axisL;

  const conditions = [
    { x: 0.1, label: 'Fixed-local\nattention', value: 'extinction', sub: '✗' },
    { x: 0.5, label: 'Convolution', value: 'rob = 0.981', sub: 'life without\nintegration' },
    { x: 0.85, label: 'Evolvable\nattention', value: 'rob = 1.001', sub: 'life at\nthreshold' },
  ];

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img"
      aria-label="Three substrate conditions ordered by integration capacity: fixed-local attention, convolution, evolvable attention.">

      <text x={w / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        The Attention Ordering (V12)
      </text>
      <text x={w / 2} y={38} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        attention is necessary but not sufficient
      </text>

      {/* Axis */}
      <line x1={axisL} y1={axisY} x2={axisR} y2={axisY}
        stroke="var(--d-fg)" strokeWidth={0.75} />
      <path d={arrowPath([axisR, axisY], 0, 5)}
        stroke="var(--d-fg)" strokeWidth={0.75} fill="none" />
      <text x={(axisL + axisR) / 2} y={axisY + 48} textAnchor="middle" fontSize={9}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        integration capacity →
      </text>

      {/* Biological threshold line at rob=1.0 */}
      {(() => {
        const threshX = axisL + 0.68 * axisW;
        return (
          <g>
            <line x1={threshX} y1={axisY - 40} x2={threshX} y2={axisY + 5}
              stroke="var(--d-line)" strokeWidth={0.5} strokeDasharray="3,2" />
            <text x={threshX + 3} y={axisY - 43} fontSize={7.5} fontStyle="italic"
              fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
              rob = 1.0
            </text>
          </g>
        );
      })()}

      {/* Conditions */}
      {conditions.map(({ x, label, value, sub }) => {
        const px = axisL + x * axisW;
        return (
          <g key={label}>
            {/* Tick */}
            <line x1={px} y1={axisY - 5} x2={px} y2={axisY + 5}
              stroke="var(--d-fg)" strokeWidth={0.75} />

            {/* Dot */}
            <circle cx={px} cy={axisY} r={4}
              fill="var(--d-fg)" fillOpacity={0.15}
              stroke="var(--d-fg)" strokeWidth={0.75} />

            {/* Label above */}
            {label.split('\n').map((line, i) => (
              <text key={i} x={px} y={axisY - 22 - (label.split('\n').length - 1 - i) * 12}
                textAnchor="middle" fontSize={9} fontWeight={500}
                fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
                {line}
              </text>
            ))}

            {/* Value below */}
            <text x={px} y={axisY + 18} textAnchor="middle" fontSize={8}
              fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              {value}
            </text>

            {/* Sub-description */}
            {sub.split('\n').map((line, i) => (
              <text key={i} x={px} y={axisY + 30 + i * 11}
                textAnchor="middle" fontSize={7.5} fontStyle="italic"
                fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
                {line}
              </text>
            ))}
          </g>
        );
      })}

      {/* +2.0pp shift annotation */}
      {(() => {
        const x1 = axisL + 0.5 * axisW;
        const x2 = axisL + 0.85 * axisW;
        const y = axisY - 52;
        return (
          <g>
            <line x1={x1} y1={y} x2={x2} y2={y}
              stroke="var(--d-fg)" strokeWidth={0.5} />
            <path d={arrowPath([x2, y], 0, 4)}
              stroke="var(--d-fg)" strokeWidth={0.5} fill="none" />
            <text x={(x1 + x2) / 2} y={y - 5} textAnchor="middle" fontSize={7.5}
              fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              +2.0pp
            </text>
          </g>
        );
      })()}

      {/* Bottom note */}
      <text x={w / 2} y={h - 6} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Wrong kind of attention is worse than no attention at all
      </text>
    </svg>
  );
}
