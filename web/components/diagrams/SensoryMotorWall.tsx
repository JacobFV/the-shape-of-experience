import { arrowPath } from './utils';

/**
 * Sensory-Motor Coupling Wall — Lenia vs. V20 Protocell.
 *
 * Shows the architectural difference: one-way influence vs.
 * closed action→environment→observation loop.
 */
export default function SensoryMotorWall() {
  const w = 520, h = 300;
  const leftCx = 135, rightCx = 385;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img"
      aria-label="Sensory-motor coupling wall: Lenia has no causal loop; V20 protocell has a closed loop.">

      <text x={w / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        The Sensory-Motor Coupling Wall
      </text>
      <text x={w / 2} y={38} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        ρ_sync ≈ 0 → ρ_sync = 0.21
      </text>

      {/* Divider */}
      <line x1={w / 2} y1={50} x2={w / 2} y2={h - 10}
        stroke="var(--d-line)" strokeWidth={0.5} strokeDasharray="4,4" />

      {/* ── LEFT: Lenia ── */}
      <text x={leftCx} y={60} textAnchor="middle" fontSize={11} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Lenia (V11–V18)
      </text>
      <text x={leftCx} y={73} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        wall intact
      </text>

      {/* Environment box */}
      <rect x={leftCx - 55} y={90} width={110} height={45} rx={5}
        fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />
      <text x={leftCx} y={112} textAnchor="middle" dominantBaseline="central"
        fontSize={10} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Environment
      </text>

      {/* Pattern inside environment */}
      <rect x={leftCx - 30} y={155} width={60} height={35} rx={4}
        fill="var(--d-fg)" fillOpacity={0.05}
        stroke="var(--d-fg)" strokeWidth={0.75} strokeDasharray="3,2" />
      <text x={leftCx} y={172} textAnchor="middle" dominantBaseline="central"
        fontSize={9} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Pattern
      </text>

      {/* One-way arrow: env → pattern */}
      <line x1={leftCx} y1={135} x2={leftCx} y2={152}
        stroke="var(--d-line)" strokeWidth={0.75} />
      <path d={arrowPath([leftCx, 152], 90, 5)}
        stroke="var(--d-line)" strokeWidth={0.75} fill="none" />

      {/* Label: one-way */}
      <text x={leftCx + 40} y={143} textAnchor="start" fontSize={8} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        one-way influence
      </text>
      <text x={leftCx + 40} y={153} textAnchor="start" fontSize={8} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        no causal loop
      </text>

      {/* Metric */}
      <text x={leftCx} y={210} textAnchor="middle" fontSize={10}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        ρ_sync ≈ 0.003
      </text>

      {/* Rungs unlocked */}
      <text x={leftCx} y={232} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Unlocked: geometry, world model, memory
      </text>
      <text x={leftCx} y={244} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Blocked: counterfactual, self-model
      </text>
      <text x={leftCx} y={262} textAnchor="middle" fontSize={9}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Rungs 1–7
      </text>

      {/* ── RIGHT: V20 Protocell ── */}
      <text x={rightCx} y={60} textAnchor="middle" fontSize={11} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        V20 Protocell
      </text>
      <text x={rightCx} y={73} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        wall broken
      </text>

      {/* Closed loop: Agent → Action → Env → Observation → Agent */}
      {(() => {
        const cx = rightCx, cy = 155;
        const rx = 55, ry = 50;
        const labels = [
          { angle: -90, text: 'Agent', bold: true },
          { angle: 0, text: 'Action' },
          { angle: 90, text: 'Environment' },
          { angle: 180, text: 'Observation' },
        ];

        return (
          <g>
            {/* Elliptical loop */}
            <ellipse cx={cx} cy={cy} rx={rx} ry={ry}
              fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />

            {/* Directional arrows along the loop */}
            {[45, 135, 225, 315].map(deg => {
              const rad = (deg * Math.PI) / 180;
              const x = cx + rx * Math.cos(rad);
              const y = cy + ry * Math.sin(rad);
              const tangent = deg + 90;
              return (
                <path key={deg} d={arrowPath([x, y], tangent, 4)}
                  stroke="var(--d-fg)" strokeWidth={0.75} fill="none" />
              );
            })}

            {/* Labels */}
            {labels.map(({ angle, text, bold }) => {
              const rad = (angle * Math.PI) / 180;
              const lx = cx + (rx + 28) * Math.cos(rad);
              const ly = cy + (ry + 20) * Math.sin(rad);
              return (
                <text key={text} x={lx} y={ly}
                  textAnchor="middle" dominantBaseline="central"
                  fontSize={9} fontWeight={bold ? 600 : 400}
                  fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
                  {text}
                </text>
              );
            })}

            {/* Center label */}
            <text x={cx} y={cy - 5} textAnchor="middle" fontSize={8} fontStyle="italic"
              fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
              closed
            </text>
            <text x={cx} y={cy + 7} textAnchor="middle" fontSize={8} fontStyle="italic"
              fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
              causal loop
            </text>
          </g>
        );
      })()}

      {/* Metric */}
      <text x={rightCx} y={228} textAnchor="middle" fontSize={10}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        ρ_sync = 0.21 (70×)
      </text>

      {/* Rungs unlocked */}
      <text x={rightCx} y={248} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Unlocked: + counterfactual, self-model,
      </text>
      <text x={rightCx} y={260} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        nascent affect dynamics
      </text>
      <text x={rightCx} y={278} textAnchor="middle" fontSize={9}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Rungs 1–9
      </text>
    </svg>
  );
}
