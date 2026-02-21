import { arrowPath } from './utils';

/**
 * Somatic Fear vs. Anticipatory Anxiety.
 *
 * Two structurally distinct fear types mapped to different
 * emergence ladder rungs, with developmental prediction.
 */
export default function SomaticAnticipatory() {
  const w = 460, h = 300;
  const leftCx = 130, rightCx = 330;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img"
      aria-label="Somatic fear requires rungs 1-3; anticipatory anxiety requires rung 8 and counterfactual capacity.">

      <text x={w / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Two Kinds of Fear
      </text>
      <text x={w / 2} y={38} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        structurally distinct, developmentally ordered
      </text>

      {/* ── LEFT: Somatic Fear ── */}
      <rect x={leftCx - 70} y={55} width={140} height={150} rx={6}
        fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />

      <text x={leftCx} y={72} textAnchor="middle" fontSize={11} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Somatic Fear
      </text>

      {[
        { label: 'V −', desc: 'gradient away from viability' },
        { label: 'A +', desc: 'high processing rate' },
        { label: 'CF = 0', desc: 'present-state only' },
      ].map((item, i) => (
        <g key={i}>
          <text x={leftCx - 55} y={98 + i * 22} fontSize={10} fontWeight={600}
            fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
            {item.label}
          </text>
          <text x={leftCx - 15} y={98 + i * 22} fontSize={8.5}
            fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
            {item.desc}
          </text>
        </g>
      ))}

      <text x={leftCx} y={175} textAnchor="middle" fontSize={9}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Rungs 1–3
      </text>
      <text x={leftCx} y={192} textAnchor="middle" fontSize={8} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        no counterfactual required
      </text>

      {/* ── RIGHT: Anticipatory Anxiety ── */}
      <rect x={rightCx - 70} y={55} width={140} height={150} rx={6}
        fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />

      <text x={rightCx} y={72} textAnchor="middle" fontSize={11} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Anticipatory Anxiety
      </text>

      {[
        { label: 'V −', desc: 'gradient away from viability' },
        { label: 'A +', desc: 'high processing rate' },
        { label: 'CF > 0', desc: 'non-actual possibilities' },
      ].map((item, i) => (
        <g key={i}>
          <text x={rightCx - 55} y={98 + i * 22} fontSize={10} fontWeight={600}
            fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
            {item.label}
          </text>
          <text x={rightCx - 15} y={98 + i * 22} fontSize={8.5}
            fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
            {item.desc}
          </text>
        </g>
      ))}

      <text x={rightCx} y={175} textAnchor="middle" fontSize={9}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Rung 8
      </text>
      <text x={rightCx} y={192} textAnchor="middle" fontSize={8} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        requires mental time travel
      </text>

      {/* ── Developmental Timeline ── */}
      <line x1={50} y1={230} x2={w - 50} y2={230}
        stroke="var(--d-fg)" strokeWidth={0.75} />
      <path d={arrowPath([w - 50, 230], 0, 5)}
        stroke="var(--d-fg)" strokeWidth={0.75} fill="none" />

      {/* Birth */}
      <line x1={70} y1={225} x2={70} y2={235} stroke="var(--d-fg)" strokeWidth={0.5} />
      <text x={70} y={245} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        birth
      </text>

      {/* Somatic fear range */}
      <line x1={100} y1={222} x2={200} y2={222}
        stroke="var(--d-fg)" strokeWidth={1.5} />
      <text x={150} y={218} textAnchor="middle" fontSize={7.5}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        somatic fear
      </text>

      {/* Threshold */}
      <line x1={260} y1={218} x2={260} y2={242} stroke="var(--d-fg)" strokeWidth={1}
        strokeDasharray="3,2" />
      <text x={260} y={256} textAnchor="middle" fontSize={8}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        ~3–4 years
      </text>
      <text x={260} y={268} textAnchor="middle" fontSize={7.5} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        false belief task
      </text>

      {/* Anticipatory anxiety range */}
      <line x1={280} y1={222} x2={380} y2={222}
        stroke="var(--d-fg)" strokeWidth={1.5} />
      <text x={330} y={218} textAnchor="middle" fontSize={7.5}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        anticipatory anxiety
      </text>

      {/* Age label */}
      <text x={w - 40} y={245} textAnchor="end" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        age →
      </text>

      {/* Prediction label */}
      <text x={w / 2} y={h - 6} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Prediction: anxiety onset co-occurs with false-belief task, not before
      </text>
    </svg>
  );
}
