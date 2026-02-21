/**
 * Ordering Principle — Broader manifolds contain narrower ones.
 *
 * Two panels: safe containment (care ⊃ friendship ⊃ transaction)
 * vs. contamination (transaction swallows the rest).
 */
export default function OrderingPrinciple() {
  const w = 460, h = 260;
  const leftCx = 130, rightCx = 330;
  const cy = 145;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img"
      aria-label="Ordering principle: broader manifolds safely contain narrower ones. Inversion causes contamination.">

      <text x={w / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        The Ordering Principle
      </text>
      <text x={w / 2} y={38} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        broader manifolds can safely contain narrower ones, not vice versa
      </text>

      {/* Divider */}
      <line x1={w / 2} y1={50} x2={w / 2} y2={h - 10}
        stroke="var(--d-line)" strokeWidth={0.5} strokeDasharray="4,4" />

      {/* ── LEFT: Safe containment ── */}
      <text x={leftCx} y={62} textAnchor="middle" fontSize={10} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Safe Containment
      </text>

      {/* Nested ellipses */}
      <ellipse cx={leftCx} cy={cy} rx={95} ry={70}
        fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />
      <text x={leftCx - 60} y={cy - 55} fontSize={9}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Care
      </text>

      <ellipse cx={leftCx} cy={cy + 5} rx={65} ry={48}
        fill="none" stroke="var(--d-fg)" strokeWidth={0.75} strokeDasharray="3,2" />
      <text x={leftCx - 35} y={cy - 30} fontSize={9}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Friendship
      </text>

      <ellipse cx={leftCx} cy={cy + 10} rx={35} ry={25}
        fill="var(--d-fg)" fillOpacity={0.04}
        stroke="var(--d-fg)" strokeWidth={0.75} />
      <text x={leftCx} y={cy + 12} textAnchor="middle" fontSize={9}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Transaction
      </text>

      {/* Check mark */}
      <text x={leftCx} y={cy + 72} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        broader holds narrower
      </text>

      {/* ── RIGHT: Contamination ── */}
      <text x={rightCx} y={62} textAnchor="middle" fontSize={10} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Contamination
      </text>

      {/* Inverted: Transaction is largest */}
      <ellipse cx={rightCx} cy={cy} rx={95} ry={70}
        fill="var(--d-fg)" fillOpacity={0.04}
        stroke="var(--d-fg)" strokeWidth={0.75} />
      <text x={rightCx + 30} y={cy - 55} fontSize={9}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Transaction
      </text>

      {/* Friendship squeezed */}
      <ellipse cx={rightCx} cy={cy + 5} rx={55} ry={38}
        fill="none" stroke="var(--d-fg)" strokeWidth={0.75} strokeDasharray="3,2" />
      <text x={rightCx - 25} y={cy - 22} fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Friendship
      </text>

      {/* Care crushed */}
      <ellipse cx={rightCx} cy={cy + 10} rx={25} ry={18}
        fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />
      <text x={rightCx} y={cy + 12} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Care
      </text>

      {/* X mark */}
      <text x={rightCx} y={cy + 72} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        narrower swallows broader
      </text>

      {/* Bottom */}
      <text x={w / 2} y={h - 8} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        When the logic of transaction governs friendship, friendship dissolves
      </text>
    </svg>
  );
}
