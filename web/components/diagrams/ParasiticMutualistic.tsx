/**
 * Parasitic vs. Mutualistic Superorganisms.
 *
 * Shows viability manifold containment: when the god's manifold
 * is contained within substrate manifold (mutualistic) vs. extends
 * beyond it (parasitic).
 */
export default function ParasiticMutualistic() {
  const w = 460, h = 250;
  const leftCx = 130, rightCx = 330;
  const cy = 130;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img"
      aria-label="Mutualistic: god viability contained in substrate viability. Parasitic: god extends beyond substrate.">

      <text x={w / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Mutualistic vs. Parasitic
      </text>
      <text x={w / 2} y={38} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        viability manifold containment determines alignment
      </text>

      {/* Divider */}
      <line x1={w / 2} y1={50} x2={w / 2} y2={h - 10}
        stroke="var(--d-line)" strokeWidth={0.5} strokeDasharray="4,4" />

      {/* ── LEFT: Mutualistic ── */}
      <text x={leftCx} y={60} textAnchor="middle" fontSize={10} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Aligned
      </text>

      {/* V_substrate (large) */}
      <ellipse cx={leftCx} cy={cy} rx={85} ry={60}
        fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />
      <text x={leftCx - 50} y={cy - 48} fontSize={9}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        V_substrate
      </text>

      {/* V_god (contained) */}
      <ellipse cx={leftCx} cy={cy + 5} rx={40} ry={28}
        fill="var(--d-fg)" fillOpacity={0.05}
        stroke="var(--d-fg)" strokeWidth={0.75} strokeDasharray="3,2" />
      <text x={leftCx} y={cy + 7} textAnchor="middle" fontSize={9}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        V_god
      </text>

      {/* Gradient arrows (aligned) */}
      <line x1={leftCx - 20} y1={cy + 35} x2={leftCx - 20} y2={cy + 15}
        stroke="var(--d-fg)" strokeWidth={0.5} />
      <line x1={leftCx + 20} y1={cy + 35} x2={leftCx + 20} y2={cy + 15}
        stroke="var(--d-fg)" strokeWidth={0.5} />
      <text x={leftCx} y={cy + 46} textAnchor="middle" fontSize={7.5} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        gradients aligned
      </text>

      <text x={leftCx} y={cy + 76} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        god flourishes ⇒ substrate flourishes
      </text>

      {/* ── RIGHT: Parasitic ── */}
      <text x={rightCx} y={60} textAnchor="middle" fontSize={10} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Parasitic
      </text>

      {/* V_substrate */}
      <ellipse cx={rightCx - 15} cy={cy} rx={60} ry={55}
        fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />
      <text x={rightCx - 55} y={cy - 43} fontSize={9}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        V_substrate
      </text>

      {/* V_god (extends beyond) */}
      <ellipse cx={rightCx + 15} cy={cy} rx={70} ry={50}
        fill="var(--d-fg)" fillOpacity={0.04}
        stroke="var(--d-fg)" strokeWidth={0.75} strokeDasharray="3,2" />
      <text x={rightCx + 55} y={cy - 5} fontSize={9}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        V_god
      </text>

      {/* Non-overlapping region label */}
      <text x={rightCx + 70} y={cy + 15} textAnchor="middle" fontSize={7} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        god persists
      </text>
      <text x={rightCx + 70} y={cy + 25} textAnchor="middle" fontSize={7} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        while substrate
      </text>
      <text x={rightCx + 70} y={cy + 35} textAnchor="middle" fontSize={7} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        diminishes
      </text>

      {/* Diverging arrows */}
      <line x1={rightCx - 5} y1={cy + 30} x2={rightCx - 20} y2={cy + 15}
        stroke="var(--d-fg)" strokeWidth={0.5} />
      <line x1={rightCx + 5} y1={cy + 30} x2={rightCx + 20} y2={cy + 15}
        stroke="var(--d-fg)" strokeWidth={0.5} />
      <text x={rightCx} y={cy + 46} textAnchor="middle" fontSize={7.5} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        gradients diverge
      </text>

      <text x={rightCx} y={cy + 76} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        god flourishes ⇏ substrate flourishes
      </text>

      {/* Bottom */}
      <text x={w / 2} y={h - 8} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Diagnostic: does substrate flourishing correlate with the god's persistence?
      </text>
    </svg>
  );
}
