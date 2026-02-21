import { arrowPath, type Point, pt } from './utils';

/**
 * Reactivity vs Understanding — the rung 7/8 transition.
 *
 * Left: parallel decomposable channels (reactivity).
 * Right: coupled non-decomposable comparison (understanding).
 */
export default function ReactivityUnderstanding() {
  const w = 520, h = 340;
  const leftCx = 140, rightCx = 380;
  const topY = 70, midY = 170, botY = 270;

  const features = ['s₁', 's₂', 's₃'];
  const actions = ['a₁', 'a₂', 'a₃'];
  const gap = 50;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img"
      aria-label="Reactivity: parallel decomposable channels. Understanding: coupled non-decomposable comparison.">

      {/* Title */}
      <text x={w / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Reactivity vs. Understanding
      </text>
      <text x={w / 2} y={38} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        the rung 7 → 8 transition
      </text>

      {/* Dividing line */}
      <line x1={w / 2} y1={55} x2={w / 2} y2={h - 10}
        stroke="var(--d-line)" strokeWidth={0.5} strokeDasharray="4,4" />

      {/* ── LEFT: Reactivity ── */}
      <text x={leftCx} y={topY - 12} textAnchor="middle" fontSize={11} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Reactivity
      </text>
      <text x={leftCx} y={topY + 2} textAnchor="middle" fontSize={9}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)" fontStyle="italic">
        partitionable
      </text>

      {/* Present State boxes */}
      <text x={leftCx} y={topY + 22} textAnchor="middle" fontSize={9}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Present State
      </text>

      {features.map((f, i) => {
        const x = leftCx - gap + i * gap;
        const y = topY + 35;
        return (
          <g key={`left-feat-${i}`}>
            <rect x={x - 16} y={y} width={32} height={24} rx={4}
              fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />
            <text x={x} y={y + 12} textAnchor="middle" dominantBaseline="central"
              fontSize={11} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              {f}
            </text>
          </g>
        );
      })}

      {/* Parallel arrows — independent channels */}
      {[0, 1, 2].map(i => {
        const x = leftCx - gap + i * gap;
        const y1 = topY + 62;
        const y2 = botY - 42;
        return (
          <g key={`left-arrow-${i}`}>
            <line x1={x} y1={y1} x2={x} y2={y2}
              stroke="var(--d-line)" strokeWidth={0.75} />
            <path d={arrowPath([x, y2], 90, 5)}
              stroke="var(--d-line)" strokeWidth={0.75} fill="none" />
          </g>
        );
      })}

      {/* Action boxes */}
      <text x={leftCx} y={botY - 15} textAnchor="middle" fontSize={9}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Action
      </text>

      {actions.map((a, i) => {
        const x = leftCx - gap + i * gap;
        const y = botY - 2;
        return (
          <g key={`left-act-${i}`}>
            <rect x={x - 16} y={y} width={32} height={24} rx={4}
              fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />
            <text x={x} y={y + 12} textAnchor="middle" dominantBaseline="central"
              fontSize={11} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              {a}
            </text>
          </g>
        );
      })}

      {/* Label: channels are separable */}
      <text x={leftCx + gap + 28} y={midY} textAnchor="start" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)" fontStyle="italic">
        each channel
      </text>
      <text x={leftCx + gap + 28} y={midY + 11} textAnchor="start" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)" fontStyle="italic">
        independent
      </text>

      {/* ── RIGHT: Understanding ── */}
      <text x={rightCx} y={topY - 12} textAnchor="middle" fontSize={11} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Understanding
      </text>
      <text x={rightCx} y={topY + 2} textAnchor="middle" fontSize={9}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)" fontStyle="italic">
        non-decomposable
      </text>

      {/* Present State boxes */}
      <text x={rightCx} y={topY + 22} textAnchor="middle" fontSize={9}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Present State
      </text>

      {features.map((f, i) => {
        const x = rightCx - gap + i * gap;
        const y = topY + 35;
        return (
          <g key={`right-feat-${i}`}>
            <rect x={x - 16} y={y} width={32} height={24} rx={4}
              fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />
            <text x={x} y={y + 12} textAnchor="middle" dominantBaseline="central"
              fontSize={11} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              {f}
            </text>
          </g>
        );
      })}

      {/* Converging arrows into coupled region */}
      {[0, 1, 2].map(i => {
        const x1 = rightCx - gap + i * gap;
        const y1 = topY + 62;
        const x2 = rightCx;
        const y2 = midY - 18;
        return (
          <g key={`right-conv-${i}`}>
            <line x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="var(--d-line)" strokeWidth={0.75} />
            <path d={arrowPath([x2, y2], Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI, 4)}
              stroke="var(--d-line)" strokeWidth={0.75} fill="none" />
          </g>
        );
      })}

      {/* Central coupled region */}
      <rect x={rightCx - 55} y={midY - 18} width={110} height={50} rx={8}
        fill="var(--d-fg)" fillOpacity={0.05}
        stroke="var(--d-fg)" strokeWidth={1} />
      <text x={rightCx} y={midY + 2} textAnchor="middle" dominantBaseline="central"
        fontSize={10} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Possibility
      </text>
      <text x={rightCx} y={midY + 15} textAnchor="middle" dominantBaseline="central"
        fontSize={10} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        comparison
      </text>

      {/* Cross-connections inside the box */}
      {[[rightCx - 30, midY - 8, rightCx + 30, midY + 22],
        [rightCx - 30, midY + 22, rightCx + 30, midY - 8],
        [rightCx - 30, midY + 7, rightCx + 30, midY + 7],
      ].map(([x1, y1, x2, y2], i) => (
        <line key={`cross-${i}`} x1={x1} y1={y1} x2={x2} y2={y2}
          stroke="var(--d-line)" strokeWidth={0.3} opacity={0.4} />
      ))}

      {/* Output arrow */}
      <line x1={rightCx} y1={midY + 32} x2={rightCx} y2={botY - 42}
        stroke="var(--d-line)" strokeWidth={0.75} />
      <path d={arrowPath([rightCx, botY - 42], 90, 5)}
        stroke="var(--d-line)" strokeWidth={0.75} fill="none" />

      {/* Action box */}
      <text x={rightCx} y={botY - 15} textAnchor="middle" fontSize={9}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Action
      </text>
      <rect x={rightCx - 30} y={botY - 2} width={60} height={24} rx={4}
        fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />
      <text x={rightCx} y={botY + 10} textAnchor="middle" dominantBaseline="central"
        fontSize={11} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        a
      </text>

      {/* Bottom labels */}
      <text x={leftCx} y={h - 10} textAnchor="middle" fontSize={9}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Rungs 1–7
      </text>
      <text x={rightCx} y={h - 10} textAnchor="middle" fontSize={9}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Rung 8+
      </text>
    </svg>
  );
}
