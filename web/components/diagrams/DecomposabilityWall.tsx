import { arrowPath } from './utils';

/**
 * Decomposability Wall — Linear head vs. MLP head.
 *
 * Shows why gradient coupling through composition breaks
 * the decomposability wall that linear prediction heads cannot cross.
 */
export default function DecomposabilityWall() {
  const w = 520, h = 310;
  const leftCx = 140, rightCx = 380;

  const hUnits = 4; // hidden units
  const tUnits = 3; // targets
  const unitR = 8;
  const hY = 80, interY = 160, tY = 240;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img"
      aria-label="Linear head allows decomposable channels; MLP head forces gradient coupling through composition.">

      <text x={w / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        The Decomposability Wall
      </text>
      <text x={w / 2} y={38} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        why architecture determines integration
      </text>

      {/* Divider */}
      <line x1={w / 2} y1={50} x2={w / 2} y2={h - 10}
        stroke="var(--d-line)" strokeWidth={0.5} strokeDasharray="4,4" />

      {/* ── LEFT: Linear Head ── */}
      <text x={leftCx} y={58} textAnchor="middle" fontSize={11} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Linear Head
      </text>

      {/* Hidden units */}
      {Array.from({ length: hUnits }).map((_, i) => {
        const x = leftCx - 45 + i * 30;
        return (
          <g key={`lh-${i}`}>
            <circle cx={x} cy={hY} r={unitR}
              fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />
            <text x={x} y={hY} textAnchor="middle" dominantBaseline="central"
              fontSize={8} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              h{i + 1}
            </text>
          </g>
        );
      })}

      {/* Target units */}
      {Array.from({ length: tUnits }).map((_, j) => {
        const tx = leftCx - 30 + j * 30;
        return (
          <g key={`lt-${j}`}>
            <circle cx={tx} cy={tY} r={unitR}
              fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />
            <text x={tx} y={tY} textAnchor="middle" dominantBaseline="central"
              fontSize={8} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              t{j + 1}
            </text>
          </g>
        );
      })}

      {/* Sparse independent connections (subset can satisfy) */}
      {[[0, 0], [1, 0], [1, 1], [2, 1], [2, 2], [3, 2]].map(([hi, tj], k) => {
        const x1 = leftCx - 45 + hi * 30;
        const x2 = leftCx - 30 + tj * 30;
        return (
          <line key={`ll-${k}`} x1={x1} y1={hY + unitR + 2} x2={x2} y2={tY - unitR - 2}
            stroke="var(--d-line)" strokeWidth={0.5} />
        );
      })}

      {/* Labels */}
      <text x={leftCx} y={hY - 18} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        hidden state H
      </text>
      <text x={leftCx} y={tY + 22} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        prediction targets T
      </text>

      <text x={leftCx} y={interY} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        subset can satisfy
      </text>
      <text x={leftCx} y={interY + 13} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        → decomposable
      </text>

      {/* Phi label */}
      <text x={leftCx} y={tY + 42} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Φ ≈ 0.08
      </text>

      {/* ── RIGHT: MLP Head ── */}
      <text x={rightCx} y={58} textAnchor="middle" fontSize={11} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        MLP Head (2-layer)
      </text>

      {/* Hidden units */}
      {Array.from({ length: hUnits }).map((_, i) => {
        const x = rightCx - 45 + i * 30;
        return (
          <g key={`rh-${i}`}>
            <circle cx={x} cy={hY} r={unitR}
              fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />
            <text x={x} y={hY} textAnchor="middle" dominantBaseline="central"
              fontSize={8} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              h{i + 1}
            </text>
          </g>
        );
      })}

      {/* Intermediate layer (narrow) */}
      {Array.from({ length: 2 }).map((_, i) => {
        const x = rightCx - 15 + i * 30;
        return (
          <g key={`ri-${i}`}>
            <circle cx={x} cy={interY} r={unitR}
              fill="var(--d-fg)" fillOpacity={0.08}
              stroke="var(--d-fg)" strokeWidth={0.75} />
          </g>
        );
      })}

      {/* H → intermediate: full connections */}
      {Array.from({ length: hUnits }).map((_, hi) =>
        Array.from({ length: 2 }).map((_, ii) => {
          const x1 = rightCx - 45 + hi * 30;
          const x2 = rightCx - 15 + ii * 30;
          return (
            <line key={`rhi-${hi}-${ii}`}
              x1={x1} y1={hY + unitR + 2} x2={x2} y2={interY - unitR - 2}
              stroke="var(--d-line)" strokeWidth={0.4} />
          );
        })
      )}

      {/* Target units */}
      {Array.from({ length: tUnits }).map((_, j) => {
        const tx = rightCx - 30 + j * 30;
        return (
          <g key={`rt-${j}`}>
            <circle cx={tx} cy={tY} r={unitR}
              fill="none" stroke="var(--d-fg)" strokeWidth={0.75} />
            <text x={tx} y={tY} textAnchor="middle" dominantBaseline="central"
              fontSize={8} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              t{j + 1}
            </text>
          </g>
        );
      })}

      {/* Intermediate → T: full connections */}
      {Array.from({ length: 2 }).map((_, ii) =>
        Array.from({ length: tUnits }).map((_, tj) => {
          const x1 = rightCx - 15 + ii * 30;
          const x2 = rightCx - 30 + tj * 30;
          return (
            <line key={`rit-${ii}-${tj}`}
              x1={x1} y1={interY + unitR + 2} x2={x2} y2={tY - unitR - 2}
              stroke="var(--d-line)" strokeWidth={0.4} />
          );
        })
      )}

      {/* Labels */}
      <text x={rightCx} y={hY - 18} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        hidden state H
      </text>
      <text x={rightCx + 50} y={interY} textAnchor="start" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        bottleneck H/2
      </text>
      <text x={rightCx} y={tY + 22} textAnchor="middle" fontSize={8}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        prediction targets T
      </text>

      {/* Gradient coupling annotation */}
      <text x={rightCx - 70} y={interY - 3} textAnchor="end" fontSize={9} fontStyle="italic"
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        gradient coupling
      </text>
      <text x={rightCx - 70} y={interY + 10} textAnchor="end" fontSize={9} fontStyle="italic"
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        through composition
      </text>

      {/* Phi label */}
      <text x={rightCx} y={tY + 42} textAnchor="middle" fontSize={10}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Φ ≈ 0.25
      </text>

      {/* Bottom annotation */}
      <text x={w / 2} y={h - 6} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Not nonlinearity, not bottleneck width — gradient coupling through composition (V28)
      </text>
    </svg>
  );
}
