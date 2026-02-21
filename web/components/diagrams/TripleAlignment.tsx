import { arrowPath, type Point } from './utils';

/**
 * Triple Alignment Test — three measurement streams.
 *
 * Triangle with Structure, Signal, and Action at vertices,
 * pairwise RSA tests on edges, failure mode diagnostics below.
 */
export default function TripleAlignment() {
  const w = 420, h = 310;
  const cx = 210, cy = 130;
  const r = 80;

  // Triangle vertices (equilateral, pointing up)
  const verts: { pos: Point; label: string; sublabel: string }[] = [
    { pos: [cx, cy - r], label: 'Structure', sublabel: 'internal affect vector aᵢ' },
    { pos: [cx - r * 0.87, cy + r * 0.5], label: 'Signal', sublabel: 'VLM-translated eᵢ' },
    { pos: [cx + r * 0.87, cy + r * 0.5], label: 'Action', sublabel: 'behavioral vector bᵢ' },
  ];

  // Edge labels
  const edges = [
    { from: 0, to: 1, label: 'RSA(a, e)' },
    { from: 1, to: 2, label: 'RSA(e, b)' },
    { from: 0, to: 2, label: 'RSA(a, b)' },
  ];

  // Failure modes
  const failures = [
    { test: 'No alignment anywhere', meaning: 'affect structure not present' },
    { test: 'Structure–Action without Signal', meaning: 'communication channel broken' },
    { test: 'Signal–Action without Structure', meaning: 'behavioral mimicry without integration' },
  ];

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img"
      aria-label="Triple alignment test: structure, signal, and action must show pairwise RSA alignment.">

      <text x={cx} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        The Triple Alignment Test
      </text>
      <text x={cx} y={38} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        three measurement streams, pairwise RSA
      </text>

      {/* Triangle edges */}
      {edges.map(({ from, to, label }, i) => {
        const [x1, y1] = verts[from].pos;
        const [x2, y2] = verts[to].pos;
        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        // Offset label outward from center
        const ox = (mx - cx) * 0.35;
        const oy = (my - cy) * 0.35;
        return (
          <g key={i}>
            <line x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="var(--d-fg)" strokeWidth={0.75} />
            <text x={mx + ox} y={my + oy} textAnchor="middle" dominantBaseline="central"
              fontSize={8} fontStyle="italic" fill="var(--d-muted)"
              fontFamily="var(--font-body, Georgia, serif)">
              {label}
            </text>
          </g>
        );
      })}

      {/* Vertices */}
      {verts.map(({ pos: [x, y], label, sublabel }, i) => {
        const isTop = i === 0;
        const ly = isTop ? y - 18 : y + 18;
        const sly = isTop ? y - 6 : y + 30;
        return (
          <g key={label}>
            <circle cx={x} cy={y} r={6}
              fill="var(--d-fg)" fillOpacity={0.08}
              stroke="var(--d-fg)" strokeWidth={0.75} />
            <text x={x} y={ly} textAnchor="middle" fontSize={10} fontWeight={600}
              fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              {label}
            </text>
            <text x={x} y={sly} textAnchor="middle" fontSize={8}
              fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
              {sublabel}
            </text>
          </g>
        );
      })}

      {/* Failure mode diagnostics */}
      <line x1={50} y1={222} x2={w - 50} y2={222}
        stroke="var(--d-line)" strokeWidth={0.3} />
      <text x={cx} y={238} textAnchor="middle" fontSize={9} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Failure Mode Diagnostics
      </text>

      {failures.map((f, i) => (
        <g key={i}>
          <text x={55} y={258 + i * 16} fontSize={8.5}
            fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
            {f.test}
          </text>
          <text x={w - 55} y={258 + i * 16} textAnchor="end" fontSize={8} fontStyle="italic"
            fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
            → {f.meaning}
          </text>
        </g>
      ))}
    </svg>
  );
}
