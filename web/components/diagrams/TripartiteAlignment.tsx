import { type Point, off, mid, angleBetween, arrowPath, pt } from './utils';

/** Part 1-5: Three-way alignment between affect structure, signal, and behavior */
export default function TripartiteAlignment() {
  const y = 150;
  const boxes: { center: Point; label: string; color: string }[] = [
    { center: [130, y], label: 'Affect Structure', color: 'var(--d-red)' },
    { center: [350, y], label: 'Translated Signal', color: 'var(--d-blue)' },
    { center: [570, y], label: 'Observable Behavior', color: 'var(--d-green)' },
  ];
  const w = 130, h = 50, r = 8;

  const edge = (a: Point, b: Point, dy: number = 0) => {
    // Horizontal arrow between box edges
    const ax = a[0] + w / 2 + 8;
    const bx = b[0] - w / 2 - 8;
    const from: Point = [ax, a[1] + dy];
    const to: Point = [bx, b[1] + dy];
    const dir = angleBetween(from, to);
    const rDir = angleBetween(to, from);
    return (
      <g key={`${a[0]}-${b[0]}-${dy}`}>
        <line
          x1={from[0]} y1={from[1]} x2={to[0]} y2={to[1]}
          stroke="var(--d-line)" strokeWidth={0.75}
        />
        <path d={arrowPath(to, dir, 6)} stroke="var(--d-line)" strokeWidth={0.75} fill="none" />
        <path d={arrowPath(from, rDir, 6)} stroke="var(--d-line)" strokeWidth={0.75} fill="none" />
      </g>
    );
  };

  // Arc arrow between first and last (spanning over middle)
  const arcFrom: Point = [boxes[0].center[0] + 20, boxes[0].center[1] - h / 2 - 6];
  const arcTo: Point = [boxes[2].center[0] - 20, boxes[2].center[1] - h / 2 - 6];
  const arcMid: Point = [350, y - h / 2 - 55];

  return (
    <svg viewBox="0 0 700 280" className="diagram-svg" role="img" aria-label="Tripartite alignment: affect structure, translated signal, and observable behavior should all align">
      {/* Boxes */}
      {boxes.map(({ center, label, color }) => (
        <g key={label}>
          <rect
            x={center[0] - w / 2} y={center[1] - h / 2}
            width={w} height={h} rx={r}
            fill={color} fillOpacity={0.1}
            stroke={color} strokeWidth={0.75}
          />
          <text
            x={center[0]} y={center[1]}
            textAnchor="middle" dominantBaseline="central"
            fontSize={13} fontFamily="var(--font-body, Georgia, serif)"
            fill="var(--d-fg)"
          >
            {label}
          </text>
        </g>
      ))}

      {/* Horizontal bidirectional arrows */}
      {edge(boxes[0].center, boxes[1].center)}
      {edge(boxes[1].center, boxes[2].center)}

      {/* Arc arrow between first and last */}
      <path
        d={`M ${pt(arcFrom)} Q ${pt(arcMid)} ${pt(arcTo)}`}
        fill="none" stroke="var(--d-line)" strokeWidth={0.75}
      />
      <path
        d={arrowPath(arcTo, angleBetween(arcMid, arcTo), 6)}
        stroke="var(--d-line)" strokeWidth={0.75} fill="none"
      />
      <path
        d={arrowPath(arcFrom, angleBetween(arcMid, arcFrom), 6)}
        stroke="var(--d-line)" strokeWidth={0.75} fill="none"
      />

      {/* Caption */}
      <text
        x={350} y={250}
        textAnchor="middle" fontSize={14} fontStyle="italic"
        fill="var(--d-muted)"
        fontFamily="var(--font-body, Georgia, serif)"
      >
        All three should align
      </text>
    </svg>
  );
}
