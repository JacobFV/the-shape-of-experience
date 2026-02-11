import { type Point, arrowPath } from './utils';

const LEVELS = [
  { label: 'Phenomenal', color: 'var(--d-violet)' },
  { label: 'Psychological', color: 'var(--d-blue)' },
  { label: 'Biological', color: 'var(--d-cyan)' },
  { label: 'Chemical', color: 'var(--d-green)' },
  { label: 'Atomic', color: 'var(--d-yellow)' },
  { label: 'Subatomic', color: 'var(--d-orange)' },
  { label: 'Quantum Fields', color: 'var(--d-red)' },
  { label: '???', color: 'var(--d-muted)' },
];

/** Part 2-0: Levels of reality / reduction chain */
export default function ReductionChain() {
  const cx = 300;
  const startY = 42;
  const gap = 68;
  const boxW = 180, boxH = 40, boxR = 6;

  const levelY = (i: number) => startY + i * gap;

  return (
    <svg viewBox="0 0 520 600" className="diagram-svg" role="img"
      aria-label="Levels of reality from phenomenal to quantum fields, with 'reduces?' arrows between each level">
      {/* Levels */}
      {LEVELS.map((level, i) => {
        const y = levelY(i);
        const isDashed = i === LEVELS.length - 1;
        return (
          <g key={level.label}>
            <rect
              x={cx - boxW / 2} y={y - boxH / 2}
              width={boxW} height={boxH} rx={boxR}
              fill={level.color} fillOpacity={0.1}
              stroke={level.color} strokeWidth={0.75}
              strokeDasharray={isDashed ? '4,3' : 'none'}
            />
            <text
              x={cx} y={y}
              textAnchor="middle" dominantBaseline="central"
              fontSize={14} fill="var(--d-fg)"
              fontFamily="var(--font-body, Georgia, serif)"
              fontStyle={isDashed ? 'italic' : 'normal'}
            >
              {level.label}
            </text>
          </g>
        );
      })}

      {/* Arrows between levels */}
      {LEVELS.slice(0, -1).map((_, i) => {
        const fromY = levelY(i) + boxH / 2 + 3;
        const toY = levelY(i + 1) - boxH / 2 - 3;
        const isDashed = i === LEVELS.length - 2;
        return (
          <g key={`arrow-${i}`}>
            <line
              x1={cx} y1={fromY} x2={cx} y2={toY}
              stroke="var(--d-line)" strokeWidth={0.75}
              strokeDasharray={isDashed ? '3,3' : 'none'}
            />
            <path
              d={arrowPath([cx, toY], 90, 5)}
              stroke="var(--d-line)" strokeWidth={0.75} fill="none"
            />
            {/* "reduces?" label on first arrow only */}
            {i === 0 && (
              <text
                x={cx + boxW / 2 + 14} y={(fromY + toY) / 2}
                textAnchor="start" dominantBaseline="central"
                fontSize={11} fontStyle="italic" fill="var(--d-muted)"
                fontFamily="var(--font-body, Georgia, serif)"
              >
                reduces?
              </text>
            )}
          </g>
        );
      })}

      {/* Left brace spanning all but bottom level */}
      {(() => {
        const braceX = cx - boxW / 2 - 25;
        const braceTop = levelY(0) - boxH / 2 + 2;
        const braceBot = levelY(LEVELS.length - 2) + boxH / 2 - 2;
        const braceMid = (braceTop + braceBot) / 2;
        const indent = 10;
        return (
          <g>
            <path
              d={[
                `M ${braceX},${braceTop}`,
                `Q ${braceX - indent},${braceTop} ${braceX - indent},${braceTop + 15}`,
                `L ${braceX - indent},${braceMid - 8}`,
                `Q ${braceX - indent},${braceMid} ${braceX - indent * 2},${braceMid}`,
                `Q ${braceX - indent},${braceMid} ${braceX - indent},${braceMid + 8}`,
                `L ${braceX - indent},${braceBot - 15}`,
                `Q ${braceX - indent},${braceBot} ${braceX},${braceBot}`,
              ].join(' ')}
              fill="none" stroke="var(--d-line)" strokeWidth={0.75}
            />
            <text
              x={braceX - indent * 2 - 8} y={braceMid}
              textAnchor="end" dominantBaseline="central"
              fontSize={12} fontStyle="italic" fill="var(--d-muted)"
              fontFamily="var(--font-body, Georgia, serif)"
              transform={`rotate(-90, ${braceX - indent * 2 - 8}, ${braceMid})`}
            >
              equally real?
            </text>
          </g>
        );
      })()}
    </svg>
  );
}
