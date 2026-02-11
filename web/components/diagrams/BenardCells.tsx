import { type Point, arrowPath, polar, pt } from './utils';

/** Part 1-1: Bénard convection cells showing spontaneous structure */
export default function BenardCells() {
  const left = 80, right = 620, top = 60, bottom = 310;
  const containerW = right - left;
  const containerH = bottom - top;

  // Three convection cells, evenly spaced
  const cellCenters: Point[] = [
    [left + containerW / 6, (top + bottom) / 2],
    [left + containerW / 2, (top + bottom) / 2],
    [left + 5 * containerW / 6, (top + bottom) / 2],
  ];
  const cellRx = containerW / 6 - 15;
  const cellRy = containerH / 2 - 30;

  // Circulation arrow (nearly complete ellipse)
  const circulationPath = (cx: number, cy: number, clockwise: boolean) => {
    const rx = cellRx * 0.75;
    const ry = cellRy * 0.7;
    // Start from top of ellipse, go around, leaving a gap for arrowhead
    const sweep = clockwise ? 1 : 0;
    const startAngle = clockwise ? -95 : -85;
    const endAngle = clockwise ? -120 : -60;
    const start = polar([cx, cy], 1, startAngle);
    const sX = cx + rx * Math.cos(startAngle * Math.PI / 180);
    const sY = cy + ry * Math.sin(startAngle * Math.PI / 180);
    const eX = cx + rx * Math.cos(endAngle * Math.PI / 180);
    const eY = cy + ry * Math.sin(endAngle * Math.PI / 180);
    return {
      path: `M ${sX.toFixed(1)},${sY.toFixed(1)} A ${rx},${ry} 0 1,${sweep} ${eX.toFixed(1)},${eY.toFixed(1)}`,
      tipX: eX,
      tipY: eY,
      // Tangent direction at endpoint for arrowhead
      tipAngle: clockwise ? endAngle - 90 : endAngle + 90,
    };
  };

  // Heat source wavy lines
  const heatWaves = (count: number) => {
    const waves = [];
    for (let i = 0; i < count; i++) {
      const x = left + 30 + (i * (containerW - 60)) / (count - 1);
      waves.push(
        <path
          key={i}
          d={`M ${x},${bottom + 15} q -4,-6 0,-12 q 4,-6 0,-12`}
          fill="none" stroke="var(--d-red)" strokeWidth={0.75} opacity={0.5}
        />
      );
    }
    return waves;
  };

  return (
    <svg viewBox="0 0 700 400" className="diagram-svg" role="img"
      aria-label="Bénard convection cells: three hexagonal circulation patterns between hot and cool surfaces, demonstrating spontaneous structure formation">

      {/* Container rectangle */}
      <rect x={left} y={top} width={containerW} height={containerH}
        fill="none" stroke="var(--d-line)" strokeWidth={1} />

      {/* Temperature gradient fill */}
      <defs>
        <linearGradient id="temp-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--d-blue)" stopOpacity={0.05} />
          <stop offset="100%" stopColor="var(--d-red)" stopOpacity={0.1} />
        </linearGradient>
      </defs>
      <rect x={left + 1} y={top + 1}
        width={containerW - 2} height={containerH - 2}
        fill="url(#temp-grad)" />

      {/* Top surface label */}
      <text x={(left + right) / 2} y={top - 12}
        textAnchor="middle" fontSize={12} fill="var(--d-blue)"
        fontFamily="var(--font-body, Georgia, serif)">
        cool surface
      </text>

      {/* Bottom surface label */}
      <text x={(left + right) / 2} y={bottom + 35}
        textAnchor="middle" fontSize={12} fill="var(--d-red)"
        fontFamily="var(--font-body, Georgia, serif)">
        hot surface
      </text>

      {/* Heat source waves */}
      {heatWaves(8)}

      {/* Cell divider lines (thin) */}
      <line x1={left + containerW / 3} y1={top} x2={left + containerW / 3} y2={bottom}
        stroke="var(--d-line)" strokeWidth={0.3} strokeDasharray="4,4" opacity={0.3} />
      <line x1={left + 2 * containerW / 3} y1={top} x2={left + 2 * containerW / 3} y2={bottom}
        stroke="var(--d-line)" strokeWidth={0.3} strokeDasharray="4,4" opacity={0.3} />

      {/* Circulation arrows */}
      {cellCenters.map((center, i) => {
        const clockwise = i % 2 === 0;
        const circ = circulationPath(center[0], center[1], clockwise);
        return (
          <g key={i}>
            <path d={circ.path}
              fill="none" stroke="var(--d-blue)" strokeWidth={1} />
            <path d={arrowPath([circ.tipX, circ.tipY], circ.tipAngle, 6)}
              stroke="var(--d-blue)" strokeWidth={1} fill="none" />
          </g>
        );
      })}

      {/* Vertical flow indicators */}
      {cellCenters.map((center, i) => {
        const clockwise = i % 2 === 0;
        return (
          <g key={`flow-${i}`}>
            {/* Upward arrow in center */}
            <line x1={center[0]} y1={center[1] + cellRy * 0.4}
              x2={center[0]} y2={center[1] - cellRy * 0.4}
              stroke="var(--d-red)" strokeWidth={0.6} opacity={0.4} />
            <path d={arrowPath([center[0], center[1] - cellRy * 0.4], -90, 4)}
              stroke="var(--d-red)" strokeWidth={0.6} fill="none" opacity={0.4} />
            <text x={center[0]} y={center[1] + cellRy * 0.65}
              textAnchor="middle" fontSize={9} fill="var(--d-muted)" opacity={0.6}
              fontFamily="var(--font-body, Georgia, serif)">
              {clockwise ? 'CW' : 'CCW'}
            </text>
          </g>
        );
      })}

      {/* Title annotation */}
      <text x={(left + right) / 2} y={bottom + 60}
        textAnchor="middle" fontSize={12} fontStyle="italic" fill="var(--d-muted)"
        fontFamily="var(--font-body, Georgia, serif)">
        spontaneous convection above critical Rayleigh number
      </text>
    </svg>
  );
}
