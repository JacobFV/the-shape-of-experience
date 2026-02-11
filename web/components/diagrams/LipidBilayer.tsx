import { type Point, polar, arrowPath, pt } from './utils';

/** A single phospholipid molecule: circle head + two wavy tails */
function Molecule({ x, y, angle = 90, headColor = 'var(--d-blue)', tailColor = 'var(--d-yellow)' }: {
  x: number; y: number; angle?: number; headColor?: string; tailColor?: string;
}) {
  const rad = (angle * Math.PI) / 180;
  const headR = 4.5;
  // Head position
  const hx = x;
  const hy = y;
  // Tail direction (opposite of head facing)
  const dx = Math.cos(rad);
  const dy = Math.sin(rad);
  const tailLen = 18;
  const wobble = 3;

  // Two tails offset slightly from center
  const tail = (offsetPerp: number) => {
    const px = -dy * offsetPerp; // perpendicular offset
    const py = dx * offsetPerp;
    const sx = hx + dx * (headR + 1) + px;
    const sy = hy + dy * (headR + 1) + py;
    const mx = sx + dx * tailLen * 0.5 + wobble * -dy;
    const my = sy + dy * tailLen * 0.5 + wobble * dx;
    const ex = sx + dx * tailLen;
    const ey = sy + dy * tailLen;
    return `M ${sx.toFixed(1)},${sy.toFixed(1)} Q ${mx.toFixed(1)},${my.toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}`;
  };

  return (
    <g>
      <path d={tail(-1.5)} fill="none" stroke={tailColor} strokeWidth={1.2} strokeLinecap="round" />
      <path d={tail(1.5)} fill="none" stroke={tailColor} strokeWidth={1.2} strokeLinecap="round" />
      <circle cx={hx} cy={hy} r={headR} fill={headColor} fillOpacity={0.8} stroke={headColor} strokeWidth={0.5} />
    </g>
  );
}

/** Part 1-3: Lipid bilayer self-assembly in 3 stages */
export default function LipidBilayer() {
  const stageW = 200;
  const stageGap = 80;
  const stageY = 180;

  // Stage centers
  const s1x = 120;  // dispersed
  const s2x = s1x + stageW + stageGap; // micelle
  const s3x = s2x + stageW + stageGap; // bilayer

  // Stage 1: Dispersed molecules (random positions and angles)
  const dispersed: { x: number; y: number; angle: number }[] = [
    { x: s1x - 60, y: stageY - 50, angle: 45 },
    { x: s1x + 20, y: stageY - 70, angle: 160 },
    { x: s1x + 70, y: stageY - 30, angle: 250 },
    { x: s1x - 40, y: stageY + 10, angle: 310 },
    { x: s1x + 50, y: stageY + 30, angle: 120 },
    { x: s1x - 20, y: stageY + 60, angle: 200 },
    { x: s1x + 30, y: stageY + 70, angle: 70 },
    { x: s1x - 70, y: stageY + 40, angle: 340 },
  ];

  // Stage 2: Micelle (circular arrangement, heads out, tails in)
  const micelleR = 40;
  const micelleCount = 10;
  const micelle = Array.from({ length: micelleCount }, (_, i) => {
    const a = (i * 360) / micelleCount - 90;
    const pos = polar([s2x, stageY], micelleR, a);
    return { x: pos[0], y: pos[1], angle: a + 180 }; // tails point inward
  });

  // Stage 3: Bilayer (two rows)
  const bilayerCount = 7;
  const bilayerSpacing = 22;
  const bilayerGap = 38; // gap between leaflets
  const upperY = stageY - bilayerGap / 2;
  const lowerY = stageY + bilayerGap / 2;
  const bilayerStartX = s3x - ((bilayerCount - 1) * bilayerSpacing) / 2;

  return (
    <svg viewBox="0 0 900 380" className="diagram-svg" role="img"
      aria-label="Lipid bilayer self-assembly: dispersed phospholipids form micelles, then bilayer membranes, driven by free energy reduction">

      {/* Stage labels */}
      <text x={s1x} y={40} textAnchor="middle" fontSize={13}
        fontFamily="var(--font-body, Georgia, serif)" fill="var(--d-fg)">
        Dispersed
      </text>
      <text x={s2x} y={40} textAnchor="middle" fontSize={13}
        fontFamily="var(--font-body, Georgia, serif)" fill="var(--d-fg)">
        Micelle
      </text>
      <text x={s3x} y={40} textAnchor="middle" fontSize={13}
        fontFamily="var(--font-body, Georgia, serif)" fill="var(--d-fg)">
        Bilayer
      </text>

      {/* Water background indicators */}
      <text x={s1x} y={stageY + 110} textAnchor="middle" fontSize={10}
        fill="var(--d-cyan)" opacity={0.4}
        fontFamily="var(--font-body, Georgia, serif)">
        water
      </text>

      {/* Stage 1: Dispersed */}
      {dispersed.map((m, i) => (
        <Molecule key={`d-${i}`} x={m.x} y={m.y} angle={m.angle} />
      ))}

      {/* Arrow 1→2 */}
      {(() => {
        const ax = s1x + stageW / 2 + 10;
        const bx = s2x - stageW / 2 - 10;
        const ay = stageY;
        return (
          <g>
            <line x1={ax} y1={ay} x2={bx} y2={ay}
              stroke="var(--d-line)" strokeWidth={0.75} />
            <path d={arrowPath([bx, ay], 0, 6)}
              stroke="var(--d-line)" strokeWidth={0.75} fill="none" />
            <text x={(ax + bx) / 2} y={ay - 12}
              textAnchor="middle" fontSize={11} fontStyle="italic"
              fill="var(--d-muted)"
              fontFamily="var(--font-body, Georgia, serif)">
              ΔG &lt; 0
            </text>
          </g>
        );
      })()}

      {/* Stage 2: Micelle */}
      {/* Micelle outline */}
      <circle cx={s2x} cy={stageY} r={micelleR + 12}
        fill="none" stroke="var(--d-line)" strokeWidth={0.3}
        strokeDasharray="3,3" opacity={0.3} />
      {micelle.map((m, i) => (
        <Molecule key={`m-${i}`} x={m.x} y={m.y} angle={m.angle} />
      ))}

      {/* Arrow 2→3 */}
      {(() => {
        const ax = s2x + stageW / 2 + 10;
        const bx = s3x - stageW / 2 - 10;
        const ay = stageY;
        return (
          <g>
            <line x1={ax} y1={ay} x2={bx} y2={ay}
              stroke="var(--d-line)" strokeWidth={0.75} />
            <path d={arrowPath([bx, ay], 0, 6)}
              stroke="var(--d-line)" strokeWidth={0.75} fill="none" />
            <text x={(ax + bx) / 2} y={ay - 12}
              textAnchor="middle" fontSize={11} fontStyle="italic"
              fill="var(--d-muted)"
              fontFamily="var(--font-body, Georgia, serif)">
              c &gt; c<tspan dy={3} fontSize={9}>crit</tspan>
            </text>
          </g>
        );
      })()}

      {/* Stage 3: Bilayer */}
      {/* Upper leaflet (heads up, tails down) */}
      {Array.from({ length: bilayerCount }, (_, i) => (
        <Molecule key={`bu-${i}`}
          x={bilayerStartX + i * bilayerSpacing} y={upperY}
          angle={-90}
        />
      ))}
      {/* Lower leaflet (heads down, tails up) */}
      {Array.from({ length: bilayerCount }, (_, i) => (
        <Molecule key={`bl-${i}`}
          x={bilayerStartX + i * bilayerSpacing} y={lowerY}
          angle={90}
        />
      ))}
      {/* Labels */}
      <text x={s3x + stageW / 2 - 10} y={upperY - 25}
        textAnchor="start" fontSize={10} fill="var(--d-muted)"
        fontFamily="var(--font-body, Georgia, serif)">
        outside
      </text>
      <text x={s3x + stageW / 2 - 10} y={stageY + 4}
        textAnchor="start" fontSize={10} fill="var(--d-muted)"
        fontFamily="var(--font-body, Georgia, serif)">
        inside
      </text>
      <text x={s3x + stageW / 2 - 10} y={lowerY + 30}
        textAnchor="start" fontSize={10} fill="var(--d-muted)"
        fontFamily="var(--font-body, Georgia, serif)">
        outside
      </text>

      {/* Legend */}
      <g transform="translate(60, 340)">
        <circle cx={0} cy={0} r={4} fill="var(--d-blue)" fillOpacity={0.8} />
        <text x={10} y={1} textAnchor="start" dominantBaseline="central"
          fontSize={10} fill="var(--d-muted)"
          fontFamily="var(--font-body, Georgia, serif)">
          hydrophilic head
        </text>
        <line x1={100} y1={0} x2={120} y2={0}
          stroke="var(--d-yellow)" strokeWidth={1.5} />
        <text x={128} y={1} textAnchor="start" dominantBaseline="central"
          fontSize={10} fill="var(--d-muted)"
          fontFamily="var(--font-body, Georgia, serif)">
          hydrophobic tails
        </text>
      </g>
    </svg>
  );
}
