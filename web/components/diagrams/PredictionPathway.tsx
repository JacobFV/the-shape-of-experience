import { arrowPath } from './utils';

/**
 * Prediction-to-Integration Pathway (V22 → V27).
 *
 * Shows four experiments in sequence: prediction accuracy alone
 * doesn't produce integration. Only architectural coupling does.
 */
export default function PredictionPathway() {
  const w = 520, h = 260;

  const boxes = [
    { label: 'V22', desc: '1 target\n1 step', phi: '0.097', phiDir: '→', accuracy: '✓' },
    { label: 'V23', desc: '3 targets\n1 step', phi: '0.079', phiDir: '↓', accuracy: '✓' },
    { label: 'V24', desc: '1 target\nmulti-step', phi: 'mixed', phiDir: '→', accuracy: '✓' },
    { label: 'V27', desc: 'MLP head\n(2-layer)', phi: '0.245', phiDir: '↑', accuracy: '✓' },
  ];

  const boxW = 90, boxH = 80;
  const startX = 45;
  const spacing = (w - 2 * startX - boxW) / (boxes.length - 1);
  const topY = 60;

  // Wall position between V24 and V27
  const wallX = startX + 2.5 * spacing + boxW / 2;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="diagram-svg" role="img"
      aria-label="Prediction pathway from V22 to V27: only architectural coupling produces integration.">

      <text x={w / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Prediction → Integration?
      </text>
      <text x={w / 2} y={38} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        accuracy ≠ coupling (V22–V27)
      </text>

      {/* Wall */}
      <line x1={wallX} y1={topY - 8} x2={wallX} y2={topY + boxH + 30}
        stroke="var(--d-fg)" strokeWidth={1.5} strokeDasharray="6,3" />
      <text x={wallX} y={topY + boxH + 46} textAnchor="middle" fontSize={9} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        decomposability wall
      </text>

      {boxes.map((box, i) => {
        const x = startX + i * spacing;
        const y = topY;
        const isV27 = i === 3;

        return (
          <g key={box.label}>
            {/* Box */}
            <rect x={x} y={y} width={boxW} height={boxH} rx={5}
              fill={isV27 ? 'var(--d-fg)' : 'none'} fillOpacity={isV27 ? 0.06 : 0}
              stroke="var(--d-fg)" strokeWidth={isV27 ? 1.5 : 0.75} />

            {/* Label */}
            <text x={x + boxW / 2} y={y + 16} textAnchor="middle"
              fontSize={12} fontWeight={600} fill="var(--d-fg)"
              fontFamily="var(--font-body, Georgia, serif)">
              {box.label}
            </text>

            {/* Description lines */}
            {box.desc.split('\n').map((line, li) => (
              <text key={li} x={x + boxW / 2} y={y + 34 + li * 13} textAnchor="middle"
                fontSize={9} fill="var(--d-muted)"
                fontFamily="var(--font-body, Georgia, serif)">
                {line}
              </text>
            ))}

            {/* Phi value */}
            <text x={x + boxW / 2} y={y + boxH + 16} textAnchor="middle"
              fontSize={10} fill="var(--d-fg)"
              fontFamily="var(--font-body, Georgia, serif)">
              Φ = {box.phi}
            </text>

            {/* Arrow to next */}
            {i < boxes.length - 1 && i !== 2 && (
              <g>
                <line x1={x + boxW + 4} y1={y + boxH / 2}
                  x2={x + spacing - 4} y2={y + boxH / 2}
                  stroke="var(--d-line)" strokeWidth={0.75} />
                <path d={arrowPath([x + spacing - 4, y + boxH / 2], 0, 5)}
                  stroke="var(--d-line)" strokeWidth={0.75} fill="none" />
              </g>
            )}
          </g>
        );
      })}

      {/* Bracket under V22-V24 */}
      {(() => {
        const x1 = startX + boxW / 2 - 10;
        const x2 = startX + 2 * spacing + boxW / 2 + 10;
        const y = topY + boxH + 32;
        return (
          <g>
            <path d={`M${x1},${y} L${x1},${y + 6} L${x2},${y + 6} L${x2},${y}`}
              fill="none" stroke="var(--d-line)" strokeWidth={0.5} />
            <text x={(x1 + x2) / 2} y={y + 18} textAnchor="middle"
              fontSize={8} fontStyle="italic" fill="var(--d-muted)"
              fontFamily="var(--font-body, Georgia, serif)">
              all decomposable — any subset can satisfy the loss
            </text>
          </g>
        );
      })()}

      {/* Bottom insight */}
      <text x={w / 2} y={h - 6} textAnchor="middle" fontSize={9} fontStyle="italic"
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        Prediction accuracy is necessary but not sufficient. Coupling is the mechanism.
      </text>
    </svg>
  );
}
