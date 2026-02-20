'use client';

import { useEffect, useRef, useState } from 'react';

/**
 * Stylized top-down view of the protocell grid world.
 *
 * Shows a 16×16 portion with:
 * - Resource patches (green cells with varying density)
 * - Agents (colored dots, one focal agent highlighted)
 * - Focal agent's observation window highlighted
 * - Communication range ring
 * - Resource depletion zone
 */

// Deterministic pseudo-random (seeded)
function mulberry32(seed: number) {
  return () => {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const GRID = 16;
const CELL = 24;
const MARGIN = 45;

interface Agent {
  x: number;
  y: number;
  type: number; // 0=focal, 1=same-type, 2=other-type
  energy: number; // 0-1
}

// Generate static world
const rand = mulberry32(42);
const resources: boolean[][] = Array.from({ length: GRID }, (_, r) =>
  Array.from({ length: GRID }, (_, c) => {
    // Resource patches — clustered
    const px = c / GRID, py = r / GRID;
    const patch1 = Math.exp(-((px - 0.3) ** 2 + (py - 0.25) ** 2) / 0.02);
    const patch2 = Math.exp(-((px - 0.75) ** 2 + (py - 0.7) ** 2) / 0.03);
    const patch3 = Math.exp(-((px - 0.15) ** 2 + (py - 0.8) ** 2) / 0.015);
    return rand() < (patch1 + patch2 + patch3) * 0.7;
  })
);

const agents: Agent[] = [
  { x: 5, y: 4, type: 0, energy: 0.8 },   // focal
  { x: 3, y: 3, type: 1, energy: 0.6 },
  { x: 7, y: 5, type: 1, energy: 0.4 },
  { x: 4, y: 6, type: 1, energy: 0.7 },
  { x: 11, y: 11, type: 2, energy: 0.5 },
  { x: 13, y: 10, type: 2, energy: 0.3 },
  { x: 12, y: 12, type: 2, energy: 0.6 },
  { x: 2, y: 12, type: 1, energy: 0.55 },
  { x: 9, y: 8, type: 2, energy: 0.45 },
  { x: 14, y: 3, type: 1, energy: 0.35 },
];

const focal = agents[0];
const OBS_RADIUS = 2; // 5×5 observation window
const COMM_RADIUS = 5;

export default function ProtocellGrid() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [showOverlays, setShowOverlays] = useState(false);
  const [hoverLabel, setHoverLabel] = useState<string | null>(null);

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setShowOverlays(true);
          observer.disconnect();
        }
      },
      { threshold: 0.3 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const gridX = (c: number) => MARGIN + c * CELL + CELL / 2;
  const gridY = (r: number) => MARGIN + r * CELL + CELL / 2;
  const totalW = MARGIN * 2 + GRID * CELL;
  const totalH = MARGIN * 2 + GRID * CELL + 32;

  return (
    <svg ref={svgRef} viewBox={`0 0 ${totalW} ${totalH}`} className="diagram-svg" role="img"
      aria-label="Top-down view of protocell grid world showing resource patches, agents, observation windows, and communication ranges">

      {/* Title */}
      <text x={totalW / 2} y={18} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Protocell Grid World
      </text>
      <text x={totalW / 2} y={33} textAnchor="middle" fontSize={10}
        fill="var(--d-muted)" fontFamily="var(--font-body, Georgia, serif)">
        16×16 portion — resource patches, agents, observation windows
      </text>

      {/* Grid background */}
      <rect x={MARGIN} y={MARGIN} width={GRID * CELL} height={GRID * CELL}
        fill="var(--d-fg)" fillOpacity={0.02} stroke="var(--d-line)" strokeWidth={0.5} strokeOpacity={0.2} rx={2} />

      {/* Grid lines */}
      {Array.from({ length: GRID + 1 }, (_, i) => (
        <g key={`grid-${i}`}>
          <line x1={MARGIN + i * CELL} y1={MARGIN} x2={MARGIN + i * CELL} y2={MARGIN + GRID * CELL}
            stroke="var(--d-line)" strokeWidth={0.3} opacity={0.15} />
          <line x1={MARGIN} y1={MARGIN + i * CELL} x2={MARGIN + GRID * CELL} y2={MARGIN + i * CELL}
            stroke="var(--d-line)" strokeWidth={0.3} opacity={0.15} />
        </g>
      ))}

      {/* Resource cells */}
      {resources.map((row, r) =>
        row.map((hasResource, c) => hasResource ? (
          <rect key={`r-${r}-${c}`}
            x={MARGIN + c * CELL + 1} y={MARGIN + r * CELL + 1}
            width={CELL - 2} height={CELL - 2} rx={2}
            fill="var(--d-green)" fillOpacity={0.2 + rand() * 0.15} />
        ) : null)
      )}

      {/* Communication range ring (behind agents) */}
      {showOverlays && (
        <circle cx={gridX(focal.x)} cy={gridY(focal.y)}
          r={COMM_RADIUS * CELL}
          fill="none" stroke="var(--d-violet)" strokeWidth={1}
          strokeDasharray="6,4" opacity={0.4}
          onMouseEnter={() => setHoverLabel('Communication range (radius=5): agents within this ring can receive signals')}
          onMouseLeave={() => setHoverLabel(null)}
          style={{ cursor: 'pointer', transition: 'opacity 0.3s' }} />
      )}

      {/* Observation window highlight */}
      {showOverlays && (
        <rect
          x={MARGIN + (focal.x - OBS_RADIUS) * CELL}
          y={MARGIN + (focal.y - OBS_RADIUS) * CELL}
          width={(OBS_RADIUS * 2 + 1) * CELL}
          height={(OBS_RADIUS * 2 + 1) * CELL}
          fill="var(--d-blue)" fillOpacity={0.08}
          stroke="var(--d-blue)" strokeWidth={1.5} rx={3}
          onMouseEnter={() => setHoverLabel('Observation window (5×5): all the focal agent can see. V26 reduces this to 1×1.')}
          onMouseLeave={() => setHoverLabel(null)}
          style={{ cursor: 'pointer', transition: 'opacity 0.3s' }} />
      )}

      {/* Agents */}
      {agents.map((agent, i) => {
        const isFocal = agent.type === 0;
        const color = agent.type === 0 ? 'var(--d-orange)' :
                      agent.type === 1 ? 'var(--d-blue)' : 'var(--d-red)';
        const r = isFocal ? 8 : 6;

        return (
          <g key={`a-${i}`}
            onMouseEnter={() => setHoverLabel(
              isFocal ? `Focal agent — energy ${(agent.energy * 100).toFixed(0)}%, obs window highlighted` :
              `Agent (type ${agent.type}) — energy ${(agent.energy * 100).toFixed(0)}%`
            )}
            onMouseLeave={() => setHoverLabel(null)}
            style={{ cursor: 'pointer' }}>
            {/* Energy ring */}
            <circle cx={gridX(agent.x)} cy={gridY(agent.y)} r={r + 2}
              fill="none" stroke={color} strokeWidth={1.5}
              strokeDasharray={`${agent.energy * 2 * Math.PI * (r + 2)} ${2 * Math.PI * (r + 2)}`}
              opacity={0.5}
              transform={`rotate(-90, ${gridX(agent.x)}, ${gridY(agent.y)})`} />
            {/* Body */}
            <circle cx={gridX(agent.x)} cy={gridY(agent.y)} r={r}
              fill={color} fillOpacity={0.7} />
            {isFocal && (
              <text x={gridX(agent.x)} y={gridY(agent.y)}
                textAnchor="middle" dominantBaseline="central"
                fontSize={8} fontWeight={700} fill="var(--d-fg)">
                ★
              </text>
            )}
          </g>
        );
      })}

      {/* Labels */}
      {showOverlays && (
        <>
          <text x={MARGIN + (focal.x + OBS_RADIUS + 1) * CELL + 4}
            y={MARGIN + (focal.y - OBS_RADIUS) * CELL + 10}
            fontSize={9} fill="var(--d-blue)" fontWeight={600}
            fontFamily="var(--font-body, Georgia, serif)">
            5×5 obs
          </text>
          <text x={gridX(focal.x) + COMM_RADIUS * CELL + 4}
            y={gridY(focal.y) - 4}
            fontSize={9} fill="var(--d-violet)" fontWeight={600}
            fontFamily="var(--font-body, Georgia, serif)">
            comm range
          </text>
        </>
      )}

      {/* Legend */}
      <g transform={`translate(${MARGIN}, ${MARGIN + GRID * CELL + 8})`}>
        {[
          { color: 'var(--d-green)', label: 'Resource patch', opacity: 0.3 },
          { color: 'var(--d-orange)', label: 'Focal agent', opacity: 0.7 },
          { color: 'var(--d-blue)', label: 'Same type', opacity: 0.7 },
          { color: 'var(--d-red)', label: 'Other type', opacity: 0.7 },
        ].map((item, i) => (
          <g key={i} transform={`translate(${i * 105}, 0)`}>
            <rect x={0} y={0} width={10} height={10} rx={item.label.includes('Resource') ? 2 : 5}
              fill={item.color} fillOpacity={item.opacity} />
            <text x={14} y={9} fontSize={8.5} fill="var(--d-muted)"
              fontFamily="var(--font-body, Georgia, serif)">{item.label}</text>
          </g>
        ))}
      </g>

      {/* Hover tooltip */}
      {hoverLabel && (
        <g>
          <rect x={MARGIN} y={MARGIN + GRID * CELL + 22} width={GRID * CELL} height={22} rx={3}
            fill="var(--d-fg)" fillOpacity={0.05} stroke="var(--d-line)" strokeWidth={0.5} strokeOpacity={0.3} />
          <text x={totalW / 2} y={MARGIN + GRID * CELL + 36}
            textAnchor="middle" fontSize={9.5} fill="var(--d-fg)"
            fontFamily="var(--font-body, Georgia, serif)">
            {hoverLabel}
          </text>
        </g>
      )}
    </svg>
  );
}
