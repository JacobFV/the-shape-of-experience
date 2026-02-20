import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  spring,
  useVideoConfig,
  Easing,
} from "remotion";

/**
 * Bottleneck Furnace Animation
 *
 * Shows a grid of 256 agents going through 5 drought-recovery cycles.
 * Agents die dramatically during droughts, survivors rebuild.
 * A Φ line tracks integration climbing for HIGH seeds.
 *
 * 360 frames @ 30fps = 12 seconds
 */

// Deterministic pseudo-random
function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(42);

// Pre-generate agent positions (16x16 grid with jitter)
const AGENTS = Array.from({ length: 256 }, (_, i) => ({
  x: (i % 16) * 60 + 30 + (rand() - 0.5) * 20,
  y: Math.floor(i / 16) * 38 + 8 + (rand() - 0.5) * 14,
  deathOrder: rand(), // when this agent dies (0=first, 1=last)
  hue: 120 + rand() * 60, // green-ish base
  resilience: rand(), // how well it bounces back
}));

// Timeline: 5 drought cycles across 360 frames
// Each cycle: ~30f ramp, ~15f crash, ~27f recovery
const CYCLES = [
  { crashStart: 35, crashEnd: 55, recoveryEnd: 85, mortality: 0.93, survivors: 18 },
  { crashStart: 105, crashEnd: 125, recoveryEnd: 155, mortality: 0.95, survivors: 12 },
  { crashStart: 175, crashEnd: 195, recoveryEnd: 225, mortality: 0.97, survivors: 8 },
  { crashStart: 235, crashEnd: 255, recoveryEnd: 285, mortality: 0.94, survivors: 15 },
  { crashStart: 295, crashEnd: 315, recoveryEnd: 345, mortality: 0.98, survivors: 6 },
];

// Φ trajectory (HIGH seed)
const PHI_POINTS = [
  { f: 0, phi: 0.06 },
  { f: 35, phi: 0.07 },
  { f: 55, phi: 0.04 },
  { f: 85, phi: 0.08 },
  { f: 105, phi: 0.075 },
  { f: 125, phi: 0.035 },
  { f: 155, phi: 0.095 },
  { f: 175, phi: 0.09 },
  { f: 195, phi: 0.03 },
  { f: 225, phi: 0.11 },
  { f: 235, phi: 0.105 },
  { f: 255, phi: 0.04 },
  { f: 285, phi: 0.13 },
  { f: 295, phi: 0.125 },
  { f: 315, phi: 0.03 },
  { f: 345, phi: 0.15 },
  { f: 360, phi: 0.145 },
];

function getPhiAtFrame(frame: number): number {
  for (let i = 0; i < PHI_POINTS.length - 1; i++) {
    if (frame >= PHI_POINTS[i].f && frame <= PHI_POINTS[i + 1].f) {
      const t =
        (frame - PHI_POINTS[i].f) / (PHI_POINTS[i + 1].f - PHI_POINTS[i].f);
      return PHI_POINTS[i].phi + t * (PHI_POINTS[i + 1].phi - PHI_POINTS[i].phi);
    }
  }
  return PHI_POINTS[PHI_POINTS.length - 1].phi;
}

function getPopAtFrame(frame: number): number {
  for (const cycle of CYCLES) {
    if (frame >= cycle.crashStart && frame <= cycle.crashEnd) {
      const t = (frame - cycle.crashStart) / (cycle.crashEnd - cycle.crashStart);
      return Math.round(256 - (256 - cycle.survivors) * t);
    }
    if (frame > cycle.crashEnd && frame <= cycle.recoveryEnd) {
      const t = (frame - cycle.crashEnd) / (cycle.recoveryEnd - cycle.crashEnd);
      return Math.round(cycle.survivors + (256 - cycle.survivors) * t);
    }
  }
  return 256;
}

function isAgentAlive(agentIdx: number, frame: number): boolean {
  const agent = AGENTS[agentIdx];
  for (const cycle of CYCLES) {
    if (frame >= cycle.crashStart && frame <= cycle.recoveryEnd) {
      if (frame <= cycle.crashEnd) {
        const t = (frame - cycle.crashStart) / (cycle.crashEnd - cycle.crashStart);
        return agent.deathOrder > t * cycle.mortality;
      }
      if (frame <= cycle.recoveryEnd) {
        const t = (frame - cycle.crashEnd) / (cycle.recoveryEnd - cycle.crashEnd);
        return agent.deathOrder > (1 - t) * cycle.mortality + t * 0;
      }
    }
  }
  return true;
}

function isDrought(frame: number): boolean {
  return CYCLES.some((c) => frame >= c.crashStart && frame <= c.crashEnd);
}

export const BottleneckFurnaceVideo: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const pop = getPopAtFrame(frame);
  const phi = getPhiAtFrame(frame);
  const drought = isDrought(frame);

  // Chart dimensions
  const chartX = 680;
  const chartW = 360;
  const chartH = 200;
  const chartTop = 80;

  // Population bar
  const popBarH = 40;
  const popBarTop = chartTop + chartH + 30;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#0a0a0f",
        fontFamily: "Georgia, serif",
      }}
    >
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 20,
          left: 40,
          color: "#e0e0e0",
          fontSize: 28,
          fontWeight: 700,
        }}
      >
        The Bottleneck Furnace
      </div>
      <div
        style={{
          position: "absolute",
          top: 55,
          left: 40,
          color: "#888",
          fontSize: 16,
          fontStyle: "italic",
        }}
      >
        near-extinction forges integration — or fails to
      </div>

      {/* Agent grid (left side) */}
      <svg
        width={640}
        height={620}
        style={{ position: "absolute", left: 20, top: 80 }}
      >
        {/* Drought flash */}
        {drought && (
          <rect
            x={0}
            y={0}
            width={640}
            height={620}
            fill="#ff2020"
            opacity={0.06 + Math.sin(frame * 0.5) * 0.03}
            rx={8}
          />
        )}

        {AGENTS.map((agent, i) => {
          const alive = isAgentAlive(i, frame);
          const opacity = alive ? 0.8 : 0;
          const scale = alive ? 1 : 0;

          // Dying animation
          let dy = 0;
          if (!alive) {
            dy = 5;
          }

          return (
            <g
              key={i}
              style={{
                transition: "opacity 0.15s, transform 0.15s",
              }}
            >
              {/* Glow for alive agents */}
              {alive && (
                <circle
                  cx={agent.x}
                  cy={agent.y + dy}
                  r={12}
                  fill={`hsla(${agent.hue}, 60%, 50%, 0.15)`}
                />
              )}
              {/* Agent dot */}
              <circle
                cx={agent.x}
                cy={agent.y + dy}
                r={alive ? 5 : 2}
                fill={
                  alive
                    ? drought
                      ? `hsl(${agent.hue - 80}, 70%, 45%)`
                      : `hsl(${agent.hue}, 60%, 50%)`
                    : "#333"
                }
                opacity={opacity}
              />
            </g>
          );
        })}

        {/* Population counter */}
        <text
          x={320}
          y={615}
          textAnchor="middle"
          fill={drought ? "#f87171" : "#4ade80"}
          fontSize={20}
          fontWeight={700}
          fontFamily="Georgia, serif"
        >
          {pop} / 256 agents
          {drought ? " — DROUGHT" : ""}
        </text>
      </svg>

      {/* Φ chart (right side) */}
      <svg
        width={chartW + 40}
        height={chartH + 100}
        style={{ position: "absolute", left: chartX, top: chartTop }}
      >
        {/* Chart background */}
        <rect
          x={30}
          y={0}
          width={chartW}
          height={chartH}
          fill="#111118"
          stroke="#333"
          strokeWidth={0.5}
          rx={4}
        />

        {/* Drought bands */}
        {CYCLES.map((c, i) => {
          const x1 = 30 + (c.crashStart / 360) * chartW;
          const w = ((c.crashEnd - c.crashStart) / 360) * chartW;
          return (
            <rect
              key={i}
              x={x1}
              y={0}
              width={w}
              height={chartH}
              fill="#ff2020"
              opacity={0.1}
            />
          );
        })}

        {/* Φ line up to current frame */}
        {(() => {
          const pts = PHI_POINTS.filter((p) => p.f <= frame);
          if (pts.length < 2) return null;
          const d = pts
            .map((p, i) => {
              const x = 30 + (p.f / 360) * chartW;
              const y = chartH - (p.phi / 0.18) * chartH;
              return `${i === 0 ? "M" : "L"} ${x} ${y}`;
            })
            .join(" ");
          return (
            <path
              d={d}
              fill="none"
              stroke="#4ade80"
              strokeWidth={2.5}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          );
        })()}

        {/* Current Φ dot */}
        <circle
          cx={30 + (frame / 360) * chartW}
          cy={chartH - (phi / 0.18) * chartH}
          r={5}
          fill="#4ade80"
        />

        {/* Labels */}
        <text x={30 + chartW / 2} y={-10} textAnchor="middle" fill="#ccc" fontSize={14} fontFamily="Georgia, serif">
          Φ (integration)
        </text>
        <text x={30} y={chartH + 16} fill="#666" fontSize={10} fontFamily="Georgia, serif">
          0
        </text>
        <text x={30 + chartW} y={chartH + 16} textAnchor="end" fill="#666" fontSize={10} fontFamily="Georgia, serif">
          30 cycles
        </text>

        {/* Current Φ value */}
        <text
          x={30 + chartW / 2}
          y={chartH + 40}
          textAnchor="middle"
          fill="#4ade80"
          fontSize={22}
          fontWeight={700}
          fontFamily="Georgia, serif"
        >
          Φ = {phi.toFixed(3)}
        </text>

        {/* Annotation */}
        {frame > 250 && (
          <text
            x={30 + chartW / 2}
            y={chartH + 65}
            textAnchor="middle"
            fill="#4ade80"
            fontSize={13}
            fontFamily="Georgia, serif"
            opacity={Math.min((frame - 250) / 30, 1)}
          >
            each bounce higher than the last
          </text>
        )}
      </svg>

      {/* Cycle counter */}
      <div
        style={{
          position: "absolute",
          bottom: 30,
          right: 40,
          color: "#666",
          fontSize: 14,
        }}
      >
        {(() => {
          let cycle = 0;
          for (const c of CYCLES) {
            if (frame >= c.crashStart) cycle++;
          }
          return `Drought ${Math.min(cycle, 5)} / 5`;
        })()}
      </div>
    </AbsoluteFill>
  );
};
