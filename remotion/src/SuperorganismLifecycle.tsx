import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
  Easing,
} from "remotion";
import { THEMES, ThemeMode } from "./themes";

/**
 * SuperorganismLifecycle — Part V video
 * Shows a social pattern (god/institution) persisting while individual substrates turn over
 * Circle of agents cycling in/out; central pattern (superorganism) remains stable
 * Phi_collective shown as overlay
 */

// Seeded random
function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(42);
const NUM_AGENTS = 16;
const AGENT_RADIUS = 220;

interface Agent {
  hue: number;
  birthFrame: number;
  lifespan: number;
}

// Pre-generate agents with staggered births
const AGENTS: Agent[] = Array.from({ length: NUM_AGENTS * 4 }, (_, i) => ({
  hue: 180 + rand() * 80,
  birthFrame: Math.floor(i / NUM_AGENTS) * 80 + (i % NUM_AGENTS) * 5 + Math.floor(rand() * 15),
  lifespan: 70 + Math.floor(rand() * 40),
}));

export const SuperorganismLifecycleVideo: React.FC<{ theme?: ThemeMode }> = ({ theme }) => {
  const t = THEMES[theme ?? "dark"];
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const cx = width / 2;
  const cy = height / 2 + 20;

  // Title fade
  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Central pattern pulse
  const patternR = interpolate(
    Math.sin(frame * 0.05),
    [-1, 1],
    [55, 65],
  );

  const patternOpacity = interpolate(frame, [10, 40], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Get active agents at current frame
  const activeAgents = AGENTS.filter(
    (a) => frame >= a.birthFrame && frame < a.birthFrame + a.lifespan,
  ).slice(0, NUM_AGENTS);

  // Phi collective value (rises then stabilizes)
  const phiValue = interpolate(frame, [0, 60, 120, 360], [0.01, 0.04, 0.08, 0.09], {
    extrapolateRight: "clamp",
  });

  // Turnover rate label
  const generation = Math.floor(frame / 80);
  const turnoverLabel = `Generation ${generation + 1}`;

  // Phase labels
  const phaseLabel =
    frame < 60
      ? "Formation"
      : frame < 150
        ? "First substrate turnover"
        : frame < 240
          ? "Second turnover — pattern persists"
          : "The pattern IS the entity";

  const phaseLabelOpacity = interpolate(
    frame % 90,
    [0, 15, 75, 90],
    [0, 1, 1, 0],
    { extrapolateRight: "clamp" },
  );

  return (
    <AbsoluteFill style={{ backgroundColor: t.bg }}>
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 30,
          width: "100%",
          textAlign: "center",
          fontFamily: "Georgia, serif",
          opacity: titleOpacity,
        }}
      >
        <div
          style={{
            fontSize: 28,
            fontWeight: 700,
            color: t.text,
          }}
        >
          Superorganism Lifecycle
        </div>
        <div style={{ fontSize: 14, color: t.muted, marginTop: 4 }}>
          individual substrates turn over — the pattern persists
        </div>
      </div>

      {/* SVG scene */}
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
        }}
      >
        {/* Central pattern (superorganism) */}
        <circle
          cx={cx}
          cy={cy}
          r={patternR + 20}
          fill={t.yellow}
          fillOpacity={0.12}
          opacity={patternOpacity * 0.3}
        />
        <circle
          cx={cx}
          cy={cy}
          r={patternR}
          fill="none"
          stroke={t.yellow}
          strokeWidth={2.5}
          opacity={patternOpacity * 0.8}
        />
        <circle
          cx={cx}
          cy={cy}
          r={patternR - 15}
          fill="none"
          stroke={t.yellow}
          strokeWidth={1}
          opacity={patternOpacity * 0.4}
          strokeDasharray="4 6"
        />
        <text
          x={cx}
          y={cy - 8}
          textAnchor="middle"
          dominantBaseline="central"
          fill={t.yellow}
          fontSize={16}
          fontWeight={700}
          fontFamily="Georgia, serif"
          opacity={patternOpacity}
        >
          Pattern
        </text>
        <text
          x={cx}
          y={cy + 12}
          textAnchor="middle"
          dominantBaseline="central"
          fill={t.yellow}
          fontSize={11}
          fontFamily="Georgia, serif"
          opacity={patternOpacity * 0.7}
        >
          (the superorganism)
        </text>

        {/* Agent ring */}
        {activeAgents.map((agent, i) => {
          const angle = (i / NUM_AGENTS) * Math.PI * 2 - Math.PI / 2;
          const ax = cx + Math.cos(angle) * AGENT_RADIUS;
          const ay = cy + Math.sin(angle) * AGENT_RADIUS;

          const age = frame - agent.birthFrame;
          const life = agent.lifespan;
          const fadeIn = interpolate(age, [0, 10], [0, 1], {
            extrapolateRight: "clamp",
          });
          const fadeOut = interpolate(age, [life - 15, life], [1, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const opacity = Math.min(fadeIn, fadeOut);

          // Connection line to center
          const lineOpacity = opacity * 0.3;

          return (
            <g key={`${agent.birthFrame}-${i}`}>
              <line
                x1={ax}
                y1={ay}
                x2={cx}
                y2={cy}
                stroke={t.blue}
                strokeWidth={0.8}
                opacity={lineOpacity}
                strokeDasharray="3 5"
              />
              <circle
                cx={ax}
                cy={ay}
                r={10}
                fill={`hsl(${agent.hue}, 60%, 55%)`}
                opacity={opacity * 0.8}
              />
              <circle
                cx={ax}
                cy={ay}
                r={10}
                fill="none"
                stroke={`hsl(${agent.hue}, 70%, 65%)`}
                strokeWidth={1}
                opacity={opacity * 0.5}
              />
            </g>
          );
        })}

        {/* Phi collective readout */}
        <rect
          x={width - 200}
          y={height - 100}
          width={170}
          height={55}
          rx={6}
          fill={t.panel}
          stroke={t.green}
          strokeWidth={1}
          opacity={0.8}
        />
        <text
          x={width - 115}
          y={height - 80}
          textAnchor="middle"
          fill={t.green}
          fontSize={12}
          fontFamily="Georgia, serif"
        >
          Φ_collective
        </text>
        <text
          x={width - 115}
          y={height - 58}
          textAnchor="middle"
          fill={t.green}
          fontSize={20}
          fontWeight={700}
          fontFamily="Georgia, serif"
        >
          {phiValue.toFixed(3)}
        </text>

        {/* Generation counter */}
        <text
          x={40}
          y={height - 60}
          fill={t.muted}
          fontSize={13}
          fontFamily="Georgia, serif"
        >
          {turnoverLabel}
        </text>
      </svg>

      {/* Phase label */}
      <div
        style={{
          position: "absolute",
          bottom: 30,
          left: 0,
          width: "100%",
          textAlign: "center",
          fontFamily: "Georgia, serif",
          fontSize: 15,
          fontStyle: "italic",
          color: t.muted,
          opacity: phaseLabelOpacity,
        }}
      >
        {phaseLabel}
      </div>
    </AbsoluteFill>
  );
};
