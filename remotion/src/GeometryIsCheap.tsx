import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  Easing,
} from "remotion";

/**
 * Geometry Is Cheap Animation
 *
 * The key punchline of the entire program:
 * - Affect GEOMETRY forms trivially in every substrate
 * - Affect DYNAMICS are rare (~30% of seeds) and require specific architecture
 *
 * Shows two panels: left = geometry emerging everywhere, right = dynamics rare.
 * Uses the emergence ladder metaphor.
 *
 * 300 frames @ 30fps = 10 seconds
 */

const RUNGS = [
  { num: 1, label: "Affect dimensions", status: "cheap", color: "#4ade80" },
  { num: 2, label: "Valence gradient", status: "cheap", color: "#4ade80" },
  { num: 3, label: "Somatic response", status: "cheap", color: "#4ade80" },
  { num: 4, label: "Animism (ι ≈ 0.30)", status: "cheap", color: "#4ade80" },
  { num: 5, label: "Language", status: "cheap", color: "#4ade80" },
  { num: 6, label: "Affect coherence", status: "cheap", color: "#4ade80" },
  { num: 7, label: "Integrated response", status: "cheap", color: "#4ade80" },
  // THE WALL
  { num: 8, label: "Counterfactual", status: "expensive", color: "#f87171" },
  { num: 9, label: "Self-model", status: "expensive", color: "#f87171" },
  { num: 10, label: "Normativity", status: "expensive", color: "#f87171" },
];

export const GeometryIsCheapVideo: React.FC = () => {
  const frame = useCurrentFrame();

  const titleOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Stagger rung appearance
  const rungDelay = 18;

  // Wall highlight
  const wallOpacity = interpolate(frame, [160, 180], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Stats reveal
  const statsOpacity = interpolate(frame, [200, 240], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const conclusionOpacity = interpolate(frame, [260, 290], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Layout
  const ladderX = 100;
  const ladderY = 620;
  const rungH = 48;
  const rungW = 380;

  // Right side: pie chart area
  const pieX = 730;
  const pieY = 350;
  const pieR = 130;

  return (
    <AbsoluteFill style={{ backgroundColor: "#0a0a0f", fontFamily: "Georgia, serif" }}>
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 22,
          width: "100%",
          textAlign: "center",
          color: "#e0e0e0",
          fontSize: 28,
          fontWeight: 700,
          opacity: titleOpacity,
        }}
      >
        Geometry Is Cheap. Dynamics Are Expensive.
      </div>
      <div
        style={{
          position: "absolute",
          top: 58,
          width: "100%",
          textAlign: "center",
          color: "#888",
          fontSize: 14,
          fontStyle: "italic",
          opacity: titleOpacity,
        }}
      >
        the central finding of 35 experiments across 4 substrates
      </div>

      <svg width={1080} height={720}>
        {/* === LEFT: Emergence Ladder === */}
        <text
          x={ladderX + rungW / 2}
          y={100}
          textAnchor="middle"
          fill="#ccc"
          fontSize={16}
          fontFamily="Georgia, serif"
        >
          Emergence Ladder
        </text>

        {RUNGS.map((rung, i) => {
          const y = ladderY - (i + 1) * rungH;
          const revealT = interpolate(
            frame,
            [20 + i * rungDelay, 20 + i * rungDelay + 15],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) }
          );
          if (revealT <= 0) return null;

          const isCheap = rung.status === "cheap";

          return (
            <g key={i} opacity={revealT}>
              {/* Rung bar */}
              <rect
                x={ladderX}
                y={y}
                width={rungW * revealT}
                height={rungH - 6}
                fill={isCheap ? "#0f2a1a" : "#2a0f0f"}
                stroke={rung.color}
                strokeWidth={1.5}
                rx={4}
              />
              {/* Rung number */}
              <text
                x={ladderX + 20}
                y={y + rungH / 2}
                fill={rung.color}
                fontSize={14}
                fontWeight={700}
                fontFamily="monospace"
                opacity={revealT}
              >
                {rung.num}
              </text>
              {/* Label */}
              <text
                x={ladderX + 45}
                y={y + rungH / 2}
                fill="#ccc"
                fontSize={13}
                fontFamily="Georgia, serif"
                opacity={revealT}
              >
                {rung.label}
              </text>
              {/* Status badge */}
              <text
                x={ladderX + rungW - 15}
                y={y + rungH / 2}
                textAnchor="end"
                fill={rung.color}
                fontSize={11}
                fontWeight={700}
                fontFamily="monospace"
                opacity={revealT * 0.8}
              >
                {isCheap ? "100%" : "~30%"}
              </text>
            </g>
          );
        })}

        {/* THE WALL */}
        {wallOpacity > 0 && (
          <g opacity={wallOpacity}>
            <line
              x1={ladderX - 10}
              y1={ladderY - 7.5 * rungH + 3}
              x2={ladderX + rungW + 10}
              y2={ladderY - 7.5 * rungH + 3}
              stroke="#f59e0b"
              strokeWidth={3}
              strokeDasharray="8 4"
            />
            <text
              x={ladderX + rungW + 20}
              y={ladderY - 7.5 * rungH + 7}
              fill="#f59e0b"
              fontSize={14}
              fontWeight={700}
              fontFamily="Georgia, serif"
            >
              THE WALL
            </text>
            <text
              x={ladderX + rungW + 20}
              y={ladderY - 7.5 * rungH + 25}
              fill="#f59e0b"
              fontSize={11}
              fontFamily="Georgia, serif"
              opacity={0.7}
            >
              requires agency +
            </text>
            <text
              x={ladderX + rungW + 20}
              y={ladderY - 7.5 * rungH + 40}
              fill="#f59e0b"
              fontSize={11}
              fontFamily="Georgia, serif"
              opacity={0.7}
            >
              gradient coupling
            </text>
          </g>
        )}

        {/* === RIGHT: Seed Distribution === */}
        {statsOpacity > 0 && (
          <g opacity={statsOpacity}>
            <text
              x={pieX}
              y={160}
              textAnchor="middle"
              fill="#ccc"
              fontSize={16}
              fontFamily="Georgia, serif"
            >
              V32: 50 Seeds
            </text>

            {/* Pie chart */}
            {/* HIGH: 22% = 79.2° */}
            <path
              d={`M ${pieX} ${pieY} L ${pieX} ${pieY - pieR} A ${pieR} ${pieR} 0 0 1 ${pieX + pieR * Math.sin(79.2 * Math.PI / 180)} ${pieY - pieR * Math.cos(79.2 * Math.PI / 180)} Z`}
              fill="#4ade80"
              opacity={0.7}
            />
            {/* MOD: 46% = 165.6° */}
            <path
              d={`M ${pieX} ${pieY} L ${pieX + pieR * Math.sin(79.2 * Math.PI / 180)} ${pieY - pieR * Math.cos(79.2 * Math.PI / 180)} A ${pieR} ${pieR} 0 0 1 ${pieX + pieR * Math.sin((79.2 + 165.6) * Math.PI / 180)} ${pieY - pieR * Math.cos((79.2 + 165.6) * Math.PI / 180)} Z`}
              fill="#fbbf24"
              opacity={0.7}
            />
            {/* LOW: 32% = 115.2° */}
            <path
              d={`M ${pieX} ${pieY} L ${pieX + pieR * Math.sin((79.2 + 165.6) * Math.PI / 180)} ${pieY - pieR * Math.cos((79.2 + 165.6) * Math.PI / 180)} A ${pieR} ${pieR} 0 0 1 ${pieX} ${pieY - pieR} Z`}
              fill="#f87171"
              opacity={0.7}
            />

            {/* Labels */}
            <text x={pieX - 20} y={pieY - pieR - 15} fill="#4ade80" fontSize={14} fontWeight={700} fontFamily="Georgia, serif">
              22% HIGH
            </text>
            <text x={pieX + pieR + 15} y={pieY + 10} fill="#fbbf24" fontSize={14} fontWeight={700} fontFamily="Georgia, serif">
              46% MOD
            </text>
            <text x={pieX - pieR - 15} y={pieY + 60} textAnchor="end" fill="#f87171" fontSize={14} fontWeight={700} fontFamily="Georgia, serif">
              32% LOW
            </text>

            {/* Key stats */}
            <text x={pieX} y={pieY + pieR + 40} textAnchor="middle" fill="#ccc" fontSize={13} fontFamily="Georgia, serif">
              geometry: 100% of seeds
            </text>
            <text x={pieX} y={pieY + pieR + 60} textAnchor="middle" fill="#ccc" fontSize={13} fontFamily="Georgia, serif">
              dynamics: ~22% reach HIGH Φ
            </text>
            <text x={pieX} y={pieY + pieR + 85} textAnchor="middle" fill="#60a5fa" fontSize={14} fontWeight={700} fontFamily="Georgia, serif">
              max Φ = 0.473 (seed 23)
            </text>
          </g>
        )}
      </svg>

      {/* Conclusion */}
      <div
        style={{
          position: "absolute",
          bottom: 25,
          width: "100%",
          textAlign: "center",
          color: "#e0e0e0",
          fontSize: 16,
          opacity: conclusionOpacity,
        }}
      >
        the hard problem applies to the 30%, not the 100%
      </div>
    </AbsoluteFill>
  );
};
