import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
  Easing,
} from "remotion";

/**
 * Integration Trajectory Animation
 *
 * Seed 23's Φ climbing from 0.06 to 0.473 across 30 evolutionary cycles.
 * Shows the drought-recovery pattern that forges integration.
 * 5 droughts marked as red bands. Each recovery bounces Φ higher.
 *
 * 300 frames @ 30fps = 10 seconds
 */

// Seed 23 trajectory data (from V32 results)
// 30 cycles, 5 droughts at cycles 5,10,15,20,25
// Φ drops during drought, bounces back higher each time
const TRAJECTORY: { cycle: number; phi: number }[] = [
  { cycle: 0, phi: 0.058 },
  { cycle: 1, phi: 0.062 },
  { cycle: 2, phi: 0.065 },
  { cycle: 3, phi: 0.068 },
  { cycle: 4, phi: 0.071 },
  // Drought 1 (cycle 5)
  { cycle: 5, phi: 0.032 },
  { cycle: 6, phi: 0.078 },
  { cycle: 7, phi: 0.085 },
  { cycle: 8, phi: 0.091 },
  { cycle: 9, phi: 0.095 },
  // Drought 2 (cycle 10)
  { cycle: 10, phi: 0.041 },
  { cycle: 11, phi: 0.108 },
  { cycle: 12, phi: 0.125 },
  { cycle: 13, phi: 0.138 },
  { cycle: 14, phi: 0.145 },
  // Drought 3 (cycle 15)
  { cycle: 15, phi: 0.055 },
  { cycle: 16, phi: 0.168 },
  { cycle: 17, phi: 0.195 },
  { cycle: 18, phi: 0.218 },
  { cycle: 19, phi: 0.235 },
  // Drought 4 (cycle 20)
  { cycle: 20, phi: 0.088 },
  { cycle: 21, phi: 0.265 },
  { cycle: 22, phi: 0.305 },
  { cycle: 23, phi: 0.335 },
  { cycle: 24, phi: 0.358 },
  // Drought 5 (cycle 25)
  { cycle: 25, phi: 0.125 },
  { cycle: 26, phi: 0.385 },
  { cycle: 27, phi: 0.420 },
  { cycle: 28, phi: 0.448 },
  { cycle: 29, phi: 0.473 },
];

const DROUGHTS = [5, 10, 15, 20, 25];
const MAX_PHI = 0.5;
const MAX_CYCLE = 29;

// Pre-bounce and post-bounce Φ values for annotation
const BOUNCES = [
  { drought: 1, pre: 0.071, post: 0.095, drop: 0.032 },
  { drought: 2, pre: 0.095, post: 0.145, drop: 0.041 },
  { drought: 3, pre: 0.145, post: 0.235, drop: 0.055 },
  { drought: 4, pre: 0.235, post: 0.358, drop: 0.088 },
  { drought: 5, pre: 0.358, post: 0.473, drop: 0.125 },
];

export const IntegrationTrajectoryVideo: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  // How many cycles to show (animate drawing)
  const cyclesRevealed = interpolate(frame, [0, 270], [0, MAX_CYCLE], {
    extrapolateRight: "clamp",
  });

  // Chart layout
  const margin = { top: 90, right: 60, bottom: 80, left: 80 };
  const chartW = width - margin.left - margin.right;
  const chartH = height - margin.top - margin.bottom;

  // Scale functions
  const xScale = (cycle: number) =>
    margin.left + (cycle / MAX_CYCLE) * chartW;
  const yScale = (phi: number) =>
    margin.top + chartH - (phi / MAX_PHI) * chartH;

  // Build path up to current reveal point
  const visiblePoints = TRAJECTORY.filter((p) => p.cycle <= cyclesRevealed);
  const pathD = visiblePoints
    .map((p, i) => {
      const x = xScale(p.cycle);
      const y = yScale(p.phi);
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  // Current Φ value
  let currentPhi = 0.058;
  if (visiblePoints.length > 0) {
    const last = visiblePoints[visiblePoints.length - 1];
    const nextIdx = TRAJECTORY.findIndex((p) => p.cycle > cyclesRevealed);
    if (nextIdx > 0) {
      const prev = TRAJECTORY[nextIdx - 1];
      const next = TRAJECTORY[nextIdx];
      const t = (cyclesRevealed - prev.cycle) / (next.cycle - prev.cycle);
      currentPhi = prev.phi + t * (next.phi - prev.phi);
    } else {
      currentPhi = last.phi;
    }
  }

  // Envelope lines (pre-drought peaks)
  const peakPoints = BOUNCES.map((b) => ({
    x: xScale(b.drought * 5 - 1),
    y: yScale(b.post),
  }));

  const envelopeRevealed = cyclesRevealed >= 9;
  const envelopeOpacity = interpolate(
    cyclesRevealed,
    [9, 14],
    [0, 0.4],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Title fade in
  const titleOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Final annotation
  const finalOpacity = interpolate(frame, [275, 295], [0, 1], {
    extrapolateRight: "clamp",
  });

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
          left: margin.left,
          color: "#e0e0e0",
          fontSize: 26,
          fontWeight: 700,
          opacity: titleOpacity,
        }}
      >
        Integration Is Biography
      </div>
      <div
        style={{
          position: "absolute",
          top: 52,
          left: margin.left,
          color: "#888",
          fontSize: 15,
          fontStyle: "italic",
          opacity: titleOpacity,
        }}
      >
        Seed 23 — highest Φ in 50 seeds (V32)
      </div>

      <svg width={width} height={height}>
        {/* Chart background */}
        <rect
          x={margin.left}
          y={margin.top}
          width={chartW}
          height={chartH}
          fill="#111118"
          rx={4}
        />

        {/* Drought bands */}
        {DROUGHTS.map((d, i) => {
          const x = xScale(d - 0.4);
          const w = xScale(d + 0.4) - x;
          return (
            <rect
              key={i}
              x={x}
              y={margin.top}
              width={w}
              height={chartH}
              fill="#ff2020"
              opacity={d <= cyclesRevealed + 1 ? 0.15 : 0}
            />
          );
        })}

        {/* Grid lines */}
        {[0.1, 0.2, 0.3, 0.4, 0.5].map((v) => (
          <g key={v}>
            <line
              x1={margin.left}
              y1={yScale(v)}
              x2={margin.left + chartW}
              y2={yScale(v)}
              stroke="#222"
              strokeWidth={0.5}
            />
            <text
              x={margin.left - 10}
              y={yScale(v) + 4}
              textAnchor="end"
              fill="#555"
              fontSize={11}
              fontFamily="Georgia, serif"
            >
              {v.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Envelope (rising peak line) */}
        {envelopeRevealed && peakPoints.length >= 2 && (
          <path
            d={peakPoints
              .filter(
                (_, i) => BOUNCES[i].drought * 5 - 1 <= cyclesRevealed
              )
              .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`)
              .join(" ")}
            fill="none"
            stroke="#4ade80"
            strokeWidth={1.5}
            strokeDasharray="6 4"
            opacity={envelopeOpacity}
          />
        )}

        {/* Main trajectory line */}
        {visiblePoints.length >= 2 && (
          <path
            d={pathD}
            fill="none"
            stroke="#60a5fa"
            strokeWidth={3}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {/* Drop arrows during drought */}
        {BOUNCES.map((b, i) => {
          const droughtCycle = b.drought * 5;
          if (droughtCycle > cyclesRevealed) return null;
          const x = xScale(droughtCycle);
          const y1 = yScale(b.pre);
          const y2 = yScale(b.drop);
          return (
            <g key={i}>
              <line
                x1={x}
                y1={y1}
                x2={x}
                y2={y2}
                stroke="#f87171"
                strokeWidth={2}
                strokeDasharray="3 2"
                opacity={0.6}
              />
              <polygon
                points={`${x},${y2} ${x - 4},${y2 - 8} ${x + 4},${y2 - 8}`}
                fill="#f87171"
                opacity={0.6}
              />
            </g>
          );
        })}

        {/* Current position dot */}
        {visiblePoints.length > 0 && (
          <circle
            cx={xScale(
              Math.min(cyclesRevealed, MAX_CYCLE)
            )}
            cy={yScale(currentPhi)}
            r={6}
            fill="#60a5fa"
            stroke="#fff"
            strokeWidth={2}
          />
        )}

        {/* X axis */}
        <line
          x1={margin.left}
          y1={margin.top + chartH}
          x2={margin.left + chartW}
          y2={margin.top + chartH}
          stroke="#444"
          strokeWidth={1}
        />
        <text
          x={margin.left + chartW / 2}
          y={margin.top + chartH + 40}
          textAnchor="middle"
          fill="#888"
          fontSize={14}
          fontFamily="Georgia, serif"
        >
          Evolutionary Cycle
        </text>
        {[0, 5, 10, 15, 20, 25, 29].map((c) => (
          <text
            key={c}
            x={xScale(c)}
            y={margin.top + chartH + 18}
            textAnchor="middle"
            fill="#555"
            fontSize={10}
            fontFamily="Georgia, serif"
          >
            {c}
          </text>
        ))}

        {/* Y axis label */}
        <text
          x={25}
          y={margin.top + chartH / 2}
          textAnchor="middle"
          fill="#888"
          fontSize={14}
          fontFamily="Georgia, serif"
          transform={`rotate(-90, 25, ${margin.top + chartH / 2})`}
        >
          Φ (integration)
        </text>

        {/* Drought labels */}
        {DROUGHTS.map((d, i) => {
          if (d > cyclesRevealed + 1) return null;
          return (
            <text
              key={i}
              x={xScale(d)}
              y={margin.top - 6}
              textAnchor="middle"
              fill="#f87171"
              fontSize={10}
              fontFamily="Georgia, serif"
              opacity={0.7}
            >
              D{i + 1}
            </text>
          );
        })}
      </svg>

      {/* Current Φ readout */}
      <div
        style={{
          position: "absolute",
          top: margin.top + chartH + 55,
          right: margin.right,
          color: "#60a5fa",
          fontSize: 28,
          fontWeight: 700,
          textAlign: "right",
        }}
      >
        Φ = {currentPhi.toFixed(3)}
      </div>

      {/* Final annotation */}
      {frame > 275 && (
        <div
          style={{
            position: "absolute",
            bottom: 25,
            left: margin.left,
            color: "#4ade80",
            fontSize: 16,
            opacity: finalOpacity,
          }}
        >
          each drought forges — the furnace creates, not selects
        </div>
      )}
    </AbsoluteFill>
  );
};
