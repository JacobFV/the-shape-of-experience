import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  Easing,
} from "remotion";

/**
 * Language Emergence Animation (V35)
 *
 * 10 agents develop referential communication under partial observability.
 * Shows: agents with speech bubbles, symbol entropy rising, BUT Φ staying flat.
 * The punchline: language is cheap, like geometry. Doesn't cross rung 8.
 *
 * 300 frames @ 30fps = 10 seconds
 */

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(35);

// Agent positions in a circular arrangement
const N_AGENTS = 10;
const AGENTS = Array.from({ length: N_AGENTS }, (_, i) => {
  const angle = (i / N_AGENTS) * Math.PI * 2 - Math.PI / 2;
  return {
    x: 270 + Math.cos(angle) * 160,
    y: 320 + Math.sin(angle) * 160,
    color: `hsl(${(i / N_AGENTS) * 360}, 60%, 55%)`,
    symbol: Math.floor(rand() * 8), // current emitted symbol (0-7)
  };
});

const SYMBOLS = ["▲", "●", "■", "◆", "★", "▼", "◀", "▶"];

// Symbol entropy trajectory (starts low, rises to ~2.5 bits)
const ENTROPY_DATA = [
  { t: 0, h: 0.3 },
  { t: 0.1, h: 0.5 },
  { t: 0.2, h: 0.8 },
  { t: 0.3, h: 1.1 },
  { t: 0.4, h: 1.5 },
  { t: 0.5, h: 1.8 },
  { t: 0.6, h: 2.0 },
  { t: 0.7, h: 2.2 },
  { t: 0.8, h: 2.35 },
  { t: 0.9, h: 2.42 },
  { t: 1.0, h: 2.48 },
];

// Φ trajectory (stays flat ~0.074)
const PHI_DATA = [
  { t: 0, phi: 0.072 },
  { t: 0.2, phi: 0.075 },
  { t: 0.4, phi: 0.071 },
  { t: 0.6, phi: 0.076 },
  { t: 0.8, phi: 0.073 },
  { t: 1.0, phi: 0.074 },
];

function lerp(data: { t: number; [k: string]: number }[], progress: number, key: string) {
  for (let i = 0; i < data.length - 1; i++) {
    if (progress >= data[i].t && progress <= data[i + 1].t) {
      const frac = (progress - data[i].t) / (data[i + 1].t - data[i].t);
      return data[i][key] + frac * (data[i + 1][key] - data[i][key]);
    }
  }
  return data[data.length - 1][key];
}

export const LanguageEmergenceVideo: React.FC = () => {
  const frame = useCurrentFrame();

  const progress = interpolate(frame, [20, 250], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const titleOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateRight: "clamp",
  });

  const punchlineOpacity = interpolate(frame, [255, 290], [0, 1], {
    extrapolateRight: "clamp",
  });

  const currentEntropy = lerp(ENTROPY_DATA, progress, "h");
  const currentPhi = lerp(PHI_DATA, progress, "phi");

  // Communication lines between nearby agents (pulsing)
  const commPulse = Math.sin(frame * 0.15) * 0.5 + 0.5;
  const commActive = progress > 0.1;

  // Symbol cycling (changes over time)
  const symbolCycle = Math.floor(frame / 20);

  // Chart dimensions
  const chartX = 560;
  const chartY = 130;
  const chartW = 460;
  const chartH = 200;

  const xScale = (t: number) => chartX + 30 + t * (chartW - 60);
  const yScaleEntropy = (h: number) => chartY + chartH - (h / 3.0) * chartH;
  const yScalePhi = (phi: number) => chartY + chartH + 220 - (phi / 0.12) * 180;

  function buildPath(data: { t: number; [k: string]: number }[], key: string, yFn: (v: number) => number, maxT: number) {
    const pts = data.filter((d) => d.t <= maxT);
    if (pts.length < 2) return "";
    return pts
      .map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.t)} ${yFn(p[key])}`)
      .join(" ");
  }

  return (
    <AbsoluteFill style={{ backgroundColor: "#0a0a0f", fontFamily: "Georgia, serif" }}>
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 22,
          left: 40,
          color: "#e0e0e0",
          fontSize: 26,
          fontWeight: 700,
          opacity: titleOpacity,
        }}
      >
        Language Is Cheap
      </div>
      <div
        style={{
          position: "absolute",
          top: 54,
          left: 40,
          color: "#888",
          fontSize: 14,
          fontStyle: "italic",
          opacity: titleOpacity,
        }}
      >
        V35: referential communication emerges in 10/10 seeds
      </div>

      <svg width={1080} height={720}>
        {/* Agent circle (left side) */}
        {/* Observation radius circles (partial obs) */}
        {AGENTS.map((a, i) => (
          <circle
            key={`obs-${i}`}
            cx={a.x}
            cy={a.y}
            r={30}
            fill="none"
            stroke={a.color}
            strokeWidth={0.5}
            strokeDasharray="3 3"
            opacity={0.2 * progress}
          />
        ))}

        {/* Communication lines */}
        {commActive &&
          AGENTS.map((a, i) =>
            AGENTS.slice(i + 1).map((b, j) => {
              const dist = Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
              if (dist > 250) return null;
              return (
                <line
                  key={`comm-${i}-${j}`}
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke="#a78bfa"
                  strokeWidth={1}
                  opacity={commPulse * 0.2 * progress}
                />
              );
            })
          )}

        {/* Agents */}
        {AGENTS.map((a, i) => {
          const agentSymbol = SYMBOLS[(a.symbol + symbolCycle + i) % 8];
          return (
            <g key={i}>
              {/* Agent body */}
              <circle
                cx={a.x}
                cy={a.y}
                r={18}
                fill="#1a1a24"
                stroke={a.color}
                strokeWidth={2}
              />
              {/* Symbol bubble */}
              {progress > 0.15 && (
                <g opacity={Math.min((progress - 0.15) * 4, 0.9)}>
                  <rect
                    x={a.x + 14}
                    y={a.y - 28}
                    width={22}
                    height={22}
                    fill="#1e1e2e"
                    stroke="#a78bfa"
                    strokeWidth={1}
                    rx={4}
                  />
                  <text
                    x={a.x + 25}
                    y={a.y - 12}
                    textAnchor="middle"
                    fill="#a78bfa"
                    fontSize={14}
                    fontFamily="monospace"
                  >
                    {agentSymbol}
                  </text>
                </g>
              )}
            </g>
          );
        })}

        {/* Label */}
        <text
          x={270}
          y={530}
          textAnchor="middle"
          fill="#a78bfa"
          fontSize={14}
          fontFamily="Georgia, serif"
          opacity={progress > 0.2 ? 1 : 0}
        >
          8 discrete symbols, comm_radius &gt; obs_radius
        </text>

        {/* --- Charts (right side) --- */}

        {/* Symbol Entropy chart */}
        <rect x={chartX} y={chartY} width={chartW} height={chartH} fill="#111118" rx={4} />
        <text
          x={chartX + chartW / 2}
          y={chartY - 8}
          textAnchor="middle"
          fill="#a78bfa"
          fontSize={14}
          fontFamily="Georgia, serif"
        >
          Symbol Entropy (bits)
        </text>
        {/* Max entropy line */}
        <line
          x1={chartX + 30}
          y1={yScaleEntropy(3.0)}
          x2={chartX + chartW - 30}
          y2={yScaleEntropy(3.0)}
          stroke="#a78bfa"
          strokeWidth={0.5}
          strokeDasharray="4 4"
          opacity={0.3}
        />
        <text x={chartX + chartW - 25} y={yScaleEntropy(3.0) + 4} fill="#a78bfa" fontSize={9} fontFamily="monospace" opacity={0.4}>
          max=3.0
        </text>
        {/* Entropy line */}
        <path
          d={buildPath(ENTROPY_DATA, "h", yScaleEntropy, progress)}
          fill="none"
          stroke="#a78bfa"
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        {progress > 0 && (
          <circle
            cx={xScale(progress)}
            cy={yScaleEntropy(currentEntropy)}
            r={5}
            fill="#a78bfa"
          />
        )}
        {/* Value readout */}
        <text
          x={chartX + chartW - 10}
          y={chartY + chartH + 20}
          textAnchor="end"
          fill="#a78bfa"
          fontSize={18}
          fontWeight={700}
          fontFamily="Georgia, serif"
        >
          H = {currentEntropy.toFixed(2)} bits
        </text>

        {/* Φ chart (below) */}
        <rect x={chartX} y={chartY + chartH + 60} width={chartW} height={180} fill="#111118" rx={4} />
        <text
          x={chartX + chartW / 2}
          y={chartY + chartH + 52}
          textAnchor="middle"
          fill="#4ade80"
          fontSize={14}
          fontFamily="Georgia, serif"
        >
          Φ (integration)
        </text>
        {/* V27 baseline */}
        <line
          x1={chartX + 30}
          y1={yScalePhi(0.091)}
          x2={chartX + chartW - 30}
          y2={yScalePhi(0.091)}
          stroke="#4ade80"
          strokeWidth={0.5}
          strokeDasharray="4 4"
          opacity={0.3}
        />
        <text x={chartX + chartW - 25} y={yScalePhi(0.091) + 4} fill="#4ade80" fontSize={9} fontFamily="monospace" opacity={0.4}>
          V27=0.091
        </text>
        {/* Φ line */}
        <path
          d={buildPath(PHI_DATA, "phi", yScalePhi, progress)}
          fill="none"
          stroke="#4ade80"
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        {progress > 0 && (
          <circle
            cx={xScale(Math.min(progress, 1))}
            cy={yScalePhi(currentPhi)}
            r={5}
            fill="#4ade80"
          />
        )}
        {/* Value readout */}
        <text
          x={chartX + chartW - 10}
          y={chartY + chartH + 260}
          textAnchor="end"
          fill="#4ade80"
          fontSize={18}
          fontWeight={700}
          fontFamily="Georgia, serif"
        >
          Φ = {currentPhi.toFixed(3)}
        </text>

        {/* Flat annotation */}
        {progress > 0.7 && (
          <text
            x={chartX + chartW / 2}
            y={chartY + chartH + 280}
            textAnchor="middle"
            fill="#f87171"
            fontSize={12}
            fontFamily="Georgia, serif"
            opacity={Math.min((progress - 0.7) * 3, 1)}
          >
            ← flat. language doesn't lift integration.
          </text>
        )}
      </svg>

      {/* Punchline */}
      <div
        style={{
          position: "absolute",
          bottom: 30,
          width: "100%",
          textAlign: "center",
          color: "#e0e0e0",
          fontSize: 16,
          opacity: punchlineOpacity,
        }}
      >
        language sits at rung 4-5 — like geometry, an inevitability of survival under information asymmetry
      </div>
    </AbsoluteFill>
  );
};
