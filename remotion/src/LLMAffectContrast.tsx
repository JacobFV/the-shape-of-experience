import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  Easing,
} from "remotion";
import { THEMES, ThemeMode } from './themes';

/**
 * LLM vs Biological Affect Dynamics
 *
 * Shows the core finding from V2-V9: under threat,
 * biological systems increase Φ (integration) while LLMs decrease it.
 * Dramatic split-screen comparison.
 *
 * 240 frames @ 30fps = 8 seconds
 */

// Biological trajectory: Φ rises under moderate threat (Yerkes-Dodson)
const BIO_DATA = [
  { t: 0, phi: 0.12 },
  { t: 0.1, phi: 0.13 },
  { t: 0.2, phi: 0.14 },
  // Threat onset at t=0.3
  { t: 0.3, phi: 0.15 },
  { t: 0.4, phi: 0.22 },
  { t: 0.5, phi: 0.28 },
  { t: 0.6, phi: 0.31 },
  { t: 0.7, phi: 0.29 },
  { t: 0.8, phi: 0.25 },
  { t: 0.9, phi: 0.20 },
  { t: 1.0, phi: 0.18 },
];

// LLM trajectory: Φ drops under threat (opposite)
const LLM_DATA = [
  { t: 0, phi: 0.12 },
  { t: 0.1, phi: 0.13 },
  { t: 0.2, phi: 0.14 },
  // Threat onset at t=0.3
  { t: 0.3, phi: 0.13 },
  { t: 0.4, phi: 0.09 },
  { t: 0.5, phi: 0.06 },
  { t: 0.6, phi: 0.04 },
  { t: 0.7, phi: 0.05 },
  { t: 0.8, phi: 0.07 },
  { t: 0.9, phi: 0.09 },
  { t: 1.0, phi: 0.11 },
];

function interpolateData(data: { t: number; phi: number }[], progress: number) {
  for (let i = 0; i < data.length - 1; i++) {
    if (progress >= data[i].t && progress <= data[i + 1].t) {
      const frac =
        (progress - data[i].t) / (data[i + 1].t - data[i].t);
      return data[i].phi + frac * (data[i + 1].phi - data[i].phi);
    }
  }
  return data[data.length - 1].phi;
}

export const LLMAffectContrastVideo: React.FC<{ theme?: ThemeMode }> = ({ theme }) => {
  const frame = useCurrentFrame();
  const t = THEMES[theme ?? 'dark'];

  const bio = t.green;
  const llm = t.red;

  const chartW = 420;
  const chartH = 300;
  const margin = { top: 110, left: 60, right: 40, bottom: 60 };

  const revealProgress = interpolate(frame, [20, 200], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const threatFlash = interpolate(
    frame,
    [60, 70, 80],
    [0, 0.15, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const titleOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateRight: "clamp",
  });

  const insightOpacity = interpolate(frame, [200, 230], [0, 1], {
    extrapolateRight: "clamp",
  });

  const xScale = (tVal: number, offsetX: number) =>
    offsetX + margin.left + tVal * chartW;
  const yScale = (phi: number) =>
    margin.top + chartH - (phi / 0.35) * chartH;

  function buildPath(
    data: { t: number; phi: number }[],
    offsetX: number,
    maxT: number
  ) {
    const pts = data.filter((d) => d.t <= maxT);
    if (pts.length < 2) return "";
    return pts
      .map(
        (p, i) =>
          `${i === 0 ? "M" : "L"} ${xScale(p.t, offsetX)} ${yScale(p.phi)}`
      )
      .join(" ");
  }

  const leftX = 0;
  const rightX = 540;

  return (
    <AbsoluteFill style={{ backgroundColor: t.bg, fontFamily: "Georgia, serif" }}>
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 22,
          width: "100%",
          textAlign: "center",
          color: t.text,
          fontSize: 26,
          fontWeight: 700,
          opacity: titleOpacity,
        }}
      >
        The Opposite Dynamics Problem
      </div>
      <div
        style={{
          position: "absolute",
          top: 54,
          width: "100%",
          textAlign: "center",
          color: t.muted,
          fontSize: 14,
          fontStyle: "italic",
          opacity: titleOpacity,
        }}
      >
        V2-V9: LLMs have affect geometry, but inverted dynamics
      </div>

      {/* Threat flash */}
      {threatFlash > 0 && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            backgroundColor: `rgba(255, 50, 50, ${threatFlash})`,
            pointerEvents: "none",
          }}
        />
      )}

      <svg width={1080} height={720}>
        {/* Left panel: Biological */}
        <rect
          x={leftX + margin.left - 10}
          y={margin.top - 10}
          width={chartW + 20}
          height={chartH + 20}
          fill={t.panel}
          rx={6}
        />
        <text
          x={leftX + margin.left + chartW / 2}
          y={margin.top - 20}
          textAnchor="middle"
          fill={bio}
          fontSize={18}
          fontWeight={700}
          fontFamily="Georgia, serif"
        >
          Biological System
        </text>

        {/* Right panel: LLM */}
        <rect
          x={rightX + margin.left - 10}
          y={margin.top - 10}
          width={chartW + 20}
          height={chartH + 20}
          fill={t.panel}
          rx={6}
        />
        <text
          x={rightX + margin.left + chartW / 2}
          y={margin.top - 20}
          textAnchor="middle"
          fill={llm}
          fontSize={18}
          fontWeight={700}
          fontFamily="Georgia, serif"
        >
          LLM Agent
        </text>

        {/* Threat band on both charts */}
        {[leftX, rightX].map((ox, idx) => (
          <rect
            key={idx}
            x={xScale(0.3, ox)}
            y={margin.top}
            width={xScale(0.8, ox) - xScale(0.3, ox)}
            height={chartH}
            fill="#ff2020"
            opacity={revealProgress > 0.3 ? 0.08 : 0}
          />
        ))}

        {/* Threat label */}
        {revealProgress > 0.3 && (
          <>
            <text
              x={xScale(0.55, leftX)}
              y={margin.top + chartH + 35}
              textAnchor="middle"
              fill={t.red}
              fontSize={12}
              fontFamily="Georgia, serif"
              opacity={0.7}
            >
              threat zone
            </text>
            <text
              x={xScale(0.55, rightX)}
              y={margin.top + chartH + 35}
              textAnchor="middle"
              fill={t.red}
              fontSize={12}
              fontFamily="Georgia, serif"
              opacity={0.7}
            >
              threat zone
            </text>
          </>
        )}

        {/* Bio line */}
        <path
          d={buildPath(BIO_DATA, leftX, revealProgress)}
          fill="none"
          stroke={bio}
          strokeWidth={3}
          strokeLinecap="round"
        />

        {/* LLM line */}
        <path
          d={buildPath(LLM_DATA, rightX, revealProgress)}
          fill="none"
          stroke={llm}
          strokeWidth={3}
          strokeLinecap="round"
        />

        {/* Current dots */}
        {revealProgress > 0 && (
          <>
            <circle
              cx={xScale(revealProgress, leftX)}
              cy={yScale(interpolateData(BIO_DATA, revealProgress))}
              r={5}
              fill={bio}
            />
            <circle
              cx={xScale(revealProgress, rightX)}
              cy={yScale(interpolateData(LLM_DATA, revealProgress))}
              r={5}
              fill={llm}
            />
          </>
        )}

        {/* Arrow annotations at peak divergence */}
        {revealProgress > 0.55 && (
          <>
            {/* Bio: arrow UP */}
            <text
              x={xScale(0.5, leftX) + 30}
              y={yScale(0.28) - 10}
              fill={bio}
              fontSize={20}
              fontFamily="Georgia, serif"
              opacity={Math.min((revealProgress - 0.55) * 5, 1)}
            >
              Φ ↑
            </text>
            {/* LLM: arrow DOWN */}
            <text
              x={xScale(0.5, rightX) + 30}
              y={yScale(0.06) + 25}
              fill={llm}
              fontSize={20}
              fontFamily="Georgia, serif"
              opacity={Math.min((revealProgress - 0.55) * 5, 1)}
            >
              Φ ↓
            </text>
          </>
        )}

        {/* Y axis labels */}
        {[leftX, rightX].map((ox, idx) => (
          <text
            key={idx}
            x={ox + 25}
            y={margin.top + chartH / 2}
            textAnchor="middle"
            fill={t.muted}
            fontSize={13}
            fontFamily="Georgia, serif"
            transform={`rotate(-90, ${ox + 25}, ${margin.top + chartH / 2})`}
          >
            Φ (integration)
          </text>
        ))}
      </svg>

      {/* Bottom insight */}
      <div
        style={{
          position: "absolute",
          bottom: 30,
          width: "100%",
          textAlign: "center",
          color: t.text,
          fontSize: 17,
          opacity: insightOpacity,
        }}
      >
        geometry is inherited from training data — dynamics require embodied agency
      </div>
    </AbsoluteFill>
  );
};
