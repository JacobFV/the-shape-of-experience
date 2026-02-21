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
 * NecessityArgument — Introduction video
 * 3-step animated argument: physics → information → affect
 * Each step builds on previous with visual accumulation
 * Ends with the six affect dimensions appearing as inevitable coordinates
 */

interface Step {
  label: string;
  premise: string;
  conclusion: string;
  color: string;
  icon: string;
}

const DIMS = ["V", "A", "Phi", "R", "CF", "SM"] as const;

function getSteps(t: typeof THEMES.dark): Step[] {
  return [
    {
      label: "Physics",
      premise: "Far-from-equilibrium systems must maintain boundaries",
      conclusion: "→ thermodynamic necessity",
      color: t.green,
      icon: "⚛",
    },
    {
      label: "Information",
      premise: "Boundary maintenance requires tracking state under uncertainty",
      conclusion: "→ computational necessity",
      color: t.blue,
      icon: "⊕",
    },
    {
      label: "Affect",
      premise: "State tracking under resource constraints produces recurring geometry",
      conclusion: "→ structural necessity",
      color: t.yellow,
      icon: "◆",
    },
  ];
}

function getDimensions(t: typeof THEMES.dark) {
  return [
    { label: "Valence", short: "V", color: t.red },
    { label: "Arousal", short: "A", color: t.orange },
    { label: "Integration", short: "Φ", color: t.yellow },
    { label: "Eff. Rank", short: "r", color: t.green },
    { label: "CF Weight", short: "CF", color: t.cyan },
    { label: "Self-Model", short: "SM", color: t.blue },
  ];
}

export const NecessityArgumentVideo: React.FC<{ theme?: ThemeMode }> = ({ theme }) => {
  const t = THEMES[theme ?? "dark"];
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();

  const STEPS = getSteps(t);
  const DIMENSIONS = getDimensions(t);

  // Phase timing: each step gets ~80 frames, then dimensions appear
  const step0Start = 20;
  const step1Start = 100;
  const step2Start = 180;
  const dimStart = 260;

  const titleOp = interpolate(frame, [0, 20], [0, 1], { extrapolateRight: "clamp" });

  // Step card geometry
  const cardW = 440;
  const cardH = 80;
  const cardX = (width - cardW) / 2;
  const cardGap = 14;

  return (
    <AbsoluteFill style={{ backgroundColor: t.bg }}>
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 28,
          width: "100%",
          textAlign: "center",
          fontFamily: "Georgia, serif",
          opacity: titleOp,
        }}
      >
        <div style={{ fontSize: 28, fontWeight: 700, color: t.text }}>
          The Necessity Argument
        </div>
        <div style={{ fontSize: 14, color: t.muted, marginTop: 4 }}>
          why affect is inevitable, not accidental
        </div>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}
      >
        {/* Step cards */}
        {STEPS.map((step, i) => {
          const stepStart = [step0Start, step1Start, step2Start][i];
          const y = 100 + i * (cardH + cardGap);

          const buildOp = interpolate(frame, [stepStart, stepStart + 20], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          });
          const slideX = interpolate(frame, [stepStart, stepStart + 25], [-30, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          });

          return (
            <g key={i} opacity={buildOp} transform={`translate(${slideX}, 0)`}>
              {/* Card background */}
              <rect
                x={cardX}
                y={y}
                width={cardW}
                height={cardH}
                rx={8}
                fill={step.color}
                fillOpacity={0.06}
                stroke={step.color}
                strokeWidth={1.2}
              />

              {/* Step number + icon */}
              <text
                x={cardX + 30}
                y={y + cardH / 2 - 8}
                textAnchor="middle"
                dominantBaseline="central"
                fill={step.color}
                fontSize={22}
              >
                {step.icon}
              </text>
              <text
                x={cardX + 30}
                y={y + cardH / 2 + 14}
                textAnchor="middle"
                fill={step.color}
                fontSize={10}
                fontWeight={700}
                fontFamily="Georgia, serif"
              >
                {i + 1}
              </text>

              {/* Label */}
              <text
                x={cardX + 60}
                y={y + 24}
                fill={step.color}
                fontSize={16}
                fontWeight={700}
                fontFamily="Georgia, serif"
              >
                {step.label}
              </text>

              {/* Premise */}
              <text
                x={cardX + 60}
                y={y + 44}
                fill={t.text}
                fontSize={11}
                fontFamily="Georgia, serif"
              >
                {step.premise}
              </text>

              {/* Conclusion */}
              <text
                x={cardX + 60}
                y={y + 62}
                fill={t.muted}
                fontSize={10}
                fontStyle="italic"
                fontFamily="Georgia, serif"
              >
                {step.conclusion}
              </text>

              {/* Connecting arrow to next step */}
              {i < STEPS.length - 1 && (
                <g opacity={interpolate(frame, [stepStart + 30, stepStart + 50], [0, 0.5], {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                })}>
                  <line
                    x1={cardX + cardW / 2}
                    y1={y + cardH}
                    x2={cardX + cardW / 2}
                    y2={y + cardH + cardGap}
                    stroke={step.color}
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                  />
                  <text
                    x={cardX + cardW / 2 + 12}
                    y={y + cardH + cardGap / 2 + 1}
                    fill={t.muted}
                    fontSize={10}
                    fontFamily="Georgia, serif"
                  >
                    therefore
                  </text>
                </g>
              )}
            </g>
          );
        })}

        {/* Dimensions appearing at the end */}
        {frame >= dimStart && (
          <g>
            {/* "therefore" arrow from step 3 to dimensions */}
            <line
              x1={width / 2}
              y1={100 + 3 * (cardH + cardGap) - cardGap + 5}
              x2={width / 2}
              y2={100 + 3 * (cardH + cardGap) - cardGap + 25}
              stroke={t.yellow}
              strokeWidth={1.5}
              opacity={interpolate(frame, [dimStart, dimStart + 15], [0, 0.5], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              })}
            />

            {/* Six dimension badges */}
            {DIMENSIONS.map((dim, i) => {
              const dimOp = interpolate(
                frame,
                [dimStart + i * 8, dimStart + i * 8 + 15],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
              );
              const badgeW = 100;
              const badgeH = 40;
              const cols = 6;
              const totalW = cols * badgeW + (cols - 1) * 8;
              const startX = (width - totalW) / 2;
              const bx = startX + i * (badgeW + 8);
              const by = 395;

              return (
                <g key={dim.short} opacity={dimOp}>
                  <rect
                    x={bx}
                    y={by}
                    width={badgeW}
                    height={badgeH}
                    rx={6}
                    fill={dim.color}
                    fillOpacity={0.12}
                    stroke={dim.color}
                    strokeWidth={1}
                  />
                  <text
                    x={bx + badgeW / 2}
                    y={by + 14}
                    textAnchor="middle"
                    fill={dim.color}
                    fontSize={14}
                    fontWeight={700}
                    fontFamily="Georgia, serif"
                  >
                    {dim.short}
                  </text>
                  <text
                    x={bx + badgeW / 2}
                    y={by + 30}
                    textAnchor="middle"
                    fill={t.muted}
                    fontSize={8.5}
                    fontFamily="Georgia, serif"
                  >
                    {dim.label}
                  </text>
                </g>
              );
            })}

            {/* Final label */}
            <text
              x={width / 2}
              y={455}
              textAnchor="middle"
              fill={t.muted}
              fontSize={13}
              fontStyle="italic"
              fontFamily="Georgia, serif"
              opacity={interpolate(frame, [dimStart + 50, dimStart + 70], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              })}
            >
              six inevitable coordinates of any viable system
            </text>
          </g>
        )}
      </svg>
    </AbsoluteFill>
  );
};
