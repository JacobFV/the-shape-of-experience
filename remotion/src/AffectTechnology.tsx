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
 * AffectTechnology — Part III video
 * Shows how cultural forms (music, ritual, meditation, psychedelics)
 * modulate ι along a spectrum.
 * Animated ι slider + radar chart morphing with each modulation.
 */

interface CulturalForm {
  name: string;
  iotaTarget: number;
  color: string;
  // 6D affect profile: V, A, Φ, r_eff, CF, SM (0-1)
  profile: number[];
  frameStart: number;
}

const AXES = ["V", "A", "Φ", "r", "CF", "SM"];

function getForms(t: typeof THEMES.dark): CulturalForm[] {
  return [
    {
      name: "Baseline",
      iotaTarget: 0.5,
      color: t.muted,
      profile: [0.5, 0.4, 0.4, 0.5, 0.3, 0.5],
      frameStart: 0,
    },
    {
      name: "Music",
      iotaTarget: 0.35,
      color: t.pink,
      profile: [0.7, 0.6, 0.6, 0.7, 0.3, 0.3],
      frameStart: 50,
    },
    {
      name: "Ritual",
      iotaTarget: 0.2,
      color: t.yellow,
      profile: [0.6, 0.7, 0.8, 0.4, 0.2, 0.2],
      frameStart: 120,
    },
    {
      name: "Meditation",
      iotaTarget: 0.15,
      color: t.blue,
      profile: [0.6, 0.2, 0.9, 0.3, 0.1, 0.1],
      frameStart: 190,
    },
    {
      name: "Psychedelics",
      iotaTarget: 0.05,
      color: t.cyan,
      profile: [0.5, 0.9, 0.5, 1.0, 0.9, 0.1],
      frameStart: 260,
    },
  ];
}

export const AffectTechnologyVideo: React.FC<{ theme?: ThemeMode }> = ({ theme }) => {
  const t = THEMES[theme ?? "dark"];
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();

  const FORMS = getForms(t);

  // Determine current form based on frame
  let currentIdx = 0;
  for (let i = FORMS.length - 1; i >= 0; i--) {
    if (frame >= FORMS[i].frameStart) {
      currentIdx = i;
      break;
    }
  }

  const currentForm = FORMS[currentIdx];
  const nextForm = FORMS[Math.min(currentIdx + 1, FORMS.length - 1)];

  // Transition progress
  const transFrames = 30;
  const transProgress =
    currentIdx < FORMS.length - 1
      ? interpolate(
          frame,
          [currentForm.frameStart, currentForm.frameStart + transFrames],
          [0, 1],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) },
        )
      : 1;

  const prevForm = currentIdx > 0 ? FORMS[currentIdx - 1] : FORMS[0];

  // Interpolated values
  const iota =
    prevForm.iotaTarget + (currentForm.iotaTarget - prevForm.iotaTarget) * transProgress;
  const profile = currentForm.profile.map((v, i) =>
    prevForm.profile[i] + (v - prevForm.profile[i]) * transProgress,
  );

  // Title
  const titleOp = interpolate(frame, [0, 20], [0, 1], { extrapolateRight: "clamp" });

  // Radar chart geometry
  const radarCx = width / 2;
  const radarCy = height / 2 + 30;
  const radarR = 140;

  const radarPoint = (axisIdx: number, value: number): [number, number] => {
    const angle = ((axisIdx / 6) * Math.PI * 2) - Math.PI / 2;
    return [
      radarCx + Math.cos(angle) * value * radarR,
      radarCy + Math.sin(angle) * value * radarR,
    ];
  };

  const profilePath = profile
    .map((v, i) => {
      const [x, y] = radarPoint(i, v);
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ") + " Z";

  // Grid polygons
  const gridLevels = [0.25, 0.5, 0.75, 1.0];

  // ι bar position
  const iotaBarX = 60;
  const iotaBarY = 180;
  const iotaBarH = 340;
  const iotaBarW = 24;
  const iotaFillH = iota * iotaBarH;

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
        <div style={{ fontSize: 26, fontWeight: 700, color: t.text }}>
          Affect Technologies
        </div>
        <div style={{ fontSize: 13, color: t.muted, marginTop: 4 }}>
          cultural forms as ι modulators
        </div>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}
      >
        {/* ι slider bar */}
        <text
          x={iotaBarX + iotaBarW / 2}
          y={iotaBarY - 20}
          textAnchor="middle"
          fill={t.text}
          fontSize={14}
          fontWeight={700}
          fontFamily="Georgia, serif"
        >
          ι
        </text>
        {/* Bar background */}
        <rect
          x={iotaBarX}
          y={iotaBarY}
          width={iotaBarW}
          height={iotaBarH}
          rx={4}
          fill={t.panel}
          stroke={t.border}
          strokeWidth={1}
        />
        {/* Bar fill (bottom-up, since low ι = more participatory) */}
        <rect
          x={iotaBarX + 2}
          y={iotaBarY + iotaBarH - iotaFillH + 2}
          width={iotaBarW - 4}
          height={iotaFillH - 4}
          rx={3}
          fill={t.blue}
          opacity={0.6}
        />
        {/* Value label */}
        <text
          x={iotaBarX + iotaBarW / 2}
          y={iotaBarY + iotaBarH + 20}
          textAnchor="middle"
          fill={t.blue}
          fontSize={14}
          fontWeight={700}
          fontFamily="Georgia, serif"
        >
          {iota.toFixed(2)}
        </text>
        {/* Scale labels */}
        <text x={iotaBarX - 8} y={iotaBarY + 6} textAnchor="end" fill={t.muted} fontSize={9}>
          0.0
        </text>
        <text x={iotaBarX - 8} y={iotaBarY + iotaBarH + 4} textAnchor="end" fill={t.muted} fontSize={9}>
          1.0
        </text>
        <text
          x={iotaBarX - 8}
          y={iotaBarY + iotaBarH * 0.3}
          textAnchor="end"
          fill={t.green}
          fontSize={8}
          fontFamily="Georgia, serif"
        >
          participatory
        </text>
        <text
          x={iotaBarX - 8}
          y={iotaBarY + iotaBarH * 0.8}
          textAnchor="end"
          fill={t.red}
          fontSize={8}
          fontFamily="Georgia, serif"
        >
          mechanistic
        </text>

        {/* Radar chart grid */}
        {gridLevels.map((level) => {
          const pts = AXES.map((_, i) => radarPoint(i, level));
          const d =
            pts.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x} ${y}`).join(" ") + " Z";
          return (
            <path
              key={level}
              d={d}
              fill="none"
              stroke={t.border}
              strokeWidth={level === 1 ? 0.8 : 0.4}
            />
          );
        })}

        {/* Axis spokes and labels */}
        {AXES.map((label, i) => {
          const [ex, ey] = radarPoint(i, 1.0);
          const [lx, ly] = radarPoint(i, 1.15);
          return (
            <g key={label}>
              <line
                x1={radarCx}
                y1={radarCy}
                x2={ex}
                y2={ey}
                stroke={t.border}
                strokeWidth={0.5}
              />
              <text
                x={lx}
                y={ly}
                textAnchor="middle"
                dominantBaseline="central"
                fill={t.muted}
                fontSize={12}
                fontFamily="Georgia, serif"
              >
                {label}
              </text>
            </g>
          );
        })}

        {/* Profile polygon */}
        <path
          d={profilePath}
          fill={currentForm.color}
          fillOpacity={0.2}
          stroke={currentForm.color}
          strokeWidth={2}
        />

        {/* Dots at vertices */}
        {profile.map((v, i) => {
          const [x, y] = radarPoint(i, v);
          return (
            <circle key={i} cx={x} cy={y} r={4} fill={currentForm.color} opacity={0.9} />
          );
        })}

        {/* Current form label */}
        <text
          x={radarCx}
          y={radarCy + radarR + 40}
          textAnchor="middle"
          fill={currentForm.color}
          fontSize={22}
          fontWeight={700}
          fontFamily="Georgia, serif"
        >
          {currentForm.name}
        </text>
      </svg>
    </AbsoluteFill>
  );
};
