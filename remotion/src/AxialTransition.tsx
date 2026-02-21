import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
  Easing,
} from "remotion";
import { THEMES, ThemeMode } from './themes';

/**
 * AxialTransition — Part VI video
 * Timeline animation: pre-Axial (low ι, participatory) →
 * Axial innovations (4 shown) → post-Axial (rising ι, reflective)
 * Key figures/traditions appear at timeline markers
 */

interface AxialEvent {
  year: number;
  label: string;
  tradition: string;
  color: string;
  frameStart: number;
}

// ι trajectory over frames
const IOTA_KEYFRAMES = [
  { f: 0, iota: 0.15 },
  { f: 50, iota: 0.18 },
  { f: 90, iota: 0.25 },
  { f: 150, iota: 0.35 },
  { f: 220, iota: 0.45 },
  { f: 280, iota: 0.55 },
  { f: 360, iota: 0.65 },
];

function getIotaAtFrame(frame: number): number {
  for (let i = 0; i < IOTA_KEYFRAMES.length - 1; i++) {
    if (frame >= IOTA_KEYFRAMES[i].f && frame <= IOTA_KEYFRAMES[i + 1].f) {
      const t =
        (frame - IOTA_KEYFRAMES[i].f) /
        (IOTA_KEYFRAMES[i + 1].f - IOTA_KEYFRAMES[i].f);
      return IOTA_KEYFRAMES[i].iota + t * (IOTA_KEYFRAMES[i + 1].iota - IOTA_KEYFRAMES[i].iota);
    }
  }
  return IOTA_KEYFRAMES[IOTA_KEYFRAMES.length - 1].iota;
}

// Radar profile that changes with ι
function profileForIota(iota: number): number[] {
  // V, A, Φ, r_eff, CF, SM
  return [
    0.5 + (1 - iota) * 0.2, // V slightly higher at low ι
    0.3 + iota * 0.3, // A rises with ι
    0.7 - iota * 0.3, // Φ drops with ι
    0.4 + iota * 0.3, // r_eff rises (more analytical)
    0.2 + iota * 0.5, // CF rises (more counterfactual)
    0.2 + iota * 0.6, // SM rises sharply (self-model)
  ];
}

const AXES = ["V", "A", "Φ", "r", "CF", "SM"];

export const AxialTransitionVideo: React.FC<{ theme?: ThemeMode }> = ({ theme }) => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const t = THEMES[theme ?? 'dark'];

  const EVENTS: AxialEvent[] = [
    {
      year: -800,
      label: "Pre-Axial",
      tradition: "participatory cultures",
      color: t.green,
      frameStart: 0,
    },
    {
      year: -600,
      label: "Greek Philosophy",
      tradition: "Socrates, Plato",
      color: t.violet,
      frameStart: 60,
    },
    {
      year: -500,
      label: "Buddhism",
      tradition: "Gautama Buddha",
      color: t.yellow,
      frameStart: 110,
    },
    {
      year: -550,
      label: "Confucianism",
      tradition: "Confucius",
      color: t.pink,
      frameStart: 160,
    },
    {
      year: -600,
      label: "Hebrew Prophets",
      tradition: "Isaiah, Jeremiah",
      color: t.blue,
      frameStart: 210,
    },
    {
      year: 0,
      label: "Post-Axial",
      tradition: "reflective consciousness",
      color: t.red,
      frameStart: 280,
    },
  ];

  const titleOp = interpolate(frame, [0, 25], [0, 1], { extrapolateRight: "clamp" });

  const iota = getIotaAtFrame(frame);
  const profile = profileForIota(iota);

  // Timeline geometry
  const tlY = height - 110;
  const tlX1 = 80;
  const tlX2 = width - 80;
  const tlW = tlX2 - tlX1;

  // Progress along timeline
  const tlProgress = interpolate(frame, [0, 340], [0, 1], {
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });
  const progressX = tlX1 + tlProgress * tlW;

  // Radar chart
  const radarCx = width / 2;
  const radarCy = height / 2 - 30;
  const radarR = 120;

  const radarPoint = (axisIdx: number, value: number): [number, number] => {
    const angle = (axisIdx / 6) * Math.PI * 2 - Math.PI / 2;
    return [
      radarCx + Math.cos(angle) * value * radarR,
      radarCy + Math.sin(angle) * value * radarR,
    ];
  };

  const profilePath =
    profile
      .map((v, i) => {
        const [x, y] = radarPoint(i, v);
        return `${i === 0 ? "M" : "L"} ${x} ${y}`;
      })
      .join(" ") + " Z";

  // ι color gradient
  const iotaColor = `hsl(${120 - iota * 120}, 60%, 55%)`;

  return (
    <AbsoluteFill style={{ backgroundColor: t.bg }}>
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 22,
          width: "100%",
          textAlign: "center",
          fontFamily: "Georgia, serif",
          opacity: titleOp,
        }}
      >
        <div style={{ fontSize: 26, fontWeight: 700, color: t.text }}>
          The Axial Transition
        </div>
        <div style={{ fontSize: 13, color: t.muted, marginTop: 4 }}>
          when ι began its long rise
        </div>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}
      >
        {/* Radar grid */}
        {[0.25, 0.5, 0.75, 1.0].map((level) => {
          const pts = AXES.map((_, i) => radarPoint(i, level));
          const d =
            pts.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x} ${y}`).join(" ") + " Z";
          return (
            <path key={level} d={d} fill="none" stroke={t.border}
              strokeWidth={level === 1 ? 0.8 : 0.4} />
          );
        })}

        {/* Axis labels */}
        {AXES.map((label, i) => {
          const [ex, ey] = radarPoint(i, 1.0);
          const [lx, ly] = radarPoint(i, 1.14);
          return (
            <g key={label}>
              <line x1={radarCx} y1={radarCy} x2={ex} y2={ey}
                stroke={t.border} strokeWidth={0.5} />
              <text x={lx} y={ly} textAnchor="middle" dominantBaseline="central"
                fill={t.muted} fontSize={11} fontFamily="Georgia, serif">
                {label}
              </text>
            </g>
          );
        })}

        {/* Profile */}
        <path d={profilePath} fill={iotaColor} fillOpacity={0.2}
          stroke={iotaColor} strokeWidth={2} />
        {profile.map((v, i) => {
          const [x, y] = radarPoint(i, v);
          return <circle key={i} cx={x} cy={y} r={3.5} fill={iotaColor} />;
        })}

        {/* ι readout */}
        <text x={radarCx} y={radarCy + radarR + 28} textAnchor="middle"
          fill={iotaColor} fontSize={18} fontWeight={700} fontFamily="Georgia, serif">
          ι = {iota.toFixed(2)}
        </text>

        {/* Timeline bar */}
        <line x1={tlX1} y1={tlY} x2={tlX2} y2={tlY}
          stroke={t.muted} strokeWidth={2} />
        {/* Progress line */}
        <line x1={tlX1} y1={tlY} x2={progressX} y2={tlY}
          stroke={iotaColor} strokeWidth={3} />
        {/* Current position marker */}
        <circle cx={progressX} cy={tlY} r={5} fill={iotaColor} />

        {/* Timeline labels */}
        <text x={tlX1} y={tlY + 20} textAnchor="middle" fill={t.muted}
          fontSize={10} fontFamily="Georgia, serif">
          800 BCE
        </text>
        <text x={tlX2} y={tlY + 20} textAnchor="middle" fill={t.muted}
          fontSize={10} fontFamily="Georgia, serif">
          0 CE
        </text>

        {/* Event markers */}
        {EVENTS.map((event, i) => {
          const eventProgress = interpolate(
            event.frameStart,
            [0, 340],
            [0, 1],
            { extrapolateRight: "clamp" },
          );
          const ex = tlX1 + eventProgress * tlW;
          const isVisible = frame >= event.frameStart - 5;
          const eventOpacity = isVisible
            ? interpolate(frame, [event.frameStart - 5, event.frameStart + 15], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              })
            : 0;

          return (
            <g key={i} opacity={eventOpacity}>
              <line x1={ex} y1={tlY - 8} x2={ex} y2={tlY + 8}
                stroke={event.color} strokeWidth={2} />
              <text x={ex} y={tlY - 14} textAnchor="middle" fill={event.color}
                fontSize={11} fontWeight={700} fontFamily="Georgia, serif">
                {event.label}
              </text>
              <text x={ex} y={tlY + 34} textAnchor="middle" fill={t.muted}
                fontSize={9} fontStyle="italic" fontFamily="Georgia, serif">
                {event.tradition}
              </text>
            </g>
          );
        })}
      </svg>
    </AbsoluteFill>
  );
};
