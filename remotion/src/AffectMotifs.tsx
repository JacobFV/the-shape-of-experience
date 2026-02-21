import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  Easing,
} from "remotion";
import { THEMES, ThemeMode } from "./themes";

/**
 * Affect Motifs — Geometric Signatures of Emotion
 *
 * Radar charts showing how different emotions look in the affect space.
 * Each emotion is a distinct geometric shape defined by its constitutive dimensions.
 *
 * 360 frames @ 30fps = 12 seconds
 */

interface AffectProfile {
  name: string;
  color: string;
  // Normalized 0-1 values for each dimension
  V: number;   // Valence (0=neg, 0.5=neutral, 1=pos)
  A: number;   // Arousal
  Phi: number; // Integration
  R: number;   // Effective Rank
  CF: number;  // Counterfactual Weight
  SM: number;  // Self-Model Salience
  tagline: string;
}

const DIMS = ["V", "A", "Φ", "r_eff", "CF", "SM"] as const;
const DIM_LABELS = ["Valence", "Arousal", "Integration", "Eff. Rank", "CF Weight", "Self-Model"];

function getAffects(t: typeof THEMES.dark): AffectProfile[] {
  return [
    { name: "Joy",       color: t.green,  V: 0.95, A: 0.6, Phi: 0.9,  R: 0.95, CF: 0.3, SM: 0.15, tagline: "positive, unified, expansive, self-light" },
    { name: "Suffering", color: t.red,    V: 0.05, A: 0.7, Phi: 0.95, R: 0.1,  CF: 0.3, SM: 0.7,  tagline: "negative, hyper-integrated, collapsed" },
    { name: "Fear",      color: t.violet, V: 0.1,  A: 0.8, Phi: 0.5,  R: 0.5,  CF: 0.95,SM: 0.9,  tagline: "anticipatory, self-threatened, future-directed" },
    { name: "Anger",     color: t.orange, V: 0.1,  A: 0.95,Phi: 0.4,  R: 0.3,  CF: 0.2, SM: 0.5,  tagline: "energized, externalized, other-compressed" },
    { name: "Curiosity", color: t.teal,   V: 0.8,  A: 0.6, Phi: 0.6,  R: 0.8,  CF: 0.9, SM: 0.15, tagline: "open, branching, welcomed uncertainty" },
    { name: "Grief",     color: t.slate,  V: 0.05, A: 0.3, Phi: 0.8,  R: 0.4,  CF: 0.85,SM: 0.6,  tagline: "persistent, past-directed, unresolvable" },
    { name: "Shame",     color: t.pink,   V: 0.05, A: 0.7, Phi: 0.85, R: 0.3,  CF: 0.4, SM: 0.98, tagline: "self-exposed, integrated negative evaluation" },
  ];
}

function radarPoint(cx: number, cy: number, r: number, angle: number, value: number) {
  const rad = (angle - 90) * (Math.PI / 180);
  return {
    x: cx + r * value * Math.cos(rad),
    y: cy + r * value * Math.sin(rad),
  };
}

function RadarShape({
  cx, cy, r, values, color, opacity, strokeWidth = 2,
}: {
  cx: number; cy: number; r: number;
  values: number[]; color: string; opacity: number; strokeWidth?: number;
}) {
  const n = values.length;
  const points = values.map((v, i) => {
    const angle = (360 / n) * i;
    return radarPoint(cx, cy, r, angle, v);
  });
  const d = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ") + " Z";
  return (
    <g opacity={opacity}>
      <path d={d} fill={color} fillOpacity={0.15} stroke={color} strokeWidth={strokeWidth} />
      {points.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={3} fill={color} />
      ))}
    </g>
  );
}

export const AffectMotifsVideo: React.FC<{ theme?: ThemeMode }> = ({ theme }) => {
  const t = THEMES[theme ?? "dark"];
  const frame = useCurrentFrame();

  const AFFECTS = getAffects(t);

  const titleOp = interpolate(frame, [0, 20], [0, 1], { extrapolateRight: "clamp" });

  // Each affect appears in sequence
  const affectDelay = 40;
  const activeIdx = Math.min(
    Math.floor(interpolate(frame, [30, 30 + AFFECTS.length * affectDelay], [0, AFFECTS.length], { extrapolateRight: "clamp" })),
    AFFECTS.length - 1
  );

  // Grid reveal
  const gridOp = interpolate(frame, [10, 25], [0, 1], { extrapolateRight: "clamp" });

  // Shared radar params
  const radarCX = 300;
  const radarCY = 380;
  const radarR = 200;

  return (
    <AbsoluteFill style={{ backgroundColor: t.bg, fontFamily: "Georgia, serif" }}>
      {/* Title */}
      <div style={{
        position: "absolute", top: 22, width: "100%", textAlign: "center",
        color: t.text, fontSize: 28, fontWeight: 700, opacity: titleOp,
      }}>
        Affect Motifs: The Geometry of Emotion
      </div>
      <div style={{
        position: "absolute", top: 58, width: "100%", textAlign: "center",
        color: t.muted, fontSize: 14, fontStyle: "italic", opacity: titleOp,
      }}>
        each emotion is a distinct shape in the six-dimensional affect space
      </div>

      <svg width={1080} height={720}>
        {/* Radar grid */}
        <g opacity={gridOp}>
          {[0.25, 0.5, 0.75, 1.0].map((ring) => {
            const n = 6;
            const pts = Array.from({ length: n }, (_, i) => {
              const angle = (360 / n) * i;
              return radarPoint(radarCX, radarCY, radarR, angle, ring);
            });
            const d = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ") + " Z";
            return <path key={ring} d={d} fill="none" stroke={t.border} strokeWidth={0.5} />;
          })}
          {/* Axis lines */}
          {DIMS.map((_, i) => {
            const angle = (360 / 6) * i;
            const end = radarPoint(radarCX, radarCY, radarR, angle, 1.0);
            return <line key={i} x1={radarCX} y1={radarCY} x2={end.x} y2={end.y} stroke={t.border} strokeWidth={0.5} />;
          })}
          {/* Axis labels */}
          {DIM_LABELS.map((label, i) => {
            const angle = (360 / 6) * i;
            const p = radarPoint(radarCX, radarCY, radarR + 25, angle, 1.0);
            return (
              <text key={i} x={p.x} y={p.y} textAnchor="middle" dominantBaseline="middle"
                fill={t.muted} fontSize={11} fontFamily="Georgia, serif">
                {label}
              </text>
            );
          })}
        </g>

        {/* Ghost traces of previous affects */}
        {AFFECTS.map((affect, i) => {
          if (i > activeIdx) return null;
          const values = [affect.V, affect.A, affect.Phi, affect.R, affect.CF, affect.SM];
          const isCurrent = i === activeIdx;
          const entryFrame = 30 + i * affectDelay;
          const morphT = interpolate(frame, [entryFrame, entryFrame + 20], [0, 1], {
            extrapolateLeft: "clamp", extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          });
          const morphedValues = values.map((v) => v * morphT);
          return (
            <RadarShape
              key={i}
              cx={radarCX} cy={radarCY} r={radarR}
              values={morphedValues}
              color={affect.color}
              opacity={isCurrent ? 1 : 0.15}
              strokeWidth={isCurrent ? 2.5 : 1}
            />
          );
        })}

        {/* Right panel: affect name and description */}
        {AFFECTS.map((affect, i) => {
          if (i > activeIdx) return null;
          const entryFrame = 30 + i * affectDelay;
          const textOp = interpolate(frame, [entryFrame + 5, entryFrame + 20], [0, 1], {
            extrapolateLeft: "clamp", extrapolateRight: "clamp",
          });
          const isCurrent = i === activeIdx;
          if (!isCurrent) return null;
          return (
            <g key={i} opacity={textOp}>
              {/* Name */}
              <text x={640} y={150} fill={affect.color} fontSize={36} fontWeight={700} fontFamily="Georgia, serif">
                {affect.name}
              </text>
              {/* Tagline */}
              <text x={640} y={185} fill={t.muted} fontSize={14} fontFamily="Georgia, serif" fontStyle="italic">
                {affect.tagline}
              </text>
              {/* Dimension bars */}
              {[
                { label: "V", val: affect.V, desc: affect.V > 0.5 ? "positive" : "negative" },
                { label: "A", val: affect.A, desc: affect.A > 0.7 ? "high" : affect.A < 0.4 ? "low" : "moderate" },
                { label: "Φ", val: affect.Phi, desc: affect.Phi > 0.7 ? "integrated" : "modular" },
                { label: "r_eff", val: affect.R, desc: affect.R > 0.7 ? "expansive" : "collapsed" },
                { label: "CF", val: affect.CF, desc: affect.CF > 0.7 ? "future-directed" : "present" },
                { label: "SM", val: affect.SM, desc: affect.SM > 0.7 ? "self-salient" : "self-light" },
              ].map((dim, j) => {
                const barY = 220 + j * 50;
                const barW = 300;
                const fillW = barW * dim.val;
                return (
                  <g key={j}>
                    <text x={640} y={barY} fill={t.muted} fontSize={12} fontFamily="monospace">{dim.label}</text>
                    <rect x={695} y={barY - 12} width={barW} height={18} fill={t.panel} rx={3} />
                    <rect x={695} y={barY - 12} width={fillW} height={18} fill={affect.color} opacity={0.6} rx={3} />
                    <text x={700 + barW + 10} y={barY} fill={t.muted} fontSize={11} fontFamily="Georgia, serif">{dim.desc}</text>
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* Bottom legend dots */}
        {AFFECTS.map((affect, i) => {
          const dotOp = i <= activeIdx ? 1 : 0.2;
          const dotX = 200 + i * 100;
          return (
            <g key={i} opacity={dotOp}>
              <circle cx={dotX} cy={660} r={6} fill={affect.color} />
              <text x={dotX} y={685} textAnchor="middle" fill={affect.color} fontSize={10} fontFamily="Georgia, serif">
                {affect.name}
              </text>
            </g>
          );
        })}
      </svg>
    </AbsoluteFill>
  );
};
