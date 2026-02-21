import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  Easing,
} from "remotion";
import { THEMES, ThemeMode } from './themes';

/**
 * Developmental Ordering — Emergence Ladder × Human Development
 *
 * Maps the 10-rung emergence ladder to human developmental timeline (birth → 6 years).
 * Shows which capacities emerge when, with the rung 8 wall as the key prediction.
 *
 * 360 frames @ 30fps = 12 seconds
 */

interface DevRung {
  rung: number;
  label: string;
  ageRange: string;
  ageStart: number; // months
  ageEnd: number;   // months
  color: string;
  evidence: string;
}

const MONTHS_MAX = 72;
const TIMELINE_X = 180;
const TIMELINE_W = 780;
const TIMELINE_Y = 600;

export const DevelopmentalOrderingVideo: React.FC<{ theme?: ThemeMode }> = ({ theme }) => {
  const frame = useCurrentFrame();
  const t = THEMES[theme ?? 'dark'];

  const RUNGS: DevRung[] = [
    { rung: 1, label: "Mood & Arousal", ageRange: "Birth", ageStart: 0, ageEnd: 1, color: t.green, evidence: "approach/withdrawal from day 1" },
    { rung: 2, label: "Habituation", ageRange: "0–3 mo", ageStart: 0, ageEnd: 3, color: t.green, evidence: "novelty preference" },
    { rung: 3, label: "Somatic fear", ageRange: "3–6 mo", ageStart: 3, ageEnd: 6, color: t.green, evidence: "startle, V↓ to threat" },
    { rung: 4, label: "Animism", ageRange: "12–18 mo", ageStart: 12, ageEnd: 18, color: t.green, evidence: "Heider-Simmel agency" },
    { rung: 5, label: "Emotional coherence", ageRange: "18–36 mo", ageStart: 18, ageEnd: 36, color: t.green, evidence: "face-behavior match" },
    { rung: 6, label: "Temporal depth", ageRange: "24–36 mo", ageStart: 24, ageEnd: 36, color: t.yellow, evidence: "episodic memory" },
    { rung: 7, label: "Resilience under stress", ageRange: "24–48 mo", ageStart: 24, ageEnd: 48, color: t.yellow, evidence: "emotion regulation" },
    { rung: 8, label: "Counterfactual / Anxiety", ageRange: "36–54 mo", ageStart: 36, ageEnd: 54, color: t.red, evidence: "false belief + anticipatory fear" },
    { rung: 9, label: "Self-awareness", ageRange: "18–60 mo", ageStart: 18, ageEnd: 60, color: t.red, evidence: "mirror → autobiographical" },
    { rung: 10, label: "Normativity", ageRange: "48–72 mo", ageStart: 48, ageEnd: 72, color: t.red, evidence: "third-party fairness" },
  ];

  const titleOp = interpolate(frame, [0, 20], [0, 1], { extrapolateRight: "clamp" });

  // Timeline appears
  const timelineOp = interpolate(frame, [15, 30], [0, 1], { extrapolateRight: "clamp" });

  // Rungs reveal sequentially
  const rungDelay = 28;

  // Wall highlight
  const wallFrame = 30 + 7 * rungDelay;
  const wallOp = interpolate(frame, [wallFrame, wallFrame + 20], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });

  // Prediction callout
  const predOp = interpolate(frame, [wallFrame + 25, wallFrame + 45], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ backgroundColor: t.bg, fontFamily: "Georgia, serif" }}>
      {/* Title */}
      <div style={{
        position: "absolute", top: 22, width: "100%", textAlign: "center",
        color: t.text, fontSize: 28, fontWeight: 700, opacity: titleOp,
      }}>
        Emergence Ladder × Human Development
      </div>
      <div style={{
        position: "absolute", top: 58, width: "100%", textAlign: "center",
        color: t.muted, fontSize: 14, fontStyle: "italic", opacity: titleOp,
      }}>
        computational requirements predict developmental sequence
      </div>

      <svg width={1080} height={720}>
        {/* Timeline axis */}
        <g opacity={timelineOp}>
          <line x1={TIMELINE_X} y1={TIMELINE_Y} x2={TIMELINE_X + TIMELINE_W} y2={TIMELINE_Y}
            stroke={t.muted} strokeWidth={2} />
          {/* Year markers */}
          {[0, 12, 24, 36, 48, 60, 72].map((m) => {
            const x = TIMELINE_X + (m / MONTHS_MAX) * TIMELINE_W;
            return (
              <g key={m}>
                <line x1={x} y1={TIMELINE_Y - 5} x2={x} y2={TIMELINE_Y + 5} stroke={t.muted} strokeWidth={1.5} />
                <text x={x} y={TIMELINE_Y + 22} textAnchor="middle" fill={t.muted} fontSize={11} fontFamily="Georgia, serif">
                  {m === 0 ? "Birth" : `${m / 12}yr`}
                </text>
              </g>
            );
          })}
          <text x={TIMELINE_X + TIMELINE_W / 2} y={TIMELINE_Y + 42} textAnchor="middle"
            fill={t.muted} fontSize={12} fontFamily="Georgia, serif">
            Age
          </text>
        </g>

        {/* Rung bars */}
        {RUNGS.map((rung, i) => {
          const entryFrame = 30 + i * rungDelay;
          const revealT = interpolate(frame, [entryFrame, entryFrame + 18], [0, 1], {
            extrapolateLeft: "clamp", extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          });
          if (revealT <= 0) return null;

          const barY = 90 + i * 48;
          const barX = TIMELINE_X + (rung.ageStart / MONTHS_MAX) * TIMELINE_W;
          const barW = ((rung.ageEnd - rung.ageStart) / MONTHS_MAX) * TIMELINE_W;

          // Drop line from bar to timeline
          const midX = barX + barW / 2;

          return (
            <g key={i} opacity={revealT}>
              {/* Vertical drop line */}
              <line x1={midX} y1={barY + 18} x2={midX} y2={TIMELINE_Y}
                stroke={rung.color} strokeWidth={0.5} opacity={0.3} strokeDasharray="3 3" />

              {/* Bar */}
              <rect x={barX} y={barY} width={barW * revealT} height={32}
                fill={rung.color} opacity={0.15} rx={4}
                stroke={rung.color} strokeWidth={1.5} />

              {/* Rung number */}
              <text x={barX - 30} y={barY + 20} textAnchor="end"
                fill={rung.color} fontSize={14} fontWeight={700} fontFamily="monospace">
                {rung.rung}
              </text>

              {/* Label */}
              <text x={barX + 6} y={barY + 14} fill={t.text} fontSize={12} fontWeight={600}
                fontFamily="Georgia, serif" opacity={revealT}>
                {rung.label}
              </text>

              {/* Evidence text */}
              <text x={barX + 6} y={barY + 28} fill={t.muted} fontSize={9}
                fontFamily="Georgia, serif" opacity={revealT * 0.8}>
                {rung.evidence}
              </text>
            </g>
          );
        })}

        {/* THE WALL between rungs 7 and 8 */}
        {wallOp > 0 && (
          <g opacity={wallOp}>
            <line x1={90} y1={90 + 7 * 48 - 6} x2={TIMELINE_X + TIMELINE_W + 20} y2={90 + 7 * 48 - 6}
              stroke={t.yellow} strokeWidth={3} strokeDasharray="8 4" />
            <text x={60} y={90 + 7 * 48 - 2} fill={t.yellow} fontSize={11} fontWeight={700}
              fontFamily="Georgia, serif" textAnchor="end">
              WALL
            </text>
            <text x={60} y={90 + 7 * 48 + 12} fill={t.yellow} fontSize={9}
              fontFamily="Georgia, serif" textAnchor="end" opacity={0.7}>
              agency required
            </text>
          </g>
        )}

        {/* Key prediction callout */}
        {predOp > 0 && (
          <g opacity={predOp}>
            <rect x={620} y={80} width={420} height={65} fill={t.panel} rx={6}
              stroke={t.yellow} strokeWidth={1.5} />
            <text x={635} y={100} fill={t.yellow} fontSize={13} fontWeight={700} fontFamily="Georgia, serif">
              KEY PREDICTION
            </text>
            <text x={635} y={118} fill={t.text} fontSize={11} fontFamily="Georgia, serif">
              Anticipatory anxiety must co-emerge with
            </text>
            <text x={635} y={134} fill={t.text} fontSize={11} fontFamily="Georgia, serif">
              false belief task, not precede it (~age 3-4)
            </text>
          </g>
        )}

        {/* Pre-reflective / Reflective labels */}
        {wallOp > 0 && (
          <g opacity={wallOp * 0.6}>
            <text x={40} y={300} fill={t.green} fontSize={13} fontWeight={700}
              fontFamily="Georgia, serif" transform="rotate(-90, 40, 300)">
              PRE-REFLECTIVE
            </text>
            <text x={40} y={500} fill={t.red} fontSize={13} fontWeight={700}
              fontFamily="Georgia, serif" transform="rotate(-90, 40, 500)">
              REFLECTIVE
            </text>
          </g>
        )}
      </svg>
    </AbsoluteFill>
  );
};
