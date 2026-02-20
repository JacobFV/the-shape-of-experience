import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  Easing,
} from "remotion";

/**
 * Falsification Scoreboard Animation
 *
 * Dramatic reveal of prediction outcomes, one by one.
 * Green = confirmed, red = contradicted, yellow = partial/revised.
 *
 * 360 frames @ 30fps = 12 seconds
 */

interface PredictionRow {
  experiment: string;
  prediction: string;
  outcome: "confirmed" | "contradicted" | "partial" | "null" | "revised";
  brief: string;
}

const PREDICTIONS: PredictionRow[] = [
  { experiment: "V10", prediction: "Forcing functions create geometry", outcome: "contradicted", brief: "All conditions aligned" },
  { experiment: "Exp 8", prediction: "Participatory default (ι ≈ 0.30)", outcome: "confirmed", brief: "20/20 snapshots" },
  { experiment: "V19", prediction: "Furnace creates (not selects)", outcome: "confirmed", brief: "β = +0.704, p < 0.0001" },
  { experiment: "V20", prediction: "ρ wall broken by agency", outcome: "confirmed", brief: "ρ = 0.21 from cycle 0" },
  { experiment: "V27", prediction: "MLP head → Φ increase", outcome: "confirmed", brief: "Φ = 0.245 (record)" },
  { experiment: "V28", prediction: "Bottleneck width matters", outcome: "contradicted", brief: "Gradient coupling instead" },
  { experiment: "V29/31", prediction: "Social target lifts Φ", outcome: "contradicted", brief: "p = 0.93 at 10 seeds" },
  { experiment: "V30", prediction: "Dual > single prediction", outcome: "contradicted", brief: "Self colonizes shared repr" },
  { experiment: "V31", prediction: "~30% develop high Φ", outcome: "confirmed", brief: "30/30/40 split, r = 0.997" },
  { experiment: "V32", prediction: "First bounce predicts", outcome: "revised", brief: "Mean bounce predicts (ρ=0.60)" },
  { experiment: "V33", prediction: "Contrastive → counterfactual", outcome: "contradicted", brief: "Destabilizes gradient" },
  { experiment: "V34", prediction: "Φ in fitness helps", outcome: "contradicted", brief: "Mixed neg, Goodhart risk" },
  { experiment: "V35", prediction: "Language lifts Φ", outcome: "contradicted", brief: "Orthogonal (ρ = 0.07)" },
  { experiment: "V35", prediction: "Referential communication", outcome: "confirmed", brief: "10/10 seeds (100%)" },
  { experiment: "VLM", prediction: "Cross-substrate convergence", outcome: "confirmed", brief: "ρ = 0.72, p < 0.0001" },
];

const OUTCOME_COLORS: Record<string, string> = {
  confirmed: "#4ade80",
  contradicted: "#f87171",
  partial: "#fbbf24",
  null: "#888",
  revised: "#60a5fa",
};

const OUTCOME_ICONS: Record<string, string> = {
  confirmed: "✓",
  contradicted: "✗",
  partial: "~",
  null: "—",
  revised: "↻",
};

export const FalsificationScoreboardVideo: React.FC = () => {
  const frame = useCurrentFrame();

  const titleOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Stagger reveal: each row appears over ~15 frames
  const rowDelay = 20;
  const rowDuration = 12;

  // Final counts
  const confirmed = PREDICTIONS.filter((p) => p.outcome === "confirmed").length;
  const contradicted = PREDICTIONS.filter((p) => p.outcome === "contradicted").length;
  const revised = PREDICTIONS.filter(
    (p) => p.outcome === "partial" || p.outcome === "revised"
  ).length;

  const summaryOpacity = interpolate(frame, [330, 355], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const rowH = 32;
  const startY = 100;
  const colX = [40, 110, 390, 520, 670];

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
          fontSize: 26,
          fontWeight: 700,
          opacity: titleOpacity,
        }}
      >
        Falsification Scoreboard
      </div>
      <div
        style={{
          position: "absolute",
          top: 54,
          width: "100%",
          textAlign: "center",
          color: "#888",
          fontSize: 14,
          fontStyle: "italic",
          opacity: titleOpacity,
        }}
      >
        the framework survives by being wrong in specific ways
      </div>

      <svg width={1080} height={720}>
        {/* Header row */}
        <text x={colX[0]} y={startY - 8} fill="#666" fontSize={11} fontFamily="monospace">EXP</text>
        <text x={colX[1]} y={startY - 8} fill="#666" fontSize={11} fontFamily="monospace">PREDICTION</text>
        <text x={colX[3]} y={startY - 8} fill="#666" fontSize={11} fontFamily="monospace">RESULT</text>
        <text x={colX[4]} y={startY - 8} fill="#666" fontSize={11} fontFamily="monospace">DETAIL</text>
        <line x1={30} y1={startY} x2={1050} y2={startY} stroke="#333" strokeWidth={0.5} />

        {PREDICTIONS.map((row, i) => {
          const revealT = interpolate(
            frame,
            [20 + i * rowDelay, 20 + i * rowDelay + rowDuration],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) }
          );
          if (revealT <= 0) return null;

          const y = startY + 10 + (i + 1) * rowH;
          const color = OUTCOME_COLORS[row.outcome];

          return (
            <g key={i} opacity={revealT}>
              {/* Row background flash */}
              <rect
                x={30}
                y={y - rowH + 10}
                width={1020}
                height={rowH}
                fill={color}
                opacity={Math.max(0, 0.08 * (1 - (revealT - 0.8) * 5))}
                rx={2}
              />
              {/* Experiment */}
              <text x={colX[0]} y={y} fill="#ccc" fontSize={13} fontFamily="monospace">
                {row.experiment}
              </text>
              {/* Prediction */}
              <text x={colX[1]} y={y} fill="#aaa" fontSize={12} fontFamily="Georgia, serif">
                {row.prediction}
              </text>
              {/* Result icon */}
              <text x={colX[3]} y={y} fill={color} fontSize={16} fontWeight={700} fontFamily="monospace">
                {OUTCOME_ICONS[row.outcome]}
              </text>
              <text x={colX[3] + 22} y={y} fill={color} fontSize={12} fontFamily="Georgia, serif">
                {row.outcome}
              </text>
              {/* Detail */}
              <text x={colX[4]} y={y} fill="#888" fontSize={11} fontFamily="Georgia, serif">
                {row.brief}
              </text>
            </g>
          );
        })}

        {/* Summary bar */}
        {summaryOpacity > 0 && (
          <g opacity={summaryOpacity}>
            <rect x={200} y={620} width={680} height={50} fill="#111118" rx={8} stroke="#333" strokeWidth={0.5} />
            <text x={360} y={650} textAnchor="middle" fill="#4ade80" fontSize={20} fontWeight={700} fontFamily="Georgia, serif">
              {confirmed} confirmed
            </text>
            <text x={540} y={650} textAnchor="middle" fill="#f87171" fontSize={20} fontWeight={700} fontFamily="Georgia, serif">
              {contradicted} contradicted
            </text>
            <text x={720} y={650} textAnchor="middle" fill="#60a5fa" fontSize={20} fontWeight={700} fontFamily="Georgia, serif">
              {revised} revised
            </text>
          </g>
        )}
      </svg>
    </AbsoluteFill>
  );
};
