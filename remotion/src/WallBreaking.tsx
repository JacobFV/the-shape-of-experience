import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  Easing,
} from "remotion";
import { THEMES, ThemeMode } from './themes';

/**
 * Wall Breaking Animation
 *
 * Shows the two architectural walls:
 * 1. ρ wall (V13-V18 → V20): sensory-motor coupling
 * 2. Decomposability wall (V22-V24 → V27): gradient coupling
 *
 * Timeline: walls appear, then crack and break dramatically
 *
 * 300 frames @ 30fps = 10 seconds
 */

interface WallProps {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  sublabel: string;
  breakerLabel: string;
  breakerSublabel: string;
  metric: string;
  crackProgress: number;
  breakProgress: number;
  buildProgress: number;
  wallColor: string;
  crackColor: string;
  breakColor: string;
  textColor: string;
  mutedColor: string;
  blueColor: string;
}

function Wall({
  x, y, w, h, label, sublabel, breakerLabel, breakerSublabel, metric,
  crackProgress, breakProgress, buildProgress,
  wallColor, crackColor, breakColor, textColor, mutedColor, blueColor,
}: WallProps) {
  const isBreaking = breakProgress > 0;
  const isCracking = crackProgress > 0;

  // Crack lines
  const cracks = [
    { x1: w * 0.3, y1: 0, x2: w * 0.45, y2: h * 0.5 },
    { x1: w * 0.45, y1: h * 0.5, x2: w * 0.35, y2: h },
    { x1: w * 0.6, y1: 0, x2: w * 0.55, y2: h * 0.4 },
    { x1: w * 0.55, y1: h * 0.4, x2: w * 0.7, y2: h },
    { x1: w * 0.2, y1: h * 0.3, x2: w * 0.8, y2: h * 0.35 },
  ];

  // Break: wall fragments fly apart
  const fragments = [
    { cx: w * 0.2, cy: h * 0.2, dx: -40, dy: -30, rot: -25 },
    { cx: w * 0.7, cy: h * 0.15, dx: 35, dy: -40, rot: 20 },
    { cx: w * 0.4, cy: h * 0.5, dx: -20, dy: 10, rot: -15 },
    { cx: w * 0.6, cy: h * 0.6, dx: 30, dy: 20, rot: 30 },
    { cx: w * 0.3, cy: h * 0.8, dx: -25, dy: 35, rot: -20 },
    { cx: w * 0.8, cy: h * 0.7, dx: 40, dy: 25, rot: 15 },
  ];

  return (
    <g transform={`translate(${x}, ${y})`} opacity={buildProgress}>
      {/* Wall body */}
      {!isBreaking && (
        <rect
          width={w}
          height={h}
          fill={wallColor}
          stroke="#6b5a3f"
          strokeWidth={2}
          rx={3}
        />
      )}

      {/* Brick pattern */}
      {!isBreaking &&
        Array.from({ length: Math.floor(h / 25) }, (_, row) =>
          Array.from({ length: Math.floor(w / 40) }, (_, col) => (
            <rect
              key={`${row}-${col}`}
              x={col * 40 + (row % 2 === 0 ? 0 : 20) + 2}
              y={row * 25 + 2}
              width={36}
              height={21}
              fill="none"
              stroke="#5a4930"
              strokeWidth={0.5}
              rx={1}
            />
          ))
        )}

      {/* Cracks */}
      {isCracking &&
        !isBreaking &&
        cracks.map((c, i) => (
          <line
            key={i}
            x1={c.x1}
            y1={c.y1}
            x2={c.x1 + (c.x2 - c.x1) * crackProgress}
            y2={c.y1 + (c.y2 - c.y1) * crackProgress}
            stroke={crackColor}
            strokeWidth={2}
            opacity={crackProgress * 0.8}
          />
        ))}

      {/* Break fragments */}
      {isBreaking &&
        fragments.map((f, i) => (
          <rect
            key={i}
            x={f.cx - 15}
            y={f.cy - 12}
            width={30}
            height={24}
            fill={wallColor}
            stroke="#6b5a3f"
            strokeWidth={1}
            rx={2}
            opacity={1 - breakProgress}
            transform={`translate(${f.dx * breakProgress}, ${f.dy * breakProgress}) rotate(${f.rot * breakProgress}, ${f.cx}, ${f.cy})`}
          />
        ))}

      {/* Wall label */}
      <text
        x={w / 2}
        y={-25}
        textAnchor="middle"
        fill={textColor}
        fontSize={16}
        fontWeight={700}
        fontFamily="Georgia, serif"
      >
        {label}
      </text>
      <text
        x={w / 2}
        y={-8}
        textAnchor="middle"
        fill={mutedColor}
        fontSize={12}
        fontFamily="Georgia, serif"
      >
        {sublabel}
      </text>

      {/* Breaker label (after break) */}
      {isBreaking && (
        <>
          <text
            x={w / 2}
            y={h / 2 - 5}
            textAnchor="middle"
            fill={breakColor}
            fontSize={18}
            fontWeight={700}
            fontFamily="Georgia, serif"
            opacity={breakProgress}
          >
            {breakerLabel}
          </text>
          <text
            x={w / 2}
            y={h / 2 + 18}
            textAnchor="middle"
            fill={breakColor}
            fontSize={13}
            fontFamily="Georgia, serif"
            opacity={breakProgress * 0.8}
          >
            {breakerSublabel}
          </text>
          <text
            x={w / 2}
            y={h / 2 + 40}
            textAnchor="middle"
            fill={blueColor}
            fontSize={15}
            fontWeight={700}
            fontFamily="monospace"
            opacity={breakProgress}
          >
            {metric}
          </text>
        </>
      )}

      {/* Blocked experiments (left side) */}
    </g>
  );
}

export const WallBreakingVideo: React.FC<{ theme?: ThemeMode }> = ({ theme }) => {
  const frame = useCurrentFrame();
  const t = THEMES[theme ?? 'dark'];

  const wallColor = '#4a3520';
  const crackColor = t.yellow;
  const breakColor = t.green;

  // Phase 1: Build walls (0-60)
  const build1 = interpolate(frame, [10, 50], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });
  const build2 = interpolate(frame, [30, 70], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });

  // Phase 2: Crack wall 1 (80-120), break (120-150)
  const crack1 = interpolate(frame, [80, 120], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const break1 = interpolate(frame, [125, 155], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });

  // Phase 3: Crack wall 2 (170-210), break (210-240)
  const crack2 = interpolate(frame, [170, 210], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const break2 = interpolate(frame, [215, 245], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });

  const titleOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateRight: "clamp",
  });

  const finalOpacity = interpolate(frame, [260, 290], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Experiments blocked by each wall
  const wall1Blocked = ["V13", "V14", "V15", "V16", "V17", "V18"];
  const wall2Blocked = ["V22", "V23", "V24"];

  return (
    <AbsoluteFill style={{ backgroundColor: t.bg, fontFamily: "Georgia, serif" }}>
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
        Two Walls, Two Breaks
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
        architectural barriers on the path to integration
      </div>

      <svg width={1080} height={720}>
        {/* Blocked experiments - Wall 1 */}
        {wall1Blocked.map((v, i) => (
          <text
            key={v}
            x={110}
            y={180 + i * 30}
            textAnchor="middle"
            fill={break1 > 0 ? t.green : crack1 > 0 ? crackColor : t.muted}
            fontSize={13}
            fontFamily="monospace"
            opacity={build1}
          >
            {v} {break1 > 0 ? "✓" : "✗"}
          </text>
        ))}

        {/* Arrow to wall 1 */}
        <line
          x1={170}
          y1={300}
          x2={220}
          y2={300}
          stroke={t.muted}
          strokeWidth={2}
          opacity={build1}
          markerEnd="url(#arrowGray)"
        />

        {/* Wall 1 */}
        <Wall
          x={240}
          y={140}
          w={200}
          h={340}
          label="ρ Wall"
          sublabel="sensory-motor coupling ≈ 0"
          breakerLabel="V20"
          breakerSublabel="Protocell Agency"
          metric="ρ_sync = 0.21"
          buildProgress={build1}
          crackProgress={crack1}
          breakProgress={break1}
          wallColor={wallColor}
          crackColor={crackColor}
          breakColor={breakColor}
          textColor={t.text}
          mutedColor={t.muted}
          blueColor={t.blue}
        />

        {/* Arrow between walls */}
        <line
          x1={460}
          y1={300}
          x2={500}
          y2={300}
          stroke={t.muted}
          strokeWidth={2}
          opacity={build1 > 0 && build2 > 0 ? 1 : 0}
          markerEnd="url(#arrowGray)"
        />

        {/* Blocked experiments - Wall 2 */}
        {wall2Blocked.map((v, i) => (
          <text
            key={v}
            x={530}
            y={235 + i * 30}
            textAnchor="middle"
            fill={break2 > 0 ? t.green : crack2 > 0 ? crackColor : t.muted}
            fontSize={13}
            fontFamily="monospace"
            opacity={build2}
          >
            {v} {break2 > 0 ? "✓" : "✗"}
          </text>
        ))}

        {/* Arrow to wall 2 */}
        <line
          x1={570}
          y1={280}
          x2={610}
          y2={280}
          stroke={t.muted}
          strokeWidth={2}
          opacity={build2}
          markerEnd="url(#arrowGray)"
        />

        {/* Wall 2 */}
        <Wall
          x={630}
          y={140}
          w={200}
          h={340}
          label="Decomposability Wall"
          sublabel="linear gradients stay in lanes"
          breakerLabel="V27"
          breakerSublabel="MLP Head"
          metric="Φ = 0.245"
          buildProgress={build2}
          crackProgress={crack2}
          breakProgress={break2}
          wallColor={wallColor}
          crackColor={crackColor}
          breakColor={breakColor}
          textColor={t.text}
          mutedColor={t.muted}
          blueColor={t.blue}
        />

        {/* Arrow to integration */}
        {break2 > 0.5 && (
          <>
            <line
              x1={850}
              y1={300}
              x2={920}
              y2={300}
              stroke={breakColor}
              strokeWidth={3}
              opacity={(break2 - 0.5) * 2}
              markerEnd="url(#arrowGreen)"
            />
            <text
              x={970}
              y={300}
              textAnchor="middle"
              fill={breakColor}
              fontSize={22}
              fontWeight={700}
              fontFamily="Georgia, serif"
              opacity={(break2 - 0.5) * 2}
            >
              HIGH Φ
            </text>
            <text
              x={970}
              y={325}
              textAnchor="middle"
              fill={t.muted}
              fontSize={13}
              fontFamily="Georgia, serif"
              opacity={(break2 - 0.5) * 2}
            >
              (~30% of seeds)
            </text>
          </>
        )}

        {/* Markers */}
        <defs>
          <marker id="arrowGray" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill={t.muted} />
          </marker>
          <marker id="arrowGreen" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill={breakColor} />
          </marker>
        </defs>
      </svg>

      {/* Bottom annotation */}
      <div
        style={{
          position: "absolute",
          bottom: 30,
          width: "100%",
          textAlign: "center",
          color: t.text,
          fontSize: 16,
          opacity: finalOpacity,
        }}
      >
        both walls are architectural — neither can be overcome by more training, better targets, or richer environments
      </div>
    </AbsoluteFill>
  );
};
