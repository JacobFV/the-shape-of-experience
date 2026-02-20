import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

/**
 * IdentificationScope — Epilogue visual
 * Shows expanding self-model identification from narrow (body) to cosmic scale.
 * As identification expands, the viability boundary (∂V) reshapes:
 * death moves from boundary to interior point.
 */

interface ScopeLevel {
  label: string;
  radius: number;
  color: string;
  tagline: string;
  deathRelation: string;
}

const LEVELS: ScopeLevel[] = [
  {
    label: "Body",
    radius: 60,
    color: "#f87171",
    tagline: "∂V = biological death",
    deathRelation: "boundary",
  },
  {
    label: "Relationships",
    radius: 120,
    color: "#fb923c",
    tagline: "∂V includes the loved",
    deathRelation: "grief-point",
  },
  {
    label: "Community",
    radius: 180,
    color: "#facc15",
    tagline: "∂V = collective persistence",
    deathRelation: "transition",
  },
  {
    label: "Humanity",
    radius: 240,
    color: "#4ade80",
    tagline: "∂V = species continuation",
    deathRelation: "interior",
  },
  {
    label: "Pattern",
    radius: 300,
    color: "#60a5fa",
    tagline: "∂V = information conservation",
    deathRelation: "interior event",
  },
];

const TOTAL = 360;
const PHASE_DURATION = TOTAL / LEVELS.length; // 72 frames each

export const IdentificationScopeVideo: React.FC = () => {
  const frame = useCurrentFrame();

  // Title fade in
  const titleOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Current active level
  const activeIndex = Math.min(
    Math.floor(frame / PHASE_DURATION),
    LEVELS.length - 1
  );
  const phaseProgress = (frame % PHASE_DURATION) / PHASE_DURATION;

  // Center of the visualization
  const cx = 540;
  const cy = 400;

  return (
    <AbsoluteFill
      style={{
        background: "linear-gradient(180deg, #0a0a1a 0%, #111128 100%)",
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 30,
          width: "100%",
          textAlign: "center",
          opacity: titleOpacity,
        }}
      >
        <div
          style={{
            fontSize: 28,
            fontWeight: 700,
            color: "#e2e8f0",
            letterSpacing: 1,
          }}
        >
          The Scope of Identification
        </div>
        <div
          style={{
            fontSize: 14,
            color: "#94a3b8",
            marginTop: 4,
          }}
        >
          What you take yourself to be determines the shape of your viability
          manifold
        </div>
      </div>

      {/* SVG visualization */}
      <svg
        viewBox="0 0 1080 720"
        style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}
      >
        {/* Concentric rings - all levels */}
        {LEVELS.map((level, i) => {
          const isActive = i <= activeIndex;
          const isCurrentPhase = i === activeIndex;

          // Ring expansion animation
          let ringScale = 0;
          if (i < activeIndex) {
            ringScale = 1;
          } else if (isCurrentPhase) {
            ringScale = interpolate(phaseProgress, [0, 0.4], [0, 1], {
              extrapolateRight: "clamp",
              easing: Easing.out(Easing.cubic),
            });
          }

          const ringOpacity = isActive
            ? interpolate(
                ringScale,
                [0, 0.5, 1],
                [0, 0.3, i === activeIndex ? 0.6 : 0.2],
                { extrapolateRight: "clamp" }
              )
            : 0;

          // Label opacity
          const labelOpacity = isCurrentPhase
            ? interpolate(phaseProgress, [0.3, 0.5], [0, 1], {
                extrapolateRight: "clamp",
              })
            : i < activeIndex
            ? 0.4
            : 0;

          return (
            <g key={i}>
              {/* Filled ring */}
              <circle
                cx={cx}
                cy={cy}
                r={level.radius * ringScale}
                fill={level.color}
                opacity={ringOpacity}
              />
              {/* Ring border */}
              <circle
                cx={cx}
                cy={cy}
                r={level.radius * ringScale}
                fill="none"
                stroke={level.color}
                strokeWidth={isCurrentPhase ? 2.5 : 1}
                opacity={isActive ? (isCurrentPhase ? 0.9 : 0.3) : 0}
              />
              {/* Label */}
              {ringScale > 0.5 && (
                <text
                  x={cx}
                  y={cy - level.radius * ringScale + 18}
                  textAnchor="middle"
                  fill={level.color}
                  fontSize={isCurrentPhase ? 15 : 12}
                  fontWeight={isCurrentPhase ? 700 : 400}
                  opacity={labelOpacity}
                >
                  {level.label}
                </text>
              )}
            </g>
          );
        })}

        {/* Death marker - a small X that moves from boundary to interior */}
        {(() => {
          const deathY = cy + 40;
          // Death starts at the boundary of the smallest circle, then becomes interior
          const bodyRadius = LEVELS[0].radius;
          const currentMaxRadius =
            LEVELS[activeIndex].radius *
            interpolate(phaseProgress, [0, 0.4], [0, 1], {
              extrapolateRight: "clamp",
              easing: Easing.out(Easing.cubic),
            });
          const effectiveMax = Math.max(
            bodyRadius,
            activeIndex > 0 ? LEVELS[activeIndex].radius : bodyRadius
          );

          // Death cross position - on boundary of body circle
          const deathX = cx + bodyRadius;

          // Is death on the boundary or interior?
          const isInterior = activeIndex >= 2;

          const deathOpacity = interpolate(frame, [10, 30], [0, 1], {
            extrapolateRight: "clamp",
          });

          // Pulse if on boundary, steady if interior
          const pulse = isInterior
            ? 1
            : interpolate(
                Math.sin(frame * 0.15),
                [-1, 1],
                [0.6, 1],
                { extrapolateRight: "clamp" }
              );

          return (
            <g opacity={deathOpacity}>
              {/* Death marker */}
              <line
                x1={deathX - 8}
                y1={deathY - 8}
                x2={deathX + 8}
                y2={deathY + 8}
                stroke={isInterior ? "#94a3b8" : "#f87171"}
                strokeWidth={2.5}
                opacity={pulse}
              />
              <line
                x1={deathX + 8}
                y1={deathY - 8}
                x2={deathX - 8}
                y2={deathY + 8}
                stroke={isInterior ? "#94a3b8" : "#f87171"}
                strokeWidth={2.5}
                opacity={pulse}
              />
              {/* Label */}
              <text
                x={deathX}
                y={deathY + 24}
                textAnchor="middle"
                fill={isInterior ? "#94a3b8" : "#f87171"}
                fontSize={11}
                opacity={pulse * 0.8}
              >
                death
              </text>
              {/* Boundary vs interior annotation */}
              {activeIndex >= 1 && (
                <text
                  x={deathX}
                  y={deathY + 38}
                  textAnchor="middle"
                  fill={isInterior ? "#4ade80" : "#fb923c"}
                  fontSize={10}
                  fontStyle="italic"
                  opacity={0.7}
                >
                  {isInterior ? "interior to manifold" : "on boundary"}
                </text>
              )}
            </g>
          );
        })()}

        {/* Self dot at center */}
        <circle cx={cx} cy={cy} r={6} fill="#ffffff" opacity={0.9} />
        <circle
          cx={cx}
          cy={cy}
          r={10}
          fill="none"
          stroke="#ffffff"
          strokeWidth={1}
          opacity={0.4}
        />

        {/* ∂V label - the viability boundary */}
        {(() => {
          const currentLevel = LEVELS[activeIndex];
          const boundaryR = currentLevel.radius;
          const boundaryOpacity = interpolate(
            phaseProgress,
            [0.4, 0.6],
            [0, 1],
            { extrapolateRight: "clamp" }
          );
          return (
            <g opacity={boundaryOpacity}>
              {/* Dashed boundary circle */}
              <circle
                cx={cx}
                cy={cy}
                r={boundaryR}
                fill="none"
                stroke="#e2e8f0"
                strokeWidth={1.5}
                strokeDasharray="6 4"
                opacity={0.5}
              />
              <text
                x={cx + boundaryR + 12}
                y={cy - 4}
                fill="#e2e8f0"
                fontSize={13}
                fontWeight={600}
              >
                ∂V
              </text>
            </g>
          );
        })()}
      </svg>

      {/* Bottom panel - current level info */}
      <div
        style={{
          position: "absolute",
          bottom: 30,
          left: 60,
          right: 60,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-end",
        }}
      >
        {/* Tagline */}
        <div>
          <div
            style={{
              fontSize: 18,
              fontWeight: 600,
              color: LEVELS[activeIndex].color,
              opacity: interpolate(phaseProgress, [0.3, 0.5], [0, 1], {
                extrapolateRight: "clamp",
              }),
            }}
          >
            θ = {LEVELS[activeIndex].label}
          </div>
          <div
            style={{
              fontSize: 14,
              color: "#94a3b8",
              marginTop: 4,
              opacity: interpolate(phaseProgress, [0.4, 0.6], [0, 1], {
                extrapolateRight: "clamp",
              }),
            }}
          >
            {LEVELS[activeIndex].tagline}
          </div>
        </div>

        {/* Gradient indicator */}
        <div
          style={{
            textAlign: "right",
            opacity: interpolate(phaseProgress, [0.5, 0.7], [0, 1], {
              extrapolateRight: "clamp",
            }),
          }}
        >
          <div style={{ fontSize: 13, color: "#94a3b8" }}>
            gradient direction
          </div>
          <div
            style={{
              fontSize: 20,
              fontWeight: 700,
              color:
                activeIndex < 2
                  ? "#f87171"
                  : activeIndex < 3
                  ? "#facc15"
                  : "#4ade80",
            }}
          >
            {activeIndex < 2
              ? "→ dissolution"
              : activeIndex < 3
              ? "→ transition"
              : "→ continuation"}
          </div>
        </div>
      </div>

      {/* Progress dots */}
      <div
        style={{
          position: "absolute",
          bottom: 10,
          width: "100%",
          display: "flex",
          justifyContent: "center",
          gap: 8,
        }}
      >
        {LEVELS.map((level, i) => (
          <div
            key={i}
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: i <= activeIndex ? level.color : "#334155",
              transition: "background 0.3s",
            }}
          />
        ))}
      </div>
    </AbsoluteFill>
  );
};
