import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { THEMES, ThemeMode } from './themes';

/**
 * IotaHistorical — Part VI visual
 * Civilizational ι trajectory from Pre-Axial (~0.10) to Digital Age (~0.85).
 * Shows cost/benefit at each level: predictive power gained, experiential richness lost.
 */

interface Era {
  label: string;
  year: string;
  iota: number;
  color: string;
  gain: string;
  cost: string;
}

const TOTAL = 360;

// Chart dimensions
const CHART_LEFT = 120;
const CHART_RIGHT = 1000;
const CHART_TOP = 130;
const CHART_BOTTOM = 520;
const CHART_W = CHART_RIGHT - CHART_LEFT;
const CHART_H = CHART_BOTTOM - CHART_TOP;

export const IotaHistoricalVideo: React.FC<{ theme?: ThemeMode }> = ({ theme }) => {
  const frame = useCurrentFrame();
  const t = THEMES[theme ?? 'dark'];

  const ERAS: Era[] = [
    {
      label: "Pre-Axial",
      year: "< 800 BCE",
      iota: 0.1,
      color: t.green,
      gain: "world alive, meaningful",
      cost: "no analytical distance",
    },
    {
      label: "Axial Age",
      year: "800–200 BCE",
      iota: 0.25,
      color: t.green,
      gain: "voluntary ι modulation",
      cost: "self-consciousness emerges",
    },
    {
      label: "Renaissance",
      year: "1400–1600",
      iota: 0.4,
      color: t.yellow,
      gain: "perspectival awareness",
      cost: "groundlessness begins",
    },
    {
      label: "Scientific Revolution",
      year: "1600–1800",
      iota: 0.6,
      color: t.orange,
      gain: "predictive power",
      cost: "disenchantment (Weber)",
    },
    {
      label: "Industrial / Psychological",
      year: "1800–1960",
      iota: 0.72,
      color: t.red,
      gain: "material abundance",
      cost: "iron cage of rationality",
    },
    {
      label: "Digital Transition",
      year: "1990–present",
      iota: 0.85,
      color: t.red,
      gain: "global connectivity",
      cost: "meaning crisis",
    },
  ];

  const PHASE = TOTAL / ERAS.length; // 60 frames each

  const titleOp = interpolate(frame, [0, 20], [0, 1], { extrapolateRight: "clamp" });

  // Active era index
  const activeIdx = Math.min(Math.floor(frame / PHASE), ERAS.length - 1);
  const phaseProgress = (frame % PHASE) / PHASE;

  // Convert iota value to Y coordinate (0 at bottom, 1 at top)
  const iotaToY = (iota: number) => CHART_BOTTOM - iota * CHART_H;
  // Era index to X coordinate
  const idxToX = (i: number) => CHART_LEFT + (i / (ERAS.length - 1)) * CHART_W;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: t.bg,
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 24,
          width: "100%",
          textAlign: "center",
          opacity: titleOp,
        }}
      >
        <div style={{ fontSize: 26, fontWeight: 700, color: t.text, letterSpacing: 1 }}>
          Civilizational ι Trajectory
        </div>
        <div style={{ fontSize: 13, color: t.slate, marginTop: 4 }}>
          Each step gained predictive power and lost experiential richness
        </div>
      </div>

      <svg viewBox="0 0 1080 720" style={{ position: "absolute", width: "100%", height: "100%" }}>
        {/* Y axis */}
        <line
          x1={CHART_LEFT}
          y1={CHART_TOP}
          x2={CHART_LEFT}
          y2={CHART_BOTTOM}
          stroke={t.border}
          strokeWidth={1}
        />
        {/* Y axis labels */}
        {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
          <g key={v}>
            <line
              x1={CHART_LEFT - 6}
              y1={iotaToY(v)}
              x2={CHART_LEFT}
              y2={iotaToY(v)}
              stroke={t.muted}
              strokeWidth={1}
            />
            <text
              x={CHART_LEFT - 12}
              y={iotaToY(v) + 4}
              textAnchor="end"
              fill={t.slate}
              fontSize={11}
            >
              {v.toFixed(2)}
            </text>
            {/* Grid line */}
            <line
              x1={CHART_LEFT}
              y1={iotaToY(v)}
              x2={CHART_RIGHT}
              y2={iotaToY(v)}
              stroke={t.border}
              strokeWidth={0.5}
            />
          </g>
        ))}
        {/* Y axis title */}
        <text
          x={40}
          y={(CHART_TOP + CHART_BOTTOM) / 2}
          textAnchor="middle"
          fill={t.slate}
          fontSize={14}
          fontWeight={600}
          transform={`rotate(-90, 40, ${(CHART_TOP + CHART_BOTTOM) / 2})`}
        >
          ι (inhibition coefficient)
        </text>

        {/* Gradient zones */}
        {/* Participatory zone (low ι) */}
        <rect
          x={CHART_LEFT}
          y={iotaToY(0.35)}
          width={CHART_W}
          height={iotaToY(0) - iotaToY(0.35)}
          fill={t.green}
          opacity={0.04}
        />
        <text x={CHART_RIGHT + 8} y={iotaToY(0.15)} fill={t.green} fontSize={10} opacity={0.5}>
          participatory
        </text>
        {/* Mechanistic zone (high ι) */}
        <rect
          x={CHART_LEFT}
          y={iotaToY(1.0)}
          width={CHART_W}
          height={iotaToY(0.65) - iotaToY(1.0)}
          fill={t.red}
          opacity={0.04}
        />
        <text x={CHART_RIGHT + 8} y={iotaToY(0.85)} fill={t.red} fontSize={10} opacity={0.5}>
          mechanistic
        </text>

        {/* Path connecting eras */}
        {ERAS.map((era, i) => {
          if (i === 0) return null;
          const isRevealed = i <= activeIdx;
          const isAnimating = i === activeIdx;

          const x1 = idxToX(i - 1);
          const y1 = iotaToY(ERAS[i - 1].iota);
          const x2 = idxToX(i);
          const y2 = iotaToY(era.iota);

          let lineProgress = 0;
          if (isRevealed && !isAnimating) lineProgress = 1;
          else if (isAnimating) {
            lineProgress = interpolate(phaseProgress, [0, 0.3], [0, 1], {
              extrapolateRight: "clamp",
              easing: Easing.out(Easing.cubic),
            });
          }

          const dx = x2 - x1;
          const dy = y2 - y1;

          return (
            <line
              key={i}
              x1={x1}
              y1={y1}
              x2={x1 + dx * lineProgress}
              y2={y1 + dy * lineProgress}
              stroke={era.color}
              strokeWidth={2.5}
              opacity={0.8}
            />
          );
        })}

        {/* Era dots and labels */}
        {ERAS.map((era, i) => {
          const isRevealed = i <= activeIdx;
          const isCurrent = i === activeIdx;

          const dotOp = isRevealed
            ? isCurrent
              ? interpolate(phaseProgress, [0.2, 0.4], [0, 1], {
                  extrapolateRight: "clamp",
                })
              : 0.7
            : 0;

          const x = idxToX(i);
          const y = iotaToY(era.iota);

          // Pulse for current
          const pulse = isCurrent
            ? 6 + Math.sin(frame * 0.12) * 2
            : 5;

          return (
            <g key={i} opacity={dotOp}>
              {/* Glow */}
              {isCurrent && (
                <circle cx={x} cy={y} r={16} fill={era.color} opacity={0.15} />
              )}
              {/* Dot */}
              <circle cx={x} cy={y} r={pulse} fill={era.color} />
              {/* Era label below */}
              <text
                x={x}
                y={CHART_BOTTOM + 20}
                textAnchor="middle"
                fill={isCurrent ? era.color : t.slate}
                fontSize={isCurrent ? 12 : 10}
                fontWeight={isCurrent ? 700 : 400}
              >
                {era.label}
              </text>
              <text
                x={x}
                y={CHART_BOTTOM + 34}
                textAnchor="middle"
                fill={t.slate}
                fontSize={9}
              >
                {era.year}
              </text>
              {/* ι value label */}
              <text
                x={x + 14}
                y={y - 12}
                fill={era.color}
                fontSize={11}
                fontWeight={600}
                opacity={isCurrent ? 1 : 0.5}
              >
                ι ≈ {era.iota.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Bottom info panel */}
      <div
        style={{
          position: "absolute",
          bottom: 30,
          left: 60,
          right: 60,
          display: "flex",
          justifyContent: "space-between",
          opacity: interpolate(phaseProgress, [0.4, 0.6], [0, 1], {
            extrapolateRight: "clamp",
          }),
        }}
      >
        <div>
          <div style={{ fontSize: 11, color: t.slate, textTransform: "uppercase", letterSpacing: 1 }}>
            gained
          </div>
          <div style={{ fontSize: 15, color: t.green, fontWeight: 600 }}>
            {ERAS[activeIdx].gain}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 11, color: t.slate, textTransform: "uppercase", letterSpacing: 1 }}>
            lost
          </div>
          <div style={{ fontSize: 15, color: t.red, fontWeight: 600 }}>
            {ERAS[activeIdx].cost}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
