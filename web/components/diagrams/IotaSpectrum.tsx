"use client";
import { useState, useEffect, useRef } from "react";

/**
 * Iota (ι) Spectrum — Participatory ↔ Mechanistic Perception
 *
 * Interactive spectrum showing how the inhibition coefficient changes perception.
 * Hover/drag along the spectrum to see what changes at each ι value.
 */

const MARKERS = [
  { pos: 0.0, label: "Animism", sublabel: "World alive, everything agentive", color: "#4ade80" },
  { pos: 0.15, label: "Childhood default", sublabel: "Piaget's animistic stage", color: "#4ade80" },
  { pos: 0.30, label: "ι ≈ 0.30", sublabel: "Evolutionary steady state (Exp 8)", color: "#fbbf24" },
  { pos: 0.50, label: "Cultural modulation", sublabel: "Religious practice, contemplation", color: "#fbbf24" },
  { pos: 0.75, label: "Scientific training", sublabel: "Mechanistic perception learned", color: "#fb923c" },
  { pos: 1.0, label: "Pure mechanism", sublabel: "Inert matter, blind law", color: "#f87171" },
];

const DIMENSIONS = [
  { label: "Valence", lowLabel: "responsive", highLabel: "flattened" },
  { label: "Arousal", lowLabel: "coupled to world", highLabel: "dampened" },
  { label: "Integration (Φ)", lowLabel: "very high", highLabel: "modular" },
  { label: "Effective Rank", lowLabel: "high", highLabel: "variable" },
  { label: "CF Weight", lowLabel: "narrative-rich", highLabel: "present-focused" },
  { label: "Self-Model", lowLabel: "porous boundary", highLabel: "sharp boundary" },
];

export default function IotaSpectrum() {
  const [hoverPos, setHoverPos] = useState<number | null>(null);
  const [visible, setVisible] = useState(false);
  const ref = useRef<SVGSVGElement>(null);
  const barRef = useRef<SVGRectElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) setVisible(true); },
      { threshold: 0.3 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const handleMouseMove = (e: React.MouseEvent<SVGRectElement>) => {
    const rect = barRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = (e.clientX - rect.left) / rect.width;
    setHoverPos(Math.max(0, Math.min(1, x)));
  };

  const barX = 80;
  const barY = 70;
  const barW = 660;
  const barH = 40;

  const iota = hoverPos ?? 0.30; // default to evolutionary steady state

  return (
    <svg
      ref={ref}
      viewBox="0 0 820 420"
      className="diagram-svg"
      style={{ maxWidth: 820, margin: "0 auto", display: "block" }}
    >
      {/* Title */}
      <text x={410} y={30} textAnchor="middle" fill="var(--d-fg)" fontSize={16} fontWeight={700} fontFamily="Georgia, serif"
        opacity={visible ? 1 : 0} style={{ transition: "opacity 0.5s" }}>
        The Inhibition Coefficient (ι)
      </text>
      <text x={410} y={50} textAnchor="middle" fill="var(--d-muted)" fontSize={12} fontFamily="Georgia, serif"
        opacity={visible ? 1 : 0} style={{ transition: "opacity 0.5s" }}>
        hover to explore the spectrum from participatory to mechanistic perception
      </text>

      {/* Gradient bar */}
      <defs>
        <linearGradient id="iota-grad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#4ade80" />
          <stop offset="30%" stopColor="#fbbf24" />
          <stop offset="70%" stopColor="#fb923c" />
          <stop offset="100%" stopColor="#f87171" />
        </linearGradient>
      </defs>

      <rect x={barX} y={barY} width={barW} height={barH} fill="url(#iota-grad)" rx={6}
        opacity={visible ? 0.8 : 0} style={{ transition: "opacity 0.5s" }} />

      {/* Interactive overlay */}
      <rect
        ref={barRef}
        x={barX} y={barY} width={barW} height={barH}
        fill="transparent" style={{ cursor: "crosshair" }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverPos(null)}
      />

      {/* Labels at ends */}
      <text x={barX} y={barY + barH + 18} fill="var(--d-fg)" fontSize={12} fontWeight={700} fontFamily="Georgia, serif"
        opacity={visible ? 1 : 0} style={{ transition: "opacity 0.5s" }}>
        ι = 0 (participatory)
      </text>
      <text x={barX + barW} y={barY + barH + 18} textAnchor="end" fill="var(--d-fg)" fontSize={12} fontWeight={700}
        fontFamily="Georgia, serif" opacity={visible ? 1 : 0} style={{ transition: "opacity 0.5s" }}>
        ι = 1 (mechanistic)
      </text>

      {/* Marker dots */}
      {MARKERS.map((m, i) => {
        const x = barX + m.pos * barW;
        return (
          <g key={i} opacity={visible ? 1 : 0} style={{ transition: `opacity 0.5s ${i * 0.1}s` }}>
            <circle cx={x} cy={barY + barH / 2} r={5} fill={m.color} stroke="#0a0a0f" strokeWidth={2} />
            <line x1={x} y1={barY + barH} x2={x} y2={barY + barH + 30} stroke={m.color} strokeWidth={0.5} opacity={0.5} />
            <text x={x} y={barY + barH + 42} textAnchor="middle" fill={m.color} fontSize={10} fontWeight={700}
              fontFamily="Georgia, serif">
              {m.label}
            </text>
            <text x={x} y={barY + barH + 55} textAnchor="middle" fill="var(--d-muted)" fontSize={9}
              fontFamily="Georgia, serif">
              {m.sublabel}
            </text>
          </g>
        );
      })}

      {/* Current position indicator */}
      {hoverPos !== null && (
        <g>
          <line x1={barX + hoverPos * barW} y1={barY - 8} x2={barX + hoverPos * barW} y2={barY + barH + 8}
            stroke="white" strokeWidth={2} />
          <text x={barX + hoverPos * barW} y={barY - 14} textAnchor="middle" fill="white" fontSize={13}
            fontWeight={700} fontFamily="monospace">
            ι = {hoverPos.toFixed(2)}
          </text>
        </g>
      )}

      {/* Dimension effects panel */}
      <g opacity={visible ? 1 : 0} style={{ transition: "opacity 0.5s 0.3s" }}>
        <text x={410} y={175} textAnchor="middle" fill="var(--d-fg)" fontSize={13} fontWeight={700}
          fontFamily="Georgia, serif">
          Affect Dimensions at ι = {iota.toFixed(2)}
        </text>

        {DIMENSIONS.map((dim, i) => {
          const y = 200 + i * 34;
          const barInnerX = 200;
          const barInnerW = 420;
          // Interpolate label based on iota
          const lowW = barInnerW * (1 - iota);
          const highW = barInnerW * iota;

          return (
            <g key={i}>
              <text x={barInnerX - 10} y={y + 12} textAnchor="end" fill="var(--d-fg)" fontSize={11}
                fontFamily="Georgia, serif">
                {dim.label}
              </text>

              {/* Background bar */}
              <rect x={barInnerX} y={y} width={barInnerW} height={20} fill="var(--d-fg)" opacity={0.04} rx={3} />

              {/* Low-ι portion (green) */}
              <rect x={barInnerX} y={y} width={lowW} height={20} fill="#4ade80" opacity={0.2} rx={3} />
              {lowW > 60 && (
                <text x={barInnerX + 8} y={y + 14} fill="#4ade80" fontSize={9} fontFamily="Georgia, serif">
                  {dim.lowLabel}
                </text>
              )}

              {/* High-ι portion (red) */}
              <rect x={barInnerX + lowW} y={y} width={highW} height={20} fill="#f87171" opacity={0.2} rx={3} />
              {highW > 60 && (
                <text x={barInnerX + barInnerW - 8} y={y + 14} textAnchor="end" fill="#f87171" fontSize={9}
                  fontFamily="Georgia, serif">
                  {dim.highLabel}
                </text>
              )}
            </g>
          );
        })}
      </g>

      {/* Key insight */}
      <g opacity={visible ? 1 : 0} style={{ transition: "opacity 0.5s 0.6s" }}>
        <text x={410} y={410} textAnchor="middle" fill="var(--d-muted)" fontSize={11} fontFamily="Georgia, serif"
          fontStyle="italic">
          High ι reduces integration — the mechanistic worldview is genuinely less conscious (IIT)
        </text>
      </g>
    </svg>
  );
}
