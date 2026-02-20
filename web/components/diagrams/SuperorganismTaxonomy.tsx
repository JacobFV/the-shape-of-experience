"use client";
import { useState, useEffect, useRef } from "react";

/**
 * Superorganism Taxonomy — Parasitic / Aligned / Mutualistic
 *
 * Three-panel diagram showing the three viability-manifold relationships
 * between a superorganism G and its human substrate h.
 *
 * Interactive: hover for details
 */

const TYPES = [
  {
    id: "parasitic",
    title: "Parasitic (Demon)",
    subtitle: "V_G ⊄ V_h",
    color: "#f87171",
    bgColor: "#2a0f0f",
    equation: "∃s ∈ V_G : s ∉ ∩ V_h",
    desc: "Pattern thrives when humans suffer",
    examples: ["Attention economy", "Extractive cult", "Predatory lending"],
    gOffset: { x: 30, y: -20 },
    hOffset: { x: -30, y: 20 },
    overlap: 0.3,
  },
  {
    id: "aligned",
    title: "Aligned",
    subtitle: "V_G ⊆ ∩ V_h",
    color: "#fbbf24",
    bgColor: "#2a1f0f",
    equation: "V_G ⊆ ∩_h V_h",
    desc: "Pattern can only thrive if humans thrive",
    examples: ["Healthy democracy", "Functional co-op", "Open-source community"],
    gOffset: { x: 0, y: 0 },
    hOffset: { x: 0, y: 0 },
    overlap: 0.8,
  },
  {
    id: "mutualistic",
    title: "Mutualistic (God)",
    subtitle: "V_h^{with G} ⊃ V_h^{without G}",
    color: "#4ade80",
    bgColor: "#0f2a1a",
    equation: "V_h^with ⊃ V_h^without",
    desc: "Pattern expands human viability",
    examples: ["Contemplative tradition", "Scientific community", "Mutual aid network"],
    gOffset: { x: 0, y: 0 },
    hOffset: { x: 0, y: 0 },
    overlap: 0.95,
  },
];

export default function SuperorganismTaxonomy() {
  const [hovered, setHovered] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);
  const ref = useRef<SVGSVGElement>(null);

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

  const panelW = 240;
  const gap = 25;
  const totalW = panelW * 3 + gap * 2;
  const startX = (820 - totalW) / 2;

  return (
    <svg
      ref={ref}
      viewBox="0 0 820 480"
      className="diagram-svg"
      style={{ maxWidth: 820, margin: "0 auto", display: "block" }}
    >
      {/* Title */}
      <text x={410} y={30} textAnchor="middle" fill="var(--d-fg)" fontSize={16} fontWeight={700}
        fontFamily="Georgia, serif" opacity={visible ? 1 : 0} style={{ transition: "opacity 0.5s" }}>
        Superorganism Taxonomy
      </text>
      <text x={410} y={50} textAnchor="middle" fill="var(--d-muted)" fontSize={12}
        fontFamily="Georgia, serif" opacity={visible ? 1 : 0} style={{ transition: "opacity 0.5s" }}>
        the viability relationship between pattern and substrate
      </text>

      {TYPES.map((type, i) => {
        const px = startX + i * (panelW + gap);
        const cx = px + panelW / 2;
        const cy = 180;
        const isActive = hovered === null || hovered === type.id;
        const delay = i * 0.15;

        return (
          <g
            key={type.id}
            opacity={visible ? (isActive ? 1 : 0.25) : 0}
            style={{ transition: `opacity 0.4s ${delay}s`, cursor: "pointer" }}
            onMouseEnter={() => setHovered(type.id)}
            onMouseLeave={() => setHovered(null)}
          >
            {/* Panel bg */}
            <rect x={px} y={65} width={panelW} height={400} fill={type.bgColor}
              opacity={0.3} rx={8} stroke={type.color} strokeWidth={hovered === type.id ? 2 : 0.5} />

            {/* Title */}
            <text x={cx} y={85} textAnchor="middle" fill={type.color} fontSize={13}
              fontWeight={700} fontFamily="Georgia, serif">
              {type.title}
            </text>

            {/* Venn diagram */}
            {/* Human manifold V_h */}
            <circle
              cx={cx + type.hOffset.x} cy={cy + type.hOffset.y}
              r={60} fill="#60a5fa" opacity={0.1}
              stroke="#60a5fa" strokeWidth={1.5}
            />
            <text x={cx + type.hOffset.x - 35} y={cy + type.hOffset.y - 45}
              fill="#60a5fa" fontSize={10} fontFamily="Georgia, serif">
              V_h
            </text>

            {/* Superorganism manifold V_G */}
            <circle
              cx={cx + type.gOffset.x} cy={cy + type.gOffset.y}
              r={type.id === "mutualistic" ? 45 : 50}
              fill={type.color} opacity={0.1}
              stroke={type.color} strokeWidth={1.5}
              strokeDasharray={type.id === "parasitic" ? "4 3" : "none"}
            />
            <text x={cx + type.gOffset.x + (type.id === "parasitic" ? 30 : 0)}
              y={cy + type.gOffset.y - (type.id === "parasitic" ? 35 : 30)}
              fill={type.color} fontSize={10} fontWeight={700} fontFamily="Georgia, serif">
              V_G
            </text>

            {/* Conflict/alignment indicator */}
            {type.id === "parasitic" && (
              <g>
                <text x={cx + 5} y={cy + 5} textAnchor="middle" fill="#fbbf24" fontSize={20} fontWeight={700}>
                  ✕
                </text>
                <text x={cx + 5} y={cy + 20} textAnchor="middle" fill="#fbbf24" fontSize={8}
                  fontFamily="Georgia, serif">
                  conflict
                </text>
              </g>
            )}
            {type.id === "aligned" && (
              <text x={cx} y={cy + 5} textAnchor="middle" fill={type.color} fontSize={14} fontWeight={700}>
                ⊆
              </text>
            )}
            {type.id === "mutualistic" && (
              <text x={cx} y={cy + 5} textAnchor="middle" fill={type.color} fontSize={14} fontWeight={700}>
                ⊃
              </text>
            )}

            {/* Subtitle equation */}
            <text x={cx} y={cy + 80} textAnchor="middle" fill="var(--d-muted)" fontSize={10}
              fontFamily="monospace">
              {type.equation}
            </text>

            {/* Description */}
            <text x={cx} y={cy + 105} textAnchor="middle" fill="var(--d-fg)" fontSize={11}
              fontFamily="Georgia, serif" fontStyle="italic">
              {type.desc}
            </text>

            {/* Valence diagnostic */}
            <g>
              <rect x={px + 20} y={cy + 120} width={panelW - 40} height={50} fill="var(--d-fg)"
                opacity={0.04} rx={4} />
              {type.id === "parasitic" ? (
                <>
                  <text x={cx} y={cy + 138} textAnchor="middle" fill={type.color} fontSize={10}
                    fontFamily="monospace">
                    V_G {">"} 0 AND V_human {"<"} 0
                  </text>
                  <text x={cx} y={cy + 155} textAnchor="middle" fill="var(--d-muted)" fontSize={9}
                    fontFamily="Georgia, serif">
                    the demon signature
                  </text>
                </>
              ) : type.id === "aligned" ? (
                <>
                  <text x={cx} y={cy + 138} textAnchor="middle" fill={type.color} fontSize={10}
                    fontFamily="monospace">
                    V_G {">"} 0 AND V_human {">"} 0
                  </text>
                  <text x={cx} y={cy + 155} textAnchor="middle" fill="var(--d-muted)" fontSize={9}
                    fontFamily="Georgia, serif">
                    aligned viability
                  </text>
                </>
              ) : (
                <>
                  <text x={cx} y={cy + 138} textAnchor="middle" fill={type.color} fontSize={10}
                    fontFamily="monospace">
                    V_h^with {">"} V_h^without
                  </text>
                  <text x={cx} y={cy + 155} textAnchor="middle" fill="var(--d-muted)" fontSize={9}
                    fontFamily="Georgia, serif">
                    expands human viability
                  </text>
                </>
              )}
            </g>

            {/* Examples */}
            {type.examples.map((ex, j) => (
              <text key={j} x={cx} y={cy + 195 + j * 18} textAnchor="middle"
                fill="var(--d-muted)" fontSize={10} fontFamily="Georgia, serif">
                {ex}
              </text>
            ))}
          </g>
        );
      })}

      {/* Bottom insight */}
      <text x={410} y={470} textAnchor="middle" fill="var(--d-muted)" fontSize={11}
        fontFamily="Georgia, serif" fontStyle="italic"
        opacity={visible ? 1 : 0} style={{ transition: "opacity 0.5s 0.5s" }}>
        not &quot;do you serve a superorganism?&quot; but &quot;which ones, and are they gods or demons?&quot;
      </text>
    </svg>
  );
}
