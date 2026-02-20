"use client";
import { useState, useEffect, useRef } from "react";

/**
 * Contamination Dynamics — Clean vs Contaminated Relationship Manifolds
 *
 * Two-panel diagram showing:
 * Left: Clean friendship manifold (aligned gradients)
 * Right: Contaminated manifold (conflicting gradients from friendship + transaction)
 *
 * Interactive: hover to highlight gradient conflict
 */

export default function ContaminationDynamics() {
  const [hovered, setHovered] = useState<"clean" | "contaminated" | null>(null);
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

  const panelW = 350;
  const gapW = 60;
  const startX1 = 30;
  const startX2 = startX1 + panelW + gapW;

  // Arrow helper
  const Arrow = ({ x1, y1, x2, y2, color, width = 2, opacity = 0.8 }: {
    x1: number; y1: number; x2: number; y2: number;
    color: string; width?: number; opacity?: number;
  }) => {
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const headLen = 10;
    return (
      <g opacity={opacity}>
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={width} />
        <polygon
          points={`${x2},${y2} ${x2 - headLen * Math.cos(angle - 0.4)},${y2 - headLen * Math.sin(angle - 0.4)} ${x2 - headLen * Math.cos(angle + 0.4)},${y2 - headLen * Math.sin(angle + 0.4)}`}
          fill={color}
        />
      </g>
    );
  };

  return (
    <svg
      ref={ref}
      viewBox="0 0 820 440"
      className="diagram-svg"
      style={{ maxWidth: 820, margin: "0 auto", display: "block" }}
    >
      {/* Title */}
      <text x={410} y={30} textAnchor="middle" fill="var(--d-fg)" fontSize={16} fontWeight={700}
        fontFamily="Georgia, serif" opacity={visible ? 1 : 0} style={{ transition: "opacity 0.5s" }}>
        Incentive Contamination
      </text>
      <text x={410} y={50} textAnchor="middle" fill="var(--d-muted)" fontSize={12}
        fontFamily="Georgia, serif" opacity={visible ? 1 : 0} style={{ transition: "opacity 0.5s" }}>
        when relationship manifolds mix, gradients conflict
      </text>

      {/* === LEFT PANEL: Clean Friendship === */}
      <g
        opacity={visible ? (hovered === "contaminated" ? 0.3 : 1) : 0}
        style={{ transition: "opacity 0.4s", cursor: "pointer" }}
        onMouseEnter={() => setHovered("clean")}
        onMouseLeave={() => setHovered(null)}
      >
        {/* Panel background */}
        <rect x={startX1} y={65} width={panelW} height={330} fill="var(--d-fg)" opacity={0.03} rx={8} />
        <text x={startX1 + panelW / 2} y={90} textAnchor="middle" fill="#4ade80" fontSize={14}
          fontWeight={700} fontFamily="Georgia, serif">
          Clean Friendship
        </text>
        <text x={startX1 + panelW / 2} y={108} textAnchor="middle" fill="var(--d-muted)" fontSize={11}
          fontFamily="Georgia, serif">
          V_F ≡ V_A ∩ V_B
        </text>

        {/* Viability manifold - overlapping circles */}
        <circle cx={startX1 + 140} cy={240} r={90} fill="#4ade80" opacity={0.08}
          stroke="#4ade80" strokeWidth={1.5} />
        <circle cx={startX1 + 210} cy={240} r={90} fill="#4ade80" opacity={0.08}
          stroke="#4ade80" strokeWidth={1.5} />
        <text x={startX1 + 110} y={240} textAnchor="middle" fill="#4ade80" fontSize={11}
          fontFamily="Georgia, serif">V_A</text>
        <text x={startX1 + 240} y={240} textAnchor="middle" fill="#4ade80" fontSize={11}
          fontFamily="Georgia, serif">V_B</text>
        <text x={startX1 + 175} y={230} textAnchor="middle" fill="#4ade80" fontSize={12}
          fontWeight={700} fontFamily="Georgia, serif">V_F</text>

        {/* Aligned gradient arrows - both pointing toward center */}
        <Arrow x1={startX1 + 80} y1={260} x2={startX1 + 130} y2={250} color="#4ade80" />
        <Arrow x1={startX1 + 270} y1={260} x2={startX1 + 220} y2={250} color="#4ade80" />

        {/* Gradient label */}
        <text x={startX1 + 175} y={290} textAnchor="middle" fill="#4ade80" fontSize={10}
          fontFamily="Georgia, serif">
          ∇V_F aligned
        </text>

        {/* Verdict */}
        <rect x={startX1 + 75} y={340} width={200} height={35} fill="#0f2a1a" opacity={0.5} rx={6}
          stroke="#4ade80" strokeWidth={1.5} />
        <text x={startX1 + 175} y={362} textAnchor="middle" fill="#4ade80" fontSize={13}
          fontWeight={700} fontFamily="Georgia, serif">
          aesthetic clarity
        </text>
      </g>

      {/* === RIGHT PANEL: Contaminated === */}
      <g
        opacity={visible ? (hovered === "clean" ? 0.3 : 1) : 0}
        style={{ transition: "opacity 0.4s", cursor: "pointer" }}
        onMouseEnter={() => setHovered("contaminated")}
        onMouseLeave={() => setHovered(null)}
      >
        {/* Panel background */}
        <rect x={startX2} y={65} width={panelW} height={330} fill="var(--d-fg)" opacity={0.03} rx={8} />
        <text x={startX2 + panelW / 2} y={90} textAnchor="middle" fill="#f87171" fontSize={14}
          fontWeight={700} fontFamily="Georgia, serif">
          Contaminated: Friend + Transaction
        </text>
        <text x={startX2 + panelW / 2} y={108} textAnchor="middle" fill="var(--d-muted)" fontSize={11}
          fontFamily="Georgia, serif">
          ∇V_F · ∇V_T {"<"} 0
        </text>

        {/* Overlapping manifolds - different colors */}
        <circle cx={startX2 + 140} cy={230} r={80} fill="#4ade80" opacity={0.06}
          stroke="#4ade80" strokeWidth={1} strokeDasharray="4 3" />
        <circle cx={startX2 + 210} cy={230} r={80} fill="#4ade80" opacity={0.06}
          stroke="#4ade80" strokeWidth={1} strokeDasharray="4 3" />
        {/* Transaction manifold overlay */}
        <ellipse cx={startX2 + 175} cy={250} rx={70} ry={50} fill="#f87171" opacity={0.08}
          stroke="#f87171" strokeWidth={1.5} />

        <text x={startX2 + 100} y={225} textAnchor="middle" fill="#4ade80" fontSize={10}
          fontFamily="Georgia, serif" opacity={0.7}>V_F</text>
        <text x={startX2 + 175} y={270} textAnchor="middle" fill="#f87171" fontSize={11}
          fontWeight={700} fontFamily="Georgia, serif">V_T</text>

        {/* Conflicting gradient arrows */}
        {/* Friendship gradient → toward mutual flourishing (inward) */}
        <Arrow x1={startX2 + 100} y1={200} x2={startX2 + 150} y2={215} color="#4ade80" opacity={0.6} />
        {/* Transaction gradient → toward balanced exchange (outward/different direction) */}
        <Arrow x1={startX2 + 150} y1={215} x2={startX2 + 120} y2={260} color="#f87171" />

        {/* Friendship gradient right side */}
        <Arrow x1={startX2 + 250} y1={200} x2={startX2 + 200} y2={215} color="#4ade80" opacity={0.6} />
        {/* Transaction gradient conflicts */}
        <Arrow x1={startX2 + 200} y1={215} x2={startX2 + 230} y2={260} color="#f87171" />

        {/* Conflict markers */}
        <text x={startX2 + 148} y={240} fill="#fbbf24" fontSize={16} fontWeight={700}>✕</text>
        <text x={startX2 + 198} y={240} fill="#fbbf24" fontSize={16} fontWeight={700}>✕</text>

        {/* Gradient label */}
        <text x={startX2 + 175} y={300} textAnchor="middle" fill="#f87171" fontSize={10}
          fontFamily="Georgia, serif">
          gradients conflict — valence uncomputable
        </text>

        {/* Verdict */}
        <rect x={startX2 + 75} y={340} width={200} height={35} fill="#2a0f0f" opacity={0.5} rx={6}
          stroke="#f87171" strokeWidth={1.5} />
        <text x={startX2 + 175} y={362} textAnchor="middle" fill="#f87171" fontSize={13}
          fontWeight={700} fontFamily="Georgia, serif">
          social nausea
        </text>
      </g>

      {/* Center divider */}
      <line x1={startX1 + panelW + gapW / 2} y1={80} x2={startX1 + panelW + gapW / 2} y2={380}
        stroke="var(--d-line)" strokeWidth={0.5} opacity={visible ? 0.3 : 0}
        style={{ transition: "opacity 0.5s" }} />
      <text x={startX1 + panelW + gapW / 2} y={410} textAnchor="middle" fill="var(--d-muted)"
        fontSize={10} fontFamily="Georgia, serif" fontStyle="italic"
        opacity={visible ? 1 : 0} style={{ transition: "opacity 0.5s" }}>
        vs
      </text>

      {/* Bottom insight */}
      <text x={410} y={435} textAnchor="middle" fill="var(--d-muted)" fontSize={11}
        fontFamily="Georgia, serif" fontStyle="italic"
        opacity={visible ? 1 : 0} style={{ transition: "opacity 0.5s 0.4s" }}>
        the detection system responds to the shadow manifold, not the surface action
      </text>
    </svg>
  );
}
