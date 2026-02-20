"use client";
import { useState, useEffect, useRef } from "react";

/**
 * V19 Bottleneck Furnace Mechanism
 *
 * Shows the three-phase fork experiment:
 *   Phase 1: shared evolution → Phase 2: fork into 3 conditions → Phase 3: novel test
 *   BOTTLENECK > GRADUAL > CONTROL at novel stress
 *
 * Interactive: hover to see condition details
 */

const PHASES = [
  { label: "Phase 1", sublabel: "Shared Evolution", cycles: "10 cycles", x: 70, w: 200 },
  { label: "Phase 2", sublabel: "Forked Conditions", cycles: "10 cycles", x: 310, w: 260 },
  { label: "Phase 3", sublabel: "Novel Test", cycles: "5 cycles", x: 610, w: 160 },
];

const CONDITIONS = [
  {
    id: "bottleneck",
    label: "BOTTLENECK",
    sublabel: ">90% mortality",
    color: "#f87171",
    y: 100,
    result: "rob = 1.116",
    beta: "β = +0.704",
    pval: "p < 0.0001",
    verdict: "CREATION",
  },
  {
    id: "gradual",
    label: "GRADUAL",
    sublabel: "8% regen reduction",
    color: "#fbbf24",
    y: 210,
    result: "rob = 1.016",
    beta: "—",
    pval: "—",
    verdict: "NEUTRAL",
  },
  {
    id: "control",
    label: "CONTROL",
    sublabel: "No stress change",
    color: "#60a5fa",
    y: 320,
    result: "rob = 1.029",
    beta: "baseline",
    pval: "—",
    verdict: "BASELINE",
  },
];

export function FurnaceMechanism() {
  const [hovered, setHovered] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);
  const [phaseRevealed, setPhaseRevealed] = useState(0);
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting && !visible) {
          setVisible(true);
          setTimeout(() => setPhaseRevealed(1), 300);
          setTimeout(() => setPhaseRevealed(2), 800);
          setTimeout(() => setPhaseRevealed(3), 1400);
        }
      },
      { threshold: 0.3 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [visible]);

  return (
    <svg
      ref={ref}
      viewBox="0 0 820 440"
      className="diagram-svg"
      style={{ maxWidth: 820, margin: "0 auto", display: "block" }}
    >
      {/* Phase backgrounds */}
      {PHASES.map((p, i) => (
        <g key={i} opacity={phaseRevealed > i ? 1 : 0} style={{ transition: "opacity 0.5s" }}>
          <rect
            x={p.x}
            y={60}
            width={p.w}
            height={340}
            fill="var(--d-fg)"
            opacity={0.03}
            rx={8}
          />
          <text
            x={p.x + p.w / 2}
            y={45}
            textAnchor="middle"
            fill="var(--d-fg)"
            fontSize={14}
            fontWeight={700}
            fontFamily="Georgia, serif"
          >
            {p.label}
          </text>
          <text
            x={p.x + p.w / 2}
            y={415}
            textAnchor="middle"
            fill="var(--d-muted)"
            fontSize={11}
            fontFamily="Georgia, serif"
          >
            {p.sublabel} ({p.cycles})
          </text>
        </g>
      ))}

      {/* Shared evolution (Phase 1) */}
      {phaseRevealed >= 1 && (
        <g style={{ transition: "opacity 0.5s" }}>
          <rect
            x={100}
            y={185}
            width={140}
            height={70}
            fill="var(--d-fg)"
            opacity={0.08}
            rx={6}
            stroke="var(--d-line)"
            strokeWidth={1}
          />
          <text x={170} y={215} textAnchor="middle" fill="var(--d-fg)" fontSize={13} fontWeight={700} fontFamily="Georgia, serif">
            Shared Evolution
          </text>
          <text x={170} y={235} textAnchor="middle" fill="var(--d-muted)" fontSize={11} fontFamily="Georgia, serif">
            All seeds identical
          </text>
        </g>
      )}

      {/* Fork arrows and conditions (Phase 2) */}
      {phaseRevealed >= 2 && CONDITIONS.map((c) => (
        <g
          key={c.id}
          onMouseEnter={() => setHovered(c.id)}
          onMouseLeave={() => setHovered(null)}
          style={{ cursor: "pointer", transition: "opacity 0.3s" }}
          opacity={hovered === null || hovered === c.id ? 1 : 0.3}
        >
          {/* Fork arrow */}
          <path
            d={`M 240 220 C 280 220, 290 ${c.y + 35}, 330 ${c.y + 35}`}
            fill="none"
            stroke={c.color}
            strokeWidth={2}
            opacity={0.6}
          />

          {/* Condition box */}
          <rect
            x={330}
            y={c.y}
            width={220}
            height={70}
            fill={hovered === c.id ? c.color : "var(--d-fg)"}
            opacity={hovered === c.id ? 0.15 : 0.08}
            rx={6}
            stroke={c.color}
            strokeWidth={hovered === c.id ? 2 : 1}
          />
          <text x={340} y={c.y + 25} fill={c.color} fontSize={13} fontWeight={700} fontFamily="monospace">
            {c.label}
          </text>
          <text x={340} y={c.y + 45} fill="var(--d-muted)" fontSize={11} fontFamily="Georgia, serif">
            {c.sublabel}
          </text>
          <text x={540} y={c.y + 25} textAnchor="end" fill="var(--d-fg)" fontSize={11} fontFamily="monospace" opacity={0.7}>
            {c.result}
          </text>
        </g>
      ))}

      {/* Novel test arrows and results (Phase 3) */}
      {phaseRevealed >= 3 && CONDITIONS.map((c) => (
        <g
          key={`result-${c.id}`}
          opacity={hovered === null || hovered === c.id ? 1 : 0.3}
          style={{ transition: "opacity 0.3s" }}
        >
          {/* Arrow to test */}
          <line
            x1={550}
            y1={c.y + 35}
            x2={620}
            y2={c.y + 35}
            stroke={c.color}
            strokeWidth={2}
            opacity={0.5}
            markerEnd={`url(#arrow-${c.id})`}
          />

          {/* Result box */}
          <rect
            x={625}
            y={c.y + 5}
            width={140}
            height={60}
            fill={c.verdict === "CREATION" ? "#0f2a1a" : "var(--d-fg)"}
            opacity={c.verdict === "CREATION" ? 0.4 : 0.08}
            rx={6}
            stroke={c.verdict === "CREATION" ? "#4ade80" : "var(--d-line)"}
            strokeWidth={c.verdict === "CREATION" ? 2 : 1}
          />
          <text
            x={695}
            y={c.y + 28}
            textAnchor="middle"
            fill={c.verdict === "CREATION" ? "#4ade80" : "var(--d-muted)"}
            fontSize={12}
            fontWeight={700}
            fontFamily="monospace"
          >
            {c.verdict}
          </text>
          <text x={695} y={c.y + 48} textAnchor="middle" fill="var(--d-muted)" fontSize={10} fontFamily="monospace">
            {c.beta} {c.pval !== "—" ? `(${c.pval})` : ""}
          </text>
        </g>
      ))}

      {/* Arrow markers */}
      <defs>
        {CONDITIONS.map((c) => (
          <marker
            key={c.id}
            id={`arrow-${c.id}`}
            markerWidth="8"
            markerHeight="6"
            refX="8"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0, 8 3, 0 6" fill={c.color} opacity={0.6} />
          </marker>
        ))}
      </defs>

      {/* Hover detail tooltip */}
      {hovered && (() => {
        const c = CONDITIONS.find((c) => c.id === hovered)!;
        return (
          <g>
            <rect x={250} y={390} width={320} height={35} fill="var(--d-fg)" opacity={0.1} rx={4} />
            <text x={410} y={412} textAnchor="middle" fill={c.color} fontSize={12} fontFamily="Georgia, serif">
              {c.label}: novel-stress robustness = {c.result.split("= ")[1]} | {c.beta !== "—" ? c.beta : "no β"} | {c.verdict}
            </text>
          </g>
        );
      })()}
    </svg>
  );
}
