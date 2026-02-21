'use client';

import { useState, useEffect, useRef } from 'react';
import { type Point, arrowPath } from './utils';

/**
 * WhatContinues — Epilogue diagram
 * Pattern propagation: individual → influence → cultural pattern → information conservation
 * Geometric representation of how affect structure persists through substrate change
 */

interface Stage {
  id: string;
  label: string;
  sublabel: string;
  x: number;
  color: string;
}

const STAGES: Stage[] = [
  { id: 'individual', label: 'Individual', sublabel: 'embodied pattern', x: 80, color: 'var(--d-blue)' },
  { id: 'influence', label: 'Influence', sublabel: 'affect resonance', x: 230, color: 'var(--d-green)' },
  { id: 'cultural', label: 'Cultural Pattern', sublabel: 'distributed memory', x: 400, color: 'var(--d-orange)' },
  { id: 'information', label: 'Information', sublabel: 'geometric invariant', x: 560, color: 'var(--d-violet)' },
];

const W = 640;
const H = 340;
const CY = 160;
const NODE_R = 40;

export default function WhatContinues() {
  const [hovered, setHovered] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setVisible(true); obs.disconnect(); } },
      { threshold: 0.3 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${W} ${H}`}
      className={`diagram-svg${hovered ? ' has-focus' : ''}`}
      role="img"
      aria-label="Pattern propagation from individual through influence to cultural pattern to information conservation"
    >
      {/* Title */}
      <text x={W / 2} y={28} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}
        fontFamily="var(--font-body, Georgia, serif)">
        What Continues
      </text>
      <text x={W / 2} y={46} textAnchor="middle" fill="var(--d-muted)" fontSize={10.5}
        fontFamily="var(--font-body, Georgia, serif)">
        affect structure persists through substrate change
      </text>

      {/* Connecting arrows between stages */}
      {STAGES.slice(0, -1).map((stage, i) => {
        const next = STAGES[i + 1];
        const x1 = stage.x + NODE_R + 6;
        const x2 = next.x - NODE_R - 6;
        const midX = (x1 + x2) / 2;
        return (
          <g key={`arrow-${i}`}
            opacity={visible ? 0.5 : 0}
            style={{ transition: `opacity 0.6s ease ${0.3 + i * 0.3}s` }}
          >
            <line x1={x1} y1={CY} x2={x2} y2={CY}
              stroke="var(--d-line)" strokeWidth={1.2}
              strokeDasharray="6 4" />
            <path d={arrowPath([x2, CY], 0, 6)}
              stroke="var(--d-line)" strokeWidth={1.2} fill="none" />
            {/* Transform label */}
            <text x={midX} y={CY - 12} textAnchor="middle"
              fill="var(--d-muted)" fontSize={8.5} fontStyle="italic"
              fontFamily="var(--font-body, Georgia, serif)">
              {i === 0 ? 'shapes others' : i === 1 ? 'becomes tradition' : 'conserves form'}
            </text>
          </g>
        );
      })}

      {/* Stage nodes */}
      {STAGES.map((stage, i) => {
        const isFocused = hovered === stage.id;
        const isDimmed = hovered !== null && !isFocused;

        return (
          <g key={stage.id}
            className={`interactive${isFocused ? ' focused' : ''}`}
            onMouseEnter={() => setHovered(stage.id)}
            onMouseLeave={() => setHovered(null)}
            style={{
              opacity: visible ? (isDimmed ? 0.2 : 1) : 0,
              transition: `opacity 0.5s ease ${i * 0.2}s`,
              cursor: 'pointer',
            }}
          >
            {/* Node circle */}
            <circle cx={stage.x} cy={CY} r={NODE_R}
              fill={stage.color} fillOpacity={isFocused ? 0.2 : 0.08}
              stroke={stage.color} strokeWidth={isFocused ? 2 : 1}
              style={{ transition: 'fill-opacity 0.2s, stroke-width 0.2s' }}
            />
            {/* Label */}
            <text x={stage.x} y={CY - 4} textAnchor="middle" dominantBaseline="central"
              fill={stage.color} fontSize={11} fontWeight={700}
              fontFamily="var(--font-body, Georgia, serif)">
              {stage.label}
            </text>
            {/* Sublabel */}
            <text x={stage.x} y={CY + 12} textAnchor="middle" dominantBaseline="central"
              fill="var(--d-muted)" fontSize={8.5}
              fontFamily="var(--font-body, Georgia, serif)">
              {stage.sublabel}
            </text>
          </g>
        );
      })}

      {/* Substrate change indicator */}
      <g opacity={visible ? 0.6 : 0} style={{ transition: 'opacity 1s ease 1.2s' }}>
        <line x1={80} y1={CY + 65} x2={560} y2={CY + 65}
          stroke="var(--d-line)" strokeWidth={0.5} strokeDasharray="3 3" />
        <text x={W / 2} y={CY + 80} textAnchor="middle"
          fill="var(--d-muted)" fontSize={9.5}
          fontFamily="var(--font-body, Georgia, serif)">
          substrate changes — pattern persists
        </text>

        {/* Small substrate icons below each stage */}
        {STAGES.map((stage, i) => (
          <g key={`sub-${i}`}>
            <rect x={stage.x - 15} y={CY + 55} width={30} height={8} rx={2}
              fill={stage.color} fillOpacity={0.15}
              stroke={stage.color} strokeWidth={0.5} />
            {i > 0 && (
              <rect x={stage.x - 12} y={CY + 57} width={24} height={4} rx={1}
                fill={stage.color} fillOpacity={0.3} />
            )}
          </g>
        ))}
      </g>

      {/* Bottom insight */}
      <text x={W / 2} y={H - 25} textAnchor="middle" fill="var(--d-muted)" fontSize={10.5}
        fontStyle="italic" fontFamily="var(--font-body, Georgia, serif)"
        opacity={visible ? 1 : 0} style={{ transition: 'opacity 1s ease 1.5s' }}>
        what you are is a pattern — and patterns propagate
      </text>
    </svg>
  );
}
