'use client';

import { useState, useEffect, useRef } from 'react';

/**
 * ContaminationFlow — Part IV diagram
 * Animated on scroll: shows two manifolds (friendship + transaction)
 * separating then overlapping, with gradient conflict emerging
 */

const W = 640;
const H = 350;
const CX = W / 2;
const CY = 185;

export default function ContaminationFlow() {
  const [phase, setPhase] = useState(0); // 0=separated, 1=approaching, 2=overlapping
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

  // Auto-advance phases
  useEffect(() => {
    if (!visible) return;
    const t1 = setTimeout(() => setPhase(1), 800);
    const t2 = setTimeout(() => setPhase(2), 2000);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [visible]);

  // Manifold positions based on phase
  const sep = phase === 0 ? 140 : phase === 1 ? 80 : 30;
  const friendX = CX - sep;
  const transX = CX + sep;

  const conflictOpacity = phase === 2 ? 1 : 0;

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${W} ${H}`}
      className="diagram-svg"
      role="img"
      aria-label="Animation showing friendship and transaction manifolds colliding to produce gradient conflict"
    >
      {/* Title */}
      <text x={CX} y={26} textAnchor="middle" fill="var(--d-fg)" fontSize={15} fontWeight={700}
        fontFamily="var(--font-body, Georgia, serif)"
        opacity={visible ? 1 : 0} style={{ transition: 'opacity 0.5s' }}>
        How Contamination Happens
      </text>
      <text x={CX} y={44} textAnchor="middle" fill="var(--d-muted)" fontSize={10.5}
        fontFamily="var(--font-body, Georgia, serif)"
        opacity={visible ? 1 : 0} style={{ transition: 'opacity 0.5s' }}>
        when two relationship manifolds are forced to overlap
      </text>

      {/* Friendship manifold */}
      <g style={{ transition: 'transform 1s ease-in-out' }}>
        <ellipse cx={friendX} cy={CY} rx={100} ry={75}
          fill="var(--d-green)" fillOpacity={0.1}
          stroke="var(--d-green)" strokeWidth={1.5}
          style={{ transition: 'cx 1s ease-in-out' }}
        />
        <text x={friendX - 40} y={CY - 30} textAnchor="middle"
          fill="var(--d-green)" fontSize={12} fontWeight={700}
          fontFamily="var(--font-body, Georgia, serif)"
          style={{ transition: 'x 1s ease-in-out' }}>
          Friendship
        </text>
        {/* Friendship gradient arrow (toward center = mutual flourishing) */}
        <line x1={friendX - 50} y1={CY + 15} x2={friendX - 10} y2={CY + 5}
          stroke="var(--d-green)" strokeWidth={2} markerEnd="url(#arrowGreen)"
          opacity={0.7} style={{ transition: 'all 1s ease-in-out' }}
        />
        <text x={friendX - 30} y={CY + 35} textAnchor="middle"
          fill="var(--d-green)" fontSize={9}
          fontFamily="var(--font-body, Georgia, serif)">
          ∇V_F
        </text>
      </g>

      {/* Transaction manifold */}
      <g style={{ transition: 'transform 1s ease-in-out' }}>
        <ellipse cx={transX} cy={CY} rx={100} ry={75}
          fill="var(--d-orange)" fillOpacity={0.1}
          stroke="var(--d-orange)" strokeWidth={1.5}
          style={{ transition: 'cx 1s ease-in-out' }}
        />
        <text x={transX + 40} y={CY - 30} textAnchor="middle"
          fill="var(--d-orange)" fontSize={12} fontWeight={700}
          fontFamily="var(--font-body, Georgia, serif)"
          style={{ transition: 'x 1s ease-in-out' }}>
          Transaction
        </text>
        {/* Transaction gradient arrow (toward balanced exchange = outward) */}
        <line x1={transX + 10} y1={CY + 5} x2={transX + 50} y2={CY + 15}
          stroke="var(--d-orange)" strokeWidth={2} markerEnd="url(#arrowOrange)"
          opacity={0.7} style={{ transition: 'all 1s ease-in-out' }}
        />
        <text x={transX + 30} y={CY + 35} textAnchor="middle"
          fill="var(--d-orange)" fontSize={9}
          fontFamily="var(--font-body, Georgia, serif)">
          ∇V_T
        </text>
      </g>

      {/* Conflict zone (only visible in phase 2) */}
      <g opacity={conflictOpacity} style={{ transition: 'opacity 0.8s ease-in 0.3s' }}>
        <ellipse cx={CX} cy={CY} rx={40} ry={55}
          fill="var(--d-red)" fillOpacity={0.12}
          stroke="var(--d-red)" strokeWidth={1.5} strokeDasharray="4 3" />
        <text x={CX} y={CY - 8} textAnchor="middle"
          fill="var(--d-red)" fontSize={16} fontWeight={700}>
          ✕
        </text>
        <text x={CX} y={CY + 12} textAnchor="middle"
          fill="var(--d-red)" fontSize={9.5} fontWeight={600}
          fontFamily="var(--font-body, Georgia, serif)">
          gradient conflict
        </text>
      </g>

      {/* Phase label */}
      <text x={CX} y={H - 40} textAnchor="middle" fill="var(--d-muted)" fontSize={11}
        fontStyle="italic" fontFamily="var(--font-body, Georgia, serif)"
        opacity={visible ? 1 : 0} style={{ transition: 'opacity 0.5s' }}>
        {phase === 0 ? 'distinct relationship types — clean gradients' :
         phase === 1 ? 'forced proximity...' :
         '∇V_F · ∇V_T < 0 — valence becomes uncomputable'}
      </text>

      {/* Arrow markers */}
      <defs>
        <marker id="arrowGreen" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="var(--d-green)" />
        </marker>
        <marker id="arrowOrange" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="var(--d-orange)" />
        </marker>
      </defs>

      {/* Bottom note */}
      <text x={CX} y={H - 18} textAnchor="middle" fill="var(--d-muted)" fontSize={9}
        fontFamily="var(--font-body, Georgia, serif)"
        opacity={visible ? 1 : 0} style={{ transition: 'opacity 0.5s' }}>
        the detection system responds to the shadow manifold, not the surface action
      </text>
    </svg>
  );
}
