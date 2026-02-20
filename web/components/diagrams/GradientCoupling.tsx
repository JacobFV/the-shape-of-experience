'use client';

import { useEffect, useRef, useState } from 'react';
import { type Point, arrowPath, pt, smoothOpen } from './utils';

/**
 * Animated diagram showing why linear readouts are decomposable
 * and 2-layer MLP readouts force gradient coupling.
 *
 * Left panel: linear head — gradients flow independently to each hidden unit.
 * Right panel: MLP head — gradients couple through the shared hidden layer.
 */
export default function GradientCoupling() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [animate, setAnimate] = useState(false);
  const [phase, setPhase] = useState(0); // 0=idle, 1=linear flow, 2=mlp flow

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setAnimate(true); observer.disconnect(); } },
      { threshold: 0.3 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!animate) return;
    // Phase sequence: 0 → 1 (linear) → 2 (MLP), with delays
    const t1 = setTimeout(() => setPhase(1), 400);
    const t2 = setTimeout(() => setPhase(2), 2200);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [animate]);

  // Layout
  const leftCx = 155;
  const rightCx = 415;
  const panelW = 230;

  // Hidden units (h_1 through h_4) — same positions for both panels
  const hiddenY = 70;
  const hiddenGap = 42;
  const nHidden = 4;
  const hiddenXs = (cx: number) =>
    Array.from({ length: nHidden }, (_, i) => cx - ((nHidden - 1) * hiddenGap) / 2 + i * hiddenGap);

  // MLP intermediate layer (m_1, m_2) — only right panel
  const midY = 150;
  const nMid = 2;
  const midXs = Array.from({ length: nMid }, (_, i) => rightCx - ((nMid - 1) * 50) / 2 + i * 50);

  // Output node
  const outY = 215;

  // Gradient flow paths (animated)
  const gradientPath = (fromX: number, fromY: number, toX: number, toY: number): string => {
    return `M ${fromX} ${fromY} L ${toX} ${toY}`;
  };

  const gradColor = (active: boolean, coupled: boolean) => {
    if (!active) return 'var(--d-muted)';
    return coupled ? 'var(--d-green)' : 'var(--d-blue)';
  };

  return (
    <svg ref={svgRef} viewBox="0 0 570 310" className="diagram-svg" role="img"
      aria-label="Gradient coupling: linear head sends independent gradients to each hidden unit; MLP head couples all units through shared intermediate layer">

      {/* Panel labels */}
      <text x={leftCx} y={20} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        Linear Head (V22)
      </text>
      <text x={rightCx} y={20} textAnchor="middle" fontSize={13} fontWeight={600}
        fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
        MLP Head (V27)
      </text>

      {/* Divider */}
      <line x1={285} y1={30} x2={285} y2={280} stroke="var(--d-line)" strokeWidth={0.5} strokeDasharray="4,4" opacity={0.3} />

      {/* ========== LEFT PANEL: Linear ========== */}
      <g>
        {/* Hidden units */}
        {hiddenXs(leftCx).map((x, i) => (
          <g key={`lh-${i}`}>
            <circle cx={x} cy={hiddenY} r={14}
              fill="var(--d-blue)" fillOpacity={0.12}
              stroke="var(--d-blue)" strokeWidth={1} />
            <text x={x} y={hiddenY} textAnchor="middle" dominantBaseline="central"
              fontSize={10} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              h<tspan dy={3} fontSize={7}>{i + 1}</tspan>
            </text>
          </g>
        ))}

        {/* Output */}
        <circle cx={leftCx} cy={outY} r={16}
          fill="var(--d-fg)" fillOpacity={0.08}
          stroke="var(--d-fg)" strokeWidth={1} />
        <text x={leftCx} y={outY} textAnchor="middle" dominantBaseline="central"
          fontSize={10} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
          ŷ
        </text>

        {/* Forward connections: h → ŷ (thin) */}
        {hiddenXs(leftCx).map((x, i) => (
          <line key={`lf-${i}`}
            x1={x} y1={hiddenY + 14} x2={leftCx} y2={outY - 16}
            stroke="var(--d-line)" strokeWidth={0.5} opacity={0.3}
          />
        ))}

        {/* Weight labels */}
        {hiddenXs(leftCx).map((x, i) => {
          const mx = (x + leftCx) / 2;
          const my = (hiddenY + 14 + outY - 16) / 2;
          return (
            <text key={`lw-${i}`} x={mx + (i < 2 ? -8 : 8)} y={my}
              textAnchor="middle" dominantBaseline="central"
              fontSize={9} fill="var(--d-muted)" fontStyle="italic"
              fontFamily="var(--font-body, Georgia, serif)">
              w<tspan dy={2} fontSize={6}>{i + 1}</tspan>
            </text>
          );
        })}

        {/* Gradient arrows (animated): ∂L/∂h_i = 2(ŷ-y)·w_i — independent per unit */}
        {hiddenXs(leftCx).map((x, i) => {
          const active = phase >= 1;
          return (
            <g key={`lg-${i}`} style={{
              opacity: active ? 1 : 0,
              transition: `opacity 0.5s ease ${i * 0.15}s`,
            }}>
              <line
                x1={leftCx + (x - leftCx) * 0.2} y1={outY - 25}
                x2={x} y2={hiddenY + 20}
                stroke="var(--d-blue)" strokeWidth={2} opacity={0.5}
                strokeDasharray="4,3"
              />
              <path
                d={arrowPath([x, hiddenY + 20], -90 + (x - leftCx) * 0.3, 5)}
                stroke="var(--d-blue)" strokeWidth={1.5} fill="none"
              />
            </g>
          );
        })}

        {/* "Independent" label */}
        <text x={leftCx} y={outY + 38} textAnchor="middle" fontSize={10}
          fill="var(--d-blue)"
          fontFamily="var(--font-body, Georgia, serif)"
          style={{ opacity: phase >= 1 ? 1 : 0, transition: 'opacity 0.6s ease 0.6s' }}>
          ∂L/∂h_i = 2(ŷ−y)·w_i
        </text>
        <text x={leftCx} y={outY + 52} textAnchor="middle" fontSize={10}
          fill="var(--d-blue)" fontWeight={600}
          fontFamily="var(--font-body, Georgia, serif)"
          style={{ opacity: phase >= 1 ? 1 : 0, transition: 'opacity 0.6s ease 0.8s' }}>
          each unit independent — decomposable
        </text>
      </g>

      {/* ========== RIGHT PANEL: MLP ========== */}
      <g>
        {/* Hidden units */}
        {hiddenXs(rightCx).map((x, i) => (
          <g key={`rh-${i}`}>
            <circle cx={x} cy={hiddenY} r={14}
              fill="var(--d-green)" fillOpacity={0.12}
              stroke="var(--d-green)" strokeWidth={1} />
            <text x={x} y={hiddenY} textAnchor="middle" dominantBaseline="central"
              fontSize={10} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              h<tspan dy={3} fontSize={7}>{i + 1}</tspan>
            </text>
          </g>
        ))}

        {/* Intermediate layer */}
        {midXs.map((x, i) => (
          <g key={`rm-${i}`}>
            <circle cx={x} cy={midY} r={12}
              fill="var(--d-orange)" fillOpacity={0.12}
              stroke="var(--d-orange)" strokeWidth={1} />
            <text x={x} y={midY} textAnchor="middle" dominantBaseline="central"
              fontSize={9} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
              m<tspan dy={2} fontSize={6}>{i + 1}</tspan>
            </text>
          </g>
        ))}

        {/* Output */}
        <circle cx={rightCx} cy={outY} r={16}
          fill="var(--d-fg)" fillOpacity={0.08}
          stroke="var(--d-fg)" strokeWidth={1} />
        <text x={rightCx} y={outY} textAnchor="middle" dominantBaseline="central"
          fontSize={10} fill="var(--d-fg)" fontFamily="var(--font-body, Georgia, serif)">
          ŷ
        </text>

        {/* Forward connections: h → m (thin) */}
        {hiddenXs(rightCx).map((hx, hi) =>
          midXs.map((mx, mi) => (
            <line key={`rf1-${hi}-${mi}`}
              x1={hx} y1={hiddenY + 14} x2={mx} y2={midY - 12}
              stroke="var(--d-line)" strokeWidth={0.4} opacity={0.25}
            />
          ))
        )}

        {/* Forward connections: m → ŷ (thin) */}
        {midXs.map((mx, mi) => (
          <line key={`rf2-${mi}`}
            x1={mx} y1={midY + 12} x2={rightCx} y2={outY - 16}
            stroke="var(--d-line)" strokeWidth={0.4} opacity={0.25}
          />
        ))}

        {/* Layer labels */}
        <text x={rightCx + panelW / 2 - 5} y={hiddenY} textAnchor="start" fontSize={9}
          fill="var(--d-muted)" fontStyle="italic" fontFamily="var(--font-body, Georgia, serif)">
          W₁
        </text>
        <text x={rightCx + panelW / 2 - 5} y={midY} textAnchor="start" fontSize={9}
          fill="var(--d-muted)" fontStyle="italic" fontFamily="var(--font-body, Georgia, serif)">
          tanh
        </text>
        <text x={rightCx + panelW / 2 - 5} y={(midY + outY) / 2} textAnchor="start" fontSize={9}
          fill="var(--d-muted)" fontStyle="italic" fontFamily="var(--font-body, Georgia, serif)">
          W₂
        </text>

        {/* Gradient arrows (animated): coupling through shared intermediate */}
        {/* First: ŷ → m (all m units receive gradient) */}
        {midXs.map((mx, mi) => (
          <g key={`rg1-${mi}`} style={{
            opacity: phase >= 2 ? 1 : 0,
            transition: `opacity 0.4s ease ${mi * 0.15}s`,
          }}>
            <line
              x1={rightCx + (mx - rightCx) * 0.3} y1={outY - 22}
              x2={mx} y2={midY + 16}
              stroke="var(--d-orange)" strokeWidth={2} opacity={0.5}
              strokeDasharray="4,3"
            />
            <path
              d={arrowPath([mx, midY + 16], -90 + (mx - rightCx) * 0.4, 4)}
              stroke="var(--d-orange)" strokeWidth={1.5} fill="none"
            />
          </g>
        ))}

        {/* Then: m → h (EACH m fans out to ALL h units) */}
        {midXs.map((mx, mi) =>
          hiddenXs(rightCx).map((hx, hi) => (
            <g key={`rg2-${mi}-${hi}`} style={{
              opacity: phase >= 2 ? 1 : 0,
              transition: `opacity 0.4s ease ${0.5 + (mi * nHidden + hi) * 0.08}s`,
            }}>
              <line
                x1={mx} y1={midY - 16}
                x2={hx} y2={hiddenY + 18}
                stroke="var(--d-green)" strokeWidth={1.8} opacity={0.4}
                strokeDasharray="3,3"
              />
              <path
                d={arrowPath([hx, hiddenY + 18], -90 + (hx - mx) * 0.4, 4)}
                stroke="var(--d-green)" strokeWidth={1.3} fill="none"
              />
            </g>
          ))
        )}

        {/* "Coupled" label */}
        <text x={rightCx} y={outY + 38} textAnchor="middle" fontSize={10}
          fill="var(--d-green)"
          fontFamily="var(--font-body, Georgia, serif)"
          style={{ opacity: phase >= 2 ? 1 : 0, transition: 'opacity 0.6s ease 1s' }}>
          ∂L/∂h_i = ... · W₂ᵀ · diag(1−tanh²) · W₁
        </text>
        <text x={rightCx} y={outY + 52} textAnchor="middle" fontSize={10}
          fill="var(--d-green)" fontWeight={600}
          fontFamily="var(--font-body, Georgia, serif)"
          style={{ opacity: phase >= 2 ? 1 : 0, transition: 'opacity 0.6s ease 1.2s' }}>
          every unit coupled — non-decomposable
        </text>
      </g>

      {/* Bottom comparison */}
      <g style={{ opacity: phase >= 2 ? 1 : 0, transition: 'opacity 0.8s ease 1.5s' }}>
        <text x={leftCx} y={290} textAnchor="middle" fontSize={11}
          fill="var(--d-blue)" fontFamily="var(--font-body, Georgia, serif)">
          Φ ≈ 0.097
        </text>
        <text x={rightCx} y={290} textAnchor="middle" fontSize={11}
          fill="var(--d-green)" fontFamily="var(--font-body, Georgia, serif)">
          Φ ≈ 0.245 (2.5×)
        </text>
      </g>
    </svg>
  );
}
