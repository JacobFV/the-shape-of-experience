'use client';

import { useEffect, useRef, useState } from 'react';
import { type Point, smoothClosed, smoothOpen, arrowPath, angleBetween } from './utils';

/** Part 1-2: Viability manifold with animated trajectories */
export default function ViabilityManifold() {
  const svgRef = useRef<SVGSVGElement>(null);
  const viableRef = useRef<SVGPathElement>(null);
  const dissolveRef = useRef<SVGPathElement>(null);
  const [animate, setAnimate] = useState(false);

  // Trigger animation when diagram enters viewport
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

  // Set up stroke-dasharray from measured path lengths
  useEffect(() => {
    if (!animate) return;
    for (const ref of [viableRef, dissolveRef]) {
      const path = ref.current;
      if (!path) continue;
      const len = path.getTotalLength();
      path.style.strokeDasharray = `${len}`;
      path.style.strokeDashoffset = `${len}`;
      // Force reflow then animate
      path.getBoundingClientRect();
      path.style.transition = 'stroke-dashoffset 2.2s ease-in-out';
      path.style.strokeDashoffset = '0';
    }
  }, [animate]);

  // Manifold boundary (organic blob shape)
  const blobPoints: Point[] = [
    [200, 150], [270, 100], [370, 110], [430, 160],
    [450, 240], [430, 320], [370, 370], [280, 360],
    [190, 320], [170, 250],
  ];

  // Attractor inside manifold
  const attractor: Point = [310, 235];

  // Viable trajectory (stays inside, spirals toward attractor)
  const viablePath: Point[] = [
    [220, 170], [260, 145], [320, 140], [380, 170],
    [400, 220], [380, 270], [340, 280], [310, 255],
    [320, 240],
  ];

  // Dissolution trajectory (exits the manifold)
  const dissolvePath: Point[] = [
    [250, 310], [280, 340], [330, 355], [400, 360],
    [460, 350], [510, 330], [540, 300],
  ];

  // Axes
  const axO: Point = [90, 420];

  return (
    <svg ref={svgRef} viewBox="0 0 600 470" className="diagram-svg" role="img"
      aria-label="Viability manifold: a bounded region where a system can persist, with a viable trajectory spiraling inward and a dissolution trajectory exiting the boundary">

      {/* Axes */}
      <line x1={axO[0]} y1={axO[1]} x2={axO[0] + 100} y2={axO[1]}
        stroke="var(--d-line)" strokeWidth={0.75} />
      <path d={arrowPath([axO[0] + 100, axO[1]], 0, 5)}
        stroke="var(--d-line)" strokeWidth={0.75} fill="none" />
      <line x1={axO[0]} y1={axO[1]} x2={axO[0]} y2={axO[1] - 80}
        stroke="var(--d-line)" strokeWidth={0.75} />
      <path d={arrowPath([axO[0], axO[1] - 80], -90, 5)}
        stroke="var(--d-line)" strokeWidth={0.75} fill="none" />
      <text x={axO[0] + 105} y={axO[1] + 5}
        textAnchor="start" dominantBaseline="central" fontSize={14}
        fontStyle="italic" fill="var(--d-muted)"
        fontFamily="var(--font-body, Georgia, serif)">
        x<tspan dy={3} fontSize={10}>1</tspan>
      </text>
      <text x={axO[0] - 5} y={axO[1] - 85}
        textAnchor="end" dominantBaseline="central" fontSize={14}
        fontStyle="italic" fill="var(--d-muted)"
        fontFamily="var(--font-body, Georgia, serif)">
        x<tspan dy={3} fontSize={10}>2</tspan>
      </text>

      {/* Manifold fill */}
      <path d={smoothClosed(blobPoints, 0.3)}
        fill="var(--d-blue)" fillOpacity={0.06} stroke="none" />

      {/* Manifold boundary */}
      <path d={smoothClosed(blobPoints, 0.3)}
        fill="none" stroke="var(--d-blue)" strokeWidth={1.2} />

      {/* Manifold label */}
      <text x={460} y={130} textAnchor="start" fontSize={16}
        fontStyle="italic" fill="var(--d-blue)"
        fontFamily="var(--font-body, Georgia, serif)">
        V
      </text>
      <text x={180} y={200} textAnchor="end" fontSize={12}
        fontStyle="italic" fill="var(--d-blue)" opacity={0.7}
        fontFamily="var(--font-body, Georgia, serif)">
        ∂V
      </text>

      {/* Attractor */}
      <circle cx={attractor[0]} cy={attractor[1]} r={3.5} fill="var(--d-fg)" />
      <circle cx={attractor[0]} cy={attractor[1]} r={12}
        fill="none" stroke="var(--d-fg)" strokeWidth={0.5}
        strokeDasharray="2,2" />

      {/* Viable trajectory (green, animated) */}
      <path ref={viableRef} d={smoothOpen(viablePath, 0.3)}
        fill="none" stroke="var(--d-green)" strokeWidth={1.3}
        strokeLinecap="round" />
      {/* Arrowhead — visible only after animation */}
      {animate && (() => {
        const n = viablePath.length;
        const tip = viablePath[n - 1];
        const prev = viablePath[n - 2];
        return <path d={arrowPath(tip, angleBetween(prev, tip), 6)}
          stroke="var(--d-green)" strokeWidth={1.3} fill="none"
          style={{ opacity: 0, animation: 'fade-in 0.3s ease 2.2s forwards' }} />;
      })()}
      {/* Start dot */}
      <circle cx={viablePath[0][0]} cy={viablePath[0][1]} r={3}
        fill="var(--d-green)" />
      <text x={viablePath[0][0] - 10} y={viablePath[0][1] - 12}
        textAnchor="middle" fontSize={11.5} fill="var(--d-green)"
        fontFamily="var(--font-body, Georgia, serif)">
        viable
      </text>

      {/* Dissolution trajectory (red, animated with delay) */}
      <path ref={dissolveRef} d={smoothOpen(dissolvePath, 0.3)}
        fill="none" stroke="var(--d-red)" strokeWidth={1.3}
        strokeLinecap="round"
        style={animate ? { transitionDelay: '0.8s' } : undefined} />
      {animate && (() => {
        const n = dissolvePath.length;
        const tip = dissolvePath[n - 1];
        const prev = dissolvePath[n - 2];
        return <path d={arrowPath(tip, angleBetween(prev, tip), 6)}
          stroke="var(--d-red)" strokeWidth={1.3} fill="none"
          style={{ opacity: 0, animation: 'fade-in 0.3s ease 3s forwards' }} />;
      })()}
      <circle cx={dissolvePath[0][0]} cy={dissolvePath[0][1]} r={3}
        fill="var(--d-red)" />
      <text x={530} y={285} textAnchor="start" fontSize={11.5}
        fill="var(--d-red)"
        fontFamily="var(--font-body, Georgia, serif)">
        dissolution
      </text>
    </svg>
  );
}
