import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
  Easing,
} from "remotion";
import { THEMES, ThemeMode } from './themes';

/**
 * Gradient Coupling Animation
 *
 * Shows why 2-layer MLP breaks the decomposability wall.
 * Left: Linear head (gradients stay in their lane → decomposable)
 * Right: MLP head (gradients couple through composition → integrated)
 *
 * Phase 1 (0-100): Build the architectures
 * Phase 2 (100-220): Animate forward pass + backprop
 * Phase 3 (220-300): Show Φ result
 *
 * 300 frames @ 30fps = 10 seconds
 */

interface NeuronPos {
  x: number;
  y: number;
  label?: string;
}

export const GradientCouplingVideo: React.FC<{ theme?: ThemeMode }> = ({ theme }) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const t = THEMES[theme ?? 'dark'];

  // Derived colors
  const hidden = t.blue;
  const output = t.green;
  const gradient = t.yellow;
  const gradientCoupled = t.pink;
  const loss = t.red;
  const neuron = '#1e293b';
  const neuronActive = '#334155';
  const wire = '#444';

  function drawNeuron(
    n: NeuronPos,
    r: number,
    fill: string,
    stroke: string,
    opacity: number
  ) {
    return (
      <g key={`${n.x}-${n.y}`} opacity={opacity}>
        <circle
          cx={n.x}
          cy={n.y}
          r={r}
          fill={fill}
          stroke={stroke}
          strokeWidth={1.5}
        />
        {n.label && (
          <text
            x={n.x}
            y={n.y + 4}
            textAnchor="middle"
            fill={t.text}
            fontSize={9}
            fontFamily="monospace"
          >
            {n.label}
          </text>
        )}
      </g>
    );
  }

  // Layout: two panels side by side
  const panelW = 460;
  const panelH = 440;
  const gap = 40;
  const leftX = (width - panelW * 2 - gap) / 2;
  const rightX = leftX + panelW + gap;
  const panelY = 100;

  // Phase progress
  const buildProgress = interpolate(frame, [0, 90], [0, 1], {
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });
  const forwardProgress = interpolate(frame, [95, 160], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const backpropProgress = interpolate(frame, [165, 230], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const resultProgress = interpolate(frame, [240, 280], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });

  // Neural network layouts
  const hiddenY = panelY + 100;
  const headY = panelY + 220;
  const outputY = panelY + 330;

  // 4 hidden neurons
  const makeHidden = (baseX: number): NeuronPos[] =>
    Array.from({ length: 4 }, (_, i) => ({
      x: baseX + 100 + i * 80,
      y: hiddenY,
      label: `h${i + 1}`,
    }));

  // Linear: 2 output neurons (direct connection)
  const linearOutputs: NeuronPos[] = [
    { x: leftX + 180, y: outputY, label: "o₁" },
    { x: leftX + 280, y: outputY, label: "o₂" },
  ];

  // MLP: 3 intermediate neurons + 2 outputs
  const mlpMiddle: NeuronPos[] = [
    { x: rightX + 140, y: headY, label: "m₁" },
    { x: rightX + 230, y: headY, label: "m₂" },
    { x: rightX + 320, y: headY, label: "m₃" },
  ];
  const mlpOutputs: NeuronPos[] = [
    { x: rightX + 180, y: outputY, label: "o₁" },
    { x: rightX + 280, y: outputY, label: "o₂" },
  ];

  const leftHidden = makeHidden(leftX);
  const rightHidden = makeHidden(rightX);

  // Gradient pulse animation
  const pulsePhase = backpropProgress * Math.PI * 2;

  // Draw connection with optional gradient flow
  function drawConnection(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    isForward: boolean,
    isCoupled: boolean,
    flowProgress: number,
    key: string
  ) {
    const opacity = buildProgress;
    const baseColor = isCoupled ? gradientCoupled : wire;
    const flowColor = isCoupled ? gradientCoupled : gradient;

    // Flow dot position (forward: top→bottom, backprop: bottom→top)
    const dotT = isForward ? flowProgress : 1 - flowProgress;
    const dotX = x1 + (x2 - x1) * dotT;
    const dotY = y1 + (y2 - y1) * dotT;

    return (
      <g key={key}>
        <line
          x1={x1}
          y1={y1}
          x2={x2}
          y2={y2}
          stroke={baseColor}
          strokeWidth={isCoupled ? 2 : 1}
          opacity={opacity * 0.4}
        />
        {flowProgress > 0 && flowProgress < 1 && (
          <circle
            cx={dotX}
            cy={dotY}
            r={isCoupled ? 4 : 3}
            fill={flowColor}
            opacity={0.9}
          />
        )}
      </g>
    );
  }

  // Whether we're in forward or backprop phase
  const isForwardPhase = frame >= 95 && frame < 165;
  const isBackpropPhase = frame >= 165 && frame < 235;

  // Flow progress for connections
  const connectionFlow = isForwardPhase
    ? forwardProgress
    : isBackpropPhase
    ? backpropProgress
    : 0;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: t.bg,
        fontFamily: "Georgia, serif",
      }}
    >
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 20,
          left: 40,
          color: t.text,
          fontSize: 26,
          fontWeight: 700,
          opacity: interpolate(frame, [0, 20], [0, 1], {
            extrapolateRight: "clamp",
          }),
        }}
      >
        The Decomposability Wall
      </div>
      <div
        style={{
          position: "absolute",
          top: 52,
          left: 40,
          color: t.muted,
          fontSize: 15,
          fontStyle: "italic",
          opacity: interpolate(frame, [0, 20], [0, 1], {
            extrapolateRight: "clamp",
          }),
        }}
      >
        why gradient coupling through composition breaks it (V27/V28)
      </div>

      <svg width={width} height={height}>
        {/* Panel backgrounds */}
        <rect
          x={leftX}
          y={panelY}
          width={panelW}
          height={panelH}
          fill={t.panel}
          stroke={t.border}
          strokeWidth={0.5}
          rx={6}
          opacity={buildProgress}
        />
        <rect
          x={rightX}
          y={panelY}
          width={panelW}
          height={panelH}
          fill={t.panel}
          stroke={t.border}
          strokeWidth={0.5}
          rx={6}
          opacity={buildProgress}
        />

        {/* Panel labels */}
        <text
          x={leftX + panelW / 2}
          y={panelY + 30}
          textAnchor="middle"
          fill={t.muted}
          fontSize={16}
          fontFamily="Georgia, serif"
          opacity={buildProgress}
        >
          Linear Head (V22)
        </text>
        <text
          x={rightX + panelW / 2}
          y={panelY + 30}
          textAnchor="middle"
          fill={t.muted}
          fontSize={16}
          fontFamily="Georgia, serif"
          opacity={buildProgress}
        >
          MLP Head (V27)
        </text>

        {/* === LINEAR SIDE === */}

        {/* Layer labels */}
        <text
          x={leftX + 30}
          y={hiddenY + 4}
          fill={t.muted}
          fontSize={10}
          fontFamily="Georgia, serif"
          opacity={buildProgress}
        >
          hidden
        </text>
        <text
          x={leftX + 30}
          y={outputY + 4}
          fill={t.muted}
          fontSize={10}
          fontFamily="Georgia, serif"
          opacity={buildProgress}
        >
          output
        </text>

        {/* Linear connections: each hidden → each output (independent lanes) */}
        {leftHidden.map((h, hi) =>
          linearOutputs.map((o, oi) =>
            drawConnection(
              h.x,
              h.y + 15,
              o.x,
              o.y - 15,
              isForwardPhase,
              false,
              connectionFlow,
              `lin-${hi}-${oi}`
            )
          )
        )}

        {/* Hidden neurons */}
        {leftHidden.map((n) =>
          drawNeuron(n, 15, neuron, hidden, buildProgress)
        )}
        {/* Output neurons */}
        {linearOutputs.map((n) =>
          drawNeuron(n, 15, neuron, output, buildProgress)
        )}

        {/* Gradient labels for linear */}
        {isBackpropPhase && (
          <>
            <text
              x={leftX + panelW / 2}
              y={hiddenY + 55}
              textAnchor="middle"
              fill={gradient}
              fontSize={12}
              fontFamily="monospace"
              opacity={backpropProgress * 0.8}
            >
              ∂L/∂h = W_out · ∂L/∂o
            </text>
            <text
              x={leftX + panelW / 2}
              y={hiddenY + 72}
              textAnchor="middle"
              fill={t.muted}
              fontSize={11}
              fontFamily="Georgia, serif"
              opacity={backpropProgress * 0.6}
            >
              gradients stay in independent lanes
            </text>

            {/* Show independent gradient arrows */}
            {linearOutputs.map((o, oi) => {
              const targetH = leftHidden[oi * 2];
              if (!targetH) return null;
              return (
                <line
                  key={`grad-lin-${oi}`}
                  x1={o.x}
                  y1={o.y - 18}
                  x2={targetH.x}
                  y2={targetH.y + 18}
                  stroke={gradient}
                  strokeWidth={2}
                  strokeDasharray="4 3"
                  opacity={backpropProgress * 0.6}
                  markerEnd="url(#arrowAmber)"
                />
              );
            })}
          </>
        )}

        {/* === MLP SIDE === */}

        {/* Layer labels */}
        <text
          x={rightX + 30}
          y={hiddenY + 4}
          fill={t.muted}
          fontSize={10}
          fontFamily="Georgia, serif"
          opacity={buildProgress}
        >
          hidden
        </text>
        <text
          x={rightX + 30}
          y={headY + 4}
          fill={t.muted}
          fontSize={10}
          fontFamily="Georgia, serif"
          opacity={buildProgress}
        >
          middle
        </text>
        <text
          x={rightX + 30}
          y={outputY + 4}
          fill={t.muted}
          fontSize={10}
          fontFamily="Georgia, serif"
          opacity={buildProgress}
        >
          output
        </text>

        {/* Hidden → Middle connections (ALL-to-ALL = coupling!) */}
        {rightHidden.map((h, hi) =>
          mlpMiddle.map((m, mi) =>
            drawConnection(
              h.x,
              h.y + 15,
              m.x,
              m.y - 15,
              isForwardPhase,
              isBackpropPhase,
              connectionFlow,
              `mlp-hm-${hi}-${mi}`
            )
          )
        )}

        {/* Middle → Output connections */}
        {mlpMiddle.map((m, mi) =>
          mlpOutputs.map((o, oi) =>
            drawConnection(
              m.x,
              m.y + 15,
              o.x,
              o.y - 15,
              isForwardPhase,
              isBackpropPhase,
              connectionFlow,
              `mlp-mo-${mi}-${oi}`
            )
          )
        )}

        {/* MLP neurons */}
        {rightHidden.map((n) =>
          drawNeuron(n, 15, neuron, hidden, buildProgress)
        )}
        {mlpMiddle.map((n) =>
          drawNeuron(
            n,
            13,
            isBackpropPhase ? '#2d1525' : neuron,
            isBackpropPhase ? gradientCoupled : t.violet,
            buildProgress
          )
        )}
        {mlpOutputs.map((n) =>
          drawNeuron(n, 15, neuron, output, buildProgress)
        )}

        {/* ReLU label on middle layer */}
        {mlpMiddle.map((m, i) => (
          <text
            key={`relu-${i}`}
            x={m.x}
            y={m.y + 26}
            textAnchor="middle"
            fill={t.violet}
            fontSize={9}
            fontFamily="monospace"
            opacity={buildProgress * 0.6}
          >
            ReLU
          </text>
        ))}

        {/* Gradient labels for MLP */}
        {isBackpropPhase && (
          <>
            <text
              x={rightX + panelW / 2}
              y={headY + 50}
              textAnchor="middle"
              fill={gradientCoupled}
              fontSize={12}
              fontFamily="monospace"
              opacity={backpropProgress * 0.8}
            >
              ∂L/∂h = W₁ · diag(σ') · W₂ · ∂L/∂o
            </text>
            <text
              x={rightX + panelW / 2}
              y={headY + 67}
              textAnchor="middle"
              fill={t.muted}
              fontSize={11}
              fontFamily="Georgia, serif"
              opacity={backpropProgress * 0.6}
            >
              composition couples ALL hidden units
            </text>

            {/* Show coupled gradient arrows (all connected) */}
            {mlpOutputs.map((o, oi) =>
              mlpMiddle.map((m, mi) => (
                <line
                  key={`grad-mlp-${oi}-${mi}`}
                  x1={o.x}
                  y1={o.y - 18}
                  x2={m.x}
                  y2={m.y + 18}
                  stroke={gradientCoupled}
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  opacity={backpropProgress * 0.4}
                />
              ))
            )}
          </>
        )}

        {/* Arrow markers */}
        <defs>
          <marker
            id="arrowAmber"
            markerWidth="8"
            markerHeight="6"
            refX="8"
            refY="3"
            orient="auto"
          >
            <polygon
              points="0 0, 8 3, 0 6"
              fill={gradient}
              opacity={0.6}
            />
          </marker>
        </defs>

        {/* === RESULTS === */}
        {resultProgress > 0 && (
          <>
            {/* Linear result */}
            <rect
              x={leftX + 60}
              y={panelY + panelH - 60}
              width={panelW - 120}
              height={40}
              fill={t.panel}
              rx={6}
              opacity={resultProgress}
            />
            <text
              x={leftX + panelW / 2}
              y={panelY + panelH - 34}
              textAnchor="middle"
              fill={t.red}
              fontSize={18}
              fontWeight={700}
              fontFamily="Georgia, serif"
              opacity={resultProgress}
            >
              Φ ≈ 0.08 — decomposable
            </text>

            {/* MLP result */}
            <rect
              x={rightX + 60}
              y={panelY + panelH - 60}
              width={panelW - 120}
              height={40}
              fill={t.panel}
              rx={6}
              opacity={resultProgress}
            />
            <text
              x={rightX + panelW / 2}
              y={panelY + panelH - 34}
              textAnchor="middle"
              fill={t.green}
              fontSize={18}
              fontWeight={700}
              fontFamily="Georgia, serif"
              opacity={resultProgress}
            >
              Φ ≈ 0.25 — integrated
            </text>
          </>
        )}
      </svg>

      {/* Phase indicator */}
      <div
        style={{
          position: "absolute",
          bottom: 25,
          left: 40,
          color: t.muted,
          fontSize: 13,
        }}
      >
        {frame < 95
          ? "architecture"
          : frame < 165
          ? "forward pass →"
          : frame < 235
          ? "← backpropagation"
          : "result"}
      </div>

      {/* Bottom annotation */}
      {resultProgress > 0.5 && (
        <div
          style={{
            position: "absolute",
            bottom: 25,
            right: 40,
            color: t.green,
            fontSize: 14,
            opacity: (resultProgress - 0.5) * 2,
          }}
        >
          not nonlinearity, not bottleneck width — gradient coupling through
          composition
        </div>
      )}
    </AbsoluteFill>
  );
};
