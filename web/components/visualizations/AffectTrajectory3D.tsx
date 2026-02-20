'use client';

import { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TrajectoryPoint {
  x: number;
  y: number;
  z: number;
  cycle: number;
  step: number;
  energy: number;
  condition: 'normal' | 'drought';
}

interface FocalStep {
  step: number;
  position: number[];
  energy: number;
  energy_delta: number;
  prediction: number;
  prediction_error: number;
  move_direction: number;
  consume: number;
  emit: number;
  hidden_state: number[];
  local_resources: number[][];
  local_signals: number[][];
  local_agents: number[][];
}

interface CycleData {
  meta: {
    cycle: number;
    is_drought: boolean;
    condition: string;
    n_alive_start: number;
    n_alive_end: number;
    mortality: number;
    phi: number;
    segment_affect: {
      valence: number;
      arousal: number;
      effective_rank: number;
    };
  };
  focal_timeline: FocalStep[];
  env_snapshots: {
    step: number;
    resource_grid: number[][];
    signal_grid: number[][];
    agents: { pos: number[]; energy: number }[];
    n_alive: number;
  }[];
}

interface TrajectoryData {
  pca_explained_variance: number[];
  points: TrajectoryPoint[];
}

interface AffectTrajectory3DProps {
  trajectoryUrl?: string;
  cycleUrls?: Record<number, string>;
  demoMode?: boolean;
}

// ---------------------------------------------------------------------------
// Color utilities
// ---------------------------------------------------------------------------

function conditionColor(condition: string, energy: number): THREE.Color {
  if (condition === 'drought') {
    // Red range — darker for lower energy
    const r = 0.6 + energy * 0.4;
    return new THREE.Color(r, 0.15, 0.1);
  }
  // Normal — green to cyan range
  const g = 0.4 + energy * 0.4;
  return new THREE.Color(0.1, g, 0.3 + energy * 0.2);
}

function timeColor(t: number): THREE.Color {
  // Time gradient: blue → cyan → green → yellow → orange
  if (t < 0.25) {
    return new THREE.Color().setHSL(0.6 - t * 0.8, 0.7, 0.5);
  } else if (t < 0.5) {
    return new THREE.Color().setHSL(0.4 - (t - 0.25) * 1.2, 0.7, 0.5);
  } else if (t < 0.75) {
    return new THREE.Color().setHSL(0.15 - (t - 0.5) * 0.2, 0.8, 0.5);
  }
  return new THREE.Color().setHSL(0.08 - (t - 0.75) * 0.12, 0.9, 0.45);
}

// ---------------------------------------------------------------------------
// 3D Scene Components
// ---------------------------------------------------------------------------

function TrajectoryPath({
  points,
  currentIdx,
}: {
  points: TrajectoryPoint[];
  currentIdx: number;
}) {
  const ghostRef = useRef<THREE.Line>(null);
  const trailRef = useRef<THREE.Line>(null);

  const colors = useMemo(() => {
    return points.map((p) => conditionColor(p.condition, p.energy));
  }, [points]);

  // Ghost: full trajectory, faint
  const ghostLine = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(points.length * 3);
    const colorArr = new Float32Array(points.length * 3);
    points.forEach((p, i) => {
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
      const c = colors[i];
      colorArr[i * 3] = c.r;
      colorArr[i * 3 + 1] = c.g;
      colorArr[i * 3 + 2] = c.b;
    });
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colorArr, 3));
    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.15,
    });
    return new THREE.Line(geometry, material);
  }, [points, colors]);

  // Trail: up to currentIdx, bright
  useEffect(() => {
    if (!trailRef.current) return;
    const n = Math.min(currentIdx + 1, points.length);
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(n * 3);
    const colorArr = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const p = points[i];
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
      const c = colors[i];
      colorArr[i * 3] = c.r;
      colorArr[i * 3 + 1] = c.g;
      colorArr[i * 3 + 2] = c.b;
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colorArr, 3));
    trailRef.current.geometry.dispose();
    trailRef.current.geometry = geometry;
  }, [points, colors, currentIdx]);

  const trailLine = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(3), 3)
    );
    const material = new THREE.LineBasicMaterial({ vertexColors: true });
    return new THREE.Line(geometry, material);
  }, []);

  return (
    <group>
      <primitive object={ghostLine} />
      <primitive ref={trailRef} object={trailLine} />
    </group>
  );
}

function CurrentPoint({
  point,
  index,
  total,
}: {
  point: TrajectoryPoint;
  index: number;
  total: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.5;
    }
    if (glowRef.current) {
      const scale = 1.0 + Math.sin(Date.now() * 0.003) * 0.15;
      glowRef.current.scale.setScalar(scale);
    }
  });

  const color = conditionColor(point.condition, point.energy);

  return (
    <group position={[point.x, point.y, point.z]}>
      {/* Glow */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} />
      </mesh>
      {/* Core */}
      <mesh ref={meshRef}>
        <octahedronGeometry args={[0.06, 0]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.5}
        />
      </mesh>
    </group>
  );
}

function DroughtMarkers({ points }: { points: TrajectoryPoint[] }) {
  // Find drought onset/offset transitions
  const markers: { position: THREE.Vector3; label: string }[] = [];
  let inDrought = false;
  let droughtCount = 0;

  for (let i = 1; i < points.length; i++) {
    if (points[i].condition === 'drought' && !inDrought) {
      droughtCount++;
      inDrought = true;
      markers.push({
        position: new THREE.Vector3(points[i].x, points[i].y + 0.3, points[i].z),
        label: `drought ${droughtCount}`,
      });
    } else if (points[i].condition !== 'drought' && inDrought) {
      inDrought = false;
    }
  }

  return (
    <group>
      {markers.map((m, i) => (
        <Text
          key={i}
          position={m.position}
          fontSize={0.08}
          color="#cc3333"
          anchorX="center"
          anchorY="bottom"
          font="/fonts/inter-regular.woff"
        >
          {m.label}
        </Text>
      ))}
    </group>
  );
}

function AxisLabels({ variance }: { variance: number[] }) {
  const labels = variance.map(
    (v, i) => `PC${i + 1} (${(v * 100).toFixed(0)}%)`
  );

  return (
    <group>
      <Text
        position={[2.5, 0, 0]}
        fontSize={0.07}
        color="#888"
        anchorX="center"
      >
        {labels[0] || 'PC1'}
      </Text>
      <Text
        position={[0, 2.5, 0]}
        fontSize={0.07}
        color="#888"
        anchorX="center"
      >
        {labels[1] || 'PC2'}
      </Text>
      <Text
        position={[0, 0, 2.5]}
        fontSize={0.07}
        color="#888"
        anchorX="center"
      >
        {labels[2] || 'PC3'}
      </Text>
    </group>
  );
}

function Scene({
  trajectory,
  currentIdx,
  playing,
  setCurrentIdx,
  speed,
}: {
  trajectory: TrajectoryData;
  currentIdx: number;
  playing: boolean;
  setCurrentIdx: (fn: (prev: number) => number) => void;
  speed: number;
}) {
  useFrame((_, delta) => {
    if (playing && trajectory.points.length > 0) {
      setCurrentIdx((prev) => {
        const next = prev + delta * speed * 30;
        if (next >= trajectory.points.length - 1) {
          return 0; // Loop
        }
        return next;
      });
    }
  });

  const currentPoint = trajectory.points[Math.floor(currentIdx)] || trajectory.points[0];

  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={0.6} />

      <TrajectoryPath
        points={trajectory.points}
        currentIdx={Math.floor(currentIdx)}
      />

      {currentPoint && (
        <CurrentPoint
          point={currentPoint}
          index={Math.floor(currentIdx)}
          total={trajectory.points.length}
        />
      )}

      <DroughtMarkers points={trajectory.points} />
      <AxisLabels variance={trajectory.pca_explained_variance} />

      {/* Grid helper */}
      <gridHelper args={[4, 20, '#333', '#222']} rotation={[0, 0, 0]} />

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={1}
        maxDistance={10}
      />
    </>
  );
}

// ---------------------------------------------------------------------------
// 2D Egocentric View (Canvas)
// ---------------------------------------------------------------------------

function EgocentricView({
  step,
  cycle,
}: {
  step: FocalStep | null;
  cycle: CycleData | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !step) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = canvas.width;
    const cellSize = size / 5;

    // Clear
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, size, size);

    // Draw 5x5 observation grid
    for (let r = 0; r < 5; r++) {
      for (let c = 0; c < 5; c++) {
        const x = c * cellSize;
        const y = r * cellSize;

        // Resources (green channel)
        const resource = step.local_resources[r][c];
        const signal = step.local_signals[r][c];
        const agents = step.local_agents[r][c];

        // Background: resource density
        const rg = Math.min(1, resource * 2);
        ctx.fillStyle = `rgb(${Math.floor(30 + rg * 40)}, ${Math.floor(40 + rg * 180)}, ${Math.floor(30 + rg * 60)})`;
        ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);

        // Signal overlay (blue glow)
        if (signal > 0.01) {
          ctx.fillStyle = `rgba(60, 120, 255, ${Math.min(0.6, signal * 3)})`;
          ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
        }

        // Agent markers
        if (agents > 0 && !(r === 2 && c === 2)) {
          ctx.fillStyle = '#ff8844';
          ctx.beginPath();
          ctx.arc(
            x + cellSize / 2,
            y + cellSize / 2,
            cellSize * 0.2,
            0,
            Math.PI * 2
          );
          ctx.fill();
        }
      }
    }

    // Focal agent (center)
    const cx = 2.5 * cellSize;
    const cy = 2.5 * cellSize;
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(cx, cy, cellSize * 0.25, 0, Math.PI * 2);
    ctx.fill();

    // Movement direction arrow
    const dirs: Record<number, [number, number]> = {
      0: [0, 0],
      1: [0, -1],
      2: [0, 1],
      3: [-1, 0],
      4: [1, 0],
    };
    const [dx, dy] = dirs[step.move_direction] || [0, 0];
    if (dx !== 0 || dy !== 0) {
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + dx * cellSize * 0.4, cy + dy * cellSize * 0.4);
      ctx.stroke();

      // Arrowhead
      const angle = Math.atan2(dy, dx);
      ctx.beginPath();
      ctx.moveTo(
        cx + dx * cellSize * 0.4,
        cy + dy * cellSize * 0.4
      );
      ctx.lineTo(
        cx + dx * cellSize * 0.4 - Math.cos(angle - 0.5) * 6,
        cy + dy * cellSize * 0.4 - Math.sin(angle - 0.5) * 6
      );
      ctx.lineTo(
        cx + dx * cellSize * 0.4 - Math.cos(angle + 0.5) * 6,
        cy + dy * cellSize * 0.4 - Math.sin(angle + 0.5) * 6
      );
      ctx.closePath();
      ctx.fillStyle = '#ffff00';
      ctx.fill();
    }

    // Energy bar
    const barWidth = size - 20;
    const barHeight = 8;
    const barX = 10;
    const barY = size - 15;
    ctx.fillStyle = '#333';
    ctx.fillRect(barX, barY, barWidth, barHeight);
    ctx.fillStyle = step.energy > 0.5
      ? '#44cc44'
      : step.energy > 0.2
      ? '#ccaa44'
      : '#cc4444';
    ctx.fillRect(barX, barY, barWidth * Math.min(1, step.energy / 2), barHeight);

  }, [step]);

  return (
    <canvas
      ref={canvasRef}
      width={200}
      height={200}
      style={{
        width: 200,
        height: 200,
        borderRadius: 8,
        border: '1px solid var(--border, #333)',
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Demo data generator
// ---------------------------------------------------------------------------

function generateDemoTrajectory(): TrajectoryData {
  const points: TrajectoryPoint[] = [];
  const nCycles = 15;
  const stepsPerCycle = 50;

  let x = 0, y = 0, z = 0;
  let vx = 0, vy = 0, vz = 0;

  for (let cycle = 0; cycle < nCycles; cycle++) {
    const isDrought = [5, 10, 15, 20, 25].includes(cycle * 2);

    for (let step = 0; step < stepsPerCycle; step++) {
      const t = (cycle * stepsPerCycle + step) / (nCycles * stepsPerCycle);

      // Brownian motion with drift
      vx += (Math.random() - 0.5) * 0.08;
      vy += (Math.random() - 0.5) * 0.08;
      vz += (Math.random() - 0.5) * 0.08;

      // Attractor toward center
      vx -= x * 0.01;
      vy -= y * 0.01;
      vz -= z * 0.01;

      // Damping
      vx *= 0.95;
      vy *= 0.95;
      vz *= 0.95;

      // Drought perturbation
      if (isDrought) {
        vx += Math.sin(step * 0.1) * 0.05;
        vy -= 0.02;
      }

      x += vx;
      y += vy;
      z += vz;

      const energy = isDrought
        ? 0.3 + Math.random() * 0.3
        : 0.6 + Math.random() * 0.4;

      points.push({
        x, y, z,
        cycle: cycle * 2,
        step,
        energy,
        condition: isDrought ? 'drought' : 'normal',
      });
    }
  }

  return {
    pca_explained_variance: [0.35, 0.22, 0.15],
    points,
  };
}

function generateDemoFocalStep(
  idx: number,
  condition: string
): FocalStep {
  const isDrought = condition === 'drought';
  const resource = isDrought ? Math.random() * 0.1 : 0.3 + Math.random() * 0.5;

  const localResources = Array(5).fill(null).map(() =>
    Array(5).fill(null).map(() =>
      isDrought ? Math.random() * 0.05 : resource + (Math.random() - 0.5) * 0.2
    )
  );

  return {
    step: idx,
    position: [64 + Math.floor(Math.random() * 10), 64 + Math.floor(Math.random() * 10)],
    energy: isDrought ? 0.3 + Math.random() * 0.4 : 0.8 + Math.random() * 0.4,
    energy_delta: isDrought ? -0.01 + Math.random() * 0.005 : -0.005 + Math.random() * 0.015,
    prediction: Math.random() * 0.02 - 0.01,
    prediction_error: Math.random() * 0.01,
    move_direction: Math.floor(Math.random() * 5),
    consume: Math.random() > 0.3 ? 1 : 0,
    emit: Math.random() > 0.8 ? 1 : 0,
    hidden_state: Array(16).fill(0).map(() => Math.random() * 2 - 1),
    local_resources: localResources,
    local_signals: Array(5).fill(null).map(() =>
      Array(5).fill(null).map(() => Math.random() * 0.1)
    ),
    local_agents: Array(5).fill(null).map(() =>
      Array(5).fill(null).map(() => Math.random() > 0.85 ? 1 : 0)
    ),
  };
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function AffectTrajectory3D({
  trajectoryUrl,
  cycleUrls,
  demoMode = true,
}: AffectTrajectory3DProps) {
  const [trajectory, setTrajectory] = useState<TrajectoryData | null>(null);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [mounted, setMounted] = useState(false);

  // Load data
  useEffect(() => {
    setMounted(true);

    if (demoMode || !trajectoryUrl) {
      setTrajectory(generateDemoTrajectory());
      return;
    }

    fetch(trajectoryUrl)
      .then((r) => r.json())
      .then(setTrajectory)
      .catch(console.error);
  }, [trajectoryUrl, demoMode]);

  // Current point data
  const currentPoint = trajectory?.points[Math.floor(currentIdx)];
  const currentFocalStep = useMemo(() => {
    if (!currentPoint) return null;
    if (demoMode) {
      return generateDemoFocalStep(
        Math.floor(currentIdx),
        currentPoint.condition
      );
    }
    return null;
  }, [currentPoint, currentIdx, demoMode]);

  const handleScrub = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setCurrentIdx(Number(e.target.value));
    },
    []
  );

  if (!mounted || !trajectory) {
    return (
      <div style={{
        height: 500,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'var(--text-secondary, #888)',
        fontFamily: 'var(--font-mono, monospace)',
        fontSize: '0.85rem',
      }}>
        Loading trajectory data...
      </div>
    );
  }

  const totalPoints = trajectory.points.length;
  const idx = Math.floor(currentIdx);

  return (
    <div
      className="affect-trajectory-3d"
      style={{
        width: '100%',
        maxWidth: 900,
        margin: '2rem auto',
        fontFamily: 'var(--font-sans, system-ui)',
      }}
    >
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'baseline',
        marginBottom: '0.5rem',
        padding: '0 0.25rem',
      }}>
        <span style={{
          fontSize: '0.75rem',
          color: 'var(--text-secondary, #888)',
          fontFamily: 'var(--font-mono, monospace)',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        }}>
          Egocentric Affect Trajectory
        </span>
        <span style={{
          fontSize: '0.7rem',
          color: 'var(--text-secondary, #666)',
          fontFamily: 'var(--font-mono, monospace)',
        }}>
          {demoMode ? 'demo data' : `seed 23 | ${totalPoints} points`}
        </span>
      </div>

      {/* Main visualization area */}
      <div style={{
        display: 'flex',
        gap: '1rem',
        alignItems: 'flex-start',
        flexWrap: 'wrap',
      }}>
        {/* 3D view */}
        <div style={{
          flex: '1 1 500px',
          height: 420,
          borderRadius: 8,
          overflow: 'hidden',
          border: '1px solid var(--border, #333)',
          background: '#0a0a0a',
        }}>
          <Canvas
            camera={{ position: [3, 2.5, 3], fov: 50 }}
            style={{ background: '#0a0a0a' }}
          >
            <Scene
              trajectory={trajectory}
              currentIdx={currentIdx}
              playing={playing}
              setCurrentIdx={setCurrentIdx}
              speed={speed}
            />
          </Canvas>
        </div>

        {/* Side panel */}
        <div style={{
          flex: '0 0 220px',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem',
        }}>
          {/* Egocentric view */}
          <div>
            <div style={{
              fontSize: '0.65rem',
              color: 'var(--text-secondary, #888)',
              marginBottom: 4,
              fontFamily: 'var(--font-mono, monospace)',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}>
              5x5 observation window
            </div>
            <EgocentricView step={currentFocalStep} cycle={null} />
          </div>

          {/* Metadata */}
          {currentPoint && (
            <div style={{
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono, monospace)',
              color: 'var(--text, #ccc)',
              lineHeight: 1.6,
              padding: '0.5rem',
              background: 'var(--bg-sidebar, #111)',
              borderRadius: 6,
              border: '1px solid var(--border, #333)',
            }}>
              <div>
                <span style={{ color: 'var(--text-secondary, #888)' }}>cycle </span>
                {currentPoint.cycle}
              </div>
              <div>
                <span style={{ color: 'var(--text-secondary, #888)' }}>condition </span>
                <span style={{
                  color: currentPoint.condition === 'drought' ? '#cc4444' : '#44aa44',
                  fontWeight: 600,
                }}>
                  {currentPoint.condition}
                </span>
              </div>
              <div>
                <span style={{ color: 'var(--text-secondary, #888)' }}>energy </span>
                {currentPoint.energy.toFixed(3)}
              </div>
              {currentFocalStep && (
                <>
                  <div>
                    <span style={{ color: 'var(--text-secondary, #888)' }}>pred error </span>
                    {currentFocalStep.prediction_error.toFixed(5)}
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-secondary, #888)' }}>position </span>
                    [{currentFocalStep.position[0]}, {currentFocalStep.position[1]}]
                  </div>
                </>
              )}
            </div>
          )}

          {/* Legend */}
          <div style={{
            fontSize: '0.65rem',
            fontFamily: 'var(--font-mono, monospace)',
            color: 'var(--text-secondary, #888)',
            lineHeight: 1.8,
          }}>
            <div>
              <span style={{ color: '#44aa44' }}>&#9679;</span> normal
            </div>
            <div>
              <span style={{ color: '#cc4444' }}>&#9679;</span> drought
            </div>
            <div>
              <span style={{ color: '#ffffff' }}>&#9670;</span> current state
            </div>
          </div>
        </div>
      </div>

      {/* Playback controls */}
      <div style={{
        marginTop: '0.75rem',
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        padding: '0.5rem',
        background: 'var(--bg-sidebar, #111)',
        borderRadius: 6,
        border: '1px solid var(--border, #333)',
      }}>
        <button
          onClick={() => setPlaying(!playing)}
          style={{
            background: 'none',
            border: '1px solid var(--border, #444)',
            color: 'var(--text, #ccc)',
            padding: '4px 12px',
            borderRadius: 4,
            cursor: 'pointer',
            fontFamily: 'var(--font-mono, monospace)',
            fontSize: '0.75rem',
          }}
        >
          {playing ? 'pause' : 'play'}
        </button>

        <input
          type="range"
          min={0}
          max={totalPoints - 1}
          value={Math.floor(currentIdx)}
          onChange={handleScrub}
          style={{
            flex: 1,
            accentColor: 'var(--accent, #888)',
          }}
        />

        <select
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          style={{
            background: 'var(--bg, #111)',
            border: '1px solid var(--border, #444)',
            color: 'var(--text, #ccc)',
            padding: '2px 6px',
            borderRadius: 4,
            fontFamily: 'var(--font-mono, monospace)',
            fontSize: '0.7rem',
          }}
        >
          <option value={0.25}>0.25x</option>
          <option value={0.5}>0.5x</option>
          <option value={1}>1x</option>
          <option value={2}>2x</option>
          <option value={4}>4x</option>
        </select>

        <span style={{
          fontSize: '0.65rem',
          fontFamily: 'var(--font-mono, monospace)',
          color: 'var(--text-secondary, #666)',
          minWidth: 80,
          textAlign: 'right',
        }}>
          {idx + 1} / {totalPoints}
        </span>
      </div>
    </div>
  );
}
