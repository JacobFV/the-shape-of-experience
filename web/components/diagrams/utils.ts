/** Positioning utilities for native SVG diagrams */

export type Point = [number, number];

/** Offset from center at angle and distance. SVG coords: 0°=right, 90°=down */
export const polar = (c: Point, r: number, deg: number): Point => {
  const rad = (deg * Math.PI) / 180;
  return [c[0] + r * Math.cos(rad), c[1] + r * Math.sin(rad)];
};

/** Midpoint between two points */
export const mid = (a: Point, b: Point): Point => [
  (a[0] + b[0]) / 2,
  (a[1] + b[1]) / 2,
];

/** Linear interpolation */
export const lerp = (a: Point, b: Point, t: number): Point => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
];

/** Offset a point */
export const off = (p: Point, dx: number, dy: number): Point => [
  p[0] + dx,
  p[1] + dy,
];

/** Distance between two points */
export const dist = (a: Point, b: Point): number =>
  Math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2);

/** Angle in degrees from a to b */
export const angleBetween = (a: Point, b: Point): number =>
  (Math.atan2(b[1] - a[1], b[0] - a[0]) * 180) / Math.PI;

/** Format point for SVG path */
export const pt = (p: Point): string =>
  `${p[0].toFixed(1)},${p[1].toFixed(1)}`;

/** Arrowhead path at tip pointing in dirDeg direction */
export const arrowPath = (
  tip: Point,
  dirDeg: number,
  size: number = 7,
): string => {
  const w1 = polar(tip, size, dirDeg + 155);
  const w2 = polar(tip, size, dirDeg - 155);
  return `M${pt(w1)} L${pt(tip)} L${pt(w2)}`;
};

/** Smooth closed curve through points (Catmull-Rom → cubic bezier) */
export const smoothClosed = (
  points: Point[],
  tension: number = 0.25,
): string => {
  const n = points.length;
  if (n < 3) return '';
  let d = `M ${pt(points[0])}`;
  for (let i = 0; i < n; i++) {
    const p0 = points[(i - 1 + n) % n];
    const p1 = points[i];
    const p2 = points[(i + 1) % n];
    const p3 = points[(i + 2) % n];
    const cp1: Point = [
      p1[0] + (p2[0] - p0[0]) * tension,
      p1[1] + (p2[1] - p0[1]) * tension,
    ];
    const cp2: Point = [
      p2[0] - (p3[0] - p1[0]) * tension,
      p2[1] - (p3[1] - p1[1]) * tension,
    ];
    d += ` C ${pt(cp1)} ${pt(cp2)} ${pt(p2)}`;
  }
  return d + ' Z';
};

/** Smooth open curve through points */
export const smoothOpen = (
  points: Point[],
  tension: number = 0.25,
): string => {
  const n = points.length;
  if (n < 2) return '';
  if (n === 2)
    return `M ${pt(points[0])} L ${pt(points[1])}`;
  let d = `M ${pt(points[0])}`;
  for (let i = 0; i < n - 1; i++) {
    const p0 = points[Math.max(i - 1, 0)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(i + 2, n - 1)];
    const cp1: Point = [
      p1[0] + (p2[0] - p0[0]) * tension,
      p1[1] + (p2[1] - p0[1]) * tension,
    ];
    const cp2: Point = [
      p2[0] - (p3[0] - p1[0]) * tension,
      p2[1] - (p3[1] - p1[1]) * tension,
    ];
    d += ` C ${pt(cp1)} ${pt(cp2)} ${pt(p2)}`;
  }
  return d;
};

/** Common SVG text style props */
export const textStyle = (
  fontSize: number = 13,
  anchor: string = 'middle',
) => ({
  fontSize,
  textAnchor: anchor as 'start' | 'middle' | 'end',
  dominantBaseline: 'central' as const,
  fontFamily: 'var(--font-body, Georgia, serif)',
});
