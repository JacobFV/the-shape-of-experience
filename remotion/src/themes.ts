/**
 * Shared theme palettes for Remotion compositions.
 * Light palette mirrors the --d-* CSS variables from globals.css
 * for visual consistency between SVG diagrams and rendered videos.
 */

export type ThemeMode = 'dark' | 'light';

export interface ThemePalette {
  bg: string;
  panel: string;
  text: string;
  muted: string;
  border: string;
  green: string;
  red: string;
  blue: string;
  yellow: string;
  orange: string;
  violet: string;
  pink: string;
  cyan: string;
  teal: string;
  slate: string;
}

export const THEMES: Record<ThemeMode, ThemePalette> = {
  dark: {
    bg: '#0a0a0f',
    panel: '#111118',
    text: '#e0e0e0',
    muted: '#888',
    border: '#333',
    green: '#4ade80',
    red: '#f87171',
    blue: '#60a5fa',
    yellow: '#fbbf24',
    orange: '#fb923c',
    violet: '#a78bfa',
    pink: '#f472b6',
    cyan: '#2dd4bf',
    teal: '#38bdf8',
    slate: '#94a3b8',
  },
  light: {
    bg: '#fafaf8',
    panel: '#f0efeb',
    text: '#1a1a1a',
    muted: '#666',
    border: '#ccc',
    green: '#1e8449',
    red: '#b03a2e',
    blue: '#2471a3',
    yellow: '#b7950b',
    orange: '#ca6f1e',
    violet: '#7d3c98',
    pink: '#9b2c5e',
    cyan: '#148f77',
    teal: '#117a65',
    slate: '#64748b',
  },
};
