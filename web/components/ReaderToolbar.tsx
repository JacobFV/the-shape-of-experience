'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { usePathname } from 'next/navigation';
import UserButton from './UserButton';
import SearchOverlay from './SearchOverlay';

type ThemeMode = 'light' | 'dark' | 'system';
type FontSize = 'small' | 'medium' | 'large' | 'xlarge';
type FontFamily = 'georgia' | 'palatino' | 'system-sans' | 'inter' | 'mono';
type AccentPreset = 'blue' | 'warm' | 'forest' | 'plum';

const FONT_SIZES: Record<FontSize, number> = {
  small: 15,
  medium: 17,
  large: 19,
  xlarge: 21,
};

const FONT_ORDER: FontSize[] = ['small', 'medium', 'large', 'xlarge'];

const FONT_FAMILIES: Record<FontFamily, { label: string; stack: string }> = {
  'georgia': { label: 'Georgia', stack: "Georgia, 'Times New Roman', serif" },
  'palatino': { label: 'Palatino', stack: "'Palatino Linotype', 'Book Antiqua', Palatino, serif" },
  'system-sans': { label: 'System Sans', stack: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif" },
  'inter': { label: 'Inter', stack: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif" },
  'mono': { label: 'Monospace', stack: "'SF Mono', 'Fira Code', 'JetBrains Mono', monospace" },
};

const ACCENT_PRESETS: Record<AccentPreset, { label: string; accent: string; accentLight: string; darkAccent: string; darkAccentLight: string }> = {
  'blue': { label: 'Blue', accent: '#2c5aa0', accentLight: '#e8f0fe', darkAccent: '#6b9edd', darkAccentLight: '#2a3a52' },
  'warm': { label: 'Warm', accent: '#b07520', accentLight: '#fef3e2', darkAccent: '#d4932a', darkAccentLight: '#3a2a18' },
  'forest': { label: 'Forest', accent: '#3a7a3a', accentLight: '#e8f5e8', darkAccent: '#6ab06a', darkAccentLight: '#1e2a1e' },
  'plum': { label: 'Plum', accent: '#7744aa', accentLight: '#f3e8fe', darkAccent: '#b088dd', darkAccentLight: '#261e30' },
};

function getTheme(): ThemeMode {
  if (typeof window === 'undefined') return 'system';
  return (localStorage.getItem('soe-theme') as ThemeMode) || 'system';
}

function getFontSize(): FontSize {
  if (typeof window === 'undefined') return 'medium';
  return (localStorage.getItem('soe-font-size') as FontSize) || 'medium';
}

function getFontFamily(): FontFamily {
  if (typeof window === 'undefined') return 'georgia';
  return (localStorage.getItem('soe-font-family') as FontFamily) || 'georgia';
}

function getAccent(): AccentPreset {
  if (typeof window === 'undefined') return 'blue';
  return (localStorage.getItem('soe-accent') as AccentPreset) || 'blue';
}

function applyTheme(mode: ThemeMode) {
  const root = document.documentElement;
  if (mode === 'system') {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
  } else {
    root.setAttribute('data-theme', mode);
  }
}

function applyFontSize(size: FontSize) {
  document.documentElement.style.fontSize = `${FONT_SIZES[size]}px`;
}

function applyFontFamily(family: FontFamily) {
  const stack = FONT_FAMILIES[family]?.stack || FONT_FAMILIES.georgia.stack;
  document.documentElement.style.setProperty('--font-body', stack);
}

function applyAccent(preset: AccentPreset) {
  const p = ACCENT_PRESETS[preset] || ACCENT_PRESETS.blue;
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  document.documentElement.style.setProperty('--accent', isDark ? p.darkAccent : p.accent);
  document.documentElement.style.setProperty('--accent-light', isDark ? p.darkAccentLight : p.accentLight);
}

export default function ReaderToolbar() {
  const pathname = usePathname();
  const [theme, setTheme] = useState<ThemeMode>('system');
  const [fontSize, setFontSize] = useState<FontSize>('medium');
  const [fontFamily, setFontFamily] = useState<FontFamily>('georgia');
  const [accent, setAccent] = useState<AccentPreset>('blue');
  const [showFontPicker, setShowFontPicker] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const fontPickerRef = useRef<HTMLDivElement>(null);

  const pathBase = pathname.replace(/^\//, '').split('/')[0];
  const READING_SLUGS = ['introduction', 'part-1', 'part-2', 'part-3', 'part-4', 'part-5', 'epilogue'];
  const isReadingPage = READING_SLUGS.includes(pathBase);
  const slug = pathBase;

  useEffect(() => {
    const t = getTheme();
    const f = getFontSize();
    const ff = getFontFamily();
    const a = getAccent();
    setTheme(t);
    setFontSize(f);
    setFontFamily(ff);
    setAccent(a);
    applyTheme(t);
    applyFontSize(f);
    applyFontFamily(ff);
    applyAccent(a);

    // Listen for system theme changes
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onMqChange = () => {
      if (getTheme() === 'system') applyTheme('system');
      applyAccent(getAccent());
    };
    mq.addEventListener('change', onMqChange);
    return () => mq.removeEventListener('change', onMqChange);
  }, []);

  // Close dropdowns on outside click
  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (fontPickerRef.current && !fontPickerRef.current.contains(e.target as Node)) {
        setShowFontPicker(false);
      }
    }
    if (showFontPicker) document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [showFontPicker]);

  const selectTheme = useCallback((mode: ThemeMode) => {
    setTheme(mode);
    localStorage.setItem('soe-theme', mode);
    applyTheme(mode);
    // Re-apply accent for the new theme
    setTimeout(() => applyAccent(getAccent()), 0);
  }, []);

  const cycleTheme = useCallback(() => {
    const order: ThemeMode[] = ['light', 'dark', 'system'];
    const next = order[(order.indexOf(theme) + 1) % order.length];
    selectTheme(next);
  }, [theme, selectTheme]);

  const changeFontSize = useCallback((dir: -1 | 1) => {
    const idx = FONT_ORDER.indexOf(fontSize);
    const next = FONT_ORDER[Math.max(0, Math.min(FONT_ORDER.length - 1, idx + dir))];
    setFontSize(next);
    localStorage.setItem('soe-font-size', next);
    applyFontSize(next);
  }, [fontSize]);

  const changeFontFamily = useCallback((family: FontFamily) => {
    setFontFamily(family);
    localStorage.setItem('soe-font-family', family);
    applyFontFamily(family);
    setShowFontPicker(false);
  }, []);

  const changeAccent = useCallback((preset: AccentPreset) => {
    setAccent(preset);
    localStorage.setItem('soe-accent', preset);
    applyAccent(preset);
  }, []);

  return (
    <div className="reader-toolbar">
      {/* Search */}
      <button
        className="reader-toolbar-btn"
        onClick={() => setSearchOpen(true)}
        title="Search"
        aria-label="Search"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <circle cx="11" cy="11" r="8" />
          <path d="M21 21l-4.35-4.35" />
        </svg>
      </button>

      {/* Theme toggle - single cycling button */}
      <button
        className="reader-toolbar-btn theme-cycle-btn"
        onClick={cycleTheme}
        title={`Theme: ${theme}`}
        aria-label={`Theme: ${theme}. Click to cycle.`}
      >
        {theme === 'light' && (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <circle cx="12" cy="12" r="5" />
            <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
          </svg>
        )}
        {theme === 'dark' && (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
          </svg>
        )}
        {theme === 'system' && (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <rect x="2" y="3" width="20" height="14" rx="2" />
            <path d="M8 21h8M12 17v4" />
          </svg>
        )}
      </button>

      {/* Font family picker */}
      <div className="font-picker" ref={fontPickerRef}>
        <button
          className="reader-toolbar-btn"
          onClick={() => setShowFontPicker(!showFontPicker)}
          title={`Font: ${FONT_FAMILIES[fontFamily].label}`}
          aria-label="Change font family"
          style={{ fontFamily: FONT_FAMILIES[fontFamily].stack, fontSize: '0.8rem' }}
        >
          Aa
        </button>
        {showFontPicker && (
          <div className="font-picker-menu">
            {(Object.entries(FONT_FAMILIES) as [FontFamily, { label: string; stack: string }][]).map(([key, { label, stack }]) => (
              <button
                key={key}
                className={`font-picker-option ${fontFamily === key ? 'active' : ''}`}
                onClick={() => changeFontFamily(key)}
                style={{ fontFamily: stack }}
              >
                {label}
              </button>
            ))}
            <div style={{ borderTop: '1px solid var(--border)', margin: '4px 0', padding: '4px 6px 0' }}>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-sans)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '4px' }}>Accent</div>
              <div className="accent-picker">
                {(Object.entries(ACCENT_PRESETS) as [AccentPreset, typeof ACCENT_PRESETS[AccentPreset]][]).map(([key, p]) => (
                  <button
                    key={key}
                    className={`accent-dot ${accent === key ? 'active' : ''}`}
                    onClick={() => changeAccent(key)}
                    title={p.label}
                    aria-label={`Accent: ${p.label}`}
                  >
                    <div className="accent-dot-inner" style={{ background: p.accent }} />
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Font size - small A / big A */}
      <div className="fontsize-toggle">
        <button
          className="fontsize-btn fontsize-btn-small"
          onClick={() => changeFontSize(-1)}
          disabled={fontSize === 'small'}
          title="Decrease font size"
          aria-label="Decrease font size"
        >A</button>
        <button
          className="fontsize-btn fontsize-btn-large"
          onClick={() => changeFontSize(1)}
          disabled={fontSize === 'xlarge'}
          title="Increase font size"
          aria-label="Increase font size"
        >A</button>
      </div>

      {/* Chat */}
      <button
        className="reader-toolbar-btn"
        onClick={() => {
          window.dispatchEvent(new CustomEvent('open-chat', {
            detail: { slug, contextType: isReadingPage ? 'page' : 'book' },
          }));
        }}
        title="Chat about this page"
        aria-label="Chat about this page"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
        </svg>
      </button>

      {/* User */}
      <UserButton />

      <SearchOverlay open={searchOpen} onClose={() => setSearchOpen(false)} />
    </div>
  );
}
