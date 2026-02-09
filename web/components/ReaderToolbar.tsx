'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { usePathname, useRouter } from 'next/navigation';

type ThemeMode = 'light' | 'dark' | 'system';
type FontSize = 'small' | 'medium' | 'large' | 'xlarge';

const FONT_SIZES: Record<FontSize, number> = {
  small: 15,
  medium: 17,
  large: 19,
  xlarge: 21,
};

const FONT_ORDER: FontSize[] = ['small', 'medium', 'large', 'xlarge'];

interface Bookmark {
  id: string;
  slug: string;
  scrollY: number;
  nearestHeadingId: string;
  nearestHeadingText: string;
  createdAt: number;
}

function getTheme(): ThemeMode {
  if (typeof window === 'undefined') return 'system';
  return (localStorage.getItem('soe-theme') as ThemeMode) || 'system';
}

function getFontSize(): FontSize {
  if (typeof window === 'undefined') return 'medium';
  return (localStorage.getItem('soe-font-size') as FontSize) || 'medium';
}

function getBookmarks(): Bookmark[] {
  if (typeof window === 'undefined') return [];
  try {
    return JSON.parse(localStorage.getItem('soe-bookmarks') || '[]');
  } catch { return []; }
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

function findNearestHeading(): { id: string; text: string } {
  const headings = document.querySelectorAll<HTMLElement>('.chapter-content h1[id], .chapter-content h2[id], .chapter-content h3[id]');
  const scrollY = window.scrollY + 100;
  let nearest = { id: '', text: 'Start of chapter' };
  for (const h of headings) {
    if (h.offsetTop <= scrollY) {
      nearest = { id: h.id, text: h.textContent?.trim() || '' };
    }
  }
  return nearest;
}

export default function ReaderToolbar() {
  const pathname = usePathname();
  const router = useRouter();
  const [theme, setTheme] = useState<ThemeMode>('system');
  const [fontSize, setFontSize] = useState<FontSize>('medium');
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
  const [showBookmarks, setShowBookmarks] = useState(false);
  const [justBookmarked, setJustBookmarked] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const isChapter = pathname !== '/' && pathname !== '';
  const slug = pathname.replace(/^\//, '');

  useEffect(() => {
    const t = getTheme();
    const f = getFontSize();
    setTheme(t);
    setFontSize(f);
    applyTheme(t);
    applyFontSize(f);
    setBookmarks(getBookmarks());

    // Listen for system theme changes
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onMqChange = () => { if (getTheme() === 'system') applyTheme('system'); };
    mq.addEventListener('change', onMqChange);
    return () => mq.removeEventListener('change', onMqChange);
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowBookmarks(false);
      }
    }
    if (showBookmarks) document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [showBookmarks]);

  const cycleTheme = useCallback(() => {
    const order: ThemeMode[] = ['light', 'dark', 'system'];
    const next = order[(order.indexOf(theme) + 1) % order.length];
    setTheme(next);
    localStorage.setItem('soe-theme', next);
    applyTheme(next);
  }, [theme]);

  const changeFontSize = useCallback((dir: -1 | 1) => {
    const idx = FONT_ORDER.indexOf(fontSize);
    const next = FONT_ORDER[Math.max(0, Math.min(FONT_ORDER.length - 1, idx + dir))];
    setFontSize(next);
    localStorage.setItem('soe-font-size', next);
    applyFontSize(next);
  }, [fontSize]);

  const addBookmark = useCallback(() => {
    if (!isChapter) return;
    const heading = findNearestHeading();
    const bm: Bookmark = {
      id: `bm-${Date.now()}`,
      slug,
      scrollY: window.scrollY,
      nearestHeadingId: heading.id,
      nearestHeadingText: heading.text,
      createdAt: Date.now(),
    };
    const updated = [...getBookmarks(), bm];
    localStorage.setItem('soe-bookmarks', JSON.stringify(updated));
    setBookmarks(updated);
    setJustBookmarked(true);
    setTimeout(() => setJustBookmarked(false), 1500);
  }, [isChapter, slug]);

  const removeBookmark = useCallback((id: string) => {
    const updated = getBookmarks().filter(b => b.id !== id);
    localStorage.setItem('soe-bookmarks', JSON.stringify(updated));
    setBookmarks(updated);
  }, []);

  const navigateToBookmark = useCallback((bm: Bookmark) => {
    setShowBookmarks(false);
    if (bm.slug === slug) {
      // Same page — just scroll
      if (bm.nearestHeadingId) {
        const el = document.getElementById(bm.nearestHeadingId);
        if (el) { el.scrollIntoView({ behavior: 'smooth' }); return; }
      }
      window.scrollTo({ top: bm.scrollY, behavior: 'smooth' });
    } else {
      // Navigate then scroll (via hash or scrollY)
      if (bm.nearestHeadingId) {
        router.push(`/${bm.slug}#${bm.nearestHeadingId}`);
      } else {
        router.push(`/${bm.slug}`);
        // Scroll after navigation
        setTimeout(() => window.scrollTo(0, bm.scrollY), 500);
      }
    }
  }, [slug, router]);

  const themeIcon = theme === 'dark' ? '\u263E' : theme === 'light' ? '\u2600' : '\u25D1';
  const themeLabel = theme === 'dark' ? 'Dark' : theme === 'light' ? 'Light' : 'System';

  // Group bookmarks by slug
  const grouped: Record<string, Bookmark[]> = {};
  for (const bm of bookmarks) {
    (grouped[bm.slug] ||= []).push(bm);
  }

  return (
    <div className="reader-toolbar">
      {/* Theme toggle */}
      <button
        className="reader-toolbar-btn"
        onClick={cycleTheme}
        title={`Theme: ${themeLabel}`}
        aria-label={`Theme: ${themeLabel}`}
      >
        {themeIcon}
      </button>

      {/* Font size */}
      <button
        className="reader-toolbar-btn"
        onClick={() => changeFontSize(-1)}
        disabled={fontSize === 'small'}
        title="Decrease font size"
        aria-label="Decrease font size"
      >
        A<span className="font-size-minus">&minus;</span>
      </button>
      <button
        className="reader-toolbar-btn"
        onClick={() => changeFontSize(1)}
        disabled={fontSize === 'xlarge'}
        title="Increase font size"
        aria-label="Increase font size"
      >
        A<span className="font-size-plus">+</span>
      </button>

      {/* Bookmark */}
      <div className="reader-toolbar-dropdown" ref={dropdownRef}>
        <button
          className={`reader-toolbar-btn ${justBookmarked ? 'bookmark-flash' : ''}`}
          onClick={isChapter ? addBookmark : () => setShowBookmarks(!showBookmarks)}
          title={isChapter ? 'Add bookmark' : 'View bookmarks'}
          aria-label={isChapter ? 'Add bookmark' : 'View bookmarks'}
        >
          {justBookmarked ? '\u2605' : '\u2606'}
        </button>
        {bookmarks.length > 0 && (
          <button
            className="reader-toolbar-btn reader-toolbar-btn-small"
            onClick={() => setShowBookmarks(!showBookmarks)}
            title="View bookmarks"
            aria-label="View bookmarks"
          >
            <span className="bookmark-count">{bookmarks.length}</span>
          </button>
        )}
        {showBookmarks && (
          <div className="reader-toolbar-menu">
            {bookmarks.length === 0 ? (
              <div className="reader-toolbar-menu-empty">No bookmarks yet</div>
            ) : (
              Object.entries(grouped).map(([bmSlug, bms]) => (
                <div key={bmSlug}>
                  <div className="reader-toolbar-menu-group">{bmSlug}</div>
                  {bms.map(bm => (
                    <div key={bm.id} className="reader-toolbar-menu-item">
                      <button
                        className="reader-toolbar-menu-link"
                        onClick={() => navigateToBookmark(bm)}
                      >
                        {bm.nearestHeadingText || 'Start'}
                      </button>
                      <button
                        className="reader-toolbar-menu-remove"
                        onClick={() => removeBookmark(bm.id)}
                        title="Remove bookmark"
                        aria-label="Remove bookmark"
                      >
                        &times;
                      </button>
                    </div>
                  ))}
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}
