'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import UserButton from './UserButton';
import SearchOverlay from './SearchOverlay';
import { useAnnotations, type Annotation } from '@/lib/hooks/useAnnotations';

type ThemeMode = 'light' | 'dark' | 'system';
type FontSize = 'small' | 'medium' | 'large' | 'xlarge';

const FONT_SIZES: Record<FontSize, number> = {
  small: 15,
  medium: 17,
  large: 19,
  xlarge: 21,
};

const FONT_ORDER: FontSize[] = ['small', 'medium', 'large', 'xlarge'];

function getTheme(): ThemeMode {
  if (typeof window === 'undefined') return 'system';
  return (localStorage.getItem('soe-theme') as ThemeMode) || 'system';
}

function getFontSize(): FontSize {
  if (typeof window === 'undefined') return 'medium';
  return (localStorage.getItem('soe-font-size') as FontSize) || 'medium';
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
  const [showBookmarks, setShowBookmarks] = useState(false);
  const [justBookmarked, setJustBookmarked] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const slug = pathname.replace(/^\//, '');
  const READING_SLUGS = ['introduction', 'part-1', 'part-2', 'part-3', 'part-4', 'part-5', 'epilogue'];
  const isReadingPage = READING_SLUGS.includes(slug);

  const { items: allAnnotations, add: addAnnotation, remove: removeAnnotation } = useAnnotations();

  // Bookmarks are annotations with empty exact
  const bookmarks = allAnnotations.filter((a) => !a.exact);

  useEffect(() => {
    const t = getTheme();
    const f = getFontSize();
    setTheme(t);
    setFontSize(f);
    applyTheme(t);
    applyFontSize(f);

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

  const selectTheme = useCallback((mode: ThemeMode) => {
    setTheme(mode);
    localStorage.setItem('soe-theme', mode);
    applyTheme(mode);
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

  const addBookmark = useCallback(async () => {
    if (!isReadingPage) return;
    const heading = findNearestHeading();
    await addAnnotation({
      slug,
      nearestHeadingId: heading.id,
      nearestHeadingText: heading.text,
      prefix: '',
      exact: '',
      suffix: '',
      note: '',
    });
    setJustBookmarked(true);
    setTimeout(() => setJustBookmarked(false), 1500);
  }, [isReadingPage, slug, addAnnotation]);

  const navigateToBookmark = useCallback((bm: Annotation) => {
    setShowBookmarks(false);
    if (bm.slug === slug) {
      // Same page — just scroll
      if (bm.nearestHeadingId) {
        const el = document.getElementById(bm.nearestHeadingId);
        if (el) { el.scrollIntoView({ behavior: 'smooth' }); return; }
      }
    } else {
      // Navigate then scroll (via hash)
      if (bm.nearestHeadingId) {
        router.push(`/${bm.slug}#${bm.nearestHeadingId}`);
      } else {
        router.push(`/${bm.slug}`);
      }
    }
  }, [slug, router]);

  // Group bookmarks by slug
  const grouped: Record<string, Annotation[]> = {};
  for (const bm of bookmarks) {
    (grouped[bm.slug] ||= []).push(bm);
  }

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

      {/* User */}
      <UserButton />

      {/* Bookmark */}
      <div className="reader-toolbar-dropdown" ref={dropdownRef}>
        <button
          className={`reader-toolbar-btn ${justBookmarked ? 'bookmark-flash' : ''}`}
          onClick={isReadingPage ? addBookmark : () => setShowBookmarks(!showBookmarks)}
          title={isReadingPage ? 'Add bookmark' : 'View bookmarks'}
          aria-label={isReadingPage ? 'Add bookmark' : 'View bookmarks'}
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
                        onClick={() => removeAnnotation(bm.id)}
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

      <SearchOverlay open={searchOpen} onClose={() => setSearchOpen(false)} />
    </div>
  );
}
