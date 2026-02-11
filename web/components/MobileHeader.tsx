'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { useSession, signOut } from 'next-auth/react';
import { useMobileUI } from '../lib/MobileUIContext';
import { useBookmarks, type Bookmark } from '../lib/hooks/useBookmarks';

type ThemeMode = 'light' | 'dark' | 'system';
type FontSize = 'small' | 'medium' | 'large' | 'xlarge';

const FONT_SIZES: Record<FontSize, number> = { small: 15, medium: 17, large: 19, xlarge: 21 };
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
  const headings = document.querySelectorAll<HTMLElement>(
    '.chapter-content h1[id], .chapter-content h2[id], .chapter-content h3[id]'
  );
  const scrollY = window.scrollY + 100;
  let nearest = { id: '', text: 'Start of chapter' };
  for (const h of headings) {
    if (h.offsetTop <= scrollY) nearest = { id: h.id, text: h.textContent?.trim() || '' };
  }
  return nearest;
}

export default function MobileHeader() {
  const { sidebarOpen, setSidebarOpen, audioAvailable, audioStarted, audioToggleRef } = useMobileUI();
  const [menuOpen, setMenuOpen] = useState(false);
  const [theme, setTheme] = useState<ThemeMode>('system');
  const [fontSize, setFontSize] = useState<FontSize>('medium');
  const [showBookmarks, setShowBookmarks] = useState(false);
  const [justBookmarked, setJustBookmarked] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const pathname = usePathname();
  const router = useRouter();
  const { data: session, status } = useSession();
  const { items: bookmarks, add: addBookmarkApi, remove: removeBookmark } = useBookmarks();

  const isChapter = pathname !== '/' && pathname !== '';
  const slug = pathname.replace(/^\//, '');

  useEffect(() => {
    setTheme(getTheme());
    setFontSize(getFontSize());
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onMqChange = () => { if (getTheme() === 'system') applyTheme('system'); };
    mq.addEventListener('change', onMqChange);
    return () => mq.removeEventListener('change', onMqChange);
  }, []);

  useEffect(() => {
    setMenuOpen(false);
    setShowBookmarks(false);
  }, [pathname]);

  useEffect(() => {
    if (!menuOpen) return;
    function onMouseDown(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
        setShowBookmarks(false);
      }
    }
    document.addEventListener('mousedown', onMouseDown);
    return () => document.removeEventListener('mousedown', onMouseDown);
  }, [menuOpen]);

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

  const addBookmark = useCallback(async () => {
    if (!isChapter) return;
    const heading = findNearestHeading();
    await addBookmarkApi({
      slug,
      scrollY: window.scrollY,
      nearestHeadingId: heading.id,
      nearestHeadingText: heading.text,
    });
    setJustBookmarked(true);
    setTimeout(() => setJustBookmarked(false), 1500);
  }, [isChapter, slug, addBookmarkApi]);

  const navigateToBookmark = useCallback((bm: Bookmark) => {
    setMenuOpen(false);
    setShowBookmarks(false);
    if (bm.slug === slug) {
      if (bm.nearestHeadingId) {
        const el = document.getElementById(bm.nearestHeadingId);
        if (el) { el.scrollIntoView({ behavior: 'smooth' }); return; }
      }
      window.scrollTo({ top: bm.scrollY, behavior: 'smooth' });
    } else {
      if (bm.nearestHeadingId) {
        router.push(`/${bm.slug}#${bm.nearestHeadingId}`);
      } else {
        router.push(`/${bm.slug}`);
        setTimeout(() => window.scrollTo(0, bm.scrollY), 500);
      }
    }
  }, [slug, router]);

  const themeIcon = theme === 'dark' ? '\u263E' : theme === 'light' ? '\u2600\uFE0E' : '\u25D1';
  const themeLabel = theme === 'dark' ? 'Dark' : theme === 'light' ? 'Light' : 'System';

  const grouped: Record<string, Bookmark[]> = {};
  for (const bm of bookmarks) (grouped[bm.slug] ||= []).push(bm);

  return (
    <header className="mobile-header">
      <div className="mobile-header-bar">
        <button
          className="mobile-header-btn"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          aria-label={sidebarOpen ? 'Close menu' : 'Open menu'}
        >
          {sidebarOpen ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M3 12h18M3 6h18M3 18h18" />
            </svg>
          )}
        </button>

        <div className="mobile-header-center" />

        <div className="mobile-header-right" ref={menuRef}>
          {audioAvailable && !audioStarted && (
            <button
              className="mobile-header-btn mobile-header-play"
              onClick={() => audioToggleRef.current?.()}
              aria-label="Play audio"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z" />
              </svg>
            </button>
          )}

          <button
            className="mobile-header-btn"
            onClick={() => { setMenuOpen(!menuOpen); if (menuOpen) setShowBookmarks(false); }}
            aria-label="Settings"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <circle cx="12" cy="5" r="1.5" />
              <circle cx="12" cy="12" r="1.5" />
              <circle cx="12" cy="19" r="1.5" />
            </svg>
          </button>

          {menuOpen && (
            <div className="mobile-header-menu">
              <button className="mobile-menu-item" onClick={cycleTheme}>
                <span className="mobile-menu-icon">{themeIcon}</span>
                <span>Theme: {themeLabel}</span>
              </button>

              <div className="mobile-menu-item mobile-menu-fontsize">
                <span className="mobile-menu-icon">A</span>
                <span>Text size</span>
                <div className="mobile-menu-fontsize-controls">
                  <button
                    onClick={() => changeFontSize(-1)}
                    disabled={fontSize === 'small'}
                    aria-label="Decrease font size"
                  >&minus;</button>
                  <button
                    onClick={() => changeFontSize(1)}
                    disabled={fontSize === 'xlarge'}
                    aria-label="Increase font size"
                  >+</button>
                </div>
              </div>

              {isChapter && (
                <button className="mobile-menu-item" onClick={addBookmark}>
                  <span className="mobile-menu-icon">{justBookmarked ? '\u2605' : '\u2606'}</span>
                  <span>{justBookmarked ? 'Bookmarked!' : 'Add bookmark'}</span>
                </button>
              )}

              {bookmarks.length > 0 && (
                <>
                  <button className="mobile-menu-item" onClick={() => setShowBookmarks(!showBookmarks)}>
                    <span className="mobile-menu-icon">{'\u2630'}</span>
                    <span>Bookmarks ({bookmarks.length})</span>
                    <span className="mobile-menu-chevron">{showBookmarks ? '\u25B2' : '\u25BC'}</span>
                  </button>
                  {showBookmarks && (
                    <div className="mobile-menu-bookmarks">
                      {Object.entries(grouped).map(([bmSlug, bms]) => (
                        <div key={bmSlug}>
                          <div className="mobile-menu-bm-group">{bmSlug}</div>
                          {bms.map(bm => (
                            <div key={bm.id} className="mobile-menu-bm-item">
                              <button onClick={() => navigateToBookmark(bm)}>
                                {bm.nearestHeadingText || 'Start'}
                              </button>
                              <button
                                className="mobile-menu-bm-remove"
                                onClick={() => removeBookmark(bm.id)}
                                aria-label="Remove bookmark"
                              >&times;</button>
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}

              <div className="mobile-menu-divider" />

              {status === 'authenticated' && session?.user ? (
                <>
                  <div className="mobile-menu-user-header">
                    {session.user.image ? (
                      <img src={session.user.image} alt="" width={24} height={24} style={{ borderRadius: '50%' }} />
                    ) : (
                      <span className="mobile-menu-user-initial">
                        {(session.user.name?.[0] || '?').toUpperCase()}
                      </span>
                    )}
                    <div>
                      <div className="mobile-menu-user-name">{session.user.name}</div>
                      <div className="mobile-menu-user-email">{session.user.email}</div>
                    </div>
                  </div>
                  <button className="mobile-menu-item" onClick={() => { setMenuOpen(false); router.push('/library'); }}>
                    Library
                  </button>
                  <button className="mobile-menu-item" onClick={() => { setMenuOpen(false); router.push('/settings'); }}>
                    Settings
                  </button>
                  <button className="mobile-menu-item mobile-menu-signout" onClick={() => signOut({ callbackUrl: '/' })}>
                    Sign out
                  </button>
                </>
              ) : (
                <button className="mobile-menu-item" onClick={() => { setMenuOpen(false); router.push('/login'); }}>
                  <span className="mobile-menu-icon">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" />
                      <circle cx="12" cy="7" r="4" />
                    </svg>
                  </span>
                  <span>Sign in</span>
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
