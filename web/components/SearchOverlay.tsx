'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';

interface SearchSection {
  headingId: string;
  heading: string;
  text: string;
}

interface SearchChapter {
  slug: string;
  title: string;
  sections: SearchSection[];
}

interface SearchResult {
  slug: string;
  chapterTitle: string;
  heading: string;
  headingId: string;
  snippet: string;
  score: number;
}

interface SearchOverlayProps {
  open: boolean;
  onClose: () => void;
}

export default function SearchOverlay({ open, onClose }: SearchOverlayProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [index, setIndex] = useState<SearchChapter[] | null>(null);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  // Load search index on first open
  useEffect(() => {
    if (!open || index) return;
    setLoading(true);
    fetch('/search-index.json')
      .then(r => r.json())
      .then((data: SearchChapter[]) => {
        setIndex(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [open, index]);

  // Focus input when opened
  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 100);
    } else {
      setQuery('');
      setResults([]);
    }
  }, [open]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const search = useCallback((q: string) => {
    setQuery(q);
    if (!index || q.trim().length < 2) {
      setResults([]);
      return;
    }

    const terms = q.toLowerCase().split(/\s+/).filter(t => t.length >= 2);
    const found: SearchResult[] = [];

    for (const ch of index) {
      for (const sec of ch.sections) {
        const textLower = sec.text.toLowerCase();
        let score = 0;
        let matchIdx = -1;

        for (const term of terms) {
          const idx = textLower.indexOf(term);
          if (idx !== -1) {
            score += 1;
            if (matchIdx === -1 || idx < matchIdx) matchIdx = idx;
          }
        }

        if (score > 0 && matchIdx !== -1) {
          // Extract snippet around match
          const start = Math.max(0, matchIdx - 40);
          const end = Math.min(sec.text.length, matchIdx + 120);
          const snippet =
            (start > 0 ? '...' : '') +
            sec.text.slice(start, end) +
            (end < sec.text.length ? '...' : '');

          found.push({
            slug: ch.slug,
            chapterTitle: ch.title,
            heading: sec.heading,
            headingId: sec.headingId,
            snippet,
            score,
          });
        }
      }
    }

    // Sort by score desc, limit to 20
    found.sort((a, b) => b.score - a.score);
    setResults(found.slice(0, 20));
  }, [index]);

  const navigate = useCallback((result: SearchResult) => {
    onClose();
    const hash = result.headingId ? `#${result.headingId}` : '';
    router.push(`/${result.slug}${hash}`);
  }, [onClose, router]);

  if (!open) return null;

  return (
    <div className="search-overlay" onClick={onClose}>
      <div className="search-container" onClick={e => e.stopPropagation()}>
        <div className="search-input-wrap">
          <svg className="search-input-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <circle cx="11" cy="11" r="8" />
            <path d="M21 21l-4.35-4.35" />
          </svg>
          <input
            ref={inputRef}
            type="text"
            className="search-input"
            placeholder="Search chapters..."
            value={query}
            onChange={e => search(e.target.value)}
          />
          <button className="search-close" onClick={onClose} aria-label="Close search">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="search-results">
          {loading && <div className="search-empty">Loading...</div>}
          {!loading && query.length >= 2 && results.length === 0 && (
            <div className="search-empty">No results</div>
          )}
          {results.map((r, i) => (
            <button key={i} className="search-result" onClick={() => navigate(r)}>
              <div className="search-result-chapter">{r.chapterTitle}</div>
              {r.heading && <div className="search-result-heading">{r.heading}</div>}
              <div className="search-result-snippet">{r.snippet}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
