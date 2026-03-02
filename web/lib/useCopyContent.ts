'use client';

import { useState, useCallback } from 'react';
import { usePathname } from 'next/navigation';
import { chapters } from './chapter-data';

interface SearchSection {
  headingId: string;
  heading: string;
  text: string;
}

interface SearchEntry {
  slug: string;
  title: string;
  sections: SearchSection[];
}

let searchIndexCache: SearchEntry[] | null = null;

async function loadSearchIndex(): Promise<SearchEntry[]> {
  if (searchIndexCache) return searchIndexCache;
  const res = await fetch('/search-index.json');
  searchIndexCache = await res.json();
  return searchIndexCache!;
}

export function useCopyContent() {
  const pathname = usePathname();
  const [toast, setToast] = useState<string | null>(null);

  const slug = pathname.replace(/^\//, '').split('/')[0];
  const chapter = chapters.find(ch => ch.slug === slug);

  const isReadingPage = !!chapter;

  const pageTitle = typeof document !== 'undefined'
    ? document.querySelector('.chapter-title')?.textContent || chapter?.shortTitle || ''
    : chapter?.shortTitle || '';

  const partTitle = chapter?.shortTitle || '';

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2000);
  }, []);

  const copyPage = useCallback(async () => {
    const content = document.querySelector('.chapter-content');
    if (!content) return;
    const text = (content as HTMLElement).innerText;
    try {
      await navigator.clipboard.writeText(text);
      showToast('Page copied');
    } catch {
      showToast('Failed to copy');
    }
  }, [showToast]);

  const copyPart = useCallback(async () => {
    if (!slug) return;
    try {
      const index = await loadSearchIndex();
      const entry = index.find(e => e.slug === slug);
      if (!entry) {
        showToast('No content found');
        return;
      }
      const parts = entry.sections
        .map(s => {
          const heading = s.heading ? `# ${s.heading}\n\n` : '';
          return heading + s.text;
        });
      const text = `${entry.title}\n\n${parts.join('\n\n')}`;
      await navigator.clipboard.writeText(text);
      showToast('Part copied');
    } catch {
      showToast('Failed to copy');
    }
  }, [slug, showToast]);

  return { copyPage, copyPart, pageTitle, partTitle, isReadingPage, toast };
}
