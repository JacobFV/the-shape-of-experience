'use client';

import { useState, useCallback, useMemo } from 'react';
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

interface MetadataEntry {
  slug: string;
  title: string;
  sections: { level: number; id: string; text: string }[];
}

let searchIndexCache: SearchEntry[] | null = null;
let metadataCache: MetadataEntry[] | null = null;

async function loadSearchIndex(): Promise<SearchEntry[]> {
  if (searchIndexCache) return searchIndexCache;
  const res = await fetch('/search-index.json');
  searchIndexCache = await res.json();
  return searchIndexCache!;
}

async function loadMetadata(): Promise<MetadataEntry[]> {
  if (metadataCache) return metadataCache;
  const res = await fetch('/metadata.json');
  metadataCache = await res.json();
  return metadataCache!;
}

export function useCopyContent() {
  const pathname = usePathname();
  const [toast, setToast] = useState<string | null>(null);
  const [hasSubsections, setHasSubsections] = useState(false);

  const slug = pathname.replace(/^\//, '').split('/')[0];
  const chapter = chapters.find(ch => ch.slug === slug);

  const isReadingPage = !!chapter;

  const pageTitle = typeof document !== 'undefined'
    ? document.querySelector('.chapter-title')?.textContent || chapter?.shortTitle || ''
    : chapter?.shortTitle || '';

  const partTitle = chapter?.shortTitle || '';

  // Check if current chapter has level-1 sections (i.e., is split into pages)
  useMemo(() => {
    if (!slug || !isReadingPage) return;
    loadMetadata().then(meta => {
      const ch = meta.find(c => c.slug === slug);
      const l1Count = ch?.sections.filter(s => s.level === 1).length ?? 0;
      setHasSubsections(l1Count > 0);
    });
  }, [slug, isReadingPage]);

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

  const copyBook = useCallback(async () => {
    try {
      const index = await loadSearchIndex();
      const allText = index.map(entry => {
        const parts = entry.sections
          .map(s => {
            const heading = s.heading ? `## ${s.heading}\n\n` : '';
            return heading + s.text;
          });
        return `# ${entry.title}\n\n${parts.join('\n\n')}`;
      });
      const text = `The Shape of Experience\n\n${allText.join('\n\n---\n\n')}`;
      await navigator.clipboard.writeText(text);
      showToast('Book copied');
    } catch {
      showToast('Failed to copy');
    }
  }, [showToast]);

  return { copyPage, copyPart, copyBook, pageTitle, partTitle, isReadingPage, hasSubsections, toast };
}
