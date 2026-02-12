'use client';

import { useState, useEffect, useCallback } from 'react';
import { useSession } from 'next-auth/react';

export interface Annotation {
  id: string;
  slug: string;
  nearestHeadingId: string;
  nearestHeadingText: string;
  prefix: string;
  exact: string;
  suffix: string;
  note: string;
  isPublished: boolean;
  createdAt: number | string | Date;
}

function getLocalHighlights(slug: string): Annotation[] {
  try {
    const raw = JSON.parse(localStorage.getItem(`soe-highlights-${slug}`) || '[]');
    return raw.map((h: Record<string, unknown>) => ({
      ...h,
      nearestHeadingText: (h.nearestHeadingText as string) || '',
      note: (h.note as string) || '',
      isPublished: false,
    }));
  } catch {
    return [];
  }
}

function saveLocalHighlights(slug: string, items: Annotation[]) {
  localStorage.setItem(`soe-highlights-${slug}`, JSON.stringify(items));
}

export function useAnnotations(slug?: string) {
  const { data: session, status } = useSession();
  const isAuth = status === 'authenticated' && !!session?.user;
  const [items, setItems] = useState<Annotation[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch annotations
  useEffect(() => {
    if (status === 'loading') return;

    if (isAuth && slug) {
      fetch(`/api/annotations?slug=${encodeURIComponent(slug)}`)
        .then((r) => r.json())
        .then((data) => { setItems(data); setLoading(false); })
        .catch(() => setLoading(false));
    } else if (isAuth) {
      fetch('/api/annotations')
        .then((r) => r.json())
        .then((data) => { setItems(data); setLoading(false); })
        .catch(() => setLoading(false));
    } else if (slug) {
      setItems(getLocalHighlights(slug));
      setLoading(false);
    } else {
      // Unauthenticated, no slug — gather all local highlights across slugs
      const all: Annotation[] = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key?.startsWith('soe-highlights-')) {
          const s = key.replace('soe-highlights-', '');
          try {
            const raw = JSON.parse(localStorage.getItem(key) || '[]');
            for (const h of raw) {
              all.push({ ...h, slug: s, nearestHeadingText: h.nearestHeadingText || '', note: h.note || '', isPublished: false });
            }
          } catch { /* ignore */ }
        }
      }
      setItems(all);
      setLoading(false);
    }
  }, [slug, isAuth, status]);

  const add = useCallback(
    async (data: Omit<Annotation, 'id' | 'isPublished' | 'createdAt'>) => {
      if (isAuth) {
        const res = await fetch('/api/annotations', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        });
        const annotation = await res.json();
        setItems((prev) => [...prev, annotation]);
        return annotation;
      } else {
        const annotation: Annotation = {
          ...data,
          id: `hl-${Date.now()}`,
          isPublished: false,
          createdAt: Date.now(),
        };
        const updated = [...items, annotation];
        setItems(updated);
        if (data.slug) saveLocalHighlights(data.slug, updated.filter((a) => a.slug === data.slug));
        return annotation;
      }
    },
    [isAuth, items]
  );

  const update = useCallback(
    async (id: string, data: { note?: string; isPublished?: boolean }) => {
      if (isAuth) {
        const res = await fetch(`/api/annotations/${id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        });
        const updated = await res.json();
        setItems((prev) => prev.map((a) => (a.id === id ? { ...a, ...updated } : a)));
      } else {
        setItems((prev) => {
          const updated = prev.map((a) => (a.id === id ? { ...a, ...data } : a));
          if (slug) saveLocalHighlights(slug, updated);
          return updated;
        });
      }
    },
    [isAuth, slug]
  );

  const remove = useCallback(
    async (id: string) => {
      if (isAuth) {
        await fetch(`/api/annotations/${id}`, { method: 'DELETE' });
      }
      setItems((prev) => {
        const updated = prev.filter((a) => a.id !== id);
        if (!isAuth && slug) saveLocalHighlights(slug, updated);
        return updated;
      });
    },
    [isAuth, slug]
  );

  return { items, loading, add, update, remove, isAuth };
}
