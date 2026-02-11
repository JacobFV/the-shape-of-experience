'use client';

import { useState, useEffect, useCallback } from 'react';
import { useSession } from 'next-auth/react';

export interface Bookmark {
  id: string;
  slug: string;
  scrollY: number;
  nearestHeadingId: string;
  nearestHeadingText: string;
  createdAt: number | string | Date;
}

function getLocalBookmarks(): Bookmark[] {
  try {
    return JSON.parse(localStorage.getItem('soe-bookmarks') || '[]');
  } catch {
    return [];
  }
}

function saveLocalBookmarks(items: Bookmark[]) {
  localStorage.setItem('soe-bookmarks', JSON.stringify(items));
}

export function useBookmarks() {
  const { data: session, status } = useSession();
  const isAuth = status === 'authenticated' && !!session?.user;
  const [items, setItems] = useState<Bookmark[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (status === 'loading') return;

    if (isAuth) {
      fetch('/api/bookmarks')
        .then((r) => r.json())
        .then((data) => { setItems(data); setLoading(false); })
        .catch(() => setLoading(false));
    } else {
      setItems(getLocalBookmarks());
      setLoading(false);
    }
  }, [isAuth, status]);

  const add = useCallback(
    async (data: Omit<Bookmark, 'id' | 'createdAt'>) => {
      if (isAuth) {
        const res = await fetch('/api/bookmarks', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        });
        const bookmark = await res.json();
        setItems((prev) => [...prev, bookmark]);
        return bookmark;
      } else {
        const bookmark: Bookmark = {
          ...data,
          id: `bm-${Date.now()}`,
          createdAt: Date.now(),
        };
        const updated = [...items, bookmark];
        setItems(updated);
        saveLocalBookmarks(updated);
        return bookmark;
      }
    },
    [isAuth, items]
  );

  const remove = useCallback(
    async (id: string) => {
      if (isAuth) {
        await fetch(`/api/bookmarks/${id}`, { method: 'DELETE' });
      }
      setItems((prev) => {
        const updated = prev.filter((b) => b.id !== id);
        if (!isAuth) saveLocalBookmarks(updated);
        return updated;
      });
    },
    [isAuth]
  );

  return { items, loading, add, remove, isAuth };
}
