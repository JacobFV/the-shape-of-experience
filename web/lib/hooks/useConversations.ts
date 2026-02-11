'use client';

import { useState, useEffect, useCallback } from 'react';
import { useSession } from 'next-auth/react';

export interface Conversation {
  id: string;
  slug: string | null;
  title: string;
  contextType: string;
  contextExact: string;
  contextHeadingId: string;
  isPublished: boolean;
  createdAt: string;
  updatedAt: string;
}

export function useConversations(slug?: string) {
  const { data: session, status } = useSession();
  const isAuth = status === 'authenticated' && !!session?.user;
  const [items, setItems] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    if (!isAuth) { setItems([]); setLoading(false); return; }
    try {
      const url = slug
        ? `/api/conversations?slug=${encodeURIComponent(slug)}`
        : '/api/conversations';
      const res = await fetch(url);
      if (res.ok) setItems(await res.json());
    } catch { /* ignore */ }
    setLoading(false);
  }, [isAuth, slug]);

  useEffect(() => {
    if (status === 'loading') return;
    refresh();
  }, [status, refresh]);

  const updateConversation = useCallback(
    async (id: string, data: { title?: string; isPublished?: boolean }) => {
      const res = await fetch(`/api/conversations/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (res.ok) {
        const updated = await res.json();
        setItems((prev) => prev.map((c) => (c.id === id ? { ...c, ...updated } : c)));
      }
    },
    []
  );

  const removeConversation = useCallback(
    async (id: string) => {
      await fetch(`/api/conversations/${id}`, { method: 'DELETE' });
      setItems((prev) => prev.filter((c) => c.id !== id));
    },
    []
  );

  return { items, loading, isAuth, refresh, updateConversation, removeConversation };
}
