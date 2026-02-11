'use client';

import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';

export default function SyncOnLogin() {
  const { data: session, status } = useSession();
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    if (status !== 'authenticated' || !session?.user) return;

    // Check if we've already synced
    const syncKey = `soe-synced-${session.user.id}`;
    if (localStorage.getItem(syncKey)) return;

    // Collect all localStorage highlights
    const highlights: Array<Record<string, unknown>> = [];
    const bookmarks: Array<Record<string, unknown>> = [];

    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (!key) continue;

      if (key.startsWith('soe-highlights-')) {
        const slug = key.replace('soe-highlights-', '');
        try {
          const items = JSON.parse(localStorage.getItem(key) || '[]');
          for (const item of items) {
            highlights.push({ ...item, slug });
          }
        } catch { /* ignore */ }
      }
    }

    try {
      const bms = JSON.parse(localStorage.getItem('soe-bookmarks') || '[]');
      bookmarks.push(...bms);
    } catch { /* ignore */ }

    if (highlights.length === 0 && bookmarks.length === 0) {
      localStorage.setItem(syncKey, '1');
      return;
    }

    // Sync to server
    fetch('/api/sync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ highlights, bookmarks }),
    })
      .then((r) => r.json())
      .then((data) => {
        localStorage.setItem(syncKey, '1');

        // Clear synced data from localStorage
        for (let i = localStorage.length - 1; i >= 0; i--) {
          const key = localStorage.key(i);
          if (key?.startsWith('soe-highlights-')) {
            localStorage.removeItem(key);
          }
        }
        localStorage.removeItem('soe-bookmarks');

        const total = (data.imported?.annotations || 0) + (data.imported?.bookmarks || 0);
        if (total > 0) {
          setToast(`Synced ${total} items to your account`);
          setTimeout(() => setToast(null), 3000);
        }
      })
      .catch(() => {});
  }, [status, session]);

  if (!toast) return null;
  return <div className="toast">{toast}</div>;
}
