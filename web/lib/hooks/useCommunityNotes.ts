'use client';

import { useState, useEffect } from 'react';

export interface CommunityNote {
  id: string;
  nearestHeadingId: string;
  prefix: string;
  exact: string;
  suffix: string;
  note: string;
  createdAt: string;
  userName: string;
  userImage: string | null;
}

export function useCommunityNotes(slug: string, enabled: boolean = true) {
  const [notes, setNotes] = useState<CommunityNote[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!enabled || !slug) {
      setNotes([]);
      setLoading(false);
      return;
    }

    fetch(`/api/annotations/community?slug=${encodeURIComponent(slug)}`)
      .then((r) => r.json())
      .then((data) => {
        setNotes(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [slug, enabled]);

  return { notes, loading };
}
