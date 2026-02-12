'use client';

import { useState, useEffect } from 'react';

export interface ReactionData {
  emoji: string;
  count: number;
  userReacted: boolean;
}

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
  commentCount: number;
  reactions: ReactionData[];
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

  const updateNoteReactions = (noteId: string, reactions: ReactionData[]) => {
    setNotes((prev) =>
      prev.map((n) => (n.id === noteId ? { ...n, reactions } : n))
    );
  };

  const incrementCommentCount = (noteId: string, delta: number) => {
    setNotes((prev) =>
      prev.map((n) =>
        n.id === noteId ? { ...n, commentCount: n.commentCount + delta } : n
      )
    );
  };

  return { notes, loading, updateNoteReactions, incrementCommentCount };
}
