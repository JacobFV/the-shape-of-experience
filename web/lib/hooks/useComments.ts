'use client';

import { useState, useCallback } from 'react';
import type { ReactionData } from './useCommunityNotes';

export interface Comment {
  id: string;
  content: string;
  createdAt: string;
  userId: string;
  userName: string;
  userImage: string | null;
  reactions: ReactionData[];
}

export function useComments(annotationId: string) {
  const [comments, setComments] = useState<Comment[]>([]);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    if (loaded) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/annotations/${annotationId}/comments`);
      if (res.ok) {
        const data = await res.json();
        setComments(data);
        setLoaded(true);
      }
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, [annotationId, loaded]);

  const addComment = useCallback(
    async (content: string) => {
      const res = await fetch(`/api/annotations/${annotationId}/comments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      });
      if (res.ok) {
        const comment = await res.json();
        setComments((prev) => [...prev, comment]);
        return comment;
      }
      return null;
    },
    [annotationId]
  );

  const deleteComment = useCallback(
    async (commentId: string) => {
      const res = await fetch(
        `/api/annotations/${annotationId}/comments/${commentId}`,
        { method: 'DELETE' }
      );
      if (res.ok) {
        setComments((prev) => prev.filter((c) => c.id !== commentId));
      }
    },
    [annotationId]
  );

  const updateCommentReactions = useCallback(
    (commentId: string, reactions: ReactionData[]) => {
      setComments((prev) =>
        prev.map((c) => (c.id === commentId ? { ...c, reactions } : c))
      );
    },
    []
  );

  return { comments, loading, loaded, load, addComment, deleteComment, updateCommentReactions };
}
