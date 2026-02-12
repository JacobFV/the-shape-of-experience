'use client';

import { useState, useCallback } from 'react';
import type { ReactionData } from './useCommunityNotes';

export function useReactions(
  initialReactions: ReactionData[],
  onUpdate?: (reactions: ReactionData[]) => void
) {
  const [reactions, setReactions] = useState<ReactionData[]>(initialReactions);
  const [loading, setLoading] = useState(false);

  const toggle = useCallback(
    async (targetType: 'annotation' | 'comment', targetId: string, emoji: string) => {
      // Optimistic update
      setReactions((prev) => {
        const existing = prev.find((r) => r.emoji === emoji);
        if (existing) {
          if (existing.userReacted) {
            const newCount = existing.count - 1;
            return newCount <= 0
              ? prev.filter((r) => r.emoji !== emoji)
              : prev.map((r) =>
                  r.emoji === emoji
                    ? { ...r, count: newCount, userReacted: false }
                    : r
                );
          } else {
            return prev.map((r) =>
              r.emoji === emoji
                ? { ...r, count: r.count + 1, userReacted: true }
                : r
            );
          }
        } else {
          return [...prev, { emoji, count: 1, userReacted: true }];
        }
      });

      setLoading(true);
      try {
        const res = await fetch('/api/reactions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ targetType, targetId, emoji }),
        });
        if (res.ok) {
          const data = await res.json();
          setReactions(data);
          onUpdate?.(data);
        }
      } catch {
        // Revert would be complex; server state wins on next fetch
      } finally {
        setLoading(false);
      }
    },
    [onUpdate]
  );

  return { reactions, toggle, loading };
}
