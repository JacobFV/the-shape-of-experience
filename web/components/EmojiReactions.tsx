'use client';

import { useState } from 'react';
import { useSession } from 'next-auth/react';
import { useReactions } from '@/lib/hooks/useReactions';
import type { ReactionData } from '@/lib/hooks/useCommunityNotes';

const EMOJI_SET = ['👍', '❤️', '💡', '🤔', '💯', '👀'];

interface Props {
  targetType: 'annotation' | 'comment';
  targetId: string;
  initialReactions: ReactionData[];
  onUpdate?: (reactions: ReactionData[]) => void;
}

export default function EmojiReactions({ targetType, targetId, initialReactions, onUpdate }: Props) {
  const { status } = useSession();
  const { reactions, toggle } = useReactions(initialReactions, onUpdate);
  const [showPicker, setShowPicker] = useState(false);

  const isAuth = status === 'authenticated';

  return (
    <div className="emoji-reactions">
      {reactions.map((r) => (
        <button
          key={r.emoji}
          className={`emoji-badge ${r.userReacted ? 'reacted' : ''}`}
          onClick={() => isAuth && toggle(targetType, targetId, r.emoji)}
          disabled={!isAuth}
          title={isAuth ? 'Toggle reaction' : 'Sign in to react'}
        >
          <span className="emoji-badge-emoji">{r.emoji}</span>
          <span className="emoji-badge-count">{r.count}</span>
        </button>
      ))}
      {isAuth && (
        <div className="emoji-picker-wrapper">
          <button
            className="emoji-add-btn"
            onClick={() => setShowPicker(!showPicker)}
            title="Add reaction"
          >
            +
          </button>
          {showPicker && (
            <div className="emoji-picker">
              {EMOJI_SET.map((emoji) => (
                <button
                  key={emoji}
                  className="emoji-picker-item"
                  onClick={() => {
                    toggle(targetType, targetId, emoji);
                    setShowPicker(false);
                  }}
                >
                  {emoji}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
