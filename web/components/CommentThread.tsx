'use client';

import { useState } from 'react';
import { useSession } from 'next-auth/react';
import { useComments } from '@/lib/hooks/useComments';
import EmojiReactions from './EmojiReactions';

interface Props {
  annotationId: string;
  commentCount: number;
  onCommentCountChange?: (delta: number) => void;
}

function timeAgo(dateStr: string): string {
  const d = new Date(dateStr);
  const now = Date.now();
  const diff = now - d.getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return d.toLocaleDateString();
}

export default function CommentThread({ annotationId, commentCount, onCommentCountChange }: Props) {
  const { data: session, status } = useSession();
  const isAuth = status === 'authenticated';
  const { comments, loading, loaded, load, addComment, deleteComment, updateCommentReactions } =
    useComments(annotationId);
  const [expanded, setExpanded] = useState(false);
  const [input, setInput] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const displayCount = loaded ? comments.length : commentCount;

  const handleExpand = () => {
    if (!expanded) {
      load();
      setExpanded(true);
    } else {
      setExpanded(false);
    }
  };

  const handleSubmit = async () => {
    if (!input.trim() || submitting) return;
    setSubmitting(true);
    const comment = await addComment(input.trim());
    if (comment) {
      setInput('');
      onCommentCountChange?.(1);
    }
    setSubmitting(false);
  };

  const handleDelete = async (commentId: string) => {
    await deleteComment(commentId);
    onCommentCountChange?.(-1);
  };

  return (
    <div className="comment-thread">
      <button className="comment-thread-toggle" onClick={handleExpand}>
        {displayCount > 0
          ? `${displayCount} comment${displayCount !== 1 ? 's' : ''}`
          : isAuth
          ? 'Add comment'
          : ''}
        {displayCount > 0 && (
          <span className="comment-chevron">{expanded ? '\u25B2' : '\u25BC'}</span>
        )}
      </button>

      {expanded && (
        <div className="comment-thread-body">
          {loading && <div className="comment-loading">Loading...</div>}

          {comments.map((c) => (
            <div key={c.id} className="comment-item">
              <div className="comment-header">
                {c.userImage ? (
                  <img
                    src={c.userImage}
                    alt=""
                    className="comment-avatar-img"
                  />
                ) : (
                  <span className="comment-avatar">
                    {(c.userName?.[0] || '?').toUpperCase()}
                  </span>
                )}
                <span className="comment-author">{c.userName || 'Anonymous'}</span>
                <span className="comment-time">{timeAgo(c.createdAt)}</span>
                {isAuth && session?.user?.id === c.userId && (
                  <button
                    className="comment-delete"
                    onClick={() => handleDelete(c.id)}
                    title="Delete comment"
                  >
                    &times;
                  </button>
                )}
              </div>
              <div className="comment-content">{c.content}</div>
              <EmojiReactions
                targetType="comment"
                targetId={c.id}
                initialReactions={c.reactions}
                onUpdate={(reactions) => updateCommentReactions(c.id, reactions)}
              />
            </div>
          ))}

          {isAuth && (
            <div className="comment-input-row">
              <textarea
                className="comment-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Add a comment..."
                maxLength={1000}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit();
                }}
              />
              <button
                className="comment-submit"
                onClick={handleSubmit}
                disabled={submitting || !input.trim()}
              >
                {submitting ? '...' : 'Post'}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
