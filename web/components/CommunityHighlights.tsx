'use client';

import { useEffect, useState, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { useSession } from 'next-auth/react';
import { useCommunityNotes, CommunityNote } from '@/lib/hooks/useCommunityNotes';
import EmojiReactions from './EmojiReactions';
import CommentThread from './CommentThread';

export default function CommunityHighlights({ slug }: { slug: string }) {
  const { status } = useSession();
  const [enabled, setEnabled] = useState(true);

  useEffect(() => {
    if (status !== 'authenticated') return;
    fetch('/api/settings')
      .then((r) => r.json())
      .then((data) => {
        if (data.showCommunityNotes === false) setEnabled(false);
      })
      .catch(() => {});
  }, [status]);

  const { notes, updateNoteReactions, incrementCommentCount } = useCommunityNotes(slug, enabled);

  // Group notes by nearest heading
  const grouped = useMemo(() => {
    const map: Record<string, CommunityNote[]> = {};
    for (const note of notes) {
      const key = note.nearestHeadingId || '_top';
      (map[key] ||= []).push(note);
    }
    return map;
  }, [notes]);

  // Create/maintain portal anchor elements
  const [anchors, setAnchors] = useState<Record<string, HTMLElement>>({});

  useEffect(() => {
    // Clean up previously injected containers
    document.querySelectorAll('.community-note-portal').forEach((el) => el.remove());

    const newAnchors: Record<string, HTMLElement> = {};

    for (const headingId of Object.keys(grouped)) {
      const anchor =
        headingId === '_top'
          ? document.querySelector('.chapter-content')
          : document.getElementById(headingId);
      if (!anchor) continue;

      const container = document.createElement('div');
      container.className = 'community-note-portal';

      if (headingId === '_top') {
        anchor.prepend(container);
      } else {
        anchor.after(container);
      }

      newAnchors[headingId] = container;
    }

    setAnchors(newAnchors);

    return () => {
      document.querySelectorAll('.community-note-portal').forEach((el) => el.remove());
    };
  }, [grouped]);

  if (!notes.length) return null;

  // Render into portal containers
  return (
    <>
      {Object.entries(grouped).map(([headingId, headingNotes]) => {
        const container = anchors[headingId];
        if (!container) return null;

        return createPortal(
          <div key={headingId}>
            {headingNotes.map((note) => (
              <CommunityNoteCard
                key={note.id}
                note={note}
                onReactionsUpdate={(reactions) => updateNoteReactions(note.id, reactions)}
                onCommentCountChange={(delta) => incrementCommentCount(note.id, delta)}
              />
            ))}
          </div>,
          container
        );
      })}
    </>
  );
}

function CommunityNoteCard({
  note,
  onReactionsUpdate,
  onCommentCountChange,
}: {
  note: CommunityNote;
  onReactionsUpdate: (reactions: CommunityNote['reactions']) => void;
  onCommentCountChange: (delta: number) => void;
}) {
  const initial = (note.userName?.[0] || '?').toUpperCase();

  return (
    <div className="community-note">
      <div className="community-note-header">
        {note.userImage ? (
          <img src={note.userImage} alt="" className="community-note-avatar-img" />
        ) : (
          <span className="community-note-avatar">{initial}</span>
        )}
        <span>{note.userName || 'Anonymous'}</span>
      </div>
      {note.exact && (
        <div className="community-note-quote">
          &ldquo;{note.exact.slice(0, 80)}
          {note.exact.length > 80 ? '...' : ''}&rdquo;
        </div>
      )}
      <div className="community-note-body">{note.note || ''}</div>
      <EmojiReactions
        targetType="annotation"
        targetId={note.id}
        initialReactions={note.reactions}
        onUpdate={onReactionsUpdate}
      />
      <CommentThread
        annotationId={note.id}
        commentCount={note.commentCount}
        onCommentCountChange={onCommentCountChange}
      />
    </div>
  );
}
