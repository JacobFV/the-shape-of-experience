'use client';

import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { useCommunityNotes, CommunityNote } from '@/lib/hooks/useCommunityNotes';

export default function CommunityHighlights({ slug }: { slug: string }) {
  const { status } = useSession();
  const [enabled, setEnabled] = useState(true);

  // Check user setting for community notes
  useEffect(() => {
    if (status !== 'authenticated') return;
    fetch('/api/settings')
      .then((r) => r.json())
      .then((data) => {
        if (data.showCommunityNotes === false) setEnabled(false);
      })
      .catch(() => {});
  }, [status]);

  const { notes } = useCommunityNotes(slug, enabled);

  if (!notes.length) return null;

  // Group notes by nearest heading
  const grouped: Record<string, CommunityNote[]> = {};
  for (const note of notes) {
    const key = note.nearestHeadingId || '_top';
    (grouped[key] ||= []).push(note);
  }

  // Render notes inline after their nearest heading
  useEffect(() => {
    // Clean up previously injected community notes
    document.querySelectorAll('.community-note-injected').forEach((el) => el.remove());

    for (const [headingId, headingNotes] of Object.entries(grouped)) {
      const anchor = headingId === '_top'
        ? document.querySelector('.chapter-content')
        : document.getElementById(headingId);
      if (!anchor) continue;

      const container = document.createElement('div');
      container.className = 'community-note-injected';

      for (const note of headingNotes) {
        const div = document.createElement('div');
        div.className = 'community-note';

        const initial = (note.userName?.[0] || '?').toUpperCase();
        div.innerHTML = `
          <div class="community-note-header">
            <span class="community-note-avatar">${initial}</span>
            <span>${note.userName || 'Anonymous'}</span>
          </div>
          ${note.exact ? `<div class="community-note-quote">"${note.exact.slice(0, 80)}${note.exact.length > 80 ? '...' : ''}"</div>` : ''}
          <div class="community-note-body">${escapeHtml(note.note || '')}</div>
        `;
        container.appendChild(div);
      }

      // Insert after the heading
      if (headingId === '_top') {
        anchor.prepend(container);
      } else {
        anchor.after(container);
      }
    }

    return () => {
      document.querySelectorAll('.community-note-injected').forEach((el) => el.remove());
    };
  }, [notes]); // eslint-disable-line react-hooks/exhaustive-deps

  return null;
}

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
