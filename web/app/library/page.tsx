'use client';

import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { useAnnotations } from '@/lib/hooks/useAnnotations';

export default function LibraryPage() {
  const { status } = useSession();
  const router = useRouter();
  const { items: annotations, update, remove } = useAnnotations();

  if (status === 'loading') return <div className="app-page"><p>Loading...</p></div>;
  if (status === 'unauthenticated') {
    router.push('/login');
    return null;
  }

  // Only show highlights (annotations with text)
  const highlights = annotations.filter((a) => a.exact);

  // Group by slug
  const bySlug: Record<string, typeof highlights> = {};
  for (const a of highlights) {
    (bySlug[a.slug] ||= []).push(a);
  }

  const isEmpty = highlights.length === 0;

  return (
    <div className="app-page">
      <h1>Library</h1>

      {isEmpty ? (
        <div className="library-empty">
          No highlights yet. Select text in any chapter to highlight and annotate.
        </div>
      ) : (
        Object.entries(bySlug).map(([slug, items]) => (
          <div key={slug} className="library-group">
            <div className="library-group-title">
              <a href={`/${slug}`} style={{ color: 'inherit', textDecoration: 'none' }}>
                {slug.replace(/-/g, ' ')}
              </a>
            </div>
            {items.map((a) => (
              <div key={a.id} className="library-item">
                <div className="library-item-text">
                  <div className="library-item-exact">
                    &ldquo;{a.exact.slice(0, 120)}{a.exact.length > 120 ? '...' : ''}&rdquo;
                  </div>
                  {a.note && <div className="library-item-note">{a.note}</div>}
                </div>
                <div className="library-item-actions">
                  <button
                    onClick={() => update(a.id, { isPublished: !a.isPublished })}
                    title={a.isPublished ? 'Unpublish' : 'Publish'}
                  >
                    {a.isPublished ? 'Public' : 'Private'}
                  </button>
                  <button
                    onClick={() => remove(a.id)}
                    title="Delete"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        ))
      )}
    </div>
  );
}
