'use client';

import { useState } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { useAnnotations, Annotation } from '@/lib/hooks/useAnnotations';
import { useBookmarks, Bookmark } from '@/lib/hooks/useBookmarks';

export default function LibraryPage() {
  const { status } = useSession();
  const router = useRouter();
  const [tab, setTab] = useState<'notes' | 'bookmarks'>('notes');
  const { items: annotations, update, remove: removeAnnotation } = useAnnotations();
  const { items: bookmarks, remove: removeBookmark } = useBookmarks();

  if (status === 'loading') return <div className="app-page"><p>Loading...</p></div>;
  if (status === 'unauthenticated') {
    router.push('/login');
    return null;
  }

  // Group annotations by slug
  const annotationsBySlug: Record<string, Annotation[]> = {};
  for (const a of annotations) {
    (annotationsBySlug[a.slug] ||= []).push(a);
  }

  // Group bookmarks by slug
  const bookmarksBySlug: Record<string, Bookmark[]> = {};
  for (const b of bookmarks) {
    (bookmarksBySlug[b.slug] ||= []).push(b);
  }

  return (
    <div className="app-page">
      <h1>Library</h1>

      <div className="tabs">
        <button
          className={tab === 'notes' ? 'active' : ''}
          onClick={() => setTab('notes')}
        >
          Notes ({annotations.length})
        </button>
        <button
          className={tab === 'bookmarks' ? 'active' : ''}
          onClick={() => setTab('bookmarks')}
        >
          Bookmarks ({bookmarks.length})
        </button>
      </div>

      {tab === 'notes' && (
        annotations.length === 0 ? (
          <div className="library-empty">
            No highlights or notes yet. Select text in any chapter to start annotating.
          </div>
        ) : (
          Object.entries(annotationsBySlug).map(([slug, items]) => (
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
                      onClick={() => removeAnnotation(a.id)}
                      title="Delete"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ))
        )
      )}

      {tab === 'bookmarks' && (
        bookmarks.length === 0 ? (
          <div className="library-empty">
            No bookmarks yet. Click the star icon or select text and click Bookmark.
          </div>
        ) : (
          Object.entries(bookmarksBySlug).map(([slug, items]) => (
            <div key={slug} className="library-group">
              <div className="library-group-title">
                <a href={`/${slug}`} style={{ color: 'inherit', textDecoration: 'none' }}>
                  {slug.replace(/-/g, ' ')}
                </a>
              </div>
              {items.map((b) => (
                <div key={b.id} className="library-item">
                  <div className="library-item-text">
                    <div className="library-item-exact">
                      {b.nearestHeadingText || 'Start of chapter'}
                    </div>
                  </div>
                  <div className="library-item-actions">
                    <button
                      onClick={() => {
                        if (b.nearestHeadingId) {
                          router.push(`/${slug}#${b.nearestHeadingId}`);
                        } else {
                          router.push(`/${slug}`);
                        }
                      }}
                      title="Go to bookmark"
                    >
                      Go
                    </button>
                    <button
                      onClick={() => removeBookmark(b.id)}
                      title="Delete"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ))
        )
      )}
    </div>
  );
}
