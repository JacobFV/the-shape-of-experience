import { NextResponse } from 'next/server';
import { auth } from '@/lib/auth';
import { getDb } from '@/lib/db';
import { annotations, bookmarks } from '@/lib/db/schema';

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const db = getDb();
  const body = await req.json();
  let annotationsCount = 0;
  let bookmarksCount = 0;

  // Import highlights (annotations without notes)
  if (body.highlights && Array.isArray(body.highlights)) {
    for (const h of body.highlights) {
      await db.insert(annotations).values({
        userId: session.user.id,
        slug: h.slug,
        nearestHeadingId: h.nearestHeadingId || '',
        prefix: h.prefix || '',
        exact: h.exact,
        suffix: h.suffix || '',
        note: '',
        isPublished: false,
      });
      annotationsCount++;
    }
  }

  // Import bookmarks
  if (body.bookmarks && Array.isArray(body.bookmarks)) {
    for (const bm of body.bookmarks) {
      await db.insert(bookmarks).values({
        userId: session.user.id,
        slug: bm.slug,
        scrollY: bm.scrollY || 0,
        nearestHeadingId: bm.nearestHeadingId || '',
        nearestHeadingText: bm.nearestHeadingText || '',
      });
      bookmarksCount++;
    }
  }

  return NextResponse.json({
    imported: { annotations: annotationsCount, bookmarks: bookmarksCount },
  });
}
