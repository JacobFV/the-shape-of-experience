import { NextResponse } from 'next/server';
import { auth } from '@/lib/auth';
import { getDb } from '@/lib/db';
import { annotations } from '@/lib/db/schema';

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const db = getDb();
  const body = await req.json();
  let count = 0;

  // Import highlights/annotations
  if (body.highlights && Array.isArray(body.highlights)) {
    for (const h of body.highlights) {
      await db.insert(annotations).values({
        userId: session.user.id,
        slug: h.slug,
        nearestHeadingId: h.nearestHeadingId || '',
        nearestHeadingText: h.nearestHeadingText || '',
        prefix: h.prefix || '',
        exact: h.exact || '',
        suffix: h.suffix || '',
        note: h.note || '',
        isPublished: false,
      });
      count++;
    }
  }

  // Backward compat: old clients may still send bookmarks separately
  if (body.bookmarks && Array.isArray(body.bookmarks)) {
    for (const bm of body.bookmarks) {
      await db.insert(annotations).values({
        userId: session.user.id,
        slug: bm.slug,
        nearestHeadingId: bm.nearestHeadingId || '',
        nearestHeadingText: bm.nearestHeadingText || '',
        prefix: '',
        exact: '',
        suffix: '',
        note: '',
        isPublished: false,
      });
      count++;
    }
  }

  return NextResponse.json({
    imported: { annotations: count },
  });
}
