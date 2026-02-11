import { NextResponse } from 'next/server';
import { auth } from '@/lib/auth';
import { getDb } from '@/lib/db';
import { bookmarks } from '@/lib/db/schema';
import { eq } from 'drizzle-orm';

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const db = getDb();
  const results = await db
    .select()
    .from(bookmarks)
    .where(eq(bookmarks.userId, session.user.id));

  return NextResponse.json(results);
}

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const db = getDb();
  const body = await req.json();
  const [bookmark] = await db
    .insert(bookmarks)
    .values({
      userId: session.user.id,
      slug: body.slug,
      scrollY: body.scrollY || 0,
      nearestHeadingId: body.nearestHeadingId || '',
      nearestHeadingText: body.nearestHeadingText || '',
    })
    .returning();

  return NextResponse.json(bookmark, { status: 201 });
}
