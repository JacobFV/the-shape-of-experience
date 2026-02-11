import { NextResponse } from 'next/server';
import { auth } from '@/lib/auth';
import { getDb } from '@/lib/db';
import { conversations } from '@/lib/db/schema';
import { eq, and, desc } from 'drizzle-orm';

export async function GET(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const db = getDb();
  const { searchParams } = new URL(req.url);
  const slug = searchParams.get('slug');

  const where = slug
    ? and(eq(conversations.userId, session.user.id), eq(conversations.slug, slug))
    : eq(conversations.userId, session.user.id);

  const results = await db
    .select()
    .from(conversations)
    .where(where)
    .orderBy(desc(conversations.updatedAt));

  return NextResponse.json(results);
}
