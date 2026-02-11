import { NextResponse } from 'next/server';
import { auth } from '@/lib/auth';
import { getDb } from '@/lib/db';
import { annotations, users } from '@/lib/db/schema';
import { eq, and, ne } from 'drizzle-orm';

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const slug = searchParams.get('slug');
  if (!slug) {
    return NextResponse.json({ error: 'slug required' }, { status: 400 });
  }

  const session = await auth();
  const db = getDb();

  // Get published annotations for this slug, excluding the current user's
  let where = and(
    eq(annotations.slug, slug),
    eq(annotations.isPublished, true)
  );

  if (session?.user?.id) {
    where = and(where, ne(annotations.userId, session.user.id));
  }

  const results = await db
    .select({
      id: annotations.id,
      nearestHeadingId: annotations.nearestHeadingId,
      prefix: annotations.prefix,
      exact: annotations.exact,
      suffix: annotations.suffix,
      note: annotations.note,
      createdAt: annotations.createdAt,
      userName: users.name,
      userImage: users.image,
    })
    .from(annotations)
    .innerJoin(users, eq(annotations.userId, users.id))
    .where(where!);

  return NextResponse.json(results);
}
