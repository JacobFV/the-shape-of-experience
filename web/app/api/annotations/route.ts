import { NextResponse } from 'next/server';
import { auth } from '@/lib/auth';
import { getDb } from '@/lib/db';
import { annotations } from '@/lib/db/schema';
import { eq, and } from 'drizzle-orm';

export async function GET(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const db = getDb();
  const { searchParams } = new URL(req.url);
  const slug = searchParams.get('slug');

  const where = slug
    ? and(eq(annotations.userId, session.user.id), eq(annotations.slug, slug))
    : eq(annotations.userId, session.user.id);

  const results = await db.select().from(annotations).where(where);
  return NextResponse.json(results);
}

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const db = getDb();
  const body = await req.json();
  const [annotation] = await db
    .insert(annotations)
    .values({
      userId: session.user.id,
      slug: body.slug,
      nearestHeadingId: body.nearestHeadingId || '',
      prefix: body.prefix || '',
      exact: body.exact,
      suffix: body.suffix || '',
      note: body.note || '',
      isPublished: false,
    })
    .returning();

  return NextResponse.json(annotation, { status: 201 });
}
