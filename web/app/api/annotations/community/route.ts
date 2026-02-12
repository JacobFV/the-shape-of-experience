import { NextResponse } from 'next/server';
import { auth } from '@/lib/auth';
import { getDb } from '@/lib/db';
import { annotations, users, comments, reactions } from '@/lib/db/schema';
import { eq, and, ne, sql } from 'drizzle-orm';

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
      commentCount: sql<number>`(
        SELECT COUNT(*)::int FROM comments
        WHERE comments.annotation_id = ${annotations.id}
      )`,
    })
    .from(annotations)
    .innerJoin(users, eq(annotations.userId, users.id))
    .where(where!);

  // Batch-fetch reactions for all these annotations
  const annotationIds = results.map((r) => r.id);
  let reactionRows: { targetId: string; emoji: string; userId: string }[] = [];
  if (annotationIds.length > 0) {
    reactionRows = await db
      .select({
        targetId: reactions.targetId,
        emoji: reactions.emoji,
        userId: reactions.userId,
      })
      .from(reactions)
      .where(eq(reactions.targetType, 'annotation'));
  }

  const enriched = results.map((row) => {
    const annReactions = reactionRows.filter((r) => r.targetId === row.id);
    const emojiMap: Record<string, { count: number; userReacted: boolean }> = {};
    for (const r of annReactions) {
      if (!emojiMap[r.emoji]) emojiMap[r.emoji] = { count: 0, userReacted: false };
      emojiMap[r.emoji].count++;
      if (session?.user?.id && r.userId === session.user.id) {
        emojiMap[r.emoji].userReacted = true;
      }
    }
    return {
      ...row,
      reactions: Object.entries(emojiMap).map(([emoji, data]) => ({
        emoji,
        ...data,
      })),
    };
  });

  return NextResponse.json(enriched);
}
