import { NextResponse } from 'next/server';
import { auth } from '@/lib/auth';
import { getDb } from '@/lib/db';
import { comments, annotations, users, reactions } from '@/lib/db/schema';
import { eq, and } from 'drizzle-orm';

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: annotationId } = await params;
  const session = await auth();
  const db = getDb();

  const rows = await db
    .select({
      id: comments.id,
      content: comments.content,
      createdAt: comments.createdAt,
      userId: comments.userId,
      userName: users.name,
      userImage: users.image,
    })
    .from(comments)
    .innerJoin(users, eq(comments.userId, users.id))
    .where(eq(comments.annotationId, annotationId))
    .orderBy(comments.createdAt);

  // Batch-fetch reactions for these comments
  const commentIds = rows.map((r) => r.id);
  let reactionRows: { targetId: string; emoji: string; userId: string }[] = [];
  if (commentIds.length > 0) {
    reactionRows = await db
      .select({
        targetId: reactions.targetId,
        emoji: reactions.emoji,
        userId: reactions.userId,
      })
      .from(reactions)
      .where(eq(reactions.targetType, 'comment'));
  }

  const result = rows.map((row) => {
    const commentReactions = reactionRows.filter((r) => r.targetId === row.id);
    const emojiMap: Record<string, { count: number; userReacted: boolean }> = {};
    for (const r of commentReactions) {
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

  return NextResponse.json(result);
}

export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: annotationId } = await params;
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const db = getDb();

  // Verify annotation exists and is published
  const [annotation] = await db
    .select({ id: annotations.id, isPublished: annotations.isPublished })
    .from(annotations)
    .where(eq(annotations.id, annotationId))
    .limit(1);

  if (!annotation || !annotation.isPublished) {
    return NextResponse.json({ error: 'Not found' }, { status: 404 });
  }

  const body = await req.json();
  const content = (body.content || '').trim();
  if (!content || content.length > 1000) {
    return NextResponse.json(
      { error: 'Content required (max 1000 chars)' },
      { status: 400 }
    );
  }

  const [comment] = await db
    .insert(comments)
    .values({
      annotationId,
      userId: session.user.id,
      content,
    })
    .returning();

  // Fetch user info for response
  const [user] = await db
    .select({ name: users.name, image: users.image })
    .from(users)
    .where(eq(users.id, session.user.id))
    .limit(1);

  return NextResponse.json({
    ...comment,
    userName: user?.name,
    userImage: user?.image,
    reactions: [],
  });
}
