import { NextResponse } from 'next/server';
import { auth } from '@/lib/auth';
import { getDb } from '@/lib/db';
import { reactions } from '@/lib/db/schema';
import { eq, and } from 'drizzle-orm';

const ALLOWED_EMOJIS = ['👍', '❤️', '💡', '🤔', '💯', '👀'];

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const body = await req.json();
  const { targetType, targetId, emoji } = body;

  if (!targetType || !targetId || !emoji) {
    return NextResponse.json({ error: 'Missing fields' }, { status: 400 });
  }

  if (targetType !== 'annotation' && targetType !== 'comment') {
    return NextResponse.json({ error: 'Invalid targetType' }, { status: 400 });
  }

  if (!ALLOWED_EMOJIS.includes(emoji)) {
    return NextResponse.json({ error: 'Invalid emoji' }, { status: 400 });
  }

  const db = getDb();

  // Check if reaction exists — toggle
  const [existing] = await db
    .select({ id: reactions.id })
    .from(reactions)
    .where(
      and(
        eq(reactions.userId, session.user.id),
        eq(reactions.targetType, targetType),
        eq(reactions.targetId, targetId),
        eq(reactions.emoji, emoji)
      )
    )
    .limit(1);

  if (existing) {
    await db.delete(reactions).where(eq(reactions.id, existing.id));
  } else {
    await db.insert(reactions).values({
      userId: session.user.id,
      targetType,
      targetId,
      emoji,
    });
  }

  // Return updated counts for this target
  const allReactions = await db
    .select({
      emoji: reactions.emoji,
      userId: reactions.userId,
    })
    .from(reactions)
    .where(
      and(
        eq(reactions.targetType, targetType),
        eq(reactions.targetId, targetId)
      )
    );

  const emojiMap: Record<string, { count: number; userReacted: boolean }> = {};
  for (const r of allReactions) {
    if (!emojiMap[r.emoji]) emojiMap[r.emoji] = { count: 0, userReacted: false };
    emojiMap[r.emoji].count++;
    if (r.userId === session.user.id) emojiMap[r.emoji].userReacted = true;
  }

  return NextResponse.json(
    Object.entries(emojiMap).map(([emoji, data]) => ({ emoji, ...data }))
  );
}
