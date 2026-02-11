import { NextResponse } from 'next/server';
import { getDb } from '@/lib/db';
import { conversations, messages, users, userSettings } from '@/lib/db/schema';
import { eq, and, desc, asc } from 'drizzle-orm';

export async function GET(req: Request) {
  const db = getDb();
  const { searchParams } = new URL(req.url);
  const slug = searchParams.get('slug');

  const where = slug
    ? and(eq(conversations.isPublished, true), eq(conversations.slug, slug))
    : eq(conversations.isPublished, true);

  const convos = await db
    .select({
      id: conversations.id,
      userId: conversations.userId,
      slug: conversations.slug,
      title: conversations.title,
      contextType: conversations.contextType,
      contextExact: conversations.contextExact,
      createdAt: conversations.createdAt,
    })
    .from(conversations)
    .where(where)
    .orderBy(desc(conversations.updatedAt))
    .limit(20);

  // Get first exchange and user info for each conversation
  const results = await Promise.all(
    convos.map(async (conv) => {
      const msgs = await db
        .select()
        .from(messages)
        .where(eq(messages.conversationId, conv.id))
        .orderBy(asc(messages.createdAt))
        .limit(2);

      const [user] = await db
        .select({ name: users.name, displayName: userSettings.displayName })
        .from(users)
        .leftJoin(userSettings, eq(users.id, userSettings.userId))
        .where(eq(users.id, conv.userId));

      return {
        ...conv,
        userName: user?.displayName || user?.name || 'Anonymous',
        firstExchange: msgs,
      };
    })
  );

  return NextResponse.json(results);
}
