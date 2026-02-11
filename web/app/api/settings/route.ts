import { NextResponse } from 'next/server';
import { auth } from '@/lib/auth';
import { getDb } from '@/lib/db';
import { userSettings } from '@/lib/db/schema';
import { eq } from 'drizzle-orm';

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const db = getDb();
  const [settings] = await db
    .select()
    .from(userSettings)
    .where(eq(userSettings.userId, session.user.id))
    .limit(1);

  return NextResponse.json(
    settings || { showCommunityNotes: true, displayName: '', bio: '', profileImage: null }
  );
}

export async function PATCH(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const db = getDb();
  const body = await req.json();
  const values: Record<string, unknown> = {};
  if ('showCommunityNotes' in body) values.showCommunityNotes = body.showCommunityNotes;
  if ('displayName' in body) values.displayName = body.displayName;
  if ('bio' in body) values.bio = body.bio;
  if ('profileImage' in body) values.profileImage = body.profileImage;

  const [updated] = await db
    .insert(userSettings)
    .values({ userId: session.user.id, ...values })
    .onConflictDoUpdate({
      target: userSettings.userId,
      set: values,
    })
    .returning();

  return NextResponse.json(updated);
}
