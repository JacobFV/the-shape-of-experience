import { streamText, convertToModelMessages } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { NextResponse } from 'next/server';

// Uses OPENAI_CHAT_KEY (not OPENAI_API_KEY) — a separate key with a $50 spend
// limit so public-facing chat doesn't run up costs. OPENAI_API_KEY is reserved
// for uncapped internal uses (TTS generation, CI, etc.).
const openai = createOpenAI({ apiKey: process.env.OPENAI_CHAT_KEY });
import { auth } from '@/lib/auth';
import { getDb } from '@/lib/db';
import { conversations, messages } from '@/lib/db/schema';
import { eq } from 'drizzle-orm';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

type Section = { headingId: string; heading: string; text: string };
type ChapterIndex = { slug: string; title: string; sections: Section[] };

let searchIndex: ChapterIndex[] | null = null;

function getSearchIndex(): ChapterIndex[] {
  if (searchIndex) return searchIndex;
  const indexPath = join(process.cwd(), 'public', 'search-index.json');
  if (!existsSync(indexPath)) return [];
  try {
    searchIndex = JSON.parse(readFileSync(indexPath, 'utf-8'));
    return searchIndex!;
  } catch {
    return [];
  }
}

function buildContext(contextType: string, slug?: string, contextExact?: string): string {
  const index = getSearchIndex();

  if (contextType === 'highlight' && contextExact) {
    for (const chapter of index) {
      for (const section of chapter.sections) {
        if (section.text.includes(contextExact.slice(0, 100))) {
          return `The reader highlighted this passage from "${chapter.title}":\n\n"${contextExact}"\n\nFull section context:\n${section.text.slice(0, 1500)}`;
        }
      }
    }
    return `The reader highlighted this passage:\n\n"${contextExact}"`;
  }

  if (contextType === 'page' && slug) {
    const chapter = index.find((c) => c.slug === slug);
    if (chapter) {
      const sections = chapter.sections
        .map((s) => s.text.slice(0, 800))
        .join('\n\n---\n\n');
      return `The reader is on the chapter "${chapter.title}".\n\nChapter content (excerpts):\n${sections.slice(0, 8000)}`;
    }
  }

  const summaries = index
    .map((ch) => {
      const first = ch.sections[0];
      return first ? `**${ch.title}**: ${first.text.slice(0, 300)}...` : '';
    })
    .filter(Boolean)
    .join('\n\n');
  return `This is a book called "The Shape of Experience: A Geometric Theory of Affect for Biological and Artificial Systems". Here are excerpts from each chapter:\n\n${summaries.slice(0, 6000)}`;
}

const SYSTEM_PROMPT = `You are a helpful reading companion for "The Shape of Experience," a book about a geometric theory of affect (emotion/feeling) applicable to both biological and artificial systems.

Key concepts in the book:
- The 6D affect framework: Valence, Arousal, Integration (Φ), Effective Rank, Counterfactual Weight, Self-Model Salience
- The inhibition coefficient (ι): governs participatory vs mechanistic perception
- Viability manifolds: regions of state space where a system can persist
- Forcing functions: pressures that push systems toward affect-like processing
- Attention as measurement selection in chaotic dynamics

Be conversational but substantive. Reference specific concepts from the book when relevant. If asked about something not covered in the provided context, say so honestly. Keep responses concise (2-4 paragraphs) unless the reader asks for more detail.`;

// Extract text content from UIMessage parts
function getTextFromParts(parts: Array<{ type: string; text?: string }>): string {
  return parts
    .filter((p) => p.type === 'text' && p.text)
    .map((p) => p.text!)
    .join('');
}

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const body = await req.json();
  const { messages: chatMessages, conversationId, contextType, slug, contextExact, contextHeadingId } = body;

  const db = getDb();
  let convId = conversationId;

  // Extract text from the first/latest user message
  const userMessages = chatMessages.filter((m: { role: string }) => m.role === 'user');
  const latestUserMsg = userMessages[userMessages.length - 1];
  const latestUserText = latestUserMsg
    ? (latestUserMsg.content || getTextFromParts(latestUserMsg.parts || []))
    : '';

  // Create conversation on first message
  if (!convId && userMessages.length === 1) {
    const title = latestUserText.slice(0, 80) + (latestUserText.length > 80 ? '...' : '');
    const [conv] = await db
      .insert(conversations)
      .values({
        userId: session.user.id,
        slug: slug || null,
        title,
        contextType: contextType || 'book',
        contextExact: contextExact || '',
        contextHeadingId: contextHeadingId || '',
      })
      .returning();
    convId = conv.id;

    await db.insert(messages).values({
      conversationId: convId,
      role: 'user',
      content: latestUserText,
    });
  } else if (convId && latestUserMsg) {
    await db.insert(messages).values({
      conversationId: convId,
      role: 'user',
      content: latestUserText,
    });
  }

  const context = buildContext(contextType || 'book', slug, contextExact);

  // Convert UIMessage format to model messages
  const modelMessages = await convertToModelMessages(chatMessages);

  const result = streamText({
    model: openai('gpt-4o-mini'),
    system: `${SYSTEM_PROMPT}\n\n---\n\nContext:\n${context}`,
    messages: modelMessages,
    onFinish: async ({ text }) => {
      if (convId) {
        await db.insert(messages).values({
          conversationId: convId,
          role: 'assistant',
          content: text,
        });
        await db
          .update(conversations)
          .set({ updatedAt: new Date() })
          .where(eq(conversations.id, convId));
      }
    },
  });

  const response = result.toUIMessageStreamResponse();

  if (convId) {
    response.headers.set('X-Conversation-Id', convId);
  }

  return response;
}
