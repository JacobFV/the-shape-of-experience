import { chapters, getChapterComponent, getChapterBySlug, getAdjacentChapters, getChapterAudio } from '../../lib/chapters';
import ChapterNav from '../../components/ChapterNav';
import AudioPlayer from '../../components/AudioPlayer';
import HighlightManager from '../../components/HighlightManager';
import CommunityHighlights from '../../components/CommunityHighlights';
import { notFound } from 'next/navigation';
import type { Metadata } from 'next';

export function generateStaticParams() {
  return chapters.map(ch => ({ slug: ch.slug }));
}

type Props = {
  params: Promise<{ slug: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const chapter = getChapterBySlug(slug);
  if (!chapter) return { title: 'Not Found' };
  return {
    title: `${chapter.shortTitle} — The Shape of Experience`,
    description: chapter.title,
  };
}

export default async function ChapterPage({ params }: Props) {
  const { slug } = await params;
  const chapter = getChapterBySlug(slug);
  if (!chapter) notFound();

  const ChapterContent = await getChapterComponent(slug);
  if (!ChapterContent) notFound();

  const { prev, next } = getAdjacentChapters(slug);
  const audioSections = getChapterAudio(slug);

  return (
    <article className="chapter">
      <header className="chapter-header">
        <h1 className="chapter-title">{chapter.title}</h1>
      </header>

      {audioSections.length > 0 && (
        <AudioPlayer
          sections={audioSections}
          chapterTitle={chapter.shortTitle}
          slug={slug}
        />
      )}

      <div className="chapter-content">
        <ChapterContent />
      </div>
      <HighlightManager slug={slug} />
      <CommunityHighlights slug={slug} />

      <ChapterNav
        prev={prev ? { slug: prev.slug, shortTitle: prev.shortTitle } : undefined}
        next={next ? { slug: next.slug, shortTitle: next.shortTitle } : undefined}
      />
    </article>
  );
}
