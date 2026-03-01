import { chapters, getChapterComponent, getChapterBySlug, getChapterAudio } from '@/lib/chapters';
import {
  extractSection,
  getAllSectionParams,
  getSectionTitle,
  getAdjacentSections,
} from '@/lib/sections';
import ChapterNav from '@/components/ChapterNav';
import AudioPlayer from '@/components/AudioPlayer';
import HighlightManager from '@/components/HighlightManager';
import CommunityHighlights from '@/components/CommunityHighlights';
import CommunityConversations from '@/components/CommunityConversations';
import { notFound } from 'next/navigation';
import { ReactElement } from 'react';
import type { Metadata } from 'next';

export function generateStaticParams() {
  return getAllSectionParams();
}

type Props = {
  params: Promise<{ slug: string; section: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug, section } = await params;
  const chapter = getChapterBySlug(slug);
  const sectionTitle = getSectionTitle(slug, section);
  if (!chapter || !sectionTitle) return { title: 'Not Found' };
  return {
    title: `${sectionTitle} — ${chapter.shortTitle} — The Shape of Experience`,
    description: `${sectionTitle} — ${chapter.title}`,
  };
}

export default async function SectionPage({ params }: Props) {
  const { slug, section: sectionId } = await params;
  const chapter = getChapterBySlug(slug);
  if (!chapter) notFound();

  const ChapterContent = await getChapterComponent(slug);
  if (!ChapterContent) notFound();

  const tree = (ChapterContent as (props: object) => ReactElement)({});
  const sectionContent = extractSection(tree, sectionId);
  if (!sectionContent) notFound();

  const sectionTitle = getSectionTitle(slug, sectionId);
  const { prev, next } = getAdjacentSections(slug, sectionId);
  const audioSections = getChapterAudio(slug);

  return (
    <article className="chapter">
      <header className="chapter-header">
        <div className="chapter-subtitle">{chapter.shortTitle}</div>
        <h1 className="chapter-title">{sectionTitle}</h1>
      </header>

      {audioSections.length > 0 && (
        <AudioPlayer
          sections={audioSections}
          chapterTitle={chapter.shortTitle}
          slug={slug}
        />
      )}

      <div className="chapter-content">
        {sectionContent}
      </div>

      <HighlightManager slug={slug} />
      <CommunityHighlights slug={slug} />
      <CommunityConversations slug={slug} />

      <ChapterNav prev={prev} next={next} />
    </article>
  );
}
