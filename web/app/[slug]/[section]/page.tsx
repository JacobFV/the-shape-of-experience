import { chapters, getChapterComponent, getChapterBySlug, getChapterAudio } from '@/lib/chapters';
import {
  extractSection,
  getAllSectionParams,
  getSectionTitle,
  getAdjacentSections,
  getFirstSectionId,
} from '@/lib/sections';

function getNextChapterHref(slug: string): string | undefined {
  const idx = chapters.findIndex(c => c.slug === slug);
  const next = idx >= 0 && idx < chapters.length - 1 ? chapters[idx + 1] : undefined;
  if (!next) return undefined;
  const firstSection = getFirstSectionId(next.slug);
  return firstSection ? `/${next.slug}/${firstSection}` : `/${next.slug}`;
}
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
  const nextChapterHref = getNextChapterHref(slug);

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
          nextChapterHref={nextChapterHref}
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
