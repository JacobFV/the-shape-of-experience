import { readFileSync } from 'fs';
import { join } from 'path';
import { ReactElement, ReactNode, isValidElement, Children } from 'react';
import { Section } from '@/components/content';
import { chapters } from './chapters';

interface SectionMeta {
  level: number;
  id: string;
  text: string;
  parentSection?: string;
}

interface ChapterMeta {
  slug: string;
  title: string;
  sections: SectionMeta[];
}

let _metadataCache: ChapterMeta[] | null = null;

function loadMetadata(): ChapterMeta[] {
  if (_metadataCache) return _metadataCache;
  _metadataCache = JSON.parse(
    readFileSync(join(process.cwd(), 'public', 'metadata.json'), 'utf-8')
  );
  return _metadataCache!;
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
}

/** Extract a single section from a content component's JSX tree.
 *  Supports both level-1 (top-level) and level-2 (nested) sections. */
export function extractSection(tree: ReactElement, sectionId: string): ReactNode[] | null {
  const treeProps = tree.props as { children?: ReactNode };
  const children = Children.toArray(treeProps.children);

  let firstSectionIndex = -1;
  let matchIndex = -1;

  // First pass: look for a level-1 match
  for (let i = 0; i < children.length; i++) {
    const child = children[i];
    if (isValidElement(child) && child.type === Section) {
      const childProps = child.props as { level?: number; id?: string; title: string };
      if (childProps.level === 1) {
        if (firstSectionIndex === -1) firstSectionIndex = i;
        const id = childProps.id || slugify(childProps.title);
        if (id === sectionId) {
          matchIndex = i;
          break;
        }
      }
    }
  }

  if (matchIndex !== -1) {
    const result: ReactNode[] = [];
    // For the first section, include preamble content (e.g. <Logos>)
    if (matchIndex === firstSectionIndex) {
      for (let i = 0; i < matchIndex; i++) {
        result.push(children[i]);
      }
    }
    result.push(children[matchIndex]);
    return result;
  }

  // Second pass: look inside level-1 sections for a level-2 match
  for (let i = 0; i < children.length; i++) {
    const child = children[i];
    if (isValidElement(child) && child.type === Section) {
      const childProps = child.props as { level?: number; id?: string; title: string; children?: ReactNode };
      if (childProps.level === 1) {
        const innerChildren = Children.toArray(childProps.children);
        for (const inner of innerChildren) {
          if (isValidElement(inner) && inner.type === Section) {
            const innerProps = inner.props as { level?: number; id?: string; title: string };
            if (innerProps.level === 2) {
              const id = innerProps.id || slugify(innerProps.title);
              if (id === sectionId) {
                return [inner];
              }
            }
          }
        }
      }
    }
  }

  return null;
}

/** Get level-1 sections for a chapter */
export function getChapterSections(slug: string): SectionMeta[] {
  const metadata = loadMetadata();
  const chapter = metadata.find((ch) => ch.slug === slug);
  if (!chapter) return [];
  return chapter.sections.filter((s) => s.level === 1);
}

/** Get all routable sections (level-1 and level-2) for a chapter */
export function getAllRoutableSections(slug: string): SectionMeta[] {
  const metadata = loadMetadata();
  const chapter = metadata.find((ch) => ch.slug === slug);
  if (!chapter) return [];
  return chapter.sections.filter((s) => s.level === 1 || s.level === 2);
}

/** Get the first level-1 section id for a chapter (for redirects) */
export function getFirstSectionId(slug: string): string | null {
  const sections = getChapterSections(slug);
  return sections.length > 0 ? sections[0].id : null;
}

/** Get section title by id (searches all levels) */
export function getSectionTitle(slug: string, sectionId: string): string | null {
  const metadata = loadMetadata();
  const chapter = metadata.find((ch) => ch.slug === slug);
  if (!chapter) return null;
  const section = chapter.sections.find((s) => s.id === sectionId);
  return section ? section.text : null;
}

/** Navigation entry for prev/next */
export interface NavEntry {
  href: string;
  label: string;
}

/** Build flat navigation list across all chapters and sections */
function buildNavList(): NavEntry[] {
  const metadata = loadMetadata();
  const entries: NavEntry[] = [];

  for (const chapter of metadata) {
    const routableSections = chapter.sections.filter((s) => s.level === 1 || s.level === 2);
    const chapterInfo = chapters.find((c) => c.slug === chapter.slug);

    if (routableSections.length === 0) {
      entries.push({
        href: `/${chapter.slug}`,
        label: chapterInfo?.shortTitle || chapter.title,
      });
    } else {
      for (const s of routableSections) {
        entries.push({
          href: `/${chapter.slug}/${s.id}`,
          label: s.text,
        });
      }
    }
  }

  return entries;
}

/** Get prev/next for a section page */
export function getAdjacentSections(
  slug: string,
  sectionId: string
): { prev?: NavEntry; next?: NavEntry } {
  const navList = buildNavList();
  const currentHref = `/${slug}/${sectionId}`;
  const idx = navList.findIndex((e) => e.href === currentHref);

  if (idx === -1) return {};

  return {
    prev: idx > 0 ? navList[idx - 1] : undefined,
    next: idx < navList.length - 1 ? navList[idx + 1] : undefined,
  };
}

/** Get prev/next for a chapter page (for chapters without sections, e.g. introduction) */
export function getChapterNav(slug: string): { prev?: NavEntry; next?: NavEntry } {
  const navList = buildNavList();
  const idx = navList.findIndex((e) => e.href === `/${slug}`);
  if (idx === -1) return {};
  return {
    prev: idx > 0 ? navList[idx - 1] : undefined,
    next: idx < navList.length - 1 ? navList[idx + 1] : undefined,
  };
}

/** Get all chapter-section params for generateStaticParams */
export function getAllSectionParams(): { slug: string; section: string }[] {
  const metadata = loadMetadata();
  const params: { slug: string; section: string }[] = [];

  for (const chapter of metadata) {
    const routableSections = chapter.sections.filter((s) => s.level === 1 || s.level === 2);
    for (const s of routableSections) {
      params.push({ slug: chapter.slug, section: s.id });
    }
  }

  return params;
}

/** Map a subsection id to its parent level-1 section id */
export function getSectionParent(slug: string, subsectionId: string): string | null {
  const metadata = loadMetadata();
  const chapter = metadata.find((ch) => ch.slug === slug);
  if (!chapter) return null;

  const section = chapter.sections.find((s) => s.id === subsectionId);
  if (!section) return null;

  return section.parentSection || null;
}
