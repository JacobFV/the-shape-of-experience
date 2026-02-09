'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';

interface Section {
  level: number;
  id: string;
  text: string;
}

interface ChapterToc {
  slug: string;
  shortTitle: string;
  sections: Section[];
}

const chapters: ChapterToc[] = [
  { slug: 'introduction', shortTitle: 'Introduction', sections: [] },
  { slug: 'part-1', shortTitle: 'Part I: Foundations', sections: [] },
  { slug: 'part-2', shortTitle: 'Part II: Identity Thesis', sections: [] },
  { slug: 'part-3', shortTitle: 'Part III: Affect Signatures', sections: [] },
  { slug: 'part-4', shortTitle: 'Part IV: Interventions', sections: [] },
  { slug: 'part-5', shortTitle: 'Part V: Transcendence', sections: [] },
  { slug: 'epilogue', shortTitle: 'Epilogue', sections: [] },
];

// Load section data from generated metadata at module level
let sectionsLoaded = false;
function loadSections() {
  if (sectionsLoaded) return;
  try {
    // metadata.json is copied to public/ during build
    // But since this is client-side, we'll fetch it
  } catch { /* ignore */ }
}

export default function TableOfContents({
  onNavigate,
  sectionData,
}: {
  onNavigate?: () => void;
  sectionData?: Record<string, Section[]>;
}) {
  const pathname = usePathname();
  const currentSlug = pathname === '/' ? '' : pathname.slice(1);
  const [expandedSlug, setExpandedSlug] = useState<string | null>(currentSlug || null);

  const handleChapterClick = (slug: string) => {
    setExpandedSlug(expandedSlug === slug ? null : slug);
  };

  return (
    <nav className="toc">
      <div className="toc-title">Contents</div>
      <ul>
        {chapters.map(ch => {
          const isActive = currentSlug === ch.slug;
          const isExpanded = expandedSlug === ch.slug;
          const sections = sectionData?.[ch.slug] || [];
          const hasSections = sections.length > 0;

          return (
            <li key={ch.slug} className={isActive ? 'active' : ''}>
              <div className="toc-chapter-row">
                <Link href={`/${ch.slug}`} onClick={onNavigate}>
                  {ch.shortTitle}
                </Link>
                {hasSections && (
                  <button
                    className="toc-expand"
                    onClick={(e) => {
                      e.preventDefault();
                      handleChapterClick(ch.slug);
                    }}
                    aria-label={isExpanded ? 'Collapse sections' : 'Expand sections'}
                  >
                    {isExpanded ? '\u25BE' : '\u25B8'}
                  </button>
                )}
              </div>
              {hasSections && isExpanded && (
                <ul className="toc-sections">
                  {sections.map((s) => (
                    <li key={s.id} className={s.level === 2 ? 'toc-subsection' : ''}>
                      <Link
                        href={`/${ch.slug}#${s.id}`}
                        onClick={onNavigate}
                      >
                        {s.text}
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
      <div className="toc-pdf">
        <a href="/book.pdf" target="_blank" rel="noopener noreferrer">
          PDF Version
        </a>
      </div>
    </nav>
  );
}
