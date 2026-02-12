'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState, useEffect, useRef } from 'react';

interface Section {
  level: number;
  id: string;
  text: string;
  parentSection?: string;
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

export default function TableOfContents({
  onNavigate,
  sectionData,
}: {
  onNavigate?: () => void;
  sectionData?: Record<string, Section[]>;
}) {
  const pathname = usePathname();

  // Parse pathname: /part-1/section-id → slug=part-1, currentSection=section-id
  const pathParts = pathname.slice(1).split('/');
  const currentSlug = pathParts[0] || '';
  const currentSection = pathParts[1] || '';

  const [expandedSlug, setExpandedSlug] = useState<string | null>(currentSlug || null);
  const [activeSubsectionId, setActiveSubsectionId] = useState<string | null>(null);
  const rafRef = useRef(0);

  // Expand chapter when navigating to it
  useEffect(() => {
    if (currentSlug) setExpandedSlug(currentSlug);
  }, [currentSlug]);

  // Track which subsection the user is scrolled to (within a section page)
  useEffect(() => {
    if (!currentSection) {
      setActiveSubsectionId(null);
      return;
    }

    const sections = sectionData?.[currentSlug];
    if (!sections?.length) {
      setActiveSubsectionId(null);
      return;
    }

    // Get level-2 sections that belong to the current section page
    const subsections = sections.filter(
      (s) => s.level === 2 && s.parentSection === currentSection
    );
    if (subsections.length === 0) {
      setActiveSubsectionId(null);
      return;
    }

    function updateActive() {
      const scrollY = window.scrollY + 120;
      let lastL2: string | null = null;

      for (const s of subsections) {
        const el = document.getElementById(s.id);
        if (!el) continue;
        if (el.offsetTop > scrollY) break;
        lastL2 = s.id;
      }

      setActiveSubsectionId((prev) => (prev === lastL2 ? prev : lastL2));
    }

    function onScroll() {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(updateActive);
    }

    updateActive();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', onScroll);
      cancelAnimationFrame(rafRef.current);
    };
  }, [currentSlug, currentSection, sectionData]);

  const handleChapterClick = (slug: string) => {
    setExpandedSlug(expandedSlug === slug ? null : slug);
  };

  return (
    <nav className="toc">
      <Link href="/" className="toc-home" onClick={onNavigate}>The Shape of Experience</Link>
      <ul>
        {chapters.map(ch => {
          const isChapterActive = currentSlug === ch.slug;
          const isExpanded = expandedSlug === ch.slug;
          const sections = sectionData?.[ch.slug] || [];
          const level1Sections = sections.filter(s => s.level === 1);
          const hasSections = level1Sections.length > 0;

          return (
            <li key={ch.slug} className={isChapterActive ? 'active' : ''}>
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
                  {sections.map((s) => {
                    if (s.level === 1) {
                      // Level-1 sections link to their own page
                      const isActive = isChapterActive && currentSection === s.id;
                      return (
                        <li
                          key={s.id}
                          className={isActive ? 'toc-active' : ''}
                        >
                          <Link
                            href={`/${ch.slug}/${s.id}`}
                            onClick={onNavigate}
                          >
                            {s.text}
                          </Link>
                        </li>
                      );
                    } else if (s.level === 2) {
                      // Level-2 sections link to parent section page with hash
                      const parentId = s.parentSection;
                      const isOnParentPage = isChapterActive && currentSection === parentId;
                      const isActive = isOnParentPage && activeSubsectionId === s.id;
                      return (
                        <li
                          key={s.id}
                          className={
                            'toc-subsection' + (isActive ? ' toc-active' : '')
                          }
                        >
                          <Link
                            href={`/${ch.slug}/${parentId}#${s.id}`}
                            onClick={onNavigate}
                          >
                            {s.text}
                          </Link>
                        </li>
                      );
                    }
                    // Level 3+ not shown in TOC
                    return null;
                  })}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
      <div className="toc-pdf">
        <a href="/book.pdf" target="_blank" rel="noopener noreferrer">
          <img src="/images/cover-page.png" alt="Book cover page" className="toc-pdf-cover" />
          <span>PDF Version</span>
        </a>
      </div>
      <div className="toc-socials">
        <a href="https://github.com/JacobFV" target="_blank" rel="noopener noreferrer" aria-label="GitHub" title="GitHub">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
        </a>
        <a href="https://twitter.com/jvboiid" target="_blank" rel="noopener noreferrer" aria-label="Twitter" title="Twitter">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
        </a>
        <a href="https://linkedin.com/in/jacob-f-valdez" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" title="LinkedIn">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
        </a>
        <a href="https://jvboid.dev" target="_blank" rel="noopener noreferrer" aria-label="Website" title="Website">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>
        </a>
      </div>
      <div className="toc-legal">
        <Link href="/terms" onClick={onNavigate}>Terms</Link>
        <span>&middot;</span>
        <Link href="/privacy" onClick={onNavigate}>Privacy</Link>
      </div>
    </nav>
  );
}
