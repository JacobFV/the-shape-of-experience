import type { MetadataRoute } from 'next';
import { readFileSync } from 'fs';
import { join } from 'path';

const BASE = 'https://the-shape-of-experience.vercel.app';

interface SectionMeta {
  level: number;
  id: string;
  text: string;
}

interface ChapterMeta {
  slug: string;
  title: string;
  sections: SectionMeta[];
}

export default function sitemap(): MetadataRoute.Sitemap {
  const pages = [
    '',
    '/introduction',
    '/terms',
    '/privacy',
  ];

  const entries: MetadataRoute.Sitemap = pages.map((path) => ({
    url: `${BASE}${path}`,
    lastModified: new Date(),
    changeFrequency: path === '/terms' || path === '/privacy' ? 'yearly' : 'monthly',
    priority: path === '' ? 1.0 : path === '/terms' || path === '/privacy' ? 0.2 : 0.8,
  }));

  // Add section-level URLs from metadata
  try {
    const metadata: ChapterMeta[] = JSON.parse(
      readFileSync(join(process.cwd(), 'public', 'metadata.json'), 'utf-8')
    );
    for (const chapter of metadata) {
      const level1Sections = chapter.sections.filter((s) => s.level === 1);
      if (level1Sections.length > 0) {
        for (const s of level1Sections) {
          entries.push({
            url: `${BASE}/${chapter.slug}/${s.id}`,
            lastModified: new Date(),
            changeFrequency: 'monthly',
            priority: 0.8,
          });
        }
      }
    }
  } catch {
    // If metadata.json doesn't exist, skip section URLs
  }

  return entries;
}
