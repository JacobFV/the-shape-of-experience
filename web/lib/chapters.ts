import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

export interface Chapter {
  slug: string;
  title: string;
  shortTitle: string;
}

export interface AudioSection {
  id: string;
  title: string;
  audioUrl: string;
}

export const chapters: Chapter[] = [
  { slug: 'introduction', title: 'Introduction', shortTitle: 'Introduction' },
  { slug: 'part-1', title: 'Part I: Thermodynamic Foundations and the Ladder of Emergence', shortTitle: 'Part I: Foundations' },
  { slug: 'part-2', title: 'Part II: The Identity Thesis and the Geometry of Feeling', shortTitle: 'Part II: Identity Thesis' },
  { slug: 'part-3', title: 'Part III: Signatures of Affect Under the Existential Burden', shortTitle: 'Part III: Affect Signatures' },
  { slug: 'part-4', title: 'Part IV: Interventions Across Scale', shortTitle: 'Part IV: Interventions' },
  { slug: 'part-5', title: 'Part V: The Transcendence of the Self', shortTitle: 'Part V: Transcendence' },
  { slug: 'epilogue', title: 'Epilogue', shortTitle: 'Epilogue' },
];

export function getChapterHtml(slug: string): string {
  const htmlPath = join(process.cwd(), 'generated', 'chapters', `${slug}.html`);
  if (!existsSync(htmlPath)) {
    return '<p>Chapter not found.</p>';
  }
  return readFileSync(htmlPath, 'utf-8');
}

export function getChapterBySlug(slug: string): Chapter | undefined {
  return chapters.find(ch => ch.slug === slug);
}

export function getAdjacentChapters(slug: string): { prev?: Chapter; next?: Chapter } {
  const idx = chapters.findIndex(ch => ch.slug === slug);
  return {
    prev: idx > 0 ? chapters[idx - 1] : undefined,
    next: idx < chapters.length - 1 ? chapters[idx + 1] : undefined,
  };
}

export function getChapterAudio(slug: string): AudioSection[] {
  const manifestPath = join(process.cwd(), 'public', 'audio', 'manifest.json');
  if (!existsSync(manifestPath)) return [];
  try {
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf-8'));
    return manifest[slug] ?? [];
  } catch {
    return [];
  }
}
