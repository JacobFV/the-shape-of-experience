import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { ComponentType } from 'react';

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
  { slug: 'part-4', title: 'Part IV: The Topology of Social Bonds', shortTitle: 'Part IV: Social Bonds' },
  { slug: 'part-5', title: 'Part V: Gods and Superorganisms', shortTitle: 'Part V: Gods' },
  { slug: 'part-6', title: 'Part VI: Historical Consciousness and Transcendence', shortTitle: 'Part VI: Transcendence' },
  { slug: 'part-7', title: 'Part VII: The Empirical Program', shortTitle: 'Part VII: Empirical Program' },
  { slug: 'epilogue', title: 'Epilogue', shortTitle: 'Epilogue' },
  { slug: 'appendix-experiments', title: 'Appendix: Experiment Catalog', shortTitle: 'Experiments' },
];

const chapterModules: Record<string, () => Promise<{ default: ComponentType }>> = {
  'introduction': () => import('../content/introduction'),
  'part-1': () => import('../content/part-1'),
  'part-2': () => import('../content/part-2'),
  'part-3': () => import('../content/part-3'),
  'part-4': () => import('../content/part-4'),
  'part-5': () => import('../content/part-5'),
  'part-6': () => import('../content/part-6'),
  'part-7': () => import('../content/part-7'),
  'epilogue': () => import('../content/epilogue'),
  'appendix-experiments': () => import('../content/appendix-experiments'),
};

export async function getChapterComponent(slug: string): Promise<ComponentType | null> {
  const loader = chapterModules[slug];
  if (!loader) return null;
  const mod = await loader();
  return mod.default;
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
