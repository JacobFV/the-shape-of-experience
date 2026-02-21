import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { ComponentType } from 'react';

// Import from client-safe module and re-export
import { chapters, type Chapter } from './chapter-data';
export { chapters, type Chapter };

export interface AudioSection {
  id: string;
  title: string;
  audioUrl: string;
}

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
