#!/usr/bin/env node
/**
 * Extract section headings from TSX content files to generate metadata.json for sidebar TOC.
 * Output: public/metadata.json
 */
import { readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');
const contentDir = join(root, 'content');
const outPath = join(root, 'public', 'metadata.json');

// Import canonical chapter data — single source of truth
const chapterDataPath = join(root, 'lib', 'chapter-data.ts');
const chapterDataSrc = readFileSync(chapterDataPath, 'utf-8');
// Extract the array from the TS source (simple regex — the format is stable)
const chapterMatches = [...chapterDataSrc.matchAll(/\{\s*slug:\s*'([^']+)',\s*title:\s*'([^']+)'/g)];
const chapters = chapterMatches.map(m => ({ slug: m[1], title: m[2] }));

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '')
    .slice(0, 80);
}

function extractSections(tsx) {
  const sections = [];
  let currentParent = null;
  // Match <Section title="..." level={N}> or <Section title="..." level={N} id="...">
  const re = /<Section\s+title="([^"]+)"\s+level=\{(\d)\}(?:\s+id="([^"]+)")?/g;
  let match;
  while ((match = re.exec(tsx)) !== null) {
    const title = match[1];
    const level = parseInt(match[2]);
    const id = match[3] || slugify(title);
    const entry = { level, id, text: title };
    if (level === 1) {
      currentParent = id;
    } else if (currentParent) {
      entry.parentSection = currentParent;
    }
    sections.push(entry);
  }
  return sections;
}

const metadata = chapters.map(ch => {
  try {
    const tsx = readFileSync(join(contentDir, `${ch.slug}.tsx`), 'utf-8');
    const sections = extractSections(tsx);
    return { slug: ch.slug, title: ch.title, sections };
  } catch {
    return { slug: ch.slug, title: ch.title, sections: [] };
  }
});

writeFileSync(outPath, JSON.stringify(metadata, null, 2));
console.log(`TOC metadata: ${metadata.length} chapters, ${metadata.reduce((a, c) => a + c.sections.length, 0)} sections`);
