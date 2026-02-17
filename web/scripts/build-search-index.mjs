#!/usr/bin/env node
/**
 * Build a search index from TSX content files.
 * Strips JSX tags and extracts text content organized by section.
 * Output: public/search-index.json
 */
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');
const contentDir = join(root, 'content');
const outPath = join(root, 'public', 'search-index.json');

const chapters = [
  { slug: 'introduction', title: 'Introduction' },
  { slug: 'part-1', title: 'Part I: Foundations' },
  { slug: 'part-2', title: 'Part II: Identity Thesis' },
  { slug: 'part-3', title: 'Part III: Affect Signatures' },
  { slug: 'part-4', title: 'Part IV: Social Bonds' },
  { slug: 'part-5', title: 'Part V: Gods' },
  { slug: 'part-6', title: 'Part VI: Transcendence' },
  { slug: 'part-7', title: 'Part VII: Empirical Program' },
  { slug: 'epilogue', title: 'Epilogue' },
  { slug: 'appendix-experiments', title: 'Experiments' },
];

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '')
    .slice(0, 80);
}

function stripTsx(tsx) {
  return tsx
    // Remove JSX string expressions like {"\\math..."}
    .replace(/\{"[^"]*"\}/g, '[math]')
    // Remove JSX tags
    .replace(/<[^>]+>/g, ' ')
    // Remove import/export lines
    .replace(/^(import|export)\s.*/gm, '')
    // Remove function declaration
    .replace(/export default function.*\{/g, '')
    // Clean up
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#?\w+;/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractSections(tsx) {
  const sections = [];
  // Split by <Section title="..."> tags
  const sectionRe = /<Section\s+title="([^"]+)"\s+level=\{(\d)\}/g;
  const matches = [...tsx.matchAll(sectionRe)];

  if (matches.length === 0) {
    // No sections — treat entire content as one section
    const text = stripTsx(tsx);
    if (text.length > 20) {
      sections.push({ headingId: '', heading: '', text: text.slice(0, 2000) });
    }
    return sections;
  }

  for (let i = 0; i < matches.length; i++) {
    const start = matches[i].index;
    const end = i < matches.length - 1 ? matches[i + 1].index : tsx.length;
    const sectionTsx = tsx.slice(start, end);
    const text = stripTsx(sectionTsx);
    const heading = matches[i][1];
    const headingId = slugify(heading);

    if (text.length > 20) {
      sections.push({
        headingId,
        heading,
        text: text.slice(0, 2000),
      });
    }
  }

  return sections;
}

const index = [];

for (const ch of chapters) {
  const tsxPath = join(contentDir, `${ch.slug}.tsx`);
  if (!existsSync(tsxPath)) continue;

  const tsx = readFileSync(tsxPath, 'utf-8');
  const sections = extractSections(tsx);

  index.push({
    slug: ch.slug,
    title: ch.title,
    sections,
  });
}

writeFileSync(outPath, JSON.stringify(index));
console.log(`Search index: ${index.length} chapters, ${index.reduce((a, c) => a + c.sections.length, 0)} sections`);
