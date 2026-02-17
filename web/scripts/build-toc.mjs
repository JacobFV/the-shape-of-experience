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

const chapters = [
  { slug: 'introduction', title: 'Introduction' },
  { slug: 'part-1', title: 'Part I: Thermodynamic Foundations and the Ladder of Emergence' },
  { slug: 'part-2', title: 'Part II: The Identity Thesis and the Geometry of Feeling' },
  { slug: 'part-3', title: 'Part III: Signatures of Affect Under the Existential Burden' },
  { slug: 'part-4', title: 'Part IV: The Topology of Social Bonds' },
  { slug: 'part-5', title: 'Part V: Gods and Superorganisms' },
  { slug: 'part-6', title: 'Part VI: Historical Consciousness and Transcendence' },
  { slug: 'part-7', title: 'Part VII: The Empirical Program' },
  { slug: 'epilogue', title: 'Epilogue' },
  { slug: 'appendix-experiments', title: 'Appendix: Experiment Catalog' },
];

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
