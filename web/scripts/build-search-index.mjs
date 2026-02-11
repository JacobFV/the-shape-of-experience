#!/usr/bin/env node
/**
 * Build a search index from generated HTML chapter files.
 * Output: public/search-index.json
 */
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');
const chaptersDir = join(root, 'generated', 'chapters');
const outPath = join(root, 'public', 'search-index.json');

const chapters = [
  { slug: 'introduction', title: 'Introduction' },
  { slug: 'part-1', title: 'Part I: Foundations' },
  { slug: 'part-2', title: 'Part II: Identity Thesis' },
  { slug: 'part-3', title: 'Part III: Affect Signatures' },
  { slug: 'part-4', title: 'Part IV: Interventions' },
  { slug: 'part-5', title: 'Part V: Transcendence' },
  { slug: 'epilogue', title: 'Epilogue' },
];

function stripHtml(html) {
  return html
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#?\w+;/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractSections(html) {
  const sections = [];
  // Split by h2/h3 headings
  const headingRegex = /<h([23])\s+id="([^"]*)"[^>]*>([\s\S]*?)<\/h[23]>/gi;
  let lastIdx = 0;
  let lastHeading = { id: '', text: '', level: 1 };
  let match;

  // Collect all heading positions
  const headings = [];
  while ((match = headingRegex.exec(html)) !== null) {
    headings.push({
      index: match.index,
      level: parseInt(match[1]),
      id: match[2],
      text: stripHtml(match[3]),
    });
  }

  for (let i = 0; i < headings.length; i++) {
    const start = i === 0 ? 0 : headings[i].index;
    const end = i < headings.length - 1 ? headings[i + 1].index : html.length;
    const sectionHtml = html.slice(start, end);
    const text = stripHtml(sectionHtml);

    if (text.length > 20) {
      sections.push({
        headingId: headings[i].id,
        heading: headings[i].text,
        // Store first 500 chars for context, full text for search
        text: text.slice(0, 2000),
      });
    }
  }

  // If no headings found, treat entire content as one section
  if (sections.length === 0) {
    const text = stripHtml(html);
    if (text.length > 20) {
      sections.push({ headingId: '', heading: '', text: text.slice(0, 2000) });
    }
  }

  return sections;
}

const index = [];

for (const ch of chapters) {
  const htmlPath = join(chaptersDir, `${ch.slug}.html`);
  if (!existsSync(htmlPath)) continue;

  const html = readFileSync(htmlPath, 'utf-8');
  const sections = extractSections(html);

  index.push({
    slug: ch.slug,
    title: ch.title,
    sections,
  });
}

writeFileSync(outPath, JSON.stringify(index));
console.log(`Search index: ${index.length} chapters, ${index.reduce((a, c) => a + c.sections.length, 0)} sections`);
