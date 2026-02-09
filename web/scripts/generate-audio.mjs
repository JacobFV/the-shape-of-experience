/**
 * Generate TTS audio for each chapter section using Deepgram Aura-2.
 *
 * Usage: DEEPGRAM_API_KEY=... node scripts/generate-audio.mjs [slug]
 *
 * Reads generated HTML files, splits by <h1> boundaries, strips markup,
 * calls Deepgram REST API, writes MP3 + manifest.json.
 *
 * Caching: stores text hashes alongside MP3s; skips unchanged sections.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { createHash } from 'crypto';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const GEN = resolve(ROOT, 'generated', 'chapters');
const AUDIO_OUT = resolve(ROOT, 'public', 'audio');

const API_KEY = process.env.DEEPGRAM_API_KEY;
const MODEL = 'aura-2-thalia-en';
const CHUNK_LIMIT = 1900; // chars per API call (leave margin under 2000)

const CHAPTERS = [
  'introduction', 'part-1', 'part-2', 'part-3', 'part-4', 'part-5', 'epilogue',
];

// --- Text cleaning ---

function stripHtml(html) {
  // Remove HTML tags
  let text = html.replace(/<[^>]*>/g, ' ');
  // Decode common entities
  text = text.replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, ' ')
    .replace(/&#\d+;/g, ' ')
    .replace(/&\w+;/g, ' ');
  return text;
}

function stripMath(text) {
  // Remove display math: $$...$$ and \[...\]
  text = text.replace(/\$\$[\s\S]*?\$\$/g, ' ');
  text = text.replace(/\\\[[\s\S]*?\\\]/g, ' ');
  // Remove inline math: $...$ and \(...\)
  text = text.replace(/\$[^$]+\$/g, ' ');
  text = text.replace(/\\\([\s\S]*?\\\)/g, ' ');
  // Remove stray LaTeX commands
  text = text.replace(/\\[a-zA-Z]+\{[^}]*\}/g, ' ');
  text = text.replace(/\\[a-zA-Z]+/g, ' ');
  return text;
}

function cleanTextForTts(html) {
  let text = stripHtml(html);
  text = stripMath(text);
  // Remove figure references like [fig:X]
  text = text.replace(/\[fig:[^\]]*\]/g, '');
  // Remove footnote markers like [1], [2]
  text = text.replace(/\[\d+\]/g, '');
  // Collapse whitespace
  text = text.replace(/\s+/g, ' ').trim();
  return text;
}

// --- Section splitting ---

function splitSections(html, slug) {
  const sections = [];
  // Split at <h1 boundaries
  const h1Regex = /<h1\s[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/h1>/gi;
  const matches = [...html.matchAll(h1Regex)];

  if (matches.length === 0) {
    // No h1 headers (e.g., introduction) — treat whole content as one section
    const text = cleanTextForTts(html);
    if (text.length >= 50) {
      sections.push({ id: 'full', title: capitalizeSlug(slug), text });
    }
    return sections;
  }

  // Content before first h1
  const preContent = html.slice(0, matches[0].index);
  const preText = cleanTextForTts(preContent);
  if (preText.length >= 50) {
    sections.push({ id: 'intro', title: 'Introduction', text: preText });
  }

  // Each h1 section
  for (let i = 0; i < matches.length; i++) {
    const match = matches[i];
    const id = match[1];
    const titleHtml = match[2];
    const title = stripHtml(titleHtml).trim();

    const start = match.index;
    const end = i + 1 < matches.length ? matches[i + 1].index : html.length;
    const sectionHtml = html.slice(start, end);
    const text = cleanTextForTts(sectionHtml);

    if (text.length >= 50) {
      sections.push({ id, title, text });
    }
  }

  return sections;
}

function capitalizeSlug(slug) {
  return slug.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

// --- Chunking ---

function chunkText(text, limit = CHUNK_LIMIT) {
  if (text.length <= limit) return [text];

  const chunks = [];
  let remaining = text;

  while (remaining.length > limit) {
    // Find last sentence boundary within limit
    let breakPoint = -1;
    const searchText = remaining.slice(0, limit);

    // Try period, then semicolon, then comma, then space
    for (const sep of ['. ', '! ', '? ', '; ', ', ', ' ']) {
      const idx = searchText.lastIndexOf(sep);
      if (idx > limit * 0.3) {
        breakPoint = idx + sep.length;
        break;
      }
    }

    if (breakPoint === -1) breakPoint = limit;

    chunks.push(remaining.slice(0, breakPoint).trim());
    remaining = remaining.slice(breakPoint).trim();
  }

  if (remaining.length > 0) {
    chunks.push(remaining);
  }

  return chunks;
}

// --- Deepgram API ---

async function synthesizeChunk(text) {
  const url = `https://api.deepgram.com/v1/speak?model=${MODEL}&encoding=mp3`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Token ${API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Deepgram API error ${res.status}: ${errText}`);
  }

  return Buffer.from(await res.arrayBuffer());
}

async function synthesizeSection(text, sectionId, slugDir) {
  const chunks = chunkText(text);
  const buffers = [];

  for (let i = 0; i < chunks.length; i++) {
    console.log(`    Chunk ${i + 1}/${chunks.length} (${chunks[i].length} chars)...`);
    const buf = await synthesizeChunk(chunks[i]);
    buffers.push(buf);
    // Small delay to avoid rate limiting
    if (i < chunks.length - 1) {
      await new Promise(r => setTimeout(r, 250));
    }
  }

  // Concatenate MP3 buffers (MP3 frames are independently decodable)
  return Buffer.concat(buffers);
}

// --- Hashing / caching ---

function textHash(text) {
  return createHash('sha256').update(text).digest('hex').slice(0, 16);
}

function isCached(hashPath, currentHash) {
  if (!existsSync(hashPath)) return false;
  const stored = readFileSync(hashPath, 'utf-8').trim();
  return stored === currentHash;
}

// --- Main ---

async function processChapter(slug) {
  const htmlPath = resolve(GEN, `${slug}.html`);
  if (!existsSync(htmlPath)) {
    console.warn(`  Skipping ${slug}: HTML file not found`);
    return [];
  }

  const html = readFileSync(htmlPath, 'utf-8');
  const sections = splitSections(html, slug);

  if (sections.length === 0) {
    console.log(`  ${slug}: no sections with enough text`);
    return [];
  }

  const slugDir = resolve(AUDIO_OUT, slug);
  mkdirSync(slugDir, { recursive: true });

  const results = [];

  for (const section of sections) {
    const mp3Path = resolve(slugDir, `${section.id}.mp3`);
    const hashPath = resolve(slugDir, `${section.id}.hash`);
    const hash = textHash(section.text);

    if (isCached(hashPath, hash) && existsSync(mp3Path)) {
      console.log(`  [cached] ${section.id}: "${section.title}"`);
    } else {
      console.log(`  [generate] ${section.id}: "${section.title}" (${section.text.length} chars)`);
      const mp3 = await synthesizeSection(section.text, section.id, slugDir);
      writeFileSync(mp3Path, mp3);
      writeFileSync(hashPath, hash);
    }

    results.push({
      id: section.id,
      title: section.title,
      audioUrl: `/audio/${slug}/${section.id}.mp3`,
    });
  }

  return results;
}

async function main() {
  if (!API_KEY) {
    console.error('DEEPGRAM_API_KEY environment variable is required.');
    console.error('Set it in web/.env.local or export it.');
    process.exit(1);
  }

  // Allow filtering to specific slug(s) via CLI args
  const args = process.argv.slice(2);
  const slugs = args.length > 0 ? args.filter(s => CHAPTERS.includes(s)) : CHAPTERS;

  if (slugs.length === 0) {
    console.error(`Invalid slug(s). Available: ${CHAPTERS.join(', ')}`);
    process.exit(1);
  }

  console.log('=== Audio Generation (Deepgram Aura-2) ===\n');
  mkdirSync(AUDIO_OUT, { recursive: true });

  // Load existing manifest if we're doing partial generation
  let manifest = {};
  const manifestPath = resolve(AUDIO_OUT, 'manifest.json');
  if (existsSync(manifestPath)) {
    try {
      manifest = JSON.parse(readFileSync(manifestPath, 'utf-8'));
    } catch { /* start fresh */ }
  }

  for (const slug of slugs) {
    console.log(`\n${slug}:`);
    const sections = await processChapter(slug);
    if (sections.length > 0) {
      manifest[slug] = sections;
    }
  }

  // Write manifest
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest written to ${manifestPath}`);

  // Summary
  const totalSections = Object.values(manifest).reduce((sum, s) => sum + s.length, 0);
  console.log(`\n=== Done: ${totalSections} sections across ${Object.keys(manifest).length} chapters ===`);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
