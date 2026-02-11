/**
 * Generate TTS audio for each chapter section.
 *
 * Usage: TTS_PROVIDER=openai OPENAI_API_KEY=... node scripts/generate-audio.mjs [slug]
 *
 * Reads generated HTML files, splits by <h1> boundaries, strips markup,
 * calls TTS API with parallel concurrency, writes MP3 + manifest.json.
 *
 * Caching: stores text hashes alongside MP3s; skips unchanged sections.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { createHash } from 'crypto';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const GEN = resolve(ROOT, 'generated', 'chapters');
const AUDIO_OUT = resolve(ROOT, 'public', 'audio');

const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const TTS_PROVIDER = process.env.TTS_PROVIDER || (DEEPGRAM_API_KEY ? 'deepgram' : 'openai');

const DEEPGRAM_MODEL = 'aura-2-thalia-en';
const OPENAI_MODEL = process.env.OPENAI_TTS_MODEL || 'tts-1';
const OPENAI_VOICE = process.env.OPENAI_TTS_VOICE || 'nova';
const CHUNK_LIMIT = TTS_PROVIDER === 'openai' ? 4000 : 1900;
const CONCURRENCY = parseInt(process.env.TTS_CONCURRENCY || '15', 10);
const MAX_RETRIES = 3;

const CHAPTERS = [
  'introduction', 'part-1', 'part-2', 'part-3', 'part-4', 'part-5', 'epilogue',
];

// --- Concurrency pool ---

function createPool(concurrency) {
  let active = 0;
  const queue = [];

  function drain() {
    while (queue.length > 0 && active < concurrency) {
      active++;
      const { fn, resolve: res, reject: rej } = queue.shift();
      fn().then(res, rej).finally(() => {
        active--;
        drain();
      });
    }
  }

  return function run(fn) {
    return new Promise((resolve, reject) => {
      queue.push({ fn, resolve, reject });
      drain();
    });
  };
}

const pool = createPool(CONCURRENCY);

// --- Text cleaning ---

function stripHtml(html) {
  let text = html.replace(/<[^>]*>/g, ' ');
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
  text = text.replace(/\$\$[\s\S]*?\$\$/g, ' ');
  text = text.replace(/\\\[[\s\S]*?\\\]/g, ' ');
  text = text.replace(/\$[^$]+\$/g, ' ');
  text = text.replace(/\\\([\s\S]*?\\\)/g, ' ');
  text = text.replace(/\\[a-zA-Z]+\{[^}]*\}/g, ' ');
  text = text.replace(/\\[a-zA-Z]+/g, ' ');
  return text;
}

function cleanTextForTts(html) {
  let text = stripHtml(html);
  text = stripMath(text);
  text = text.replace(/\[fig:[^\]]*\]/g, '');
  text = text.replace(/\[\d+\]/g, '');
  text = text.replace(/\s+/g, ' ').trim();
  return text;
}

// --- Section splitting ---

function splitSections(html, slug) {
  const sections = [];
  const h1Regex = /<h1\s[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/h1>/gi;
  const matches = [...html.matchAll(h1Regex)];

  if (matches.length === 0) {
    const text = cleanTextForTts(html);
    if (text.length >= 50) {
      sections.push({ id: 'full', title: capitalizeSlug(slug), text });
    }
    return sections;
  }

  const preContent = html.slice(0, matches[0].index);
  const preText = cleanTextForTts(preContent);
  if (preText.length >= 50) {
    sections.push({ id: 'intro', title: 'Introduction', text: preText });
  }

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
    let breakPoint = -1;
    const searchText = remaining.slice(0, limit);

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

// --- TTS API with retry ---

async function synthesizeChunkWithRetry(text) {
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      return TTS_PROVIDER === 'openai'
        ? await synthesizeChunkOpenAI(text)
        : await synthesizeChunkDeepgram(text);
    } catch (err) {
      if (attempt === MAX_RETRIES) throw err;
      const delay = 1000 * Math.pow(2, attempt - 1);
      console.warn(`    Retry ${attempt}/${MAX_RETRIES} after ${delay}ms: ${err.message}`);
      await new Promise(r => setTimeout(r, delay));
    }
  }
}

async function synthesizeChunkDeepgram(text) {
  const url = `https://api.deepgram.com/v1/speak?model=${DEEPGRAM_MODEL}&encoding=mp3`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Token ${DEEPGRAM_API_KEY}`,
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

async function synthesizeChunkOpenAI(text) {
  const res = await fetch('https://api.openai.com/v1/audio/speech', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: OPENAI_MODEL,
      voice: OPENAI_VOICE,
      input: text,
      response_format: 'mp3',
    }),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`OpenAI TTS API error ${res.status}: ${errText}`);
  }

  return Buffer.from(await res.arrayBuffer());
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

async function main() {
  if (TTS_PROVIDER === 'deepgram' && !DEEPGRAM_API_KEY) {
    console.error('DEEPGRAM_API_KEY environment variable is required for Deepgram TTS.');
    process.exit(1);
  }
  if (TTS_PROVIDER === 'openai' && !OPENAI_API_KEY) {
    console.error('OPENAI_API_KEY environment variable is required for OpenAI TTS.');
    process.exit(1);
  }

  const args = process.argv.slice(2);
  const slugs = args.length > 0 ? args.filter(s => CHAPTERS.includes(s)) : CHAPTERS;

  if (slugs.length === 0) {
    console.error(`Invalid slug(s). Available: ${CHAPTERS.join(', ')}`);
    process.exit(1);
  }

  const providerLabel = TTS_PROVIDER === 'openai'
    ? `OpenAI ${OPENAI_MODEL} / ${OPENAI_VOICE}`
    : `Deepgram ${DEEPGRAM_MODEL}`;
  console.log(`=== Audio Generation (${providerLabel}, concurrency=${CONCURRENCY}) ===\n`);
  mkdirSync(AUDIO_OUT, { recursive: true });

  // Load existing manifest
  let manifest = {};
  const manifestPath = resolve(AUDIO_OUT, 'manifest.json');
  if (existsSync(manifestPath)) {
    try {
      manifest = JSON.parse(readFileSync(manifestPath, 'utf-8'));
    } catch { /* start fresh */ }
  }

  // Phase 1: Collect all work items across all chapters
  const workItems = []; // { slug, section, mp3Path, hashPath, hash, chunks }
  const metadataItems = []; // { slug, section } — for manifest (cached + generated)

  for (const slug of slugs) {
    const htmlPath = resolve(GEN, `${slug}.html`);
    if (!existsSync(htmlPath)) {
      console.warn(`Skipping ${slug}: HTML file not found`);
      continue;
    }

    const html = readFileSync(htmlPath, 'utf-8');
    const sections = splitSections(html, slug);

    if (sections.length === 0) {
      console.log(`${slug}: no sections with enough text`);
      continue;
    }

    const slugDir = resolve(AUDIO_OUT, slug);
    mkdirSync(slugDir, { recursive: true });

    for (const section of sections) {
      const mp3Path = resolve(slugDir, `${section.id}.mp3`);
      const hashPath = resolve(slugDir, `${section.id}.hash`);
      const hash = textHash(section.text);

      metadataItems.push({
        slug,
        section: {
          id: section.id,
          title: section.title,
          audioUrl: `/audio/${slug}/${section.id}.mp3`,
        },
      });

      if (isCached(hashPath, hash) && existsSync(mp3Path)) {
        console.log(`  [cached] ${slug}/${section.id}`);
        continue;
      }

      const chunks = chunkText(section.text);
      workItems.push({ slug, section, mp3Path, hashPath, hash, chunks });
    }
  }

  console.log(`\n${metadataItems.length} total sections, ${workItems.length} need generation`);
  const totalChunks = workItems.reduce((sum, w) => sum + w.chunks.length, 0);
  console.log(`${totalChunks} API calls to make (concurrency=${CONCURRENCY})\n`);

  // Phase 2: Fire all chunks through the concurrency pool
  let completed = 0;
  const startTime = Date.now();

  await Promise.all(workItems.map(async (work) => {
    const { slug, section, mp3Path, hashPath, hash, chunks } = work;
    const label = `${slug}/${section.id}`;

    // Fire all chunks for this section concurrently, maintain order
    const buffers = await Promise.all(
      chunks.map((chunk, i) =>
        pool(async () => {
          const buf = await synthesizeChunkWithRetry(chunk);
          completed++;
          const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
          console.log(`  [${completed}/${totalChunks} ${elapsed}s] ${label} chunk ${i + 1}/${chunks.length} (${chunk.length} chars)`);
          return buf;
        })
      )
    );

    const mp3 = Buffer.concat(buffers);
    writeFileSync(mp3Path, mp3);
    writeFileSync(hashPath, hash);
  }));

  // Phase 3: Build manifest
  for (const item of metadataItems) {
    if (!manifest[item.slug]) manifest[item.slug] = [];
    const existing = manifest[item.slug].find(s => s.id === item.section.id);
    if (!existing) {
      manifest[item.slug].push(item.section);
    }
  }

  // Deduplicate and maintain order per slug
  for (const slug of slugs) {
    if (manifest[slug]) {
      const seen = new Set();
      manifest[slug] = manifest[slug].filter(s => {
        if (seen.has(s.id)) return false;
        seen.add(s.id);
        return true;
      });
    }
  }

  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  const totalSections = Object.values(manifest).reduce((sum, s) => sum + s.length, 0);
  console.log(`\n=== Done: ${totalSections} sections across ${Object.keys(manifest).length} chapters in ${elapsed}s ===`);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
