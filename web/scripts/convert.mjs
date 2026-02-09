/**
 * Orchestrates the LaTeX → HTML conversion pipeline.
 *
 * 1. Copy images from book/ to web/public/images/
 * 2. Copy book.pdf to web/public/
 * 3. For each chapter: preprocess → pandoc → postprocess → generated/chapters/
 */

import { execSync } from 'child_process';
import { readFileSync, writeFileSync, mkdirSync, copyFileSync, existsSync, readdirSync } from 'fs';
import { resolve, dirname, basename, extname } from 'path';
import { fileURLToPath } from 'url';
import { preprocess, postprocess as postprocessMarkers, resetCounters } from './preprocess.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const BOOK = resolve(ROOT, '..', 'book');
const GEN = resolve(ROOT, 'generated', 'chapters');
const PUBLIC_IMAGES = resolve(ROOT, 'public', 'images');

// Find pandoc binary
function findPandoc() {
  const localBin = resolve(ROOT, 'bin', 'pandoc');
  if (existsSync(localBin)) return localBin;
  try {
    execSync('which pandoc', { stdio: 'pipe' });
    return 'pandoc';
  } catch {
    console.error('pandoc not found. Install it or run scripts/install-pandoc.sh');
    process.exit(1);
  }
}

const PANDOC = findPandoc();

// Chapter definitions: slug → source file
const CHAPTERS = [
  { slug: 'introduction', title: 'Introduction', file: 'introduction.tex' },
  { slug: 'part-1', title: 'Part I: Thermodynamic Foundations and the Ladder of Emergence', file: 'part1/chapter.tex' },
  { slug: 'part-2', title: 'Part II: The Identity Thesis and the Geometry of Feeling', file: 'part2/chapter.tex' },
  { slug: 'part-3', title: 'Part III: Signatures of Affect Under the Existential Burden', file: 'part3/chapter.tex' },
  { slug: 'part-4', title: 'Part IV: Interventions Across Scale', file: 'part4/chapter.tex' },
  { slug: 'part-5', title: 'Part V: The Transcendence of the Self', file: 'part5/chapter.tex' },
  { slug: 'epilogue', title: 'Epilogue', file: 'epilogue.tex' },
];

function copyImages() {
  console.log('Copying images...');
  mkdirSync(PUBLIC_IMAGES, { recursive: true });

  // Copy from book/images/
  const mainImages = resolve(BOOK, 'images');
  if (existsSync(mainImages)) {
    for (const f of readdirSync(mainImages)) {
      if (f.startsWith('.') || f.endsWith('.md')) continue;
      copyFileSync(resolve(mainImages, f), resolve(PUBLIC_IMAGES, f));
    }
  }

  // Copy from book/part*/images/
  for (let i = 1; i <= 5; i++) {
    const partImages = resolve(BOOK, `part${i}`, 'images');
    if (existsSync(partImages)) {
      for (const f of readdirSync(partImages)) {
        if (f.startsWith('.') || f.endsWith('.md')) continue;
        copyFileSync(resolve(partImages, f), resolve(PUBLIC_IMAGES, f));
      }
    }
  }

  console.log(`  Images copied to ${PUBLIC_IMAGES}`);
}

function copyPdf() {
  const pdfSrc = resolve(BOOK, 'book.pdf');
  const pdfDst = resolve(ROOT, 'public', 'book.pdf');
  if (existsSync(pdfSrc)) {
    copyFileSync(pdfSrc, pdfDst);
    console.log('  PDF copied to public/book.pdf');
  } else {
    console.warn('  Warning: book.pdf not found, skipping');
  }
}

function convertChapter(chapter) {
  console.log(`Converting ${chapter.slug}...`);

  const texPath = resolve(BOOK, chapter.file);
  const rawTex = readFileSync(texPath, 'utf-8');

  // Reset theorem counters for each part
  if (chapter.slug.startsWith('part-') || chapter.slug === 'introduction') {
    resetCounters();
  }

  // Preprocess
  const preprocessed = preprocess(rawTex, chapter.slug);

  // Write temp file for pandoc
  const tmpPath = resolve(GEN, `${chapter.slug}.tex`);
  writeFileSync(tmpPath, preprocessed);

  // Run pandoc
  const htmlPath = resolve(GEN, `${chapter.slug}.html`);
  try {
    execSync(
      `"${PANDOC}" "${tmpPath}" --from latex --to html --mathjax --wrap=none -o "${htmlPath}"`,
      { stdio: 'pipe', maxBuffer: 50 * 1024 * 1024 }
    );
  } catch (err) {
    console.error(`  pandoc error for ${chapter.slug}:`);
    console.error(err.stderr?.toString() || err.message);
    // Write what we have even if pandoc had warnings
    if (!existsSync(htmlPath)) {
      writeFileSync(htmlPath, `<p>Conversion error for ${chapter.slug}. See PDF version.</p>`);
    }
  }

  // Postprocess: convert markers → HTML divs, clean up
  let html = readFileSync(htmlPath, 'utf-8');
  html = postprocessMarkers(html, chapter);
  writeFileSync(htmlPath, html);

  console.log(`  → ${chapter.slug}.html`);
}

// Extract section headings from generated HTML for TOC
function extractToc(htmlPath) {
  const html = readFileSync(htmlPath, 'utf-8');
  const sections = [];
  const re = /<(h[12])\s[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/h[12]>/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    const level = parseInt(m[1][1]);
    const id = m[2];
    const text = m[3].replace(/<[^>]*>/g, '').trim();
    if (text) sections.push({ level, id, text });
  }
  return sections;
}

// Generate chapter metadata JSON (now includes section TOC)
function writeMetadata() {
  const metadata = CHAPTERS.map(ch => {
    const htmlPath = resolve(GEN, `${ch.slug}.html`);
    const sections = existsSync(htmlPath) ? extractToc(htmlPath) : [];
    return { slug: ch.slug, title: ch.title, sections };
  });
  writeFileSync(
    resolve(GEN, 'metadata.json'),
    JSON.stringify(metadata, null, 2)
  );
  console.log('  Metadata written to generated/chapters/metadata.json');
}

// Main
console.log('=== LaTeX → HTML Conversion ===\n');

mkdirSync(GEN, { recursive: true });
mkdirSync(resolve(ROOT, 'public'), { recursive: true });

copyImages();
copyPdf();
console.log('');

for (const chapter of CHAPTERS) {
  convertChapter(chapter);
}

writeMetadata();
console.log('\n=== Done ===');
