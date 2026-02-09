/**
 * Render TikZ diagrams from LaTeX source to SVG at build time.
 *
 * For each chapter, extracts tikzpicture blocks, compiles via pdflatex + pdf2svg,
 * and saves SVGs to web/public/diagrams/.
 *
 * Falls back gracefully if pdflatex or pdf2svg are not available.
 */

import { execSync } from 'child_process';
import { readFileSync, writeFileSync, mkdirSync, existsSync, unlinkSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const BOOK = resolve(ROOT, '..', 'book');
const DIAGRAMS_DIR = resolve(ROOT, 'public', 'diagrams');

const CHAPTERS = [
  { slug: 'part-1', file: 'part1/chapter.tex' },
  { slug: 'part-2', file: 'part2/chapter.tex' },
  { slug: 'part-3', file: 'part3/chapter.tex' },
  { slug: 'part-4', file: 'part4/chapter.tex' },
  { slug: 'part-5', file: 'part5/chapter.tex' },
];

// Custom macro definitions for TikZ diagrams (matching book/book.tex)
const CUSTOM_MACROS = `
\\newcommand{\\viable}{\\mathcal{V}}
\\newcommand{\\valence}{\\mathcal{V}\\hspace{-0.8pt}\\mathit{al}}
\\newcommand{\\arousal}{\\mathcal{A}\\hspace{-0.5pt}\\mathit{r}}
\\newcommand{\\intinfo}{\\Phi}
\\newcommand{\\effrank}{r_{\\text{eff}}}
\\newcommand{\\selfmodel}{\\mathcal{S}}
\\newcommand{\\worldmodel}{\\mathcal{W}}
\\newcommand{\\cfweight}{\\mathrm{CF}}
\\newcommand{\\selfsal}{\\mathrm{SM}}
\\newcommand{\\reff}{r_{\\text{eff}}}
\\newcommand{\\Val}{\\mathcal{V}\\hspace{-0.8pt}\\mathit{al}}
\\newcommand{\\Ar}{\\mathcal{A}\\hspace{-0.5pt}\\mathit{r}}
`;

function buildPreamble(tikzCode) {
  // Measure the tikz picture and set page to exact size
  return `\\documentclass{article}
\\usepackage[paperwidth=50cm,paperheight=50cm,margin=0pt]{geometry}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}
\\usetikzlibrary{arrows.meta,positioning,shapes,calc,decorations.pathmorphing,decorations.pathreplacing}
\\usepackage{amsmath,amssymb}
${CUSTOM_MACROS}
\\pagestyle{empty}
\\begin{document}
\\newsavebox{\\mybox}
\\begin{lrbox}{\\mybox}
${tikzCode}
\\end{lrbox}
\\pdfpagewidth=\\dimexpr\\wd\\mybox+2pt\\relax
\\pdfpageheight=\\dimexpr\\ht\\mybox+\\dp\\mybox+2pt\\relax
\\usebox{\\mybox}
\\end{document}
`;
}

function hasTool(name) {
  try {
    execSync(`which ${name}`, { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

function extractTikzBlocks(tex) {
  const blocks = [];
  const re = /\\begin\{tikzpicture\}([\s\S]*?)\\end\{tikzpicture\}/g;
  let match;
  while ((match = re.exec(tex)) !== null) {
    blocks.push(`\\begin{tikzpicture}${match[1]}\\end{tikzpicture}`);
  }
  return blocks;
}

function renderTikzToSvg(tikzCode, outputPath, tmpDir) {
  const texContent = buildPreamble(tikzCode);
  const texPath = resolve(tmpDir, 'diagram.tex');
  const pdfPath = resolve(tmpDir, 'diagram.pdf');

  writeFileSync(texPath, texContent);

  try {
    execSync(
      `pdflatex -interaction=nonstopmode -halt-on-error -output-directory="${tmpDir}" "${texPath}"`,
      { stdio: 'pipe', timeout: 30000 }
    );
  } catch (err) {
    const stdout = err.stdout?.toString().slice(-500) || '';
    const stderr = err.stderr?.toString().slice(-500) || '';
    console.warn(`    pdflatex failed: ${stderr || stdout || err.message}`);
    return false;
  }

  if (!existsSync(pdfPath)) return false;

  // Try conversion tools in order of preference
  const converters = [
    () => execSync(`pdf2svg "${pdfPath}" "${outputPath}"`, { stdio: 'pipe', timeout: 10000 }),
    () => execSync(`pdftocairo -svg "${pdfPath}" "${outputPath}"`, { stdio: 'pipe', timeout: 10000 }),
    () => execSync(`dvisvgm --pdf "${pdfPath}" -o "${outputPath}"`, { stdio: 'pipe', timeout: 10000 }),
  ];

  for (const convert of converters) {
    try {
      convert();
      if (existsSync(outputPath)) return true;
    } catch { /* try next */ }
  }

  console.warn(`    SVG conversion failed: no working converter`);
  return false;
}

function cleanTmpDir(tmpDir) {
  const exts = ['.tex', '.pdf', '.aux', '.log'];
  for (const ext of exts) {
    const f = resolve(tmpDir, `diagram${ext}`);
    try { unlinkSync(f); } catch {}
  }
}

export function renderAllTikz() {
  if (!hasTool('pdflatex')) {
    console.log('TikZ rendering: pdflatex not available, using placeholders');
    return { rendered: 0, total: 0 };
  }
  const hasConverter = hasTool('pdf2svg') || hasTool('pdftocairo') || hasTool('dvisvgm');
  if (!hasConverter) {
    console.log('TikZ rendering: no PDF→SVG converter (pdf2svg/pdftocairo/dvisvgm), using placeholders');
    return { rendered: 0, total: 0 };
  }

  mkdirSync(DIAGRAMS_DIR, { recursive: true });
  const tmpDir = resolve(ROOT, '.tikz-tmp');
  mkdirSync(tmpDir, { recursive: true });

  let total = 0;
  let rendered = 0;

  for (const chapter of CHAPTERS) {
    const texPath = resolve(BOOK, chapter.file);
    if (!existsSync(texPath)) continue;

    const tex = readFileSync(texPath, 'utf-8');
    const blocks = extractTikzBlocks(tex);
    if (blocks.length === 0) continue;

    console.log(`  ${chapter.slug}: ${blocks.length} diagram(s)`);

    for (let i = 0; i < blocks.length; i++) {
      total++;
      const svgPath = resolve(DIAGRAMS_DIR, `${chapter.slug}-${i}.svg`);

      // Skip if already rendered (cache)
      if (existsSync(svgPath)) {
        console.log(`    [${i}] cached`);
        rendered++;
        continue;
      }

      const ok = renderTikzToSvg(blocks[i], svgPath, tmpDir);
      if (ok) {
        rendered++;
        console.log(`    [${i}] rendered`);
      } else {
        console.log(`    [${i}] failed, will use placeholder`);
      }
      cleanTmpDir(tmpDir);
    }
  }

  // Clean up tmp dir
  try { execSync(`rm -rf "${tmpDir}"`); } catch {}

  return { rendered, total };
}

// Run directly
if (process.argv[1] && process.argv[1].includes('render-tikz')) {
  console.log('=== Rendering TikZ diagrams ===\n');
  const result = renderAllTikz();
  console.log(`\n  Rendered ${result.rendered}/${result.total} diagrams`);
  console.log('\n=== Done ===');
}
