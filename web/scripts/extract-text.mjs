/**
 * Extract plain text from TSX content files for audio generation.
 *
 * Parses TSX source files directly with regex — no React runtime needed.
 * The TSX files have highly regular structure (generated from LaTeX).
 *
 * Usage:
 *   import { extractChapterText } from './extract-text.mjs';
 *   const sections = extractChapterText('part-1');
 *   // => [{ id: 'foreword-discourse-on-origins', title: 'Foreword: ...', text: '...' }, ...]
 *
 * Or as CLI:
 *   node scripts/extract-text.mjs part-1
 */

import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONTENT_DIR = resolve(__dirname, '..', 'content');

const CHAPTERS = [
  'introduction', 'part-1', 'part-2', 'part-3', 'part-4', 'part-5', 'epilogue',
];

// --- Math symbol spoken forms ---
const MATH_SPOKEN = {
  '\\alpha': 'alpha', '\\beta': 'beta', '\\gamma': 'gamma', '\\delta': 'delta',
  '\\epsilon': 'epsilon', '\\zeta': 'zeta', '\\eta': 'eta', '\\theta': 'theta',
  '\\iota': 'iota', '\\kappa': 'kappa', '\\lambda': 'lambda', '\\mu': 'mu',
  '\\nu': 'nu', '\\xi': 'xi', '\\pi': 'pi', '\\rho': 'rho',
  '\\sigma': 'sigma', '\\tau': 'tau', '\\phi': 'phi', '\\Phi': 'Phi',
  '\\chi': 'chi', '\\psi': 'psi', '\\omega': 'omega',
  '\\Omega': 'Omega', '\\Delta': 'Delta', '\\Sigma': 'Sigma',
  '\\infty': 'infinity', '\\partial': 'partial',
  '\\viable': 'V', '\\valence': 'valence', '\\arousal': 'arousal',
  '\\intinfo': 'Phi', '\\selfmodel': 'S', '\\worldmodel': 'W',
  '\\R': 'R', '\\N': 'N', '\\Z': 'Z', '\\E': 'E',
  '\\leq': 'less than or equal to', '\\geq': 'greater than or equal to',
  '\\neq': 'not equal to', '\\approx': 'approximately',
  '\\to': 'to', '\\rightarrow': 'to', '\\leftarrow': 'from',
  '\\times': 'times', '\\cdot': 'times', '\\pm': 'plus or minus',
  '\\in': 'in', '\\subset': 'subset of', '\\cup': 'union', '\\cap': 'intersection',
};

/**
 * Convert simple inline math to spoken form.
 * Complex math (display equations) is stripped entirely.
 */
function mathToSpoken(latex) {
  if (!latex) return '';
  let text = latex;
  // Strip command arguments like \text{...}, \mathrm{...}
  text = text.replace(/\\(?:text|mathrm|mathbf|mathcal|textbf|textit|emph)\{([^}]*)\}/g, '$1');
  // Replace known symbols
  for (const [cmd, spoken] of Object.entries(MATH_SPOKEN)) {
    text = text.split(cmd).join(spoken);
  }
  // Strip remaining LaTeX commands
  text = text.replace(/\\[a-zA-Z]+/g, '');
  // Strip braces, subscripts, superscripts
  text = text.replace(/[{}^_]/g, '');
  // Clean up whitespace
  text = text.replace(/\s+/g, ' ').trim();
  return text;
}

/**
 * Decode JSX string escapes: {'"'} → " etc.
 */
function decodeJsxEscapes(text) {
  return text
    .replace(/\{"\\n"\}/g, '\n')
    .replace(/\{"\{"\}/g, '{')
    .replace(/\{"\}"\}/g, '}')
    .replace(/\{"([^"]*)"\}/g, '$1')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, ' ');
}

/**
 * Strip all JSX/HTML tags from text, keeping inner content.
 */
function stripTags(text) {
  return text.replace(/<[^>]*>/g, '');
}

/**
 * Extract sections from a TSX content file.
 * Returns array of { id, title, text } objects.
 */
export function extractChapterText(slug) {
  const filePath = resolve(CONTENT_DIR, `${slug}.tsx`);
  const source = readFileSync(filePath, 'utf-8');

  // Find the JSX body (everything inside return (<>...</>))
  const bodyMatch = source.match(/return\s*\(\s*<>([\s\S]*)<\/>\s*\)/);
  if (!bodyMatch) {
    // Introduction and others may not have sections — treat whole body as one section
    const simpleMatch = source.match(/return\s*\(\s*([\s\S]*)\s*\)\s*;\s*\}/);
    if (!simpleMatch) return [];
    const text = extractTextFromJsx(simpleMatch[1]);
    if (text.length < 50) return [];
    return [{ id: 'full', title: capitalizeSlug(slug), text }];
  }

  const body = bodyMatch[1];

  // Split into sections by <Section> tags
  const sections = [];
  const sectionRegex = /<Section\s+title="([^"]+)"\s+level=\{1\}/g;
  const matches = [...body.matchAll(sectionRegex)];

  if (matches.length === 0) {
    // No level-1 sections — treat whole content as one section
    const text = extractTextFromJsx(body);
    if (text.length >= 50) {
      sections.push({ id: 'full', title: capitalizeSlug(slug), text });
    }
    return sections;
  }

  // Content before first section
  const preContent = body.slice(0, matches[0].index);
  const preText = extractTextFromJsx(preContent);
  if (preText.length >= 50) {
    sections.push({ id: 'intro', title: 'Introduction', text: preText });
  }

  // Each section
  for (let i = 0; i < matches.length; i++) {
    const match = matches[i];
    const title = match[1];
    const id = slugify(title);
    const start = match.index;
    const end = i + 1 < matches.length ? matches[i + 1].index : body.length;
    const sectionJsx = body.slice(start, end);
    const text = extractTextFromJsx(sectionJsx);

    if (text.length >= 50) {
      sections.push({ id, title, text });
    }
  }

  return sections;
}

/**
 * Extract readable text from a chunk of JSX.
 */
function extractTextFromJsx(jsx) {
  let text = jsx;

  // Strip display math entirely: <Eq>{"..."}</Eq> and <Align>{"..."}</Align>
  text = text.replace(/<Eq>\{["`][\s\S]*?\}[\s]*<\/Eq>/g, '');
  text = text.replace(/<Align>\{["`][\s\S]*?\}[\s]*<\/Align>/g, '');

  // Convert inline math <M>{"..."}</M> to spoken form
  text = text.replace(/<M>\{"([^"]*)"\}<\/M>/g, (_, latex) => {
    const spoken = mathToSpoken(latex);
    return spoken ? ` ${spoken} ` : ' ';
  });
  // Handle backtick math strings
  text = text.replace(/<M>\{`([^`]*)`\}<\/M>/g, (_, latex) => {
    const spoken = mathToSpoken(latex);
    return spoken ? ` ${spoken} ` : ' ';
  });

  // Strip Figure and Diagram (visual only)
  text = text.replace(/<Figure\s[^>]*\/>/g, '');
  text = text.replace(/<Diagram\s[^>]*\/>/g, '');

  // Strip MarginNote
  text = text.replace(/<MarginNote>[\s\S]*?<\/MarginNote>/g, '');

  // Strip Ref tags, keep label
  text = text.replace(/<Ref\s+to="[^"]*"\s+label="([^"]*)"\s*\/>/g, '$1');
  text = text.replace(/<Ref\s+to="[^"]*"\s*\/>/g, '');

  // Strip Section opening/closing tags (keep content)
  text = text.replace(/<Section\s+title="[^"]*"[^>]*>/g, '');
  text = text.replace(/<\/Section>/g, '');

  // Strip environment wrappers (keep content) — Sidebar, Connection, etc.
  const envComponents = [
    'Sidebar', 'Connection', 'Experiment', 'OpenQuestion', 'Logos',
    'KeyResult', 'Warning', 'TodoEmpirical', 'Historical', 'Empirical',
    'Software', 'Phenomenal', 'NormativeImplication', 'EnvBox',
    'WideBreath', 'WideMargin', 'Proof', 'WebOnly',
  ];
  for (const comp of envComponents) {
    // Opening tag with props
    text = text.replace(new RegExp(`<${comp}(?:\\s[^>]*)?>`, 'g'), '');
    // Self-closing
    text = text.replace(new RegExp(`<${comp}\\s[^>]*/\\s*>`, 'g'), '');
    // Closing tag
    text = text.replace(new RegExp(`</${comp}>`, 'g'), '');
  }

  // Convert HTML elements
  text = text.replace(/<em>/g, '').replace(/<\/em>/g, '');
  text = text.replace(/<strong>/g, '').replace(/<\/strong>/g, '');
  text = text.replace(/<code>/g, '').replace(/<\/code>/g, '');
  text = text.replace(/<SmallCaps>/g, '').replace(/<\/SmallCaps>/g, '');
  text = text.replace(/<a\s[^>]*>/g, '').replace(/<\/a>/g, '');
  text = text.replace(/<blockquote>/g, '').replace(/<\/blockquote>/g, '');

  // Lists
  text = text.replace(/<[ou]l>/g, '\n').replace(/<\/[ou]l>/g, '\n');
  text = text.replace(/<li>/g, '\n').replace(/<\/li>/g, '');

  // Tables — strip entirely (hard to read as audio)
  text = text.replace(/<table[\s\S]*?<\/table>/g, '');

  // Description lists
  text = text.replace(/<dl>/g, '\n').replace(/<\/dl>/g, '\n');
  text = text.replace(/<dt>/g, '\n').replace(/<\/dt>/g, ': ');
  text = text.replace(/<dd>/g, '').replace(/<\/dd>/g, '\n');

  // Paragraphs
  text = text.replace(/<p>/g, '\n').replace(/<\/p>/g, '\n');
  text = text.replace(/<br\s*\/?>/g, '\n');

  // Any remaining tags
  text = stripTags(text);

  // Decode JSX escapes
  text = decodeJsxEscapes(text);

  // Strip JSX expression containers that might remain: {"text"}
  text = text.replace(/\{"([^"]*)"\}/g, '$1');

  // Clean up
  text = text.replace(/\n{3,}/g, '\n\n');
  text = text.replace(/\s+/g, ' ').trim();

  return text;
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
}

function capitalizeSlug(slug) {
  return slug.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

// --- CLI mode ---
if (process.argv[1] && process.argv[1].includes('extract-text')) {
  const args = process.argv.slice(2);
  const slugs = args.length > 0 ? args.filter(s => CHAPTERS.includes(s)) : CHAPTERS;

  if (slugs.length === 0) {
    console.error(`Invalid slug(s). Available: ${CHAPTERS.join(', ')}`);
    process.exit(1);
  }

  for (const slug of slugs) {
    const sections = extractChapterText(slug);
    console.log(`\n=== ${slug} (${sections.length} sections) ===`);
    for (const s of sections) {
      console.log(`  [${s.id}] ${s.title} (${s.text.length} chars)`);
      // Print first 200 chars as preview
      console.log(`    ${s.text.slice(0, 200)}...`);
    }
  }
}
