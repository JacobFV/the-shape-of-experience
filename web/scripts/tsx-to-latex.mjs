/**
 * Convert TSX content files back to LaTeX chapter bodies.
 *
 * Reads web/content/*.tsx, outputs book/generated/*.tex.
 * The book/book.tex preamble (packages, environments, macros) is hand-maintained.
 * Only chapter body content is generated.
 *
 * Usage:
 *   node scripts/tsx-to-latex.mjs           # all chapters
 *   node scripts/tsx-to-latex.mjs part-1    # single chapter
 */

import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONTENT_DIR = resolve(__dirname, '..', 'content');
const BOOK_DIR = resolve(__dirname, '..', '..', 'book');
const GEN_DIR = resolve(BOOK_DIR, 'generated');

const CHAPTERS = [
  { slug: 'introduction', texName: 'introduction' },
  { slug: 'part-1', texName: 'part1' },
  { slug: 'part-2', texName: 'part2' },
  { slug: 'part-3', texName: 'part3' },
  { slug: 'part-4', texName: 'part4' },
  { slug: 'part-5', texName: 'part5' },
  { slug: 'epilogue', texName: 'epilogue' },
];

// --- Environment mappings ---
// Maps TSX component names to LaTeX environment names
const ENV_MAP = {
  'Sidebar': 'sidebar',
  'Connection': 'connection',
  'Experiment': 'experiment',
  'OpenQuestion': 'openquestion',
  'KeyResult': 'keyresult',
  'Warning': 'warning',
  'TodoEmpirical': 'todo_empirical',
  'Historical': 'historical',
  'Empirical': 'empirical',
  'Software': 'software',
  'Phenomenal': 'phenomenal',
  'NormativeImplication': 'normimp',
};

/**
 * Convert a TSX content file to LaTeX.
 */
function convertToLatex(slug) {
  const filePath = resolve(CONTENT_DIR, `${slug}.tsx`);
  const source = readFileSync(filePath, 'utf-8');

  // Extract the JSX body from the return statement
  const bodyMatch = source.match(/return\s*\(\s*<>([\s\S]*)<\/>\s*\)/);
  let body;
  if (bodyMatch) {
    body = bodyMatch[1];
  } else {
    // Try simpler return pattern (introduction has no fragments)
    const simpleMatch = source.match(/return\s*\(\s*<>([\s\S]*)<\/>\s*\)/);
    if (simpleMatch) {
      body = simpleMatch[1];
    } else {
      // Fallback: extract between return ( and final );
      const fallback = source.match(/return\s*\(\s*([\s\S]*)\s*\)\s*;\s*\}/);
      if (!fallback) {
        console.error(`Could not extract JSX body from ${slug}`);
        return '';
      }
      body = fallback[1];
      // Strip fragment wrappers if present
      body = body.replace(/^\s*<>\s*/, '').replace(/\s*<\/>\s*$/, '');
    }
  }

  let latex = body;

  // --- Pass 1: Convert components to LaTeX ---

  // WebOnly: strip entirely (content not emitted to PDF)
  latex = latex.replace(/<WebOnly>[\s\S]*?<\/WebOnly>/g, '');

  // Section tags
  latex = latex.replace(/<Section\s+title="([^"]+)"\s+level=\{1\}[^>]*>/g, (_, title) => {
    return `\\section{${escapeLatex(title)}}`;
  });
  latex = latex.replace(/<Section\s+title="([^"]+)"\s+level=\{2\}[^>]*>/g, (_, title) => {
    return `\\subsection{${escapeLatex(title)}}`;
  });
  latex = latex.replace(/<Section\s+title="([^"]+)"\s+level=\{3\}[^>]*>/g, (_, title) => {
    return `\\subsubsection{${escapeLatex(title)}}`;
  });
  // Default level (2)
  latex = latex.replace(/<Section\s+title="([^"]+)"[^>]*>/g, (_, title) => {
    return `\\subsection{${escapeLatex(title)}}`;
  });
  latex = latex.replace(/<\/Section>/g, '');

  // Inline math: <M>{"..."}</M>
  latex = latex.replace(/<M>\{"([^"]*)"\}<\/M>/g, (_, math) => `$${unescapeJsString(math)}$`);
  latex = latex.replace(/<M>\{`([^`]*)`\}<\/M>/g, (_, math) => `$${unescapeJsString(math)}$`);

  // Display math: <Eq>{"..."}</Eq>
  latex = latex.replace(/<Eq>\{"([\s\S]*?)"\}\s*<\/Eq>/g, (_, math) => {
    return `\\begin{equation*}\n${unescapeJsString(math)}\n\\end{equation*}`;
  });
  latex = latex.replace(/<Eq>\{`([\s\S]*?)`\}\s*<\/Eq>/g, (_, math) => {
    return `\\begin{equation*}\n${unescapeJsString(math)}\n\\end{equation*}`;
  });

  // Align: <Align>{"..."}</Align>
  latex = latex.replace(/<Align>\{"([\s\S]*?)"\}\s*<\/Align>/g, (_, math) => {
    return `\\begin{equation*}\\begin{aligned}\n${unescapeJsString(math)}\n\\end{aligned}\\end{equation*}`;
  });
  latex = latex.replace(/<Align>\{`([\s\S]*?)`\}\s*<\/Align>/g, (_, math) => {
    return `\\begin{equation*}\\begin{aligned}\n${unescapeJsString(math)}\n\\end{aligned}\\end{equation*}`;
  });

  // Environment boxes with title prop
  for (const [comp, env] of Object.entries(ENV_MAP)) {
    // With title: <Sidebar title="...">
    const titleRegex = new RegExp(`<${comp}\\s+title="([^"]*)"[^>]*>`, 'g');
    latex = latex.replace(titleRegex, (_, title) => {
      if (env === 'sidebar') {
        return `\\begin{${env}}[title=${escapeLatex(title)}]`;
      }
      return `\\begin{${env}}`;
    });
    // Without title
    const noTitleRegex = new RegExp(`<${comp}(?:\\s[^>]*)?>`, 'g');
    latex = latex.replace(noTitleRegex, `\\begin{${env}}`);
    // Closing
    const closeRegex = new RegExp(`</${comp}>`, 'g');
    latex = latex.replace(closeRegex, `\\end{${env}}`);
  }

  // Logos
  latex = latex.replace(/<Logos>/g, '\\begin{logos}');
  latex = latex.replace(/<\/Logos>/g, '\\end{logos}');

  // Proof
  latex = latex.replace(/<Proof>/g, '\\begin{proof}');
  latex = latex.replace(/<\/Proof>/g, '\\end{proof}');

  // WideBreath / WideMargin
  latex = latex.replace(/<WideBreath>/g, '\\begin{widebreath}');
  latex = latex.replace(/<\/WideBreath>/g, '\\end{widebreath}');
  latex = latex.replace(/<WideMargin>/g, '\\begin{widemargin}');
  latex = latex.replace(/<\/WideMargin>/g, '\\end{widemargin}');

  // MarginNote
  latex = latex.replace(/<MarginNote>([\s\S]*?)<\/MarginNote>/g, (_, content) => {
    return `\\detail{${stripAndConvertInline(content)}}`;
  });

  // Figure
  latex = latex.replace(/<Figure\s+src="([^"]*)"\s+alt="([^"]*)"(?:\s+caption=\{<>([^]*?)<\/>\})?(?:\s+multi)?\s*\/>/g,
    (_, src, alt, caption) => {
      const lines = ['\\begin{figure}[htbp]', '\\centering'];
      lines.push(`\\includegraphics[width=0.8\\textwidth]{${src}}`);
      if (caption) {
        lines.push(`\\caption{${stripAndConvertInline(caption)}}`);
      }
      lines.push('\\end{figure}');
      return lines.join('\n');
    }
  );
  // Simpler figure patterns
  latex = latex.replace(/<Figure\s+src="([^"]*)"[^>]*\/>/g, (_, src) => {
    return `\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{${src}}\n\\end{figure}`;
  });

  // Diagram — map web SVG paths to TikZ source files
  const TIKZ_MAP = {
    '/diagrams/part-1-0.svg': 'tikz/part-1-0',
    '/diagrams/part-1-1.svg': 'tikz/part-1-1',
    '/diagrams/part-1-2.svg': 'tikz/part-1-2',
    '/diagrams/part-1-3.svg': 'tikz/part-1-3',
    '/diagrams/part-1-4.svg': 'tikz/part-1-4',
    '/diagrams/part-1-5.svg': 'tikz/part-1-5',
    '/diagrams/part-2-0.svg': 'tikz/part-2-0',
    '/diagrams/part-3-0.svg': 'tikz/part-3-0',
    '/diagrams/part-4-0.svg': 'tikz/part-4-0',
    '/diagrams/part-5-0.svg': 'tikz/part-5-0',
  };
  latex = latex.replace(/<Diagram\s+src="([^"]*)"\s*\/>/g, (_, src) => {
    const tikzFile = TIKZ_MAP[src];
    if (!tikzFile) {
      throw new Error(`No TikZ source mapping for diagram: ${src}`);
    }
    return `\\input{${tikzFile}}`;
  });

  // Ref
  latex = latex.replace(/<Ref\s+to="([^"]*)"\s+label="([^"]*)"\s*\/>/g, (_, to, label) => {
    return `\\cref{${to}}`;
  });
  latex = latex.replace(/<Ref\s+to="([^"]*)"\s*\/>/g, (_, to) => {
    return `\\cref{${to}}`;
  });

  // SmallCaps
  latex = latex.replace(/<SmallCaps>([\s\S]*?)<\/SmallCaps>/g, (_, text) => `\\textsc{${text}}`);

  // --- Pass 2: Convert HTML elements ---

  // Paragraphs: <p>...</p> → content with blank lines
  latex = latex.replace(/<p>/g, '\n');
  latex = latex.replace(/<\/p>/g, '\n');

  // Emphasis
  latex = latex.replace(/<em>([\s\S]*?)<\/em>/g, (_, text) => `\\emph{${text}}`);
  latex = latex.replace(/<strong>([\s\S]*?)<\/strong>/g, (_, text) => `\\textbf{${text}}`);
  latex = latex.replace(/<code>([\s\S]*?)<\/code>/g, (_, text) => `\\texttt{${text}}`);

  // Links
  latex = latex.replace(/<a\s+href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/g, (_, url, text) => {
    return `\\href{${url}}{${text}}`;
  });

  // Lists
  latex = latex.replace(/<ol>/g, '\\begin{enumerate}');
  latex = latex.replace(/<\/ol>/g, '\\end{enumerate}');
  latex = latex.replace(/<ul>/g, '\\begin{itemize}');
  latex = latex.replace(/<\/ul>/g, '\\end{itemize}');
  latex = latex.replace(/<li>/g, '\\item ');
  latex = latex.replace(/<\/li>/g, '');

  // Description lists
  latex = latex.replace(/<dl>/g, '\\begin{description}');
  latex = latex.replace(/<\/dl>/g, '\\end{description}');
  latex = latex.replace(/<dt>([\s\S]*?)<\/dt>/g, (_, term) => `\\item[${stripAndConvertInline(term)}] `);
  latex = latex.replace(/<dd>([\s\S]*?)<\/dd>/g, '$1\n');

  // Tables
  latex = convertTables(latex);

  // Blockquote
  latex = latex.replace(/<blockquote>/g, '\\begin{quote}');
  latex = latex.replace(/<\/blockquote>/g, '\\end{quote}');

  // Line breaks
  latex = latex.replace(/<br\s*\/?>/g, '\\\\');

  // --- Pass 3: Clean up JSX artifacts ---

  // Unescape JSX string expressions: {"text"} → text (with JS string unescaping)
  latex = latex.replace(/\{"\\n"\}/g, '\n');
  latex = latex.replace(/\{"\{"\}/g, '{');
  latex = latex.replace(/\{"\}"\}/g, '}');
  // Generic JSX expressions with string literals
  latex = latex.replace(/\{"([^"]*)"\}/g, (_, content) => unescapeJsString(content));

  // Strip any remaining JSX/HTML tags
  latex = latex.replace(/<\/?[a-zA-Z][^>]*>/g, '');

  // Clean up excessive blank lines
  latex = latex.replace(/\n{4,}/g, '\n\n\n');

  // Trim
  latex = latex.trim() + '\n';

  return latex;
}

/**
 * Convert HTML tables to LaTeX tabular.
 */
function convertTables(latex) {
  const tableRegex = /<table>([\s\S]*?)<\/table>/g;
  return latex.replace(tableRegex, (_, tableContent) => {
    const rows = [];
    const rowRegex = /<tr>([\s\S]*?)<\/tr>/g;
    let match;
    let numCols = 0;

    while ((match = rowRegex.exec(tableContent)) !== null) {
      const cells = [];
      const cellRegex = /<t[hd](?:\s[^>]*)?>([\s\S]*?)<\/t[hd]>/g;
      let cellMatch;
      while ((cellMatch = cellRegex.exec(match[1])) !== null) {
        cells.push(stripAndConvertInline(cellMatch[1]).trim());
      }
      if (cells.length > numCols) numCols = cells.length;
      rows.push(cells);
    }

    if (rows.length === 0) return '';

    const colSpec = 'l'.repeat(numCols);
    const lines = [`\\begin{tabular}{${colSpec}}`, '\\toprule'];

    for (let i = 0; i < rows.length; i++) {
      // Pad row to numCols
      while (rows[i].length < numCols) rows[i].push('');
      lines.push(rows[i].join(' & ') + ' \\\\');
      if (i === 0 && rows.length > 1) {
        lines.push('\\midrule');
      }
    }

    lines.push('\\bottomrule', '\\end{tabular}');
    return lines.join('\n');
  });
}

/**
 * Strip HTML tags and convert inline formatting for use inside LaTeX commands.
 */
function stripAndConvertInline(html) {
  let text = html;
  text = text.replace(/<em>([\s\S]*?)<\/em>/g, (_, t) => `\\emph{${t}}`);
  text = text.replace(/<strong>([\s\S]*?)<\/strong>/g, (_, t) => `\\textbf{${t}}`);
  text = text.replace(/<code>([\s\S]*?)<\/code>/g, (_, t) => `\\texttt{${t}}`);
  text = text.replace(/<M>\{"([^"]*)"\}<\/M>/g, (_, math) => `$${unescapeJsString(math)}$`);
  text = text.replace(/<[^>]*>/g, '');
  return text;
}

/**
 * Escape special LaTeX characters in plain text.
 */
function escapeLatex(text) {
  // Don't escape backslashes that are already LaTeX commands
  return text
    .replace(/&/g, '\\&')
    .replace(/%/g, '\\%')
    .replace(/#/g, '\\#');
}

/**
 * Unescape a JS string literal extracted from TSX source.
 * In JS source, \\ represents a literal \, so \\alpha → \alpha.
 * We only unescape \\\\ → \\ and quote escapes.
 * We do NOT unescape \\n or \\t since after \\\\ → \\,
 * those become valid LaTeX commands like \\neq, \\text.
 */
function unescapeJsString(text) {
  return text
    .replace(/\\\\/g, '\\')
    .replace(/\\'/g, "'")
    .replace(/\\"/g, '"');
}

/**
 * Unescape simple JSX artifacts (not full JS string escaping).
 */
function unescapeJsx(text) {
  return text;
}

// --- Main ---

function main() {
  mkdirSync(GEN_DIR, { recursive: true });

  const args = process.argv.slice(2);
  const selected = args.length > 0
    ? CHAPTERS.filter(c => args.includes(c.slug))
    : CHAPTERS;

  if (selected.length === 0) {
    console.error(`Invalid slug(s). Available: ${CHAPTERS.map(c => c.slug).join(', ')}`);
    process.exit(1);
  }

  console.log(`=== TSX → LaTeX (${selected.length} chapters) ===\n`);

  for (const { slug, texName } of selected) {
    const latex = convertToLatex(slug);
    const outPath = resolve(GEN_DIR, `${texName}.tex`);
    writeFileSync(outPath, latex);
    const lines = latex.split('\n').length;
    console.log(`  ${slug} → generated/${texName}.tex (${lines} lines)`);
  }

  console.log('\nDone.');
}

main();
