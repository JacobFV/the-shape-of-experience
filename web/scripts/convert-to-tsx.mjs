#!/usr/bin/env node
/**
 * LaTeX → TSX converter for The Shape of Experience.
 *
 * Reads LaTeX chapter files and outputs React/TSX content files
 * that use the components from @/components/content.
 *
 * Expected: ~75% clean conversion, ~20% minor fixups, ~5% manual rewrite.
 */

import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const BOOK = resolve(ROOT, '..', 'book');
const CONTENT_DIR = resolve(ROOT, 'content');

const CHAPTERS = [
  { slug: 'introduction', title: 'Introduction', shortTitle: 'Introduction', file: 'introduction.tex' },
  { slug: 'part-1', title: 'Part I: Thermodynamic Foundations and the Ladder of Emergence', shortTitle: 'Part I: Foundations', file: 'part1/chapter.tex' },
  { slug: 'part-2', title: 'Part II: The Identity Thesis and the Geometry of Feeling', shortTitle: 'Part II: Identity Thesis', file: 'part2/chapter.tex' },
  { slug: 'part-3', title: 'Part III: Signatures of Affect Under the Existential Burden', shortTitle: 'Part III: Affect Signatures', file: 'part3/chapter.tex' },
  { slug: 'part-4', title: 'Part IV: Interventions Across Scale', shortTitle: 'Part IV: Interventions', file: 'part4/chapter.tex' },
  { slug: 'part-5', title: 'Part V: The Transcendence of the Self', shortTitle: 'Part V: Transcendence', file: 'part5/chapter.tex' },
  { slug: 'epilogue', title: 'Epilogue', shortTitle: 'Epilogue', file: 'epilogue.tex' },
];

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Escape a string for use inside a JSX string literal {"..."} */
function escapeForJsx(s) {
  // We put math inside {"..."}, so we need to escape backslashes and quotes
  // Actually, in JSX {"\\alpha"}, the JS string is \alpha, which is what we want.
  // We only need to escape literal backticks and ${} if using template literals.
  // For {"..."} we need to escape " → \"
  return s.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

/** Escape math for JSX: put inside {"..."} with proper escaping */
function mathJsx(math) {
  // In JSX, {"\\alpha + \\beta"} becomes the string \alpha + \beta at runtime.
  // The backslashes in LaTeX need to be double-escaped for the JS string.
  const escaped = math.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, ' ');
  return escaped;
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '')
    .slice(0, 80);
}

function normalizeImagePath(texPath) {
  const filename = texPath.replace(/^.*\//, '');
  const webFilename = filename.replace(/\.pdf$/, '.png');
  return `/images/${webFilename}`;
}

// ── Phase 0: Strip preamble and layout commands ─────────────────────────────

function stripPreamble(tex) {
  let out = tex;

  // Strip document wrappers
  out = out.replace(/\\begin\{document\}[\s\S]*?\\mainmatter/g, '');
  out = out.replace(/\\end\{document\}/g, '');

  // Strip layout commands
  const layoutCmds = [
    /\\newgeometry\{[^}]*\}/g,
    /\\restoregeometry/g,
    /\\twocolumn(\[[^\]]*\])?/g,
    /\\onecolumn/g,
    /\\clearpage/g,
    /\\pagebreak(\[[^\]]*\])?/g,
    /\\newpage/g,
    /\\thispagestyle\{[^}]*\}/g,
    /\\pagestyle\{[^}]*\}/g,
    /\\pagenumbering\{[^}]*\}/g,
    /\\setcounter\{[^}]*\}\{[^}]*\}/g,
    /\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}/g,
    /\\raggedbottom/g,
    /\\null/g,
    /\\(Huge|huge|LARGE|Large|large|normalsize|small|footnotesize|scriptsize|tiny)\b/g,
    /\\vfill/g,
    /\\vspace\*?\{[^}]*\}/g,
    /\\hspace\*?\{[^}]*\}/g,
    /\\fa[A-Z][a-zA-Z]*/g,
    /\\centering/g,
    /\\bfseries\b/g,
    /\\itshape\b/g,
    /\\noindent\b/g,
    /\\label\{[^}]*\}/g,
  ];
  for (const re of layoutCmds) {
    out = out.replace(re, '');
  }

  // Strip comments (but not percent in math)
  out = out.replace(/(?<!\\)%.*$/gm, '');

  // Strip algorithm environments
  out = out.replace(/\\begin\{algorithm\}[\s\S]*?\\end\{algorithm\}/g, '');
  out = out.replace(/\\begin\{algorithmic\}[\s\S]*?\\end\{algorithmic\}/g, '');

  // Strip subfigure/wrapfigure/cutwin artifacts
  out = out.replace(/\\begin\{subfigure\}[\s\S]*?\\end\{subfigure\}/g, '');
  out = out.replace(/\\begin\{wrapfigure\}(\[[^\]]*\])?\{[^}]*\}\{[^}]*\}/g, '');
  out = out.replace(/\\end\{wrapfigure\}/g, '');
  out = out.replace(/\\opencmark/g, '');
  out = out.replace(/\\closecmark/g, '');

  // Strip booktabs
  out = out.replace(/\\toprule/g, '\\hline');
  out = out.replace(/\\midrule/g, '\\hline');
  out = out.replace(/\\bottomrule/g, '\\hline');
  out = out.replace(/\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}/g, '$1');

  // Strip caption (will be handled by figure processing)
  // (done later in figure phase)

  // Refs → plain text
  out = out.replace(/\\cref\{([^}]*)\}/g, '[ref]');
  out = out.replace(/\\Cref\{([^}]*)\}/g, '[Ref]');
  out = out.replace(/\\ref\{([^}]*)\}/g, '[ref]');
  out = out.replace(/\\eqref\{([^}]*)\}/g, '(ref)');

  return out;
}

// ── Phase 1: Convert environments → JSX components ──────────────────────────

const TCOLORBOX_MAP = {
  sidebar: 'Sidebar',
  connection: 'Connection',
  'connection*': 'Connection',
  historical: 'Historical',
  empirical: 'Empirical',
  todo_empirical: 'TodoEmpirical',
  openquestion: 'OpenQuestion',
  warningbox: 'Warning',
  experiment: 'Experiment',
  software: 'Software',
  logos: 'Logos',
  keyresult: 'KeyResult',
  phenomenal: 'Phenomenal',
  warning: 'Warning',
  normimp: 'NormativeImplication',
};

const ENV_TITLES = {
  connection: 'Existing Theory',
  'connection*': 'Existing Theory',
  historical: 'Historical Context',
  empirical: 'Empirical Grounding',
  todo_empirical: 'Future Empirical Work',
  openquestion: 'Open Question',
  warningbox: 'Warning',
  experiment: 'Proposed Experiment',
  software: 'Software Implementation',
  keyresult: 'Key Result',
  phenomenal: 'Phenomenal Correspondence',
  warning: 'Warning',
  normimp: 'Normative Implication',
};

function convertEnvironments(tex) {
  let out = tex;

  // Sidebar with title (strip LaTeX commands from title for JSX attribute)
  out = out.replace(
    /\\begin\{sidebar\}\[title=\{?([^}\]]*)\}?\]/g,
    (_, title) => {
      // Strip inline math and LaTeX commands from title for use as string attribute
      const cleanTitle = title
        .replace(/\$([^$]*)\$/g, '$1')  // strip $ delimiters
        .replace(/\\[a-zA-Z]+\{([^}]*)\}/g, '$1')  // \cmd{arg} → arg
        .replace(/\\[a-zA-Z]+/g, '')  // bare \cmd
        .replace(/[{}]/g, '')
        .replace(/"/g, '\\"')
        .trim();
      return `<Sidebar title="${cleanTitle}">`;
    }
  );
  out = out.replace(/\\begin\{sidebar\}(?!\[)/g, '<Sidebar>');
  out = out.replace(/\\end\{sidebar\}/g, '</Sidebar>');

  // Other tcolorbox/margin environments
  for (const [envName, compName] of Object.entries(TCOLORBOX_MAP)) {
    if (envName === 'sidebar') continue; // already handled
    const escapedName = envName.replace('*', '\\*');
    const title = ENV_TITLES[envName];

    const beginRe = new RegExp(`\\\\begin\\{${escapedName}\\}(\\[[^\\]]*\\])?`, 'g');
    const endRe = new RegExp(`\\\\end\\{${escapedName}\\}`, 'g');

    if (title && compName !== 'Logos') {
      out = out.replace(beginRe, `<${compName} title="${title}">`);
    } else {
      out = out.replace(beginRe, `<${compName}>`);
    }
    out = out.replace(endRe, `</${compName}>`);
  }

  // Layout environments
  out = out.replace(/\\begin\{widebreath\}/g, '<WideBreath>');
  out = out.replace(/\\end\{widebreath\}/g, '</WideBreath>');
  out = out.replace(/\\begin\{widemargin\}/g, '<WideMargin>');
  out = out.replace(/\\end\{widemargin\}/g, '</WideMargin>');

  // Proof
  out = out.replace(/\\begin\{proof\}/g, '<Proof>');
  out = out.replace(/\\end\{proof\}/g, '</Proof>');

  // Theorem-type environments (dissolved → bold openers)
  const THEOREM_ENVS = [
    'theorem', 'lemma', 'proposition', 'corollary',
    'definition', 'axiom', 'remark', 'example',
    'conjecture', 'hypothesis',
  ];
  for (const envName of THEOREM_ENVS) {
    const label = envName.charAt(0).toUpperCase() + envName.slice(1);
    // With optional title: \begin{hypothesis}[Some Title]
    out = out.replace(
      new RegExp(`\\\\begin\\{${envName}\\}\\[([^\\]]*)\\]`, 'g'),
      (_, title) => `<p><strong>${label}</strong> (${title}). `
    );
    // Without title
    out = out.replace(
      new RegExp(`\\\\begin\\{${envName}\\}(?!\\[)`, 'g'),
      `<p><strong>${label}.</strong> `
    );
    out = out.replace(
      new RegExp(`\\\\end\\{${envName}\\}`, 'g'),
      '</p>'
    );
  }

  // Center environment (strip, keep content)
  out = out.replace(/\\begin\{center\}/g, '');
  out = out.replace(/\\end\{center\}/g, '');

  // Quote/quotation
  out = out.replace(/\\begin\{quote\}/g, '<blockquote>');
  out = out.replace(/\\end\{quote\}/g, '</blockquote>');
  out = out.replace(/\\begin\{quotation\}/g, '<blockquote>');
  out = out.replace(/\\end\{quotation\}/g, '</blockquote>');

  return out;
}

// ── Phase 2: Convert sections → <Section> ───────────────────────────────────

function convertSections(tex) {
  let out = tex;

  function cleanTitle(raw) {
    return raw
      .replace(/\$([^$]*)\$/g, '$1')  // strip $ delimiters
      .replace(/\\[a-zA-Z]+\{([^}]*)\}/g, '$1')  // \cmd{arg} → arg
      .replace(/\\[a-zA-Z]+/g, '')  // bare \cmd
      .replace(/[{}]/g, '')
      .replace(/"/g, '\\"')
      .trim();
  }

  // \section*{Title} and \section{Title}
  out = out.replace(/\\section\*?\{((?:[^{}]|\{[^{}]*\})*)\}/g, (_, title) => {
    return `\n<Section title="${cleanTitle(title)}" level={1}>`;
  });

  out = out.replace(/\\subsection\*?\{((?:[^{}]|\{[^{}]*\})*)\}/g, (_, title) => {
    return `\n<Section title="${cleanTitle(title)}" level={2}>`;
  });

  out = out.replace(/\\subsubsection\*?\{((?:[^{}]|\{[^{}]*\})*)\}/g, (_, title) => {
    return `\n<Section title="${cleanTitle(title)}" level={3}>`;
  });

  // Now we need to close sections. This is tricky with regex alone.
  // We'll do a simple approach: insert </Section> before the next same-or-higher-level section.
  // This requires a state machine pass.
  const lines = out.split('\n');
  const result = [];
  const sectionStack = []; // stack of levels

  for (const line of lines) {
    const sectionMatch = line.match(/^<Section title="[^"]*" level=\{(\d)\}>/);
    if (sectionMatch) {
      const level = parseInt(sectionMatch[1]);
      // Close sections at same or deeper level
      while (sectionStack.length > 0 && sectionStack[sectionStack.length - 1] >= level) {
        sectionStack.pop();
        result.push('</Section>');
      }
      sectionStack.push(level);
    }
    result.push(line);
  }

  // Close any remaining open sections
  while (sectionStack.length > 0) {
    sectionStack.pop();
    result.push('</Section>');
  }

  return result.join('\n');
}

// ── Phase 3: Convert display math → <Eq>/<Align> ───────────────────────────

function convertDisplayMath(tex) {
  let out = tex;

  // \begin{equation} ... \end{equation}
  out = out.replace(/\\begin\{equation\}([\s\S]*?)\\end\{equation\}/g, (_, math) => {
    const cleaned = math.trim();
    return `<Eq>{"${mathJsx(cleaned)}"}</Eq>`;
  });

  // \begin{align} or \begin{align*}
  out = out.replace(/\\begin\{align\*?\}([\s\S]*?)\\end\{align\*?\}/g, (_, math) => {
    const cleaned = math.trim();
    return `<Align>{"${mathJsx(cleaned)}"}</Align>`;
  });

  // \begin{gather} or \begin{gather*}
  out = out.replace(/\\begin\{gather\*?\}([\s\S]*?)\\end\{gather\*?\}/g, (_, math) => {
    const cleaned = math.trim();
    return `<Eq>{"${mathJsx(cleaned)}"}</Eq>`;
  });

  // \[ ... \]
  out = out.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => {
    const cleaned = math.trim();
    return `<Eq>{"${mathJsx(cleaned)}"}</Eq>`;
  });

  // $$ ... $$ (display)
  out = out.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
    const cleaned = math.trim();
    return `<Eq>{"${mathJsx(cleaned)}"}</Eq>`;
  });

  return out;
}

// ── Phase 4: Convert inline math $...$ → <M>{"..."}</M> ────────────────────

function convertInlineMath(tex) {
  let out = tex;

  // \( ... \) → <M>
  out = out.replace(/\\\(([\s\S]*?)\\\)/g, (_, math) => {
    return `<M>{"${mathJsx(math.trim())}"}</M>`;
  });

  // $...$ (inline, but not $$)
  // Use a careful regex that matches $...$ but not $$...$$
  // We need to handle this line by line to avoid matching across paragraphs
  const lines = out.split('\n');
  const result = [];
  for (const line of lines) {
    // Skip lines that are inside <Eq> or <Align> (already converted)
    if (line.includes('<Eq>') || line.includes('<Align>')) {
      result.push(line);
      continue;
    }
    // Replace $...$ that isn't preceded/followed by another $
    let converted = line.replace(/(?<!\$)\$(?!\$)((?:[^$\\]|\\.)+?)\$(?!\$)/g, (_, math) => {
      return `<M>{"${mathJsx(math.trim())}"}</M>`;
    });
    result.push(converted);
  }
  return result.join('\n');
}

// ── Phase 5: Convert formatting commands ────────────────────────────────────

function convertFormatting(tex) {
  let out = tex;

  // \textbf{...}
  out = out.replace(/\\textbf\{((?:[^{}]|\{[^{}]*\})*)\}/g, '<strong>$1</strong>');

  // \textit{...}
  out = out.replace(/\\textit\{((?:[^{}]|\{[^{}]*\})*)\}/g, '<em>$1</em>');

  // \emph{...}
  out = out.replace(/\\emph\{((?:[^{}]|\{[^{}]*\})*)\}/g, '<em>$1</em>');

  // \textsc{...}
  out = out.replace(/\\textsc\{((?:[^{}]|\{[^{}]*\})*)\}/g, '<SmallCaps>$1</SmallCaps>');

  // \href{url}{text}
  out = out.replace(/\\href\{([^}]*)\}\{((?:[^{}]|\{[^{}]*\})*)\}/g, '<a href="$1">$2</a>');

  // \url{...}
  out = out.replace(/\\url\{([^}]*)\}/g, '<a href="$1">$1</a>');

  // \detail{...} → <MarginNote>
  out = out.replace(/\\detail\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}/g,
    '<MarginNote>$1</MarginNote>');

  // \footnote{...} → parenthetical
  out = out.replace(/\\footnote\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}/g, ' ($1)');

  // --- and -- → em/en dash
  out = out.replace(/---/g, '\u2014');
  out = out.replace(/--/g, '\u2013');

  // ~ → space
  out = out.replace(/~/g, ' ');

  // \ldots → ...
  out = out.replace(/\\ldots/g, '\u2026');
  out = out.replace(/\\dots/g, '\u2026');

  // ``...'' → "..."
  out = out.replace(/``/g, '\u201C');
  out = out.replace(/''/g, '\u201D');
  out = out.replace(/`/g, '\u2018');
  out = out.replace(/'/g, '\u2019');

  return out;
}

// ── Phase 6: Convert figures and diagrams ───────────────────────────────────

function convertFigures(tex, chapterSlug) {
  let out = tex;
  let tikzCounter = 0;

  // TikZ → <Diagram>
  out = out.replace(/\\begin\{tikzpicture\}[\s\S]*?\\end\{tikzpicture\}/g, () => {
    const src = `/diagrams/${chapterSlug}-${tikzCounter}.svg`;
    tikzCounter++;
    return `<Diagram src="${src}" />`;
  });
  out = out.replace(/\\begin\{pgfplot\}[\s\S]*?\\end\{pgfplot\}/g, () => {
    const src = `/diagrams/${chapterSlug}-${tikzCounter}.svg`;
    tikzCounter++;
    return `<Diagram src="${src}" />`;
  });

  // \begin{figure}...\end{figure}
  out = out.replace(
    /\\begin\{figure\}\*?(\[[^\]]*\])?([\s\S]*?)\\end\{figure\}\*?/g,
    (_, opts, body) => {
      const captionMatch = body.match(/\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}/);
      const caption = captionMatch ? captionMatch[1].trim() : '';
      const imgMatch = body.match(/\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}/);
      if (imgMatch) {
        const imgPath = normalizeImagePath(imgMatch[1]);
        if (caption) {
          return `<Figure src="${imgPath}" alt="${caption.replace(/"/g, '&quot;').replace(/<[^>]*>/g, '').slice(0, 100)}" caption={<>${caption}</>} />`;
        }
        return `<Figure src="${imgPath}" alt="" />`;
      }
      // Multi-image
      const imgs = [...body.matchAll(/\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}/g)];
      if (imgs.length > 0) {
        const imgTags = imgs.map(m => {
          const p = normalizeImagePath(m[1]);
          return `<Figure src="${p}" alt="" />`;
        }).join('\n');
        return imgTags;
      }
      return '';
    }
  );

  // Standalone \includegraphics
  out = out.replace(
    /\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}/g,
    (_, path) => {
      const imgPath = normalizeImagePath(path);
      return `<Figure src="${imgPath}" alt="" />`;
    }
  );

  // Strip leftover \caption
  out = out.replace(/\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}/g, '');

  return out;
}

// ── Phase 7: Convert lists ──────────────────────────────────────────────────

function convertLists(tex) {
  let out = tex;

  // Strip optional arguments from enumerate like [label=(\roman*)]
  out = out.replace(/\\begin\{enumerate\}\[[^\]]*\]/g, '\\begin{enumerate}');

  // Process list environments recursively
  // Strategy: find each list block, process items within it
  function processList(text) {
    // Find innermost list first (handles nesting)
    const listRe = /\\begin\{(itemize|enumerate)\}([\s\S]*?)\\end\{\1\}/;
    let match;
    while ((match = listRe.exec(text)) !== null) {
      const listType = match[1];
      const tag = listType === 'itemize' ? 'ul' : 'ol';
      const body = match[2];

      // Split body by \item
      const items = body.split(/\\item(?:\[([^\]]*)\])?\s*/);
      // items[0] is content before first \item (usually empty/whitespace)
      // items[1], items[2], etc. are item contents (with possible label captures)

      let listHtml = `<${tag}>\n`;

      // items array from split with capture group: [pre, label1|undefined, content1, label2|undefined, content2, ...]
      // Actually split with capture group returns: [pre, capture1, rest1, capture2, rest2, ...]
      // But our regex has optional capture, so it's: [pre, label_or_undefined, content1, label_or_undefined, content2, ...]
      for (let i = 1; i < items.length; i += 2) {
        const label = items[i]; // captured label (may be undefined)
        const content = (items[i + 1] || '').trim();
        if (!content && !label) continue;
        if (label) {
          listHtml += `<li><strong>${label}</strong> ${content}</li>\n`;
        } else {
          listHtml += `<li>${content}</li>\n`;
        }
      }
      listHtml += `</${tag}>`;

      text = text.slice(0, match.index) + listHtml + text.slice(match.index + match[0].length);
    }
    return text;
  }

  out = processList(out);
  return out;
}

// ── Phase 8: Convert tables ─────────────────────────────────────────────────

function convertTables(tex) {
  let out = tex;

  out = out.replace(/\\begin\{tabular\}\{[^}]*\}([\s\S]*?)\\end\{tabular\}/g, (_, body) => {
    let tableHtml = '<table>\n';
    const rows = body.split('\\\\').filter(r => r.trim() && !r.trim().startsWith('\\hline'));
    let isFirst = true;
    for (const row of rows) {
      if (row.includes('\\hline')) continue;
      const cells = row.split('&').map(c => c.trim().replace(/\\hline/g, ''));
      const tag = isFirst ? 'th' : 'td';
      tableHtml += '<tr>' + cells.map(c => `<${tag}>${c}</${tag}>`).join('') + '</tr>\n';
      isFirst = false;
    }
    tableHtml += '</table>';
    return tableHtml;
  });

  return out;
}

// ── Phase 9: Wrap paragraphs in <p> ─────────────────────────────────────────

function wrapParagraphs(tex) {
  const lines = tex.split('\n');
  const result = [];
  let buffer = [];
  let inBlock = 0; // depth of JSX components

  const blockTags = [
    'Section', 'Sidebar', 'Connection', 'Experiment', 'OpenQuestion',
    'Logos', 'KeyResult', 'Warning', 'TodoEmpirical', 'Historical',
    'Empirical', 'Software', 'Phenomenal', 'NormativeImplication',
    'Proof', 'WideBreath', 'WideMargin', 'Eq', 'Align', 'Diagram',
    'Figure', 'ul', 'ol', 'li', 'table', 'tr', 'th', 'td',
    'blockquote',
  ];

  function flushBuffer() {
    const text = buffer.join(' ').trim();
    if (text) {
      // Don't wrap if it's already a tag or contains only block elements
      if (text.startsWith('<') && blockTags.some(t => text.startsWith(`<${t}`))) {
        result.push(text);
      } else if (text.startsWith('<p>') || text.startsWith('</p>')) {
        result.push(text);
      } else {
        result.push(`<p>${text}</p>`);
      }
    }
    buffer = [];
  }

  for (const line of lines) {
    const trimmed = line.trim();

    // Empty line = paragraph break
    if (trimmed === '') {
      flushBuffer();
      continue;
    }

    // Track block depth
    for (const tag of blockTags) {
      const opens = (trimmed.match(new RegExp(`<${tag}[\\s>/]`, 'g')) || []).length;
      const closes = (trimmed.match(new RegExp(`</${tag}>`, 'g')) || []).length;
      inBlock += opens - closes;
    }

    // Lines that are self-contained block elements
    if (blockTags.some(t => trimmed.startsWith(`<${t}`) && trimmed.endsWith(`</${t}>`))) {
      flushBuffer();
      result.push(trimmed);
      continue;
    }

    // Lines that open/close block elements
    if (blockTags.some(t => trimmed === `<${t}>` || trimmed.startsWith(`<${t} `))) {
      flushBuffer();
      result.push(trimmed);
      continue;
    }
    if (blockTags.some(t => trimmed === `</${t}>`)) {
      flushBuffer();
      result.push(trimmed);
      continue;
    }

    // Self-closing tags like <Diagram ... />
    if (trimmed.match(/^<[A-Z]\w+[^>]*\/>$/)) {
      flushBuffer();
      result.push(trimmed);
      continue;
    }

    buffer.push(trimmed);
  }

  flushBuffer();
  return result.join('\n');
}

// ── Phase 9.5: Escape bare JSX special characters in text ────────────────────

function escapeJsxText(tex) {
  // Strategy: split the content into segments that are:
  // - JSX expressions {"..."} → leave alone
  // - JSX/HTML tags <...> → leave alone
  // - Plain text → escape { } > <
  // Use a state machine to properly handle nesting.

  const lines = tex.split('\n');
  const result = [];

  for (const line of lines) {
    let escaped = '';
    let i = 0;
    while (i < line.length) {
      // JSX expression: {"..."}
      if (line[i] === '{' && line[i + 1] === '"') {
        // Find matching "} — handle escaped quotes inside
        let j = i + 2;
        while (j < line.length) {
          if (line[j] === '\\') { j += 2; continue; } // skip escaped char
          if (line[j] === '"' && line[j + 1] === '}') { j += 2; break; }
          j++;
        }
        escaped += line.slice(i, j);
        i = j;
        continue;
      }

      // JSX expression: {<>...</>} (JSX fragment in prop)
      if (line[i] === '{' && line[i + 1] === '<') {
        // Find matching }
        let depth = 1;
        let j = i + 1;
        while (j < line.length && depth > 0) {
          if (line[j] === '{') depth++;
          if (line[j] === '}') depth--;
          j++;
        }
        escaped += line.slice(i, j);
        i = j;
        continue;
      }

      // JSX/HTML tag: <...> (opening, closing, self-closing)
      if (line[i] === '<' && (line[i + 1] === '/' || /[A-Za-z!]/.test(line[i + 1] || ''))) {
        // Find the closing > but handle nested quotes
        let j = i + 1;
        let inQuote = false;
        while (j < line.length) {
          if (line[j] === '"' && !inQuote) { inQuote = true; j++; continue; }
          if (line[j] === '"' && inQuote) { inQuote = false; j++; continue; }
          if (line[j] === '>' && !inQuote) { j++; break; }
          j++;
        }
        escaped += line.slice(i, j);
        i = j;
        continue;
      }

      // Plain text: escape JSX special chars
      const ch = line[i];
      if (ch === '{') {
        escaped += '{"{"}';
      } else if (ch === '}') {
        escaped += '{"}"}';
      } else {
        escaped += ch;
      }
      i++;
    }
    result.push(escaped);
  }

  return result.join('\n');
}

// ── Phase 10: Clean up ──────────────────────────────────────────────────────

function cleanup(tex) {
  let out = tex;

  // Remove remaining LaTeX commands that weren't caught
  out = out.replace(/\\begin\{center\}/g, '');
  out = out.replace(/\\end\{center\}/g, '');
  out = out.replace(/\\begin\{minipage\}[^]*?\\end\{minipage\}/g, '');
  out = out.replace(/\\begin\{table\}(\[[^\]]*\])?/g, '');
  out = out.replace(/\\end\{table\}/g, '');

  // Clean up extra whitespace
  out = out.replace(/\n{3,}/g, '\n\n');

  // Remove empty <p></p>
  out = out.replace(/<p>\s*<\/p>/g, '');

  // Remove stray LaTeX commands
  out = out.replace(/\\hline/g, '');
  out = out.replace(/\\\\(?!\w)/g, '');

  return out;
}

// ── Phase 11: Wrap in TSX template ──────────────────────────────────────────

function wrapTemplate(content, chapter) {
  // Collect which components are used
  const usedComponents = new Set();
  const allComponents = [
    'M', 'Eq', 'Align',
    'Section', 'Figure', 'Diagram', 'MarginNote', 'Ref', 'Proof', 'SmallCaps',
    'WideBreath', 'WideMargin',
    'Sidebar', 'Connection', 'Experiment', 'OpenQuestion', 'Logos',
    'KeyResult', 'Warning', 'TodoEmpirical', 'Historical', 'Empirical',
    'Software', 'Phenomenal', 'NormativeImplication', 'EnvBox',
  ];

  for (const comp of allComponents) {
    // Check if component is used (as opening tag)
    if (content.includes(`<${comp}`) || content.includes(`<${comp}>`)) {
      usedComponents.add(comp);
    }
  }

  const imports = usedComponents.size > 0
    ? `import { ${[...usedComponents].sort().join(', ')} } from '@/components/content';\n`
    : '';

  return `${imports}
export const metadata = {
  slug: '${chapter.slug}',
  title: '${chapter.title.replace(/'/g, "\\'")}',
  shortTitle: '${chapter.shortTitle.replace(/'/g, "\\'")}',
};

export default function ${toPascalCase(chapter.slug)}() {
  return (
    <>
${indent(content, 6)}
    </>
  );
}
`;
}

function toPascalCase(slug) {
  return slug
    .split('-')
    .map(s => s.charAt(0).toUpperCase() + s.slice(1))
    .join('');
}

function indent(text, spaces) {
  const pad = ' '.repeat(spaces);
  return text
    .split('\n')
    .map(line => line.trim() ? pad + line : '')
    .join('\n');
}

// ── Main pipeline ───────────────────────────────────────────────────────────

function convertChapter(chapter) {
  console.log(`Converting ${chapter.slug}...`);

  const texPath = resolve(BOOK, chapter.file);
  let tex = readFileSync(texPath, 'utf-8');

  // Pipeline phases
  tex = stripPreamble(tex);
  tex = convertEnvironments(tex);
  tex = convertFigures(tex, chapter.slug);
  tex = convertDisplayMath(tex);
  tex = convertInlineMath(tex);
  tex = convertFormatting(tex);
  tex = convertSections(tex);
  tex = convertLists(tex);
  tex = convertTables(tex);
  tex = wrapParagraphs(tex);
  tex = escapeJsxText(tex);
  tex = cleanup(tex);

  const tsx = wrapTemplate(tex, chapter);

  const outPath = resolve(CONTENT_DIR, `${chapter.slug}.tsx`);
  writeFileSync(outPath, tsx);
  console.log(`  → content/${chapter.slug}.tsx`);

  return tsx;
}

// Run
console.log('=== LaTeX → TSX Conversion ===\n');
mkdirSync(CONTENT_DIR, { recursive: true });

for (const chapter of CHAPTERS) {
  convertChapter(chapter);
}

console.log('\n=== Done ===');
console.log('Review each file and fix any conversion artifacts.');
