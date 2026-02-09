/**
 * LaTeX preprocessor for pandoc conversion.
 *
 * Strategy: Replace custom environments with unique MARKER tokens before pandoc,
 * then convert markers → HTML divs in postprocessing (after pandoc).
 * This avoids pandoc escaping our HTML.
 */

import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const MACROS_PATH = resolve(__dirname, 'katex-macros.json');
const macros = JSON.parse(readFileSync(MACROS_PATH, 'utf-8'));

// Build \newcommand preamble for pandoc
const macroPreamble = Object.entries(macros)
  .map(([cmd, def]) => {
    if (def.includes('#1') && def.includes('#2')) {
      return `\\newcommand{${cmd}}[2]{${def}}`;
    } else if (def.includes('#1')) {
      return `\\newcommand{${cmd}}[1]{${def}}`;
    }
    return `\\newcommand{${cmd}}{${def}}`;
  })
  .join('\n');

// Unique marker prefix (won't appear in normal text)
const M = 'XENVMARKERX';

// tcolorbox environments
const TCOLORBOX_ENVS = {
  sidebar: { class: 'env-sidebar', label: 'Sidebar' },
  connection: { class: 'env-connection', label: 'Existing Theory' },
  'connection*': { class: 'env-connection', label: 'Existing Theory' },
  historical: { class: 'env-historical', label: 'Historical Context' },
  empirical: { class: 'env-empirical', label: 'Empirical Grounding' },
  todo_empirical: { class: 'env-todo-empirical', label: 'Future Empirical Work' },
  openquestion: { class: 'env-openquestion', label: 'Open Question' },
  warningbox: { class: 'env-warningbox', label: 'Warning' },
  experiment: { class: 'env-experiment', label: 'Proposed Experiment' },
  software: { class: 'env-software', label: 'Software Implementation' },
  logos: { class: 'env-logos', label: null },
};

const MARGIN_ENVS = {
  keyresult: { class: 'env-keyresult', label: 'Key Result' },
  phenomenal: { class: 'env-phenomenal', label: 'Phenomenal Correspondence' },
  warning: { class: 'env-warning-margin', label: 'Warning' },
};

const THEOREM_ENVS = [
  'theorem', 'lemma', 'proposition', 'corollary',
  'definition', 'axiom', 'remark', 'example',
  'conjecture', 'hypothesis',
];

let theoremCounter = 0;
let sectionCounter = 0;
let tikzCounter = 0;

export function resetCounters() {
  theoremCounter = 0;
  sectionCounter = 0;
  tikzCounter = 0;
}

export function preprocess(tex, chapterSlug) {
  let out = tex;

  // Strip preamble/document wrappers
  out = out.replace(/\\begin\{document\}[\s\S]*?\\mainmatter/g, '');
  out = out.replace(/\\end\{document\}/g, '');

  // Strip layout commands
  out = out.replace(/\\newgeometry\{[^}]*\}/g, '');
  out = out.replace(/\\restoregeometry/g, '');
  out = out.replace(/\\twocolumn(\[[^\]]*\])?/g, '');
  out = out.replace(/\\onecolumn/g, '');
  out = out.replace(/\\clearpage/g, '');
  out = out.replace(/\\pagebreak(\[[^\]]*\])?/g, '');
  out = out.replace(/\\newpage/g, '');
  out = out.replace(/\\thispagestyle\{[^}]*\}/g, '');
  out = out.replace(/\\pagestyle\{[^}]*\}/g, '');
  out = out.replace(/\\pagenumbering\{[^}]*\}/g, '');
  out = out.replace(/\\setcounter\{[^}]*\}\{[^}]*\}/g, '');
  out = out.replace(/\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}/g, '');
  out = out.replace(/\\raggedbottom/g, '');
  out = out.replace(/\\null/g, '');
  out = out.replace(/\\(Huge|huge|LARGE|Large|large|normalsize|small|footnotesize|scriptsize|tiny)\b/g, '');
  out = out.replace(/\\vfill/g, '');
  out = out.replace(/\\vspace\*?\{[^}]*\}/g, '');
  out = out.replace(/\\hspace\*?\{[^}]*\}/g, '');
  out = out.replace(/\\fa[A-Z][a-zA-Z]*/g, '');
  out = out.replace(/\\centering/g, '');
  out = out.replace(/\\bfseries\b/g, '');
  out = out.replace(/\\itshape\b/g, '');

  // Handle \detail{...} → marker
  out = out.replace(/\\detail\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}/g,
    (_, content) => `${M}MARGINNOTE${M}${content}${M}ENDMARGINNOTE${M}`);

  // Layout environments → markers
  out = out.replace(/\\begin\{widebreath\}/g, `${M}WIDEBREATH${M}`);
  out = out.replace(/\\end\{widebreath\}/g, `${M}ENDWIDEBREATH${M}`);
  out = out.replace(/\\begin\{widemargin\}/g, `${M}WIDEMARGIN${M}`);
  out = out.replace(/\\end\{widemargin\}/g, `${M}ENDWIDEMARGIN${M}`);

  // Normimp
  out = out.replace(/\\begin\{normimp\}/g, `${M}NORMIMP${M}`);
  out = out.replace(/\\end\{normimp\}/g, `${M}ENDNORMIMP${M}`);

  // Track sections for theorem numbering
  out = out.replace(/\\section\*?\{/g, (match) => {
    if (!match.includes('*')) {
      sectionCounter++;
      theoremCounter = 0;
    }
    return match;
  });

  // tcolorbox environments → markers
  for (const [envName, info] of Object.entries(TCOLORBOX_ENVS)) {
    const escapedName = envName.replace('*', '\\*');

    if (envName === 'sidebar') {
      out = out.replace(
        /\\begin\{sidebar\}\[title=\{?([^}\]]*)\}?\]/g,
        (_, title) => `${M}BEGIN_${info.class}_TITLE_${title}${M}`
      );
      out = out.replace(
        /\\begin\{sidebar\}(?!\[)/g,
        `${M}BEGIN_${info.class}${M}`
      );
      out = out.replace(/\\end\{sidebar\}/g, `${M}END_${info.class}${M}`);
      continue;
    }

    const beginRe = new RegExp(`\\\\begin\\{${escapedName}\\}(\\[[^\\]]*\\])?`, 'g');
    const endRe = new RegExp(`\\\\end\\{${escapedName}\\}`, 'g');
    out = out.replace(beginRe, `${M}BEGIN_${info.class}${M}`);
    out = out.replace(endRe, `${M}END_${info.class}${M}`);
  }

  // Margin environments → markers
  for (const [envName, info] of Object.entries(MARGIN_ENVS)) {
    const beginRe = new RegExp(`\\\\begin\\{${envName}\\}`, 'g');
    const endRe = new RegExp(`\\\\end\\{${envName}\\}`, 'g');
    out = out.replace(beginRe, `${M}BEGIN_${info.class}${M}`);
    out = out.replace(endRe, `${M}END_${info.class}${M}`);
  }

  // Theorem environments → markers with numbering
  for (const envName of THEOREM_ENVS) {
    const envLabel = envName.charAt(0).toUpperCase() + envName.slice(1);

    // With optional label
    const beginWithLabelRe = new RegExp(`\\\\begin\\{${envName}\\}\\[([^\\]]*)\\]`, 'g');
    out = out.replace(beginWithLabelRe, (_, title) => {
      theoremCounter++;
      return `${M}BEGIN_THEOREM_${envName}_${sectionCounter}.${theoremCounter}_LABEL_${title}${M}`;
    });

    // Without label
    const beginRe = new RegExp(`\\\\begin\\{${envName}\\}(?!\\[)`, 'g');
    out = out.replace(beginRe, () => {
      theoremCounter++;
      return `${M}BEGIN_THEOREM_${envName}_${sectionCounter}.${theoremCounter}${M}`;
    });

    const endRe = new RegExp(`\\\\end\\{${envName}\\}`, 'g');
    out = out.replace(endRe, `${M}END_THEOREM${M}`);
  }

  // Proof
  out = out.replace(/\\begin\{proof\}/g, `${M}PROOF${M}`);
  out = out.replace(/\\end\{proof\}/g, `${M}ENDPROOF${M}`);

  // Algorithm environments → strip (show as text)
  out = out.replace(/\\begin\{algorithm\}[\s\S]*?\\end\{algorithm\}/g, '');
  out = out.replace(/\\begin\{algorithmic\}[\s\S]*?\\end\{algorithmic\}/g, '');

  // Handle figure environments before pandoc
  out = out.replace(
    /\\begin\{figure\}\*?(\[[^\]]*\])?([\s\S]*?)\\end\{figure\}\*?/g,
    (_, opts, body) => {
      const captionMatch = body.match(/\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}/);
      const caption = captionMatch ? captionMatch[1] : '';
      const imgMatch = body.match(/\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}/);
      if (imgMatch) {
        const imgPath = normalizeImagePath(imgMatch[1]);
        return `${M}FIGURE_${imgPath}_CAPTION_${caption}${M}`;
      }
      // Multi-image figures (subfigures)
      const imgs = [...body.matchAll(/\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}/g)];
      if (imgs.length > 0) {
        const imgTags = imgs.map(m => normalizeImagePath(m[1])).join('|');
        return `${M}FIGUREMULTI_${imgTags}_CAPTION_${caption}${M}`;
      }
      return '';
    }
  );

  // Handle standalone \includegraphics
  out = out.replace(
    /\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}/g,
    (_, path) => {
      const imgPath = normalizeImagePath(path);
      return `${M}IMG_${imgPath}${M}`;
    }
  );

  // Subfigure cleanup
  out = out.replace(/\\begin\{subfigure\}[^]*?\\end\{subfigure\}/g, '');

  // Handle wrapfigure
  out = out.replace(/\\begin\{wrapfigure\}(\[[^\]]*\])?\{[^}]*\}\{[^}]*\}/g, '');
  out = out.replace(/\\end\{wrapfigure\}/g, '');

  // Handle cutwin
  out = out.replace(/\\opencmark/g, '');
  out = out.replace(/\\closecmark/g, '');

  // TikZ → indexed marker (for SVG rendering)
  out = out.replace(/\\begin\{tikzpicture\}[\s\S]*?\\end\{tikzpicture\}/g,
    () => `${M}TIKZ_${tikzCounter++}${M}`);
  out = out.replace(/\\begin\{pgfplot\}[\s\S]*?\\end\{pgfplot\}/g,
    () => `${M}TIKZ_${tikzCounter++}${M}`);

  // Caption cleanup
  out = out.replace(/\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}/g, '');

  // Labels and refs
  out = out.replace(/\\label\{[^}]*\}/g, '');
  out = out.replace(/\\cref\{([^}]*)\}/g, '[ref]');
  out = out.replace(/\\Cref\{([^}]*)\}/g, '[Ref]');
  out = out.replace(/\\ref\{([^}]*)\}/g, '[ref]');
  out = out.replace(/\\eqref\{([^}]*)\}/g, '(ref)');

  // URL/href
  out = out.replace(/\\url\{([^}]*)\}/g, '\\href{$1}{$1}');

  // textsc
  out = out.replace(/\\textsc\{([^}]*)\}/g, `${M}SC_$1${M}`);

  // booktabs
  out = out.replace(/\\toprule/g, '\\hline');
  out = out.replace(/\\midrule/g, '\\hline');
  out = out.replace(/\\bottomrule/g, '\\hline');
  out = out.replace(/\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}/g, '$1');

  // Prepend macro definitions
  out = macroPreamble + '\n\n' + out;

  return out;
}

/**
 * Post-process pandoc HTML output: convert markers → proper HTML elements.
 */
export function postprocess(html, chapter) {
  let out = html;

  // Use simple string-based replacement for markers since regex character classes are tricky

  // tcolorbox + margin environments: BEGIN/END markers
  // Match: XENVMARKERXBEGIN_env-sidebar_TITLE_Some TitleXENVMARKERX
  out = out.replace(/XENVMARKERXBEGIN_(env-[a-z-]+)_TITLE_([\s\S]*?)XENVMARKERX/g,
    '<div class="$1"><div class="env-title">$2</div>');
  out = out.replace(/XENVMARKERXBEGIN_(env-[a-z-]+)XENVMARKERX/g,
    '<div class="$1">');
  out = out.replace(/XENVMARKERXEND_(env-[a-z-]+)XENVMARKERX/g,
    '</div>');

  // Add titles to environments that need them
  const envTitles = {
    'env-connection': 'Existing Theory',
    'env-historical': 'Historical Context',
    'env-empirical': 'Empirical Grounding',
    'env-todo-empirical': 'Future Empirical Work',
    'env-openquestion': 'Open Question',
    'env-warningbox': 'Warning',
    'env-experiment': 'Proposed Experiment',
    'env-software': 'Software Implementation',
    'env-keyresult': 'Key Result',
    'env-phenomenal': 'Phenomenal Correspondence',
    'env-warning-margin': 'Warning',
  };

  for (const [cls, title] of Object.entries(envTitles)) {
    out = out.replace(
      new RegExp(`<div class="${cls}">(?!<div class="env-title">)`, 'g'),
      `<div class="${cls}"><div class="env-title">${title}</div>`
    );
  }

  // Theorem environments with labels
  // Match: XENVMARKERXBEGIN_THEOREM_hypothesis_8.37_LABEL_Instability of NothingXENVMARKERX
  out = out.replace(/XENVMARKERXBEGIN_THEOREM_([a-z]+)_(\d+\.\d+)_LABEL_([\s\S]*?)XENVMARKERX/g,
    (_, type, num, label) => {
      const typeName = type.charAt(0).toUpperCase() + type.slice(1);
      return `<div class="env-theorem env-${type}"><strong>${typeName} ${num}</strong> (${label})`;
    }
  );
  // Theorem environments without labels
  out = out.replace(/XENVMARKERXBEGIN_THEOREM_([a-z]+)_(\d+\.\d+)XENVMARKERX/g,
    (_, type, num) => {
      const typeName = type.charAt(0).toUpperCase() + type.slice(1);
      return `<div class="env-theorem env-${type}"><strong>${typeName} ${num}</strong>`;
    }
  );
  out = out.replace(/XENVMARKERXEND_THEOREMXENVMARKERX/g, '</div>');

  // Proof
  out = out.replace(/XENVMARKERXPROOFXENVMARKERX/g,
    '<div class="env-proof"><em>Proof.</em> ');
  out = out.replace(/XENVMARKERXENDPROOFXENVMARKERX/g,
    ' <span class="qed">\u25a1</span></div>');

  // Layout wrappers
  out = out.replace(/XENVMARKERXWIDEBREATHXENVMARKERX/g, '<div class="wide-breath">');
  out = out.replace(/XENVMARKERXENDWIDEBREATHXENVMARKERX/g, '</div>');
  out = out.replace(/XENVMARKERXWIDEMARGINXENVMARKERX/g, '<div class="wide-margin">');
  out = out.replace(/XENVMARKERXENDWIDEMARGINXENVMARKERX/g, '</div>');

  // Normimp
  out = out.replace(/XENVMARKERXNORMIMPXENVMARKERX/g,
    '<div class="env-normimp"><strong>Normative Implication.</strong> ');
  out = out.replace(/XENVMARKERXENDNORMIMPXENVMARKERX/g, '</div>');

  // Margin notes (case-insensitive since pandoc may change case)
  out = out.replace(/XENVMARKERXMARGINNOTE?XENVMARKERX([\s\S]*?)XENVMARKERXENDMARGINNOTE?XENVMARKERX/gi,
    '<span class="margin-note">$1</span>');

  // Small caps
  out = out.replace(/XENVMARKERXSC_([\s\S]*?)XENVMARKERX/g,
    '<span class="small-caps">$1</span>');

  // Figures — caption may contain HTML/math, so match non-greedily up to the closing marker
  out = out.replace(
    /XENVMARKERXFIGURE_(\/images\/\S+?)_CAPTION_([\s\S]*?)XENVMARKERX/g,
    '<figure><img src="$1" alt="" /><figcaption>$2</figcaption></figure>'
  );
  out = out.replace(
    /XENVMARKERXFIGUREMULTI_([\s\S]*?)_CAPTION_([\s\S]*?)XENVMARKERX/g,
    (_, paths, caption) => {
      const imgs = paths.split('|').map(p => `<img src="${p}" alt="" />`).join('\n');
      return `<figure class="multi">${imgs}<figcaption>${caption}</figcaption></figure>`;
    }
  );
  out = out.replace(/XENVMARKERXIMG_(\/images\/[^\s]*)XENVMARKERX/g,
    '<img src="$1" alt="" />');

  // TikZ diagrams — use SVG if available, otherwise placeholder
  out = out.replace(/XENVMARKERXTIKZ_(\d+)XENVMARKERX/g, (_, idx) => {
    const slug = chapter?.slug || '';
    const svgPath = `/diagrams/${slug}-${idx}.svg`;
    return `<figure class="tikz-diagram"><img src="${svgPath}" alt="Diagram ${parseInt(idx) + 1}" onerror="this.parentElement.className='tikz-placeholder';this.parentElement.innerHTML='<em>[Diagram — see PDF version]</em>'" /></figure>`;
  });
  // Legacy unindexed markers
  out = out.replace(/XENVMARKERXTIKZXENVMARKERX/g,
    '<div class="tikz-placeholder"><em>[Diagram — see PDF version]</em></div>');

  // Clean up markers that might remain inside <p> tags
  // Pandoc wraps everything in <p>, but our divs need to be block-level
  // Move div open/close out of <p> tags
  out = out.replace(/<p>(<div[^>]*>)/g, '$1<p>');
  out = out.replace(/(<\/div>)<\/p>/g, '</p>$1');

  // Remove empty <p></p>
  out = out.replace(/<p>\s*<\/p>/g, '');

  // Clean any remaining LaTeX artifacts
  out = out.replace(/\\bfseries\b/g, '');
  out = out.replace(/\\itshape\b/g, '');
  out = out.replace(/\\small\b/g, '');
  out = out.replace(/\\footnotesize\b/g, '');
  out = out.replace(/\\scriptsize\b/g, '');

  // Add IDs to headings for section navigation
  out = out.replace(/<(h[1-3])([^>]*)>([\s\S]*?)<\/h[1-3]>/gi, (match, tag, attrs, content) => {
    if (attrs.includes(' id=')) return match; // already has an id
    const text = content.replace(/<[^>]*>/g, '').trim();
    const id = text
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '')
      .slice(0, 80);
    if (!id) return match;
    return `<${tag}${attrs} id="${id}">${content}</${tag}>`;
  });

  return out;
}

function normalizeImagePath(texPath) {
  const filename = texPath.replace(/^.*\//, '');
  const webFilename = filename.replace(/\.pdf$/, '.png');
  return `/images/${webFilename}`;
}

export { macroPreamble };
