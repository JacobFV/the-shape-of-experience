import type { Metadata } from 'next';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import './globals.css';
import Sidebar from './Sidebar';

export const metadata: Metadata = {
  title: 'The Shape of Experience',
  description: 'A Geometric Theory of Affect for Biological and Artificial Systems',
};

function loadSectionData(): Record<string, { level: number; id: string; text: string }[]> {
  const metaPath = join(process.cwd(), 'generated', 'chapters', 'metadata.json');
  if (!existsSync(metaPath)) return {};
  try {
    const raw = JSON.parse(readFileSync(metaPath, 'utf-8'));
    const result: Record<string, { level: number; id: string; text: string }[]> = {};
    for (const ch of raw) {
      if (ch.sections?.length) result[ch.slug] = ch.sections;
    }
    return result;
  } catch {
    return {};
  }
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const sectionData = loadSectionData();

  return (
    <html lang="en">
      <head>
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.18/dist/katex.min.css"
          crossOrigin="anonymous"
        />
        <script
          defer
          src="https://cdn.jsdelivr.net/npm/katex@0.16.18/dist/katex.min.js"
          crossOrigin="anonymous"
        />
        <script
          defer
          src="https://cdn.jsdelivr.net/npm/katex@0.16.18/dist/contrib/auto-render.min.js"
          crossOrigin="anonymous"
        />
      </head>
      <body>
        <Sidebar sectionData={sectionData} />
        <main className="main-content">
          {children}
        </main>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              document.addEventListener("DOMContentLoaded", function() {
                if (typeof renderMathInElement !== 'undefined') {
                  renderMathInElement(document.body, {
                    delimiters: [
                      {left: "$$", right: "$$", display: true},
                      {left: "\\\\[", right: "\\\\]", display: true},
                      {left: "$", right: "$", display: false},
                      {left: "\\\\(", right: "\\\\)", display: false}
                    ],
                    macros: {
                      "\\\\E": "\\\\mathbb{E}",
                      "\\\\R": "\\\\mathbb{R}",
                      "\\\\N": "\\\\mathbb{N}",
                      "\\\\Z": "\\\\mathbb{Z}",
                      "\\\\prob": "\\\\mathbb{P}",
                      "\\\\KL": "\\\\mathrm{KL}",
                      "\\\\MI": "\\\\mathrm{I}",
                      "\\\\entropy": "\\\\mathrm{H}",
                      "\\\\manifold": "\\\\mathcal{M}",
                      "\\\\viable": "\\\\mathcal{V}",
                      "\\\\belief": "\\\\mathbf{b}",
                      "\\\\state": "\\\\mathbf{s}",
                      "\\\\action": "\\\\mathbf{a}",
                      "\\\\obs": "\\\\mathbf{o}",
                      "\\\\latent": "\\\\mathbf{z}",
                      "\\\\policy": "\\\\pi",
                      "\\\\freeenergy": "\\\\mathcal{F}",
                      "\\\\intinfo": "\\\\Phi",
                      "\\\\selfmodel": "\\\\mathcal{S}",
                      "\\\\worldmodel": "\\\\mathcal{W}",
                      "\\\\effrank": "r_{\\\\text{eff}}",
                      "\\\\valence": "\\\\mathcal{V}\\\\hspace{-0.8pt}\\\\mathit{al}",
                      "\\\\arousal": "\\\\mathcal{A}\\\\hspace{-0.5pt}\\\\mathit{r}",
                      "\\\\cestructure": "\\\\mathcal{C\\\\!E}",
                      "\\\\phenom": "\\\\mathcal{P}",
                      "\\\\distinction": "\\\\delta",
                      "\\\\relation": "\\\\rho",
                      "\\\\Val": "\\\\mathcal{V}\\\\hspace{-0.8pt}\\\\mathit{al}",
                      "\\\\Ar": "\\\\mathcal{A}\\\\hspace{-0.5pt}\\\\mathit{r}",
                      "\\\\reff": "r_{\\\\text{eff}}",
                      "\\\\cfweight": "\\\\mathrm{CF}",
                      "\\\\selfsal": "\\\\mathrm{SM}",
                      "\\\\dd": "\\\\mathrm{d}"
                    },
                    throwOnError: false
                  });
                }
              });
            `,
          }}
        />
      </body>
    </html>
  );
}
