'use client';

import { useEffect, useRef } from 'react';

const KATEX_MACROS: Record<string, string> = {
  "\\E": "\\mathbb{E}",
  "\\R": "\\mathbb{R}",
  "\\N": "\\mathbb{N}",
  "\\Z": "\\mathbb{Z}",
  "\\prob": "\\mathbb{P}",
  "\\KL": "\\mathrm{KL}",
  "\\MI": "\\mathrm{I}",
  "\\entropy": "\\mathrm{H}",
  "\\manifold": "\\mathcal{M}",
  "\\viable": "\\mathcal{V}",
  "\\belief": "\\mathbf{b}",
  "\\state": "\\mathbf{s}",
  "\\action": "\\mathbf{a}",
  "\\obs": "\\mathbf{o}",
  "\\latent": "\\mathbf{z}",
  "\\policy": "\\pi",
  "\\freeenergy": "\\mathcal{F}",
  "\\intinfo": "\\Phi",
  "\\selfmodel": "\\mathcal{S}",
  "\\worldmodel": "\\mathcal{W}",
  "\\effrank": "r_{\\text{eff}}",
  "\\valence": "\\mathcal{V}\\hspace{-0.8pt}\\mathit{al}",
  "\\arousal": "\\mathcal{A}\\hspace{-0.5pt}\\mathit{r}",
  "\\cestructure": "\\mathcal{C\\!E}",
  "\\phenom": "\\mathcal{P}",
  "\\distinction": "\\delta",
  "\\relation": "\\rho",
  "\\Val": "\\mathcal{V}\\hspace{-0.8pt}\\mathit{al}",
  "\\Ar": "\\mathcal{A}\\hspace{-0.5pt}\\mathit{r}",
  "\\reff": "r_{\\text{eff}}",
  "\\cfweight": "\\mathrm{CF}",
  "\\selfsal": "\\mathrm{SM}",
  "\\dd": "\\mathrm{d}",
};

declare global {
  interface Window {
    renderMathInElement?: (el: HTMLElement, opts: Record<string, unknown>) => void;
  }
}

export default function MathRenderer({ html }: { html: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;

    function tryRender() {
      if (window.renderMathInElement && ref.current) {
        window.renderMathInElement(ref.current, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "\\[", right: "\\]", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\(", right: "\\)", display: false },
          ],
          macros: KATEX_MACROS,
          throwOnError: false,
        });
      }
    }

    // KaTeX scripts might not be loaded yet
    if (window.renderMathInElement) {
      tryRender();
    } else {
      // Poll until KaTeX is available
      const interval = setInterval(() => {
        if (window.renderMathInElement) {
          clearInterval(interval);
          tryRender();
        }
      }, 100);
      return () => clearInterval(interval);
    }
  }, [html]);

  return (
    <div
      ref={ref}
      className="chapter-content"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
