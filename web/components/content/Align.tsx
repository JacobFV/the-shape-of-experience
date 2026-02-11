import katex from 'katex';
import { katexMacros } from '../../lib/katex-macros';

export function Align({ children }: { children: string }) {
  const wrapped = `\\begin{aligned}${children}\\end{aligned}`;
  const html = katex.renderToString(wrapped, {
    displayMode: true,
    macros: katexMacros,
    throwOnError: false,
    trust: true,
  });
  return (
    <div className="math-display" dangerouslySetInnerHTML={{ __html: html }} />
  );
}
