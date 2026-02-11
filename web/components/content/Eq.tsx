import katex from 'katex';
import { katexMacros } from '../../lib/katex-macros';

export function Eq({ children }: { children: string }) {
  const html = katex.renderToString(children, {
    displayMode: true,
    macros: katexMacros,
    throwOnError: false,
    trust: true,
  });
  return (
    <div className="math-display" dangerouslySetInnerHTML={{ __html: html }} />
  );
}
