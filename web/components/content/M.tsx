import katex from 'katex';
import { katexMacros } from '../../lib/katex-macros';

export function M({ children }: { children: string }) {
  const html = katex.renderToString(children, {
    displayMode: false,
    macros: katexMacros,
    throwOnError: false,
    trust: true,
  });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
