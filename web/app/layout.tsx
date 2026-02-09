import type { Metadata } from 'next';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import './globals.css';
import Sidebar from './Sidebar';
import ReadingProgress from '../components/ReadingProgress';
import ReaderToolbar from '../components/ReaderToolbar';

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
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('soe-theme');var d=document.documentElement;if(t==='dark'||(t!=='light'&&matchMedia('(prefers-color-scheme:dark)').matches)){d.setAttribute('data-theme','dark')}else{d.setAttribute('data-theme','light')}var f=localStorage.getItem('soe-font-size');if(f){var m={small:15,medium:17,large:19,xlarge:21};if(m[f])d.style.fontSize=m[f]+'px'}}catch(e){}})();`,
          }}
        />
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
        <ReadingProgress />
        <Sidebar sectionData={sectionData} />
        <ReaderToolbar />
        <main className="main-content">
          {children}
        </main>
      </body>
    </html>
  );
}
