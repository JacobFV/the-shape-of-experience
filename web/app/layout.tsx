import type { Metadata, Viewport } from 'next';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import 'katex/dist/katex.min.css';
import './globals.css';
import Sidebar from './Sidebar';
import ReadingProgress from '../components/ReadingProgress';
import ReaderToolbar from '../components/ReaderToolbar';
import MobileHeader from '../components/MobileHeader';
import Providers from '../components/Providers';
import SyncOnLogin from '../components/SyncOnLogin';
import ChatWrapper from '../components/ChatWrapper';

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: 'cover',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#2c5aa0' },
    { media: '(prefers-color-scheme: dark)', color: '#1a1a1e' },
  ],
};

export const metadata: Metadata = {
  title: 'The Shape of Experience',
  description: 'A Geometric Theory of Affect for Biological and Artificial Systems',
  manifest: '/manifest.json',
  appleWebApp: {
    capable: true,
    title: 'Shape of Exp.',
    statusBarStyle: 'default',
  },
  icons: {
    icon: [
      { url: '/icons/favicon-32.png', sizes: '32x32', type: 'image/png' },
      { url: '/icons/icon-192.png', sizes: '192x192', type: 'image/png' },
    ],
    apple: '/icons/apple-touch-icon.png',
  },
};

function loadSectionData(): Record<string, { level: number; id: string; text: string }[]> {
  const metaPath = join(process.cwd(), 'public', 'metadata.json');
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
            __html: `(function(){try{var d=document.documentElement;var t=localStorage.getItem('soe-theme');var isDark=t==='dark'||(t!=='light'&&matchMedia('(prefers-color-scheme:dark)').matches);d.setAttribute('data-theme',isDark?'dark':'light');var f=localStorage.getItem('soe-font-size');if(f){var m={small:15,medium:17,large:19,xlarge:21};if(m[f])d.style.fontSize=m[f]+'px'}var ff=localStorage.getItem('soe-font-family');var fonts={georgia:"Georgia,'Times New Roman',serif",palatino:"'Palatino Linotype','Book Antiqua',Palatino,serif",'system-sans':"-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif",inter:"'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif",mono:"'SF Mono','Fira Code','JetBrains Mono',monospace"};if(ff&&fonts[ff])d.style.setProperty('--font-body',fonts[ff]);var ac=localStorage.getItem('soe-accent');var presets={blue:['#2c5aa0','#e8f0fe','#6b9edd','#2a3a52'],warm:['#b07520','#fef3e2','#d4932a','#3a2a18'],forest:['#3a7a3a','#e8f5e8','#6ab06a','#1e2a1e'],plum:['#7744aa','#f3e8fe','#b088dd','#261e30']};if(ac&&presets[ac]){var p=presets[ac];d.style.setProperty('--accent',isDark?p[2]:p[0]);d.style.setProperty('--accent-light',isDark?p[3]:p[1])}}catch(e){}})();`,
          }}
        />
        <script
          dangerouslySetInnerHTML={{
            __html: `if('serviceWorker' in navigator){window.addEventListener('load',function(){navigator.serviceWorker.register('/sw.js')})}`,
          }}
        />
      </head>
      <body>
        <Providers>
          <ReadingProgress />
          <MobileHeader />
          <Sidebar sectionData={sectionData} />
          <ReaderToolbar />
          <SyncOnLogin />
          <ChatWrapper />
          <main className="main-content">
            {children}
          </main>
        </Providers>
      </body>
    </html>
  );
}
