'use client';

import { useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';

export default function ReadingProgress() {
  const [progress, setProgress] = useState(0);
  const pathname = usePathname();

  const isChapter = pathname !== '/' && pathname !== '';

  useEffect(() => {
    if (!isChapter) return;

    function onScroll() {
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      if (docHeight <= 0) { setProgress(0); return; }
      setProgress(Math.min(100, Math.max(0, (scrollTop / docHeight) * 100)));
    }

    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, [isChapter, pathname]);

  if (!isChapter) return null;

  return (
    <div className="reading-progress" style={{ width: `${progress}%` }} />
  );
}
