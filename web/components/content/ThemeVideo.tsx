'use client';

import { useState, useEffect, useRef } from 'react';

interface ThemeVideoProps {
  baseName: string;
}

/**
 * Video component that swaps between dark/light video files
 * based on the current data-theme attribute on <html>.
 */
export function ThemeVideo({ baseName }: ThemeVideoProps) {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const root = document.documentElement;
    const current = root.getAttribute('data-theme');
    if (current === 'light' || current === 'dark') {
      setTheme(current);
    }

    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.attributeName === 'data-theme') {
          const val = root.getAttribute('data-theme');
          if (val === 'light' || val === 'dark') {
            setTheme(val);
          }
        }
      }
    });

    observer.observe(root, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);

  // When theme changes, restart video from current position
  useEffect(() => {
    if (videoRef.current) {
      const currentTime = videoRef.current.currentTime;
      videoRef.current.load();
      videoRef.current.currentTime = currentTime;
    }
  }, [theme]);

  return (
    <video
      ref={videoRef}
      src={`/videos/${baseName}-${theme}.mp4`}
      autoPlay
      loop
      muted
      playsInline
      style={{ width: '100%', borderRadius: 8 }}
    />
  );
}
