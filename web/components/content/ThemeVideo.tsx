'use client';

import { useState, useEffect, useRef, useCallback } from 'react';

interface ThemeVideoProps {
  baseName: string;
}

/**
 * Video component that swaps between dark/light video files
 * based on the current data-theme attribute on <html>.
 * Preserves playback position across theme switches.
 */
export function ThemeVideo({ baseName }: ThemeVideoProps) {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const videoRef = useRef<HTMLVideoElement>(null);
  const savedTime = useRef(0);

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
            // Save current playback position before theme switch
            if (videoRef.current) {
              savedTime.current = videoRef.current.currentTime;
            }
            setTheme(val);
          }
        }
      }
    });

    observer.observe(root, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);

  // Restore playback position after new source loads
  const handleLoadedData = useCallback(() => {
    if (videoRef.current && savedTime.current > 0) {
      videoRef.current.currentTime = savedTime.current;
      savedTime.current = 0;
    }
  }, []);

  return (
    <video
      ref={videoRef}
      src={`/videos/${baseName}-${theme}.mp4`}
      autoPlay
      loop
      muted
      playsInline
      onLoadedData={handleLoadedData}
      style={{ width: '100%', borderRadius: 8 }}
    />
  );
}
