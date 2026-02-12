'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useMobileUI } from '../lib/MobileUIContext';

export interface AudioSection {
  id: string;
  title: string;
  audioUrl: string;
}

interface AudioPlayerProps {
  sections: AudioSection[];
  chapterTitle: string;
  slug: string;
  nextChapterHref?: string;
}

function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function AudioPlayer({ sections, chapterTitle, slug, nextChapterHref }: AudioPlayerProps) {
  const router = useRouter();
  const audioRef = useRef<HTMLAudioElement>(null);
  const playOnLoadRef = useRef(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [everPlayed, setEverPlayed] = useState(false);
  const { setAudioAvailable, setAudioStarted, audioToggleRef } = useMobileUI();

  const current = sections[currentIndex];

  // Register with mobile UI context
  useEffect(() => {
    setAudioAvailable(true);
    document.body.dataset.hasAudio = 'true';
    return () => {
      setAudioAvailable(false);
      setAudioStarted(false);
      delete document.body.dataset.hasAudio;
      delete document.body.dataset.audioStarted;
      audioToggleRef.current = null;
    };
  }, [setAudioAvailable, setAudioStarted, audioToggleRef]);

  // Sync everPlayed to body data attribute
  useEffect(() => {
    if (everPlayed) {
      document.body.dataset.audioStarted = 'true';
      setAudioStarted(true);
    }
  }, [everPlayed, setAudioStarted]);

  // Check for auto-continue flag from previous chapter (must run before currentIndex effect)
  useEffect(() => {
    try {
      if (localStorage.getItem('audio-continue')) {
        localStorage.removeItem('audio-continue');
        playOnLoadRef.current = true;
      }
    } catch { /* ignore */ }
  }, []);

  // Restore saved position on mount (skip if auto-continuing from previous chapter)
  useEffect(() => {
    if (playOnLoadRef.current) return;
    try {
      const saved = localStorage.getItem(`audio-pos-${slug}`);
      if (saved) {
        const { index, time } = JSON.parse(saved);
        if (index >= 0 && index < sections.length) {
          setCurrentIndex(index);
          // Time will be restored when audio loads
        }
      }
    } catch { /* ignore */ }
  }, [slug, sections.length]);

  // Save position periodically
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      try {
        localStorage.setItem(`audio-pos-${slug}`, JSON.stringify({
          index: currentIndex,
          time: audioRef.current?.currentTime ?? 0,
        }));
      } catch { /* ignore */ }
    }, 3000);
    return () => clearInterval(interval);
  }, [isPlaying, currentIndex, slug]);

  // MediaSession API — integrates with OS media controls (lock screen, notification center, etc.)
  useEffect(() => {
    if (!('mediaSession' in navigator)) return;

    navigator.mediaSession.metadata = new MediaMetadata({
      title: current?.title ?? chapterTitle,
      artist: 'The Shape of Experience',
      album: chapterTitle,
    });

    navigator.mediaSession.setActionHandler('play', () => {
      audioRef.current?.play();
    });
    navigator.mediaSession.setActionHandler('pause', () => {
      audioRef.current?.pause();
    });
    navigator.mediaSession.setActionHandler('previoustrack', () => {
      if (currentIndex > 0) selectSection(currentIndex - 1);
    });
    navigator.mediaSession.setActionHandler('nexttrack', () => {
      if (currentIndex < sections.length - 1) selectSection(currentIndex + 1);
    });
    navigator.mediaSession.setActionHandler('seekto', (details) => {
      if (audioRef.current && details.seekTime != null) {
        audioRef.current.currentTime = details.seekTime;
      }
    });

    return () => {
      navigator.mediaSession.setActionHandler('play', null);
      navigator.mediaSession.setActionHandler('pause', null);
      navigator.mediaSession.setActionHandler('previoustrack', null);
      navigator.mediaSession.setActionHandler('nexttrack', null);
      navigator.mediaSession.setActionHandler('seekto', null);
    };
  }, [current, chapterTitle, currentIndex, sections.length]);

  // Update MediaSession position state
  useEffect(() => {
    if (!('mediaSession' in navigator) || !isPlaying) return;
    try {
      navigator.mediaSession.setPositionState({
        duration: duration || 0,
        playbackRate: 1,
        position: Math.min(currentTime, duration || 0),
      });
    } catch { /* some browsers don't support this */ }
  }, [currentTime, duration, isPlaying]);

  const selectSection = useCallback((index: number, play = false) => {
    if (play) playOnLoadRef.current = true;
    setCurrentIndex(index);
    setCurrentTime(0);
    setDuration(0);
    setDropdownOpen(false);
    // Audio src change will trigger load; if we were playing, continue playing
    // We set loaded false so the new source loads fresh
    setLoaded(false);
  }, []);

  // When currentIndex changes, update audio src and potentially play
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !current) return;

    const shouldPlay = isPlaying || playOnLoadRef.current;
    playOnLoadRef.current = false;

    audio.src = current.audioUrl;
    audio.load();

    // Restore saved time if this is the initial load
    try {
      const saved = localStorage.getItem(`audio-pos-${slug}`);
      if (saved) {
        const { index, time } = JSON.parse(saved);
        if (index === currentIndex && time > 0) {
          audio.currentTime = time;
        }
      }
    } catch { /* ignore */ }

    if (shouldPlay) {
      audio.play().then(() => {
        setIsPlaying(true);
        setEverPlayed(true);
      }).catch(() => setIsPlaying(false));
    }
  }, [currentIndex, current?.audioUrl]);

  // Auto-advance to next section when current one ends
  const handleEnded = useCallback(() => {
    if (currentIndex < sections.length - 1) {
      // Auto-advance to next section within chapter
      selectSection(currentIndex + 1);
      // Keep playing flag true so next section auto-plays
    } else if (nextChapterHref) {
      // Last section finished — advance to next chapter
      try {
        localStorage.setItem('audio-continue', 'true');
      } catch { /* ignore */ }
      router.push(nextChapterHref);
    } else {
      // Last section of last chapter
      setIsPlaying(false);
    }
  }, [currentIndex, sections.length, selectSection, nextChapterHref, router]);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    if (!loaded) {
      setLoaded(true);
      audio.src = current.audioUrl;
      audio.load();
    }

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.play().then(() => {
        setIsPlaying(true);
        if (!everPlayed) setEverPlayed(true);
      }).catch(() => {});
    }
  }, [isPlaying, loaded, current?.audioUrl, everPlayed]);

  // Register toggle function for mobile header
  useEffect(() => {
    audioToggleRef.current = togglePlay;
    return () => { audioToggleRef.current = null; };
  }, [togglePlay, audioToggleRef]);

  // Listen for play-section custom events (from paragraph play buttons)
  useEffect(() => {
    function onPlaySection(e: Event) {
      const detail = (e as CustomEvent).detail;
      if (!detail?.headingId) {
        // No heading — just start playing current section
        if (!isPlaying) togglePlay();
        return;
      }

      // Find the section that matches this heading
      const headingSlug = detail.headingId
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-|-$/g, '');

      let bestIdx = 0;
      for (let i = 0; i < sections.length; i++) {
        const sectionSlug = sections[i].id.toLowerCase();
        if (sectionSlug === headingSlug || headingSlug.includes(sectionSlug) || sectionSlug.includes(headingSlug)) {
          bestIdx = i;
          break;
        }
      }

      if (bestIdx !== currentIndex) {
        selectSection(bestIdx);
      }
      // Start playing
      setTimeout(() => {
        const audio = audioRef.current;
        if (audio) {
          audio.play().then(() => {
            setIsPlaying(true);
            if (!everPlayed) setEverPlayed(true);
          }).catch(() => {});
        }
      }, 100);
    }

    window.addEventListener('play-section', onPlaySection);
    return () => window.removeEventListener('play-section', onPlaySection);
  }, [sections, currentIndex, isPlaying, togglePlay, selectSection, everPlayed]);

  // Broadcast current playback state for paragraph play buttons
  useEffect(() => {
    window.dispatchEvent(new CustomEvent('audio-state-change', {
      detail: { isPlaying, currentSectionId: sections[currentIndex]?.id || '' }
    }));
  }, [isPlaying, currentIndex, sections]);

  const handleTimeUpdate = useCallback(() => {
    const audio = audioRef.current;
    if (audio) setCurrentTime(audio.currentTime);
  }, []);

  const handleLoadedMetadata = useCallback(() => {
    const audio = audioRef.current;
    if (audio) {
      setDuration(audio.duration);
      setLoaded(true);
    }
  }, []);

  const handleSeek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    if (!audio || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audio.currentTime = pct * duration;
    setCurrentTime(audio.currentTime);
  }, [duration]);

  if (!sections.length) return null;

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className={`audio-player${everPlayed ? ' audio-started' : ''}${isPlaying ? ' audio-active' : ''}`}>
      <audio
        ref={audioRef}
        preload="none"
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={handleEnded}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
      />

      <div className="audio-player-top">
        <button className="audio-play-btn" onClick={togglePlay} aria-label={isPlaying ? 'Pause' : 'Play'}>
          {isPlaying ? (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        <div className="audio-info">
          <div className="audio-section-title">{current?.title}</div>
          <div className="audio-time">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>
        </div>

        <div className="audio-controls-right">
          <button
            className="audio-prev-btn"
            onClick={() => currentIndex > 0 && selectSection(currentIndex - 1, true)}
            disabled={currentIndex === 0}
            aria-label="Previous section"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M6 6h2v12H6zm3.5 6l8.5 6V6z" />
            </svg>
          </button>
          <button
            className="audio-next-btn"
            onClick={() => currentIndex < sections.length - 1 && selectSection(currentIndex + 1, true)}
            disabled={currentIndex === sections.length - 1}
            aria-label="Next section"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M16 18h2V6h-2zm-8.5-6l8.5 6V6z" transform="scale(-1,1) translate(-24,0)" />
            </svg>
          </button>

          {sections.length > 1 && (
            <div className="audio-dropdown-wrap">
              <button
                className="audio-dropdown-btn"
                onClick={() => setDropdownOpen(!dropdownOpen)}
                aria-label="Select section"
              >
                {currentIndex + 1}/{sections.length}
                <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor" style={{ marginLeft: 4 }}>
                  <path d="M7 10l5 5 5-5z" />
                </svg>
              </button>
              {dropdownOpen && (
                <div className="audio-dropdown-menu">
                  {sections.map((s, i) => (
                    <button
                      key={s.id}
                      className={`audio-dropdown-item ${i === currentIndex ? 'active' : ''}`}
                      onClick={() => selectSection(i, true)}
                    >
                      <span className="audio-dropdown-num">{i + 1}.</span>
                      {s.title}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="audio-progress" onClick={handleSeek}>
        <div className="audio-progress-bar" style={{ width: `${progress}%` }} />
      </div>
    </div>
  );
}
