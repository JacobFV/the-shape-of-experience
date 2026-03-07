'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
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
}

function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function AudioPlayer({ sections, chapterTitle, slug }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [everPlayed, setEverPlayed] = useState(false);
  const { setAudioAvailable, setAudioStarted, setAudioPlaying, audioToggleRef } = useMobileUI();

  const current = sections[0];
  if (!current) return null;

  // Register with mobile UI context
  useEffect(() => {
    setAudioAvailable(true);
    document.body.dataset.hasAudio = 'true';
    return () => {
      setAudioAvailable(false);
      setAudioStarted(false);
      setAudioPlaying(false);
      delete document.body.dataset.hasAudio;
      delete document.body.dataset.audioStarted;
      audioToggleRef.current = null;
    };
  }, [setAudioAvailable, setAudioStarted, setAudioPlaying, audioToggleRef]);

  // Sync playing state to context for header buttons
  useEffect(() => {
    setAudioPlaying(isPlaying);
  }, [isPlaying, setAudioPlaying]);

  // Sync everPlayed to body data attribute
  useEffect(() => {
    if (everPlayed) {
      document.body.dataset.audioStarted = 'true';
      setAudioStarted(true);
    }
  }, [everPlayed, setAudioStarted]);

  // Restore saved position on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(`audio-pos-${slug}`);
      if (saved) {
        const { id, time } = JSON.parse(saved);
        if (id === current.id && time > 0) {
          const audio = audioRef.current;
          if (audio) {
            audio.src = current.audioUrl;
            audio.load();
            audio.currentTime = time;
            setLoaded(true);
          }
        }
      }
    } catch { /* ignore */ }
  }, [slug, current.id, current.audioUrl]);

  // Save position periodically
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      try {
        localStorage.setItem(`audio-pos-${slug}`, JSON.stringify({
          id: current.id,
          time: audioRef.current?.currentTime ?? 0,
        }));
      } catch { /* ignore */ }
    }, 3000);
    return () => clearInterval(interval);
  }, [isPlaying, slug, current.id]);

  // MediaSession API
  useEffect(() => {
    if (!('mediaSession' in navigator)) return;
    navigator.mediaSession.metadata = new MediaMetadata({
      title: current.title,
      artist: 'The Shape of Experience',
      album: chapterTitle,
    });
    navigator.mediaSession.setActionHandler('play', () => audioRef.current?.play());
    navigator.mediaSession.setActionHandler('pause', () => audioRef.current?.pause());
    navigator.mediaSession.setActionHandler('seekto', (details) => {
      if (audioRef.current && details.seekTime != null) {
        audioRef.current.currentTime = details.seekTime;
      }
    });
    return () => {
      navigator.mediaSession.setActionHandler('play', null);
      navigator.mediaSession.setActionHandler('pause', null);
      navigator.mediaSession.setActionHandler('seekto', null);
    };
  }, [current, chapterTitle]);

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
  }, [isPlaying, loaded, current.audioUrl, everPlayed]);

  // Register toggle for header buttons
  useEffect(() => {
    audioToggleRef.current = togglePlay;
    return () => { audioToggleRef.current = null; };
  }, [togglePlay, audioToggleRef]);

  const handleSeek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    if (!audio || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audio.currentTime = pct * duration;
    setCurrentTime(audio.currentTime);
  }, [duration]);

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className={`audio-player${everPlayed ? ' audio-started' : ''}${isPlaying ? ' audio-active' : ''}`}>
      <audio
        ref={audioRef}
        preload="none"
        onTimeUpdate={() => { if (audioRef.current) setCurrentTime(audioRef.current.currentTime); }}
        onLoadedMetadata={() => { if (audioRef.current) { setDuration(audioRef.current.duration); setLoaded(true); } }}
        onEnded={() => setIsPlaying(false)}
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

        <div className="audio-time">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>
      </div>

      <div className="audio-progress" onClick={handleSeek}>
        <div className="audio-progress-bar" style={{ width: `${progress}%` }} />
      </div>
    </div>
  );
}
