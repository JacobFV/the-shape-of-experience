'use client';

import { useState, useEffect, useRef } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { setProfileImageCache } from '../../lib/hooks/useProfileImage';

const MAX_SIZE_BYTES = 512 * 1024;
const TARGET_DIMENSION = 256;

function compressImage(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const scale = Math.min(TARGET_DIMENSION / img.width, TARGET_DIMENSION / img.height, 1);
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      // Try JPEG at decreasing quality until under limit
      for (let q = 0.9; q >= 0.3; q -= 0.1) {
        const dataUrl = canvas.toDataURL('image/jpeg', q);
        if (dataUrl.length * 0.75 <= MAX_SIZE_BYTES) {
          resolve(dataUrl);
          return;
        }
      }
      // Last resort: smaller dimensions
      const small = document.createElement('canvas');
      small.width = 128;
      small.height = 128;
      small.getContext('2d')!.drawImage(img, 0, 0, 128, 128);
      resolve(small.toDataURL('image/jpeg', 0.7));
    };
    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = URL.createObjectURL(file);
  });
}

// Preset avatars — small inline SVG data URIs
const PRESET_AVATARS = [
  // Warm gradient circle
  `data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><radialGradient id="a"><stop offset="0%" stop-color="#f5a623"/><stop offset="100%" stop-color="#d4451a"/></radialGradient></defs><circle cx="32" cy="32" r="32" fill="url(#a)"/><circle cx="32" cy="32" r="16" fill="none" stroke="#fff" stroke-width="1.5" opacity=".4"/></svg>')}`,
  // Cool gradient circle
  `data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><radialGradient id="a"><stop offset="0%" stop-color="#7eb8da"/><stop offset="100%" stop-color="#2c5282"/></radialGradient></defs><circle cx="32" cy="32" r="32" fill="url(#a)"/><circle cx="32" cy="32" r="16" fill="none" stroke="#fff" stroke-width="1.5" opacity=".4"/></svg>')}`,
  // Viability manifold (green gradient)
  `data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><radialGradient id="a"><stop offset="0%" stop-color="#68d391"/><stop offset="100%" stop-color="#276749"/></radialGradient></defs><circle cx="32" cy="32" r="32" fill="url(#a)"/><path d="M16 40 Q32 16 48 40" fill="none" stroke="#fff" stroke-width="1.5" opacity=".5"/></svg>')}`,
  // Purple nebula
  `data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><radialGradient id="a"><stop offset="0%" stop-color="#b794f4"/><stop offset="100%" stop-color="#553c9a"/></radialGradient></defs><circle cx="32" cy="32" r="32" fill="url(#a)"/><circle cx="24" cy="28" r="6" fill="#fff" opacity=".15"/><circle cx="40" cy="36" r="8" fill="#fff" opacity=".1"/></svg>')}`,
  // Topology rings
  `data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><circle cx="32" cy="32" r="32" fill="#2d3748"/><circle cx="32" cy="32" r="20" fill="none" stroke="#e2e8f0" stroke-width="1" opacity=".3"/><circle cx="32" cy="32" r="14" fill="none" stroke="#e2e8f0" stroke-width="1" opacity=".5"/><circle cx="32" cy="32" r="8" fill="none" stroke="#e2e8f0" stroke-width="1.5" opacity=".7"/><circle cx="32" cy="32" r="3" fill="#e2e8f0" opacity=".8"/></svg>')}`,
  // Sunrise
  `data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><linearGradient id="a" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#2b6cb0"/><stop offset="60%" stop-color="#ed8936"/><stop offset="100%" stop-color="#f6e05e"/></linearGradient></defs><circle cx="32" cy="32" r="32" fill="url(#a)"/><circle cx="32" cy="38" r="10" fill="#f6e05e" opacity=".6"/></svg>')}`,
  // Monochrome waves
  `data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><circle cx="32" cy="32" r="32" fill="#1a202c"/><path d="M0 36 Q16 28 32 36 Q48 44 64 36" fill="none" stroke="#a0aec0" stroke-width="1.5" opacity=".5"/><path d="M0 32 Q16 24 32 32 Q48 40 64 32" fill="none" stroke="#a0aec0" stroke-width="1" opacity=".3"/><path d="M0 28 Q16 20 32 28 Q48 36 64 28" fill="none" stroke="#a0aec0" stroke-width=".8" opacity=".2"/></svg>')}`,
  // Rose
  `data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><radialGradient id="a"><stop offset="0%" stop-color="#feb2b2"/><stop offset="100%" stop-color="#9b2c2c"/></radialGradient></defs><circle cx="32" cy="32" r="32" fill="url(#a)"/><circle cx="32" cy="32" r="10" fill="none" stroke="#fff" stroke-width="1" opacity=".3"/><circle cx="32" cy="32" r="6" fill="#fff" opacity=".15"/></svg>')}`,
];

interface Settings {
  showCommunityNotes: boolean;
  displayName: string;
  bio: string;
  profileImage: string | null;
}

export default function SettingsPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [settings, setSettings] = useState<Settings>({
    showCommunityNotes: true,
    displayName: '',
    bio: '',
    profileImage: null,
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login');
      return;
    }
    if (status !== 'authenticated') return;

    fetch('/api/settings')
      .then((r) => r.json())
      .then((data) => {
        setSettings({
          showCommunityNotes: data.showCommunityNotes ?? true,
          displayName: data.displayName || session?.user?.name || '',
          bio: data.bio || '',
          profileImage: data.profileImage || null,
        });
      })
      .catch(() => {});
  }, [status, session, router]);

  if (status === 'loading') return <div className="app-page"><p>Loading...</p></div>;
  if (status === 'unauthenticated') return null;

  async function save() {
    setSaving(true);
    await fetch('/api/settings', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    setProfileImageCache(settings.profileImage);
    setSaving(false);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  async function exportData() {
    const [annotations, bookmarks] = await Promise.all([
      fetch('/api/annotations').then((r) => r.json()),
      fetch('/api/bookmarks').then((r) => r.json()),
    ]);
    const data = JSON.stringify({ annotations, bookmarks, settings }, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'shape-of-experience-data.json';
    a.click();
    URL.revokeObjectURL(url);
  }

  const [showAvatarPicker, setShowAvatarPicker] = useState(false);
  const [compressing, setCompressing] = useState(false);

  async function handleImageUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size <= MAX_SIZE_BYTES) {
      const reader = new FileReader();
      reader.onload = () => {
        setSettings((s) => ({ ...s, profileImage: reader.result as string }));
      };
      reader.readAsDataURL(file);
    } else {
      setCompressing(true);
      try {
        const dataUrl = await compressImage(file);
        setSettings((s) => ({ ...s, profileImage: dataUrl }));
      } catch {
        alert('Could not process image');
      }
      setCompressing(false);
    }
    // Reset input so the same file can be re-selected
    e.target.value = '';
  }

  return (
    <div className="app-page">
      <h1>Settings</h1>

      <div className="settings-section">
        <h2>Profile</h2>
        <div className="settings-profile-picture">
          <div
            className="settings-avatar"
            onClick={() => fileInputRef.current?.click()}
          >
            {settings.profileImage ? (
              <img src={settings.profileImage} alt="Profile" />
            ) : session?.user?.image ? (
              <img src={session.user.image} alt="Profile" />
            ) : (
              <span className="settings-avatar-initial">
                {(session?.user?.name?.[0] || '?').toUpperCase()}
              </span>
            )}
            <div className="settings-avatar-overlay">
              {compressing ? (
                <span style={{ fontSize: '0.6rem' }}>Resizing...</span>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
                  <circle cx="12" cy="13" r="4" />
                </svg>
              )}
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            style={{ display: 'none' }}
          />
          <div className="settings-avatar-actions">
            <button
              className="settings-avatar-library-btn"
              onClick={() => setShowAvatarPicker((v) => !v)}
            >
              {showAvatarPicker ? 'Hide presets' : 'Choose a preset'}
            </button>
            {settings.profileImage && (
              <button
                className="settings-remove-avatar"
                onClick={() => setSettings({ ...settings, profileImage: null })}
              >
                Remove
              </button>
            )}
          </div>
        </div>
        {showAvatarPicker && (
          <div className="settings-avatar-grid">
            {PRESET_AVATARS.map((src, i) => (
              <button
                key={i}
                className={`settings-avatar-preset${settings.profileImage === src ? ' active' : ''}`}
                onClick={() => {
                  setSettings({ ...settings, profileImage: src });
                  setShowAvatarPicker(false);
                }}
              >
                <img src={src} alt={`Preset ${i + 1}`} />
              </button>
            ))}
          </div>
        )}
        <div className="settings-field">
          <label>Display name</label>
          <input
            type="text"
            value={settings.displayName}
            onChange={(e) => setSettings({ ...settings, displayName: e.target.value })}
          />
        </div>
        <div className="settings-field">
          <label>Bio</label>
          <textarea
            value={settings.bio}
            onChange={(e) => setSettings({ ...settings, bio: e.target.value })}
            placeholder="A short note about yourself..."
          />
        </div>
      </div>

      <div className="settings-section">
        <h2>Preferences</h2>
        <div className="settings-toggle">
          <input
            type="checkbox"
            id="community-notes"
            checked={settings.showCommunityNotes}
            onChange={(e) => setSettings({ ...settings, showCommunityNotes: e.target.checked })}
          />
          <label htmlFor="community-notes">
            Show community notes from other readers
          </label>
        </div>
      </div>

      <div className="settings-section">
        <h2>Data</h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
          Export all your highlights, notes, and bookmarks as JSON.
        </p>
        <button className="settings-save" onClick={exportData}>
          Export data
        </button>
      </div>

      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginTop: '1.5rem' }}>
        <button className="auth-submit" onClick={save} disabled={saving} style={{ marginTop: 0 }}>
          {saving ? 'Saving...' : 'Save settings'}
        </button>
        {saved && <span style={{ fontSize: '0.8rem', color: 'var(--accent)' }}>Saved</span>}
      </div>
    </div>
  );
}
