'use client';

import { useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';

interface Settings {
  showCommunityNotes: boolean;
  displayName: string;
  bio: string;
}

export default function SettingsPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [settings, setSettings] = useState<Settings>({
    showCommunityNotes: true,
    displayName: '',
    bio: '',
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

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

  return (
    <div className="app-page">
      <h1>Settings</h1>

      <div className="settings-section">
        <h2>Profile</h2>
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
