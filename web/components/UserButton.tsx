'use client';

import { useSession, signOut } from 'next-auth/react';
import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function UserButton() {
  const { data: session, status } = useSession();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const router = useRouter();

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    if (open) document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [open]);

  if (status === 'loading') return null;

  if (!session?.user) {
    return (
      <button
        className="reader-toolbar-btn"
        onClick={() => router.push('/login')}
        title="Sign in"
        aria-label="Sign in"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
          <circle cx="12" cy="7" r="4" />
        </svg>
      </button>
    );
  }

  const initial = (session.user.name?.[0] || session.user.email?.[0] || '?').toUpperCase();

  return (
    <div className="reader-toolbar-dropdown" ref={ref}>
      <button
        className="reader-toolbar-btn user-avatar-btn"
        onClick={() => setOpen(!open)}
        title={session.user.name || session.user.email || 'Account'}
        aria-label="Account menu"
      >
        {session.user.image ? (
          <img
            src={session.user.image}
            alt=""
            width={20}
            height={20}
            style={{ borderRadius: '50%' }}
          />
        ) : (
          <span className="user-avatar-initial">{initial}</span>
        )}
      </button>
      {open && (
        <div className="reader-toolbar-menu user-menu">
          <div className="user-menu-header">
            <strong>{session.user.name}</strong>
            <span>{session.user.email}</span>
          </div>
          <button
            className="reader-toolbar-menu-link"
            onClick={() => { setOpen(false); router.push('/library'); }}
          >
            Library
          </button>
          <button
            className="reader-toolbar-menu-link"
            onClick={() => { setOpen(false); router.push('/settings'); }}
          >
            Settings
          </button>
          <button
            className="reader-toolbar-menu-link user-menu-signout"
            onClick={() => signOut({ callbackUrl: '/' })}
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
