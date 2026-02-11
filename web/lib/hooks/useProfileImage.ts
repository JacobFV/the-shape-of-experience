'use client';

import { useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';

const STORAGE_KEY = 'soe-profile-image';

export function useProfileImage(): string | null {
  const { data: session, status } = useSession();
  const [profileImage, setProfileImage] = useState<string | null>(null);

  useEffect(() => {
    // Check localStorage first for immediate display
    const cached = localStorage.getItem(STORAGE_KEY);
    if (cached) setProfileImage(cached);

    if (status !== 'authenticated') return;

    // Fetch from API to get latest
    fetch('/api/settings')
      .then(r => r.json())
      .then(data => {
        if (data.profileImage) {
          setProfileImage(data.profileImage);
          localStorage.setItem(STORAGE_KEY, data.profileImage);
        } else {
          setProfileImage(null);
          localStorage.removeItem(STORAGE_KEY);
        }
      })
      .catch(() => {});
  }, [status]);

  // Fallback to session image (from OAuth)
  return profileImage || session?.user?.image || null;
}

export function setProfileImageCache(dataUrl: string | null) {
  if (dataUrl) {
    localStorage.setItem(STORAGE_KEY, dataUrl);
  } else {
    localStorage.removeItem(STORAGE_KEY);
  }
}
