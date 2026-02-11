'use client';

import { createContext, useContext, useState, useRef, type ReactNode } from 'react';

interface MobileUIContextType {
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  audioAvailable: boolean;
  setAudioAvailable: (v: boolean) => void;
  audioStarted: boolean;
  setAudioStarted: (v: boolean) => void;
  audioToggleRef: React.MutableRefObject<(() => void) | null>;
}

const MobileUIContext = createContext<MobileUIContextType | null>(null);

export function MobileUIProvider({ children }: { children: ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [audioAvailable, setAudioAvailable] = useState(false);
  const [audioStarted, setAudioStarted] = useState(false);
  const audioToggleRef = useRef<(() => void) | null>(null);

  return (
    <MobileUIContext.Provider value={{
      sidebarOpen, setSidebarOpen,
      audioAvailable, setAudioAvailable,
      audioStarted, setAudioStarted,
      audioToggleRef,
    }}>
      {children}
    </MobileUIContext.Provider>
  );
}

export function useMobileUI() {
  const ctx = useContext(MobileUIContext);
  if (!ctx) throw new Error('useMobileUI must be used within MobileUIProvider');
  return ctx;
}
