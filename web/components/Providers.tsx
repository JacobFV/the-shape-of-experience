'use client';

import { SessionProvider } from 'next-auth/react';
import { MobileUIProvider } from '../lib/MobileUIContext';

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <MobileUIProvider>
        {children}
      </MobileUIProvider>
    </SessionProvider>
  );
}
