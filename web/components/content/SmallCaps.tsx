import { ReactNode } from 'react';

export function SmallCaps({ children }: { children: ReactNode }) {
  return <span style={{ fontVariant: 'small-caps' }}>{children}</span>;
}
