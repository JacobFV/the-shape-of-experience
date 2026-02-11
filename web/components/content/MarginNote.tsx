import { ReactNode } from 'react';

export function MarginNote({ children }: { children: ReactNode }) {
  return <span className="margin-note">{children}</span>;
}
