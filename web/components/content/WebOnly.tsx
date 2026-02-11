import { ReactNode } from 'react';

export function WebOnly({ children }: { children: ReactNode }) {
  return <div className="web-only">{children}</div>;
}
