import { ReactNode } from 'react';

export function WideMargin({ children }: { children: ReactNode }) {
  return <div className="wide-margin">{children}</div>;
}
