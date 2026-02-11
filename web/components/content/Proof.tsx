import { ReactNode } from 'react';

export function Proof({ children }: { children: ReactNode }) {
  return (
    <div className="env-proof">
      <em>Proof.</em> {children} <span className="qed">{'\u25A1'}</span>
    </div>
  );
}
