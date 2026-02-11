import { ReactNode } from 'react';
import { EnvBox } from './EnvBox';

export function Phenomenal({ title, children }: { title?: string; children: ReactNode }) {
  return <EnvBox className="env-phenomenal" title={title}>{children}</EnvBox>;
}
