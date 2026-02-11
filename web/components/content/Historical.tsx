import { ReactNode } from 'react';
import { EnvBox } from './EnvBox';

export function Historical({ title, children }: { title?: string; children: ReactNode }) {
  return <EnvBox className="env-historical" title={title}>{children}</EnvBox>;
}
