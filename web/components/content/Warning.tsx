import { ReactNode } from 'react';
import { EnvBox } from './EnvBox';

export function Warning({ title, children }: { title?: string; children: ReactNode }) {
  return <EnvBox className="env-warningbox" title={title}>{children}</EnvBox>;
}
