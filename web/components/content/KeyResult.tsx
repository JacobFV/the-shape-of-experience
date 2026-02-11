import { ReactNode } from 'react';
import { EnvBox } from './EnvBox';

export function KeyResult({ title, children }: { title?: string; children: ReactNode }) {
  return <EnvBox className="env-keyresult" title={title}>{children}</EnvBox>;
}
