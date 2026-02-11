import { ReactNode } from 'react';
import { EnvBox } from './EnvBox';

export function Connection({ title, children }: { title?: string; children: ReactNode }) {
  return <EnvBox className="env-connection" title={title}>{children}</EnvBox>;
}
