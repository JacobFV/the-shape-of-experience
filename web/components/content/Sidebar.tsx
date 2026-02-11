import { ReactNode } from 'react';
import { EnvBox } from './EnvBox';

export function Sidebar({ title, children }: { title?: string; children: ReactNode }) {
  return <EnvBox className="env-sidebar" title={title}>{children}</EnvBox>;
}
