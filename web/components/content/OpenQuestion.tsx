import { ReactNode } from 'react';
import { EnvBox } from './EnvBox';

export function OpenQuestion({ title, children }: { title?: string; children: ReactNode }) {
  return <EnvBox className="env-openquestion" title={title}>{children}</EnvBox>;
}
