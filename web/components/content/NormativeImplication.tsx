import { ReactNode } from 'react';
import { EnvBox } from './EnvBox';

export function NormativeImplication({ title, children }: { title?: string; children: ReactNode }) {
  return <EnvBox className="env-normimp" title={title}>{children}</EnvBox>;
}
