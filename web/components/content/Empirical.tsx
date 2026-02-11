import { ReactNode } from 'react';
import { EnvBox } from './EnvBox';

export function Empirical({ title, children }: { title?: string; children: ReactNode }) {
  return <EnvBox className="env-empirical" title={title}>{children}</EnvBox>;
}
