import { ReactNode } from 'react';
import { EnvBox } from './EnvBox';

export function Experiment({ title, children }: { title?: string; children: ReactNode }) {
  return <EnvBox className="env-experiment" title={title}>{children}</EnvBox>;
}
