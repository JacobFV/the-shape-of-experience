import { ReactNode } from 'react';
import { EnvBox } from './EnvBox';

export function Software({ title, children }: { title?: string; children: ReactNode }) {
  return <EnvBox className="env-software" title={title}>{children}</EnvBox>;
}
