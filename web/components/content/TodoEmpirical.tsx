import { ReactNode } from 'react';
import { EnvBox } from './EnvBox';

export function TodoEmpirical({ title, children }: { title?: string; children: ReactNode }) {
  return <EnvBox className="env-todo-empirical" title={title}>{children}</EnvBox>;
}
