import { ReactNode } from 'react';

interface EnvBoxProps {
  className: string;
  title?: string;
  children: ReactNode;
}

export function EnvBox({ className, title, children }: EnvBoxProps) {
  return (
    <div className={className}>
      {title && <div className="env-title">{title}</div>}
      {children}
    </div>
  );
}
