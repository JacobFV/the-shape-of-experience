import { ReactNode } from 'react';

interface FigureProps {
  src: string;
  alt: string;
  caption?: ReactNode;
  multi?: boolean;
  children?: ReactNode;
}

export function Figure({ src, alt, caption, multi, children }: FigureProps) {
  return (
    <figure className={multi ? 'multi' : undefined}>
      {children || <img src={src} alt={alt} />}
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
}
