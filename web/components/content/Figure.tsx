import { ReactNode } from 'react';

interface FigureProps {
  src: string;
  alt: string;
  caption?: ReactNode;
  multi?: boolean;
}

export function Figure({ src, alt, caption, multi }: FigureProps) {
  return (
    <figure className={multi ? 'multi' : undefined}>
      <img src={src} alt={alt} />
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
}
