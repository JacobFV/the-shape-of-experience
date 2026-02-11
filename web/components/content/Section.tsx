import { ReactNode } from 'react';

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
}

interface SectionProps {
  title: string;
  level?: 1 | 2 | 3;
  id?: string;
  children: ReactNode;
}

export function Section({ title, level = 2, id, children }: SectionProps) {
  const sectionId = id || slugify(title);
  const Tag = `h${level}` as 'h1' | 'h2' | 'h3';

  return (
    <section>
      <Tag id={sectionId}>{title}</Tag>
      {children}
    </section>
  );
}
