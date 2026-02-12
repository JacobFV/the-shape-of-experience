import Link from 'next/link';

interface ChapterNavProps {
  prev?: { href: string; label: string };
  next?: { href: string; label: string };
}

export default function ChapterNav({ prev, next }: ChapterNavProps) {
  return (
    <nav className="chapter-nav">
      <div className="chapter-nav-prev">
        {prev && (
          <Link href={prev.href}>
            <span className="chapter-nav-arrow">&larr;</span>
            <span className="chapter-nav-label">{prev.label}</span>
          </Link>
        )}
      </div>
      <div className="chapter-nav-next">
        {next && (
          <Link href={next.href}>
            <span className="chapter-nav-label">{next.label}</span>
            <span className="chapter-nav-arrow">&rarr;</span>
          </Link>
        )}
      </div>
    </nav>
  );
}
