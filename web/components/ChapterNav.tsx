import Link from 'next/link';

interface ChapterNavProps {
  prev?: { slug: string; shortTitle: string };
  next?: { slug: string; shortTitle: string };
}

export default function ChapterNav({ prev, next }: ChapterNavProps) {
  return (
    <nav className="chapter-nav">
      <div className="chapter-nav-prev">
        {prev && (
          <Link href={`/${prev.slug}`}>
            <span className="chapter-nav-arrow">&larr;</span>
            <span className="chapter-nav-label">{prev.shortTitle}</span>
          </Link>
        )}
      </div>
      <div className="chapter-nav-next">
        {next && (
          <Link href={`/${next.slug}`}>
            <span className="chapter-nav-label">{next.shortTitle}</span>
            <span className="chapter-nav-arrow">&rarr;</span>
          </Link>
        )}
      </div>
    </nav>
  );
}
