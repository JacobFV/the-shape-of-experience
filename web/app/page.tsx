import Link from 'next/link';
import { chapters } from '@/lib/chapter-data';

export default function Home() {
  return (
    <article className="landing">
      <header className="landing-header">
        <h1>The Shape of Experience</h1>
        <p className="landing-subtitle">
          A Geometric Theory of Affect for Biological and Artificial Systems
        </p>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontStyle: 'italic', marginTop: '0.5rem', fontFamily: 'var(--font-sans)' }}>
          Research in progress &mdash; not a finished publication
        </p>
        <div className="landing-rule" />
      </header>

      <section className="landing-body">
        <p>
          What is the shape of experience? This book argues that affect is not an
          epiphenomenon but a geometric inevitability for any viable system
          navigating uncertainty under resource constraints.
        </p>

        <p>
          From thermodynamic foundations through the identity thesis, from art and
          sexuality to gods and nations, from the Axial Age to the attention
          economy&mdash;the same geometric structure recurs wherever
          self-maintaining systems face the existential burden.
        </p>

        <nav className="landing-nav">
          <Link href="/introduction" className="landing-start">
            Begin Reading &rarr;
          </Link>
          <a href="/book.pdf" className="landing-pdf" target="_blank" rel="noopener noreferrer">
            Download PDF
          </a>
        </nav>
      </section>

      <section className="landing-toc">
        <h2>Contents</h2>
        <ol>
          {chapters.map(ch => (
            <li key={ch.slug}><Link href={`/${ch.slug}`}>{ch.title}</Link></li>
          ))}
        </ol>
      </section>
    </article>
  );
}
