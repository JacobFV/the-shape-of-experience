import Link from 'next/link';

export default function Home() {
  return (
    <article className="landing">
      <header className="landing-header">
        <h1>The Shape of Experience</h1>
        <p className="landing-subtitle">
          A Geometric Theory of Affect for Biological and Artificial Systems
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
          economy&mdash;the same six-dimensional geometry recurs wherever
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
          <li><Link href="/introduction">Introduction</Link></li>
          <li><Link href="/part-1">Part I: Thermodynamic Foundations and the Ladder of Emergence</Link></li>
          <li><Link href="/part-2">Part II: The Identity Thesis and the Geometry of Feeling</Link></li>
          <li><Link href="/part-3">Part III: Signatures of Affect Under the Existential Burden</Link></li>
          <li><Link href="/part-4">Part IV: The Topology of Social Bonds</Link></li>
          <li><Link href="/part-5">Part V: Gods and Superorganisms</Link></li>
          <li><Link href="/part-6">Part VI: Historical Consciousness and Transcendence</Link></li>
          <li><Link href="/part-7">Part VII: The Empirical Program</Link></li>
          <li><Link href="/epilogue">Epilogue</Link></li>
        </ol>
      </section>
    </article>
  );
}
