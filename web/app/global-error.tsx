'use client';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html>
      <body style={{ fontFamily: 'system-ui, sans-serif', padding: '2rem', maxWidth: '600px', margin: '0 auto' }}>
        <h1 style={{ fontSize: '1.5rem' }}>Something went wrong</h1>
        <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', background: '#f5f5f5', padding: '1rem', borderRadius: '8px', fontSize: '0.85rem' }}>
          {error.message}
          {error.stack && '\n\n' + error.stack}
          {error.digest && '\n\nDigest: ' + error.digest}
        </pre>
        <button
          onClick={reset}
          style={{ marginTop: '1rem', padding: '0.5rem 1rem', background: '#2c5aa0', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
        >
          Try again
        </button>
      </body>
    </html>
  );
}
