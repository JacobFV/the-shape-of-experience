'use client';

import { ReactNode } from 'react';
import { getIllustration } from '../../content/illustrations';

interface IllustrationProps {
  /** The illustration ID from the registry */
  id: string;
  /** Override the default caption */
  caption?: ReactNode;
  /** Additional CSS class */
  className?: string;
}

/**
 * Renders an AI-generated illustration from the registry.
 *
 * Usage:
 *   <Illustration id="shadow-of-transcendence" />
 *   <Illustration id="bottleneck-furnace" caption={<>Custom caption</>} />
 *
 * Images live in /public/images/illustrations/{id}.png
 * Registry lives in web/content/illustrations.ts
 */
export function Illustration({ id, caption, className }: IllustrationProps) {
  const entry = getIllustration(id);

  if (!entry) {
    if (process.env.NODE_ENV === 'development') {
      return (
        <figure className={className}>
          <div
            style={{
              background: 'repeating-linear-gradient(45deg, #1a1a2e, #1a1a2e 10px, #16213e 10px, #16213e 20px)',
              padding: '2rem',
              borderRadius: 8,
              textAlign: 'center',
              color: '#f87171',
              fontFamily: 'monospace',
              fontSize: '0.85rem',
            }}
          >
            Unknown illustration: <strong>{id}</strong>
            <br />
            Not found in web/content/illustrations.ts
          </div>
        </figure>
      );
    }
    return null;
  }

  const src = `/images/illustrations/${entry.id}.png`;
  const resolvedCaption = caption ?? entry.caption;

  if (entry.status === 'pending') {
    // Show placeholder in development, nothing in production
    if (process.env.NODE_ENV === 'development') {
      return (
        <figure className={className}>
          <div
            style={{
              background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
              padding: '3rem 2rem',
              borderRadius: 8,
              textAlign: 'center',
              color: '#94a3b8',
              fontFamily: 'monospace',
              fontSize: '0.8rem',
              border: '1px dashed #334155',
              minHeight: 200,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.75rem',
            }}
          >
            <span style={{ fontSize: '1.5rem', opacity: 0.5 }}>🎨</span>
            <span>
              Pending: <strong>{entry.id}</strong>
            </span>
            <span style={{ maxWidth: 400, lineHeight: 1.4, opacity: 0.7 }}>
              {entry.prompt.slice(0, 120)}...
            </span>
          </div>
          {resolvedCaption && <figcaption>{resolvedCaption}</figcaption>}
        </figure>
      );
    }
    return null;
  }

  return (
    <figure className={className}>
      <img
        src={src}
        alt={entry.alt}
        style={{ width: '100%', borderRadius: 8 }}
        loading="lazy"
      />
      {resolvedCaption && <figcaption>{resolvedCaption}</figcaption>}
    </figure>
  );
}
