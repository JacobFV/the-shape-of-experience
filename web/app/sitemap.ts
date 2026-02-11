import type { MetadataRoute } from 'next';

const BASE = 'https://the-shape-of-experience.vercel.app';

export default function sitemap(): MetadataRoute.Sitemap {
  const pages = [
    '',
    '/introduction',
    '/part-1',
    '/part-2',
    '/part-3',
    '/part-4',
    '/part-5',
    '/epilogue',
    '/terms',
    '/privacy',
  ];

  return pages.map((path) => ({
    url: `${BASE}${path}`,
    lastModified: new Date(),
    changeFrequency: path === '/terms' || path === '/privacy' ? 'yearly' : 'monthly',
    priority: path === '' ? 1.0 : path === '/terms' || path === '/privacy' ? 0.2 : 0.8,
  }));
}
