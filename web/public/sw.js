const CACHE_VERSION = 'soe-v2';
const STATIC_CACHE = 'soe-static-v1';
const CDN_CACHE = 'soe-cdn-v1';

const PRECACHE_URLS = [
  '/',
  '/introduction',
  '/part-1',
  '/part-2',
  '/part-3',
  '/part-4',
  '/part-5',
  '/part-6',
  '/part-7',
  '/epilogue',
  '/appendix-experiments',
  '/icon.svg',
  '/manifest.json',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) =>
      Promise.all(
        PRECACHE_URLS.map((url) =>
          cache.add(url).catch(() => {
            // Don't let one failed URL break the whole install
          })
        )
      )
    )
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k.startsWith('soe-') && k !== CACHE_VERSION && k !== STATIC_CACHE && k !== CDN_CACHE)
          .map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.method !== 'GET') return;
  if (url.pathname.startsWith('/api/')) return;

  // CDN resources (KaTeX etc) — stale-while-revalidate
  if (url.hostname !== self.location.hostname) {
    event.respondWith(staleWhileRevalidate(request, CDN_CACHE));
    return;
  }

  // Static assets — cache-first
  if (
    url.pathname.startsWith('/_next/static/') ||
    url.pathname.startsWith('/images/') ||
    url.pathname.startsWith('/diagrams/') ||
    url.pathname.startsWith('/icons/') ||
    url.pathname.endsWith('.svg') ||
    url.pathname.endsWith('.pdf')
  ) {
    event.respondWith(cacheFirst(request, STATIC_CACHE));
    return;
  }

  // Audio — network-only (too large to cache automatically)
  if (url.pathname.startsWith('/audio/')) return;

  // HTML navigation — network-first with cache fallback
  if (request.headers.get('accept')?.includes('text/html')) {
    event.respondWith(networkFirst(request, CACHE_VERSION));
    return;
  }

  // Everything else — stale-while-revalidate
  event.respondWith(staleWhileRevalidate(request, CACHE_VERSION));
});

async function cacheFirst(request, cacheName) {
  const cached = await caches.match(request);
  if (cached) return cached;
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(cacheName);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    return new Response('', { status: 503 });
  }
}

async function networkFirst(request, cacheName) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(cacheName);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    if (cached) return cached;
    return new Response(
      '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Offline</title></head><body style="font-family:Georgia,serif;max-width:480px;margin:80px auto;text-align:center;color:#1a1a1a"><h1>You\u2019re offline</h1><p>The Shape of Experience requires a network connection for this page. Previously visited pages may still be available.</p></body></html>',
      { status: 503, headers: { 'Content-Type': 'text/html' } }
    );
  }
}

async function staleWhileRevalidate(request, cacheName) {
  const cached = await caches.match(request);
  const fetchPromise = fetch(request)
    .then((response) => {
      if (response.ok) {
        const clone = response.clone();
        caches.open(cacheName).then((cache) => cache.put(request, clone));
      }
      return response;
    })
    .catch(() => cached);
  return cached || fetchPromise;
}
