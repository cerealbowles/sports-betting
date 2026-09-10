// Minimal PWA service worker — just enough for installability plus safe
// caching of the static app shell. Deliberately does NOT cache HTML pages or
// /api/* responses: this app's whole value is live odds/scores, so serving a
// stale page from cache on a flaky connection would be worse than no cache at
// all. No offline fallback page for navigations either (see UI/UX plan) —
// on a genuine network failure with nothing cached, the browser's default
// offline error is shown; a branded offline.html is a reasonable v2 addition,
// deliberately deferred here to keep this pass scoped to installability.

const CACHE_NAME = 'spooky-shell-v1';
const SHELL_ASSETS = [
  '/static/styles.css',
  '/static/main.js',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  const isStaticAsset = url.pathname.startsWith('/static/');

  if (isStaticAsset) {
    // Cache-first: these only change on deploy, not per-request.
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return cached;
        return fetch(request).then((resp) => {
          const copy = resp.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
          return resp;
        });
      })
    );
    return;
  }

  // Network-first / pass-through for everything else (HTML pages, /api/*) —
  // never serve stale odds/scores/bet state from a cache.
  event.respondWith(fetch(request).catch(() => caches.match(request)));
});
