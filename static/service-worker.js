// Minimal PWA service worker — just enough for installability plus safe
// caching of the static app shell. Never *serves* a cached HTML page or
// /api/* response on its own: this app's whole value is live odds/scores,
// so serving a stale page from cache on a flaky connection would be worse
// than no cache at all. The one exception is opt-in: each page's last good
// HTML is kept in PAGE_CACHE, and the loader's "Show last saved" button asks
// for it by adding ?saved=1 (see static/loader.js), labelled with its time. No offline fallback page for navigations either (see UI/UX plan) —
// on a genuine network failure with nothing cached, the browser's default
// offline error is shown; a branded offline.html is a reasonable v2 addition,
// deliberately deferred here to keep this pass scoped to installability.

const CACHE_NAME = 'spooky-shell-v4';
const PAGE_CACHE = 'dm-pages-v1';
const PAGE_CACHE_MAX = 30;
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
      Promise.all(names.filter((n) => n !== CACHE_NAME && n !== PAGE_CACHE).map((n) => caches.delete(n)))
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

  if (request.mode === 'navigate') {
    if (url.searchParams.get('saved') === '1') {
      url.searchParams.delete('saved');
      event.respondWith(
        caches.open(PAGE_CACHE)
          .then((cache) => cache.match(url.href))
          .then((hit) => hit || fetch(url.href))
      );
      return;
    }
    event.respondWith(
      fetch(request).then((resp) => {
        savePage(request.url, resp.clone());
        return resp;
      }).catch(() => caches.open(CACHE_NAME).then((cache) => cache.match(request)))
    );
    return;
  }

  // Network-first / pass-through for everything else (/api/*) — never
  // serve stale odds/scores/bet state from a cache.
  event.respondWith(fetch(request).catch(() => caches.open(CACHE_NAME).then((cache) => cache.match(request))));
});

// Keep the page for "Show last saved", stamped with when it was saved. The
// loader's own cold-start page (X-DM-Loader) is never a page worth keeping.
function savePage(key, resp) {
  const type = resp.headers.get('content-type') || '';
  if (!resp.ok || resp.redirected || !type.includes('text/html') || resp.headers.get('x-dm-loader')) return;
  resp.blob().then((body) => {
    const headers = new Headers(resp.headers);
    headers.set('x-dm-saved-at', String(Date.now()));
    return caches.open(PAGE_CACHE).then((cache) =>
      cache.put(key, new Response(body, { status: resp.status, headers }))
        .then(() => cache.keys())
        .then((keys) => Promise.all(keys.slice(0, Math.max(0, keys.length - PAGE_CACHE_MAX)).map((k) => cache.delete(k))))
    );
  }).catch(() => {});
}
