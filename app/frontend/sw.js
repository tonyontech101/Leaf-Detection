/* Service worker: network-first for the app shell.
   While the local server is reachable, always fetch fresh HTML/CSS/JS so a
   redesign shows up immediately. The cache is only a fallback for offline use.
   API calls (/api/*) and thumbnails always go straight to the local server. */
const CACHE = "leaf-scanner-v3";
const SHELL = [
  "./",
  "./index.html",
  "./styles.css",
  "./app.js",
  "./manifest.webmanifest",
  "./icon.svg",
];

self.addEventListener("install", (event) => {
  // Pre-cache the shell for offline fallback, then take over immediately.
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  // Drop every older cache so stale shells can't be served again.
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Only handle same-origin GET requests; let everything else pass through.
  if (request.method !== "GET" || url.origin !== self.location.origin) return;

  // Never cache dynamic API responses or dataset thumbnails.
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/thumbs/")) {
    return;
  }

  // Network-first: fetch fresh, update the cache, fall back to cache offline.
  event.respondWith(
    fetch(request)
      .then((response) => {
        // Refresh the cached copy for offline use.
        const copy = response.clone();
        caches.open(CACHE).then((c) => c.put(request, copy)).catch(() => {});
        return response;
      })
      .catch(() =>
        caches.match(request).then((hit) => hit || caches.match("./index.html"))
      )
  );
});
