// No-flash theme bootstrap: apply .dark on <html> BEFORE React paints so
// dark-mode users never see a white flash. Reads the same localStorage key
// ('theme') and the same rule the ThemeProvider uses, so there is no reflow
// after mount. Wrapped in try/catch so a storage exception never blocks paint.
//
// Kept in an EXTERNAL same-origin file (not inline in index.html) so the app's
// Content-Security-Policy can stay strict: `default-src 'self'` permits a
// same-origin `<script src>` but blocks inline scripts, so inlining this would
// require `'unsafe-inline'` (weakens XSS defense) or a per-byte sha256 hash
// (brittle — any whitespace change silently re-breaks the no-flash bootstrap).
// Referenced as a render-blocking `<script src="/theme-init.js">` in <head>, so
// it still runs before first paint. Served from Vite's public/ dir (copied to
// the dist root verbatim, stable URL — not content-hashed).
(function () {
  try {
    var stored = localStorage.getItem("theme");
    var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    var dark =
      stored === "dark" ||
      ((stored === "system" || stored === null) && prefersDark);
    document.documentElement.classList.toggle("dark", dark);
  } catch (e) {
    /* ignore — leave light as the safe default */
  }
})();
