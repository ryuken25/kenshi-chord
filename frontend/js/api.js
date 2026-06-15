/* =================================================================
   KENSHICHORD — API CLIENT
   Wrapper untuk backend FastAPI (FASE 1).
   Kalau backend nggak reachable, callers fallback ke mock.
   ================================================================= */

(function () {
  "use strict";

  // Base URL — bisa di-override via window.KC_API_BASE
  const BASE = (window.KC_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");
  let _available = null;  // cached health check result

  async function tryFetch(path, options = {}) {
    const url = path.startsWith("http") ? path : BASE + path;
    let res;
    try {
      res = await fetch(url, {
        ...options,
        headers: {
          "Accept": "application/json",
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...(options.headers || {})
        }
      });
    } catch (e) {
      // Network error (backend down / CORS)
      const err = new Error("Network: " + e.message);
      err.networkError = true;
      throw err;
    }
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      const err = new Error(`HTTP ${res.status}: ${txt || res.statusText}`);
      err.status = res.status;
      err.body = txt;
      throw err;
    }
    if (res.status === 204) return null;
    return res.json();
  }

  const api = {
    BASE,

    /** Quick health check (cached for session). */
    async isAvailable() {
      if (_available !== null) return _available;
      try {
        const r = await tryFetch("/api/health");
        _available = (r && r.status === "ok");
      } catch (e) {
        _available = false;
      }
      return _available;
    },

    /** Submit YouTube URL. Returns { cached, song_id?, job_id?, song? }. */
    generate(youtube_url) {
      return tryFetch("/api/songs/generate", {
        method: "POST",
        body: JSON.stringify({ youtube_url })
      });
    },

    /** Get job status. */
    getJob(job_id) {
      return tryFetch(`/api/jobs/${encodeURIComponent(job_id)}`);
    },

    /** Get full song by numeric ID. */
    getSong(id) {
      return tryFetch(`/api/songs/${id}`);
    },

    /** Lookup by youtube_id. */
    getSongByYoutube(youtube_id) {
      return tryFetch(`/api/songs/by-youtube/${encodeURIComponent(youtube_id)}`);
    },

    /** List library. */
    listSongs(search) {
      const q = search ? `?search=${encodeURIComponent(search)}` : "";
      return tryFetch(`/api/songs${q}`);
    },

    /** Admin: delete. */
    deleteSong(id) {
      return tryFetch(`/api/songs/${id}`, { method: "DELETE" });
    },

    /** Reset cached health (e.g., kalau user refresh). */
    resetHealth() { _available = null; }
  };

  window.KC = window.KC || {};
  window.KC.api = api;
})();
