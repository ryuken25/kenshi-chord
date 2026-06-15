/* =================================================================
   KENSHICHORD — MAIN
   Shared utilities, header behavior
   ================================================================= */

(function () {
  "use strict";

  /** Format seconds → mm:ss.s */
  function fmtTime(sec) {
    if (!isFinite(sec) || sec < 0) sec = 0;
    const m = Math.floor(sec / 60);
    const s = sec - m * 60;
    return String(m).padStart(2, "0") + ":" + s.toFixed(1).padStart(4, "0");
  }

  /** Read URL query */
  function getQuery(key) {
    const u = new URL(window.location.href);
    return u.searchParams.get(key);
  }

  /** Smooth scroll helper */
  function scrollTo(el, offset = 0) {
    if (!el) return;
    const y = el.getBoundingClientRect().top + window.pageYOffset - offset;
    window.scrollTo({ top: y, behavior: "smooth" });
  }

  /** Debounce */
  function debounce(fn, ms = 200) {
    let t;
    return function (...args) {
      clearTimeout(t);
      t = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  /** Read YouTube ID from various URL formats */
  function extractYoutubeId(url) {
    if (!url) return null;
    try {
      const u = new URL(url);
      if (u.hostname.includes("youtu.be")) {
        return u.pathname.slice(1).split(/[/?]/)[0];
      }
      if (u.hostname.includes("youtube.com")) {
        if (u.pathname.startsWith("/embed/")) {
          return u.pathname.split("/")[2];
        }
        return u.searchParams.get("v");
      }
    } catch (_) { /* fall through */ }
    return null;
  }

  /** Quick validation: looks like a youtube url */
  function isYoutubeUrl(url) {
    return !!extractYoutubeId(url);
  }

  /** Toggle active class on segmented control */
  function bindSegmented(selector, onChange) {
    document.querySelectorAll(selector).forEach((group) => {
      group.addEventListener("click", (e) => {
        const btn = e.target.closest("button");
        if (!btn) return;
        group.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        onChange && onChange(btn);
      });
    });
  }

  /** Expose */
  window.KC = window.KC || {};
  Object.assign(window.KC, { fmtTime, getQuery, scrollTo, debounce, extractYoutubeId, isYoutubeUrl, bindSegmented });

  // Highlight active nav link
  document.addEventListener("DOMContentLoaded", () => {
    const path = window.location.pathname.split("/").pop() || "index.html";
    document.querySelectorAll(".nav a").forEach((a) => {
      if (a.getAttribute("href") === path) a.classList.add("active");
    });
  });
})();
