/* =================================================================
   KENSHICHORD — LIBRARY PAGE
   Search + filter + sort + grid.
   Try backend API dulu, fallback ke MOCK_LIBRARY kalau offline.
   ================================================================= */

(function () {
  "use strict";

  const state = {
    items: [],          // array of { id, song: { meta, ... }, cache_hit, processed_at }
    filter: { lang: null, key: null, sort: "new" },
    apiAvailable: false
  };

  function keyOf(m) { return (m.key || "C").replace(" major","").replace(" minor","m"); }

  /** Normalize API SongListItem → internal item shape (match MOCK_LIBRARY). */
  function apiToInternal(api) {
    return {
      id: api.id,
      cache_hit: false,
      processed_at: api.created_at || new Date().toISOString(),
      song: {
        meta: {
          youtube_id: api.youtube_id,
          artist:     api.artist,
          title:      api.title,
          duration_sec: api.duration_sec,
          bpm:        api.bpm,
          key:        api.key || "C major",
          language:   api.language,
          time_sig:   "4/4"
        },
        sections: [],
        lines: []
      }
    };
  }

  async function loadFromApi() {
    if (!window.KC.api) return false;
    const ok = await window.KC.api.isAvailable();
    if (!ok) return false;
    try {
      const items = await window.KC.api.listSongs();
      state.items = items.map(apiToInternal);
      state.apiAvailable = true;
      return true;
    } catch (e) {
      console.warn("API list failed, using mock:", e);
      return false;
    }
  }

  async function applySearchRemote(q) {
    // If API available, refetch with search query
    if (!state.apiAvailable || !window.KC.api) return false;
    try {
      const items = await window.KC.api.listSongs(q);
      state.items = items.map(apiToInternal);
      return true;
    } catch (e) {
      console.warn("API search failed:", e);
      return false;
    }
  }

  function applyFilters() {
    const q = (document.getElementById("search-input")?.value || "").trim().toLowerCase();
    return state.items.filter((it) => {
      const m = it.song.meta;
      if (q) {
        const hay = (m.title + " " + m.artist + " " + (m.album || "")).toLowerCase();
        if (!hay.includes(q)) return false;
      }
      if (state.filter.lang && m.language !== state.filter.lang) return false;
      if (state.filter.key && keyOf(m) !== state.filter.key) return false;
      return true;
    }).sort((a, b) => {
      if (state.filter.sort === "bpm") {
        return (a.song.meta.bpm || 0) - (b.song.meta.bpm || 0);
      }
      return new Date(b.processed_at) - new Date(a.processed_at);
    });
  }

  function render() {
    const grid = document.getElementById("library-grid");
    const data = applyFilters();

    if (data.length === 0) {
      grid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="icon" aria-hidden="true">⌕</div>
          <h3>Tidak ada lagu ditemukan</h3>
          <p>${state.apiAvailable ? "Coba kata kunci lain, atau generate lagu baru." : "Backend offline — pakai mock. Coba kata kunci lain."}</p>
        </div>`;
    } else {
      grid.innerHTML = data.map((it) => {
        const m = it.song.meta;
        const thumb = getSongThumb(it.id, m.title);
        const keyTag = keyOf(m);
        const lang = (m.language || "—").toUpperCase();
        const cacheBadge = it.cache_hit ? "CACHE HIT" : (state.apiAvailable ? "AI" : "MOCK");
        // Use numeric id if available, else youtube_id
        const link = typeof it.id === "number"
          ? `song.html?id=${it.id}`
          : `song.html?id=${encodeURIComponent(m.youtube_id)}`;
        return `
          <a class="card" href="${link}">
            <div class="card-thumb">
              <img src="${thumb}" alt="${escapeHtml(m.title)}" />
              <div class="card-thumb-overlay">
                <span class="badge badge-ai">${cacheBadge}</span>
              </div>
            </div>
            <div class="card-body">
              <h3 class="card-title">${escapeHtml(m.title)}</h3>
              <div class="card-artist">${escapeHtml(m.artist)}</div>
              <div class="card-meta">
                <span>♪ ${escapeHtml(keyTag)}</span>
                <span>${m.bpm ? Math.round(m.bpm) : "—"} BPM</span>
                <span>${lang}</span>
              </div>
            </div>
          </a>
        `;
      }).join("");
    }

    // Counts
    document.getElementById("visible-count").textContent = data.length;
    document.getElementById("total-count").textContent   = state.items.length;
    document.getElementById("cnt-all").textContent       = state.items.length;
    document.getElementById("cnt-ja").textContent       = state.items.filter(x => x.song.meta.language === "ja").length;
    document.getElementById("cnt-id").textContent       = state.items.filter(x => x.song.meta.language === "id").length;
    document.getElementById("cnt-en").textContent       = state.items.filter(x => x.song.meta.language === "en").length;

    // API badge
    const apiBadge = document.getElementById("api-status");
    if (apiBadge) apiBadge.textContent = state.apiAvailable ? "🟢 API" : "⚪ Mock";
  }

  function bindFilters() {
    document.getElementById("filters").addEventListener("click", (e) => {
      const b = e.target.closest(".filter-chip");
      if (!b) return;
      const f = b.dataset.filter;
      document.querySelectorAll(".filter-chip").forEach(x => x.classList.remove("active"));
      b.classList.add("active");

      state.filter.lang = null; state.filter.key = null; state.filter.sort = "new";
      if (f === "all")      {/* no extra filter */}
      else if (f.startsWith("lang:")) state.filter.lang = f.slice(5);
      else if (f.startsWith("key:"))  state.filter.key  = f.slice(4);
      else if (f === "sort:bpm")      state.filter.sort = "bpm";
      render();
    });
  }

  function bindSearch() {
    const input = document.getElementById("search-input");
    let pending = null;
    input.addEventListener("input", () => {
      clearTimeout(pending);
      pending = setTimeout(async () => {
        const q = input.value.trim();
        // Try remote search first
        if (state.apiAvailable && q) {
          await applySearchRemote(q);
        }
        render();
      }, 250);
    });
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
  }

  document.addEventListener("DOMContentLoaded", async () => {
    // Try API first
    const ok = await loadFromApi();
    if (!ok) {
      // Fallback to mock
      state.items = (window.MOCK_LIBRARY || []).slice();
    }
    render();
    bindFilters();
    bindSearch();
  });
})();
