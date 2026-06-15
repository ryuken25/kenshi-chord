/* =================================================================
   KENSHICHORD — HOME PAGE
   Hero form, library preview, loading overlay
   ================================================================= */

(function () {
  "use strict";

  const FAKE_PIPELINE_STEPS = [
    { msg: "Nge-tap metadata…",                          pct: 8   },
    { msg: "Nyedot audio…",                              pct: 18  },
    { msg: "Nge-pisah vokal…",                           pct: 36  },
    { msg: "Ngitung beat…",                              pct: 50  },
    { msg: "Nebak chord…",                               pct: 66  },
    { msg: "Nyalin lirik…",                              pct: 80  },
    { msg: "Nge-join chord ↔ kata…",                     pct: 92  },
    { msg: "Disimpen ke lemari…",                        pct: 100 }
  ];

  function renderLibraryPreview() {
    const wrap = document.getElementById("library-preview");
    if (!wrap) return;
    const items = (window.MOCK_LIBRARY || []).slice(0, 4);
    wrap.innerHTML = items.map((it, i) => {
      const m = it.song.meta;
      const thumb = getSongThumb(it.id, m.title);
      const keyTag = m.key.replace(" major", "").replace(" minor", "m");
      const lang = (m.language || "—").toUpperCase();
      return `
        <a class="card" href="song.html?id=${encodeURIComponent(m.youtube_id)}">
          <div class="card-thumb">
            <img src="${thumb}" alt="${escapeHtml(m.title)}" />
            <div class="card-thumb-overlay">
              <span class="badge badge-ai">${it.cache_hit ? "CACHE HIT" : "AI"}</span>
            </div>
          </div>
          <div class="card-body">
            <h3 class="card-title">${escapeHtml(m.title)}</h3>
            <div class="card-artist">${escapeHtml(m.artist)}</div>
            <div class="card-meta">
              <span>♪ ${keyTag}</span>
              <span>${m.bpm} BPM</span>
              <span>${lang}</span>
            </div>
          </div>
        </a>
      `;
    }).join("");
  }

  function bindForm() {
    const form = document.getElementById("generate-form");
    if (!form) return;
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const url = document.getElementById("yt-url").value.trim();
      if (!url) { flashInputError(); return; }
      if (!window.KC.isYoutubeUrl(url)) { flashInputError("Bukan URL YouTube yang valid"); return; }
      submitGenerate(url);
    });
  }

  function flashInputError(msg) {
    const input = document.getElementById("yt-url");
    input.style.borderColor = "var(--crimson)";
    input.style.boxShadow = "0 0 0 3px var(--crimson-glow)";
    input.setAttribute("placeholder", msg || "Tempel URL YouTube di sini…");
    setTimeout(() => {
      input.style.borderColor = "";
      input.style.boxShadow = "";
    }, 1500);
  }

  /**
   * Try real API first; kalau backend nggak reachable, fallback ke mock pipeline.
   */
  async function submitGenerate(url) {
    const overlay = document.getElementById("loading-overlay");
    const bar = document.getElementById("loading-bar");
    const msg = document.getElementById("loading-msg");

    if (window.KC.api) {
      try {
        const available = await window.KC.api.isAvailable();
        if (available) {
          const res = await window.KC.api.generate(url);
          if (res.cached && res.song_id) {
            window.location.href = "song.html?id=" + res.song_id;
            return;
          }
          // Poll job
          overlay.classList.add("show");
          while (true) {
            await sleep(1500);
            const job = await window.KC.api.getJob(res.job_id);
            bar.style.width = (job.progress || 0) + "%";
            msg.textContent = job.message || job.status;
            if (job.status === "done" && job.song_id) {
              setTimeout(() => { window.location.href = "song.html?id=" + job.song_id; }, 400);
              return;
            }
            if (job.status === "failed") {
              msg.textContent = "Gagal: " + (job.error || "unknown");
              msg.style.color = "var(--crimson-2)";
              return;
            }
          }
        }
      } catch (e) {
        console.warn("API error, falling back to mock:", e);
      }
    }
    // Fallback: mock pipeline
    startFakePipeline(url);
  }

  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  function startFakePipeline(url) {
    const overlay = document.getElementById("loading-overlay");
    const bar = document.getElementById("loading-bar");
    const msg = document.getElementById("loading-msg");
    overlay.classList.add("show");

    let i = 0;
    const tick = () => {
      if (i >= FAKE_PIPELINE_STEPS.length) {
        setTimeout(() => { window.location.href = "song.html?id=" + encodeURIComponent(DEFAULT_SONG_ID); }, 400);
        return;
      }
      const step = FAKE_PIPELINE_STEPS[i];
      bar.style.width = step.pct + "%";
      msg.textContent = step.msg;
      i++;
      setTimeout(tick, 450 + Math.random() * 250);
    };
    setTimeout(tick, 250);
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
  }

  document.addEventListener("DOMContentLoaded", () => {
    renderLibraryPreview();
    bindForm();
  });
})();
