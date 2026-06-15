/* =================================================================
   KENSHICHORD — SONG VIEW CONTROLLER
   Render lyric + chord + bar grid, wire toolbar & sync engine.
   ================================================================= */

(function () {
  "use strict";

  const state = {
    song: null,
    transposed: null,
    transpose: 0,
    speed: 1,
    instrument: "guitar",
    autoScroll: true,
    showChord: true,
    soundOn: false,        // chord click on every chord change during playback
    metronomeOn: false,    // tick on every beat during playback
    romajiMode: "off",     // "off" | "ro" | "both"
    fontScale: 1.0,        // 0.8 .. 1.4
    activeLine: null,
    activeChord: null,
    lastBeatIdx: -1,       // for metronome (avoid spam on seek)
    lastChordPlayed: null, // for auto-play (avoid repeat)
    sync: null
  };

  /* ====================== AUDIO (Web Audio API) ====================== */
  let audioCtx = null;
  function ensureAudio() {
    if (!audioCtx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return null;
      try { audioCtx = new Ctx(); } catch (e) { return null; }
    }
    if (audioCtx.state === "suspended") audioCtx.resume();
    return audioCtx;
  }
  function chordToMidiNotes(chordName) {
    const p = window.KC.parseChord(chordName);
    if (!p || p.kind !== "parsed") return [];
    const root = window.KC.noteToIdx(p.root);
    if (root == null) return [];
    const isMinor = p.modifier && p.modifier.startsWith("m") && !p.modifier.startsWith("maj");
    const third = isMinor ? 3 : 4;
    const base = 48 + root; // C3 area (good for guitar)
    return [base, base + third, base + 7];
  }
  function playChordClick(chordName) {
    const ctx = ensureAudio();
    if (!ctx) return;
    const notes = chordToMidiNotes(chordName);
    if (notes.length === 0) return;
    const now = ctx.currentTime;
    const dur = 1.1;
    notes.forEach((midi, i) => {
      const freq = 440 * Math.pow(2, (midi - 69) / 12);
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "triangle";
      osc.frequency.value = freq;
      osc.connect(gain);
      gain.connect(ctx.destination);
      const t0 = now + i * 0.012;
      gain.gain.setValueAtTime(0, t0);
      gain.gain.linearRampToValueAtTime(0.16, t0 + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.001, t0 + dur);
      osc.start(t0);
      osc.stop(t0 + dur);
    });
  }
  function playMetronomeClick(isDownbeat) {
    const ctx = ensureAudio();
    if (!ctx) return;
    const now = ctx.currentTime;
    // downbeat = higher pitch + sedikit lebih keras; regular beat = lower, lebih pelan
    const freq = isDownbeat ? 1500 : 880;
    const vol = isDownbeat ? 0.32 : 0.18;
    const dur = isDownbeat ? 0.06 : 0.04;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = freq;
    osc.connect(gain);
    gain.connect(ctx.destination);
    gain.gain.setValueAtTime(vol, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + dur);
    osc.start(now);
    osc.stop(now + dur + 0.01);
  }

  /* ====================== BEAT HELPERS ====================== */
  function getBeatsPerBar(song) {
    const sig = (song && song.meta && song.meta.time_sig) || "4/4";
    return parseInt(sig.split("/")[0], 10) || 4;
  }
  function findBeatIdx(beats, t) {
    if (!beats || beats.length === 0) return -1;
    // Linear scan dari belakang (umumnya < 200 beats)
    for (let i = beats.length - 1; i >= 0; i--) {
      if (beats[i] <= t) return i;
    }
    return -1;
  }

  /* ====================== INIT ====================== */
  async function init() {
    // Resolve the song id/y param. When absent AND the API is reachable,
    // redirect to the most recent real song so the bare "Song" navbar link
    // never shows the Soran Bushi mock fallback (Bug 0.2).
    let idParam = window.KC.getQuery("id") || window.KC.getQuery("y");
    if (!idParam) {
      const resolved = await resolveNoParam();
      if (resolved) {
        // resolveNoParam already called location.replace — abort this init.
        return;
      }
      // API down + no mock: show empty state. Otherwise fall through to
      // getMockSong(DEFAULT_SONG_ID) so dev-without-backend still works.
      idParam = DEFAULT_SONG_ID;
    }
    const song = await loadSong(idParam);
    state.song = song;
    state.transposed = song;

    renderMeta();
    renderViewer();
    initToolbar();
    initSync();
    applyRomajiMode();
    applyChordVisibility();
    applyFontScale();
  }

  /**
   * When song.html is loaded with no `id`/`y` query param, decide what to do:
   *   - API up + ≥1 song   → redirect to song.html?id={most_recent_id}
   *   - API up + 0 songs   → return false (caller shows empty state)
   *   - API down           → return false (caller falls back to mock)
   * Returns true iff a redirect was issued.
   */
  async function resolveNoParam() {
    if (!window.KC.api) return false;
    let available = false;
    try {
      available = await window.KC.api.isAvailable();
    } catch (e) {
      return false;
    }
    if (!available) return false;
    try {
      const items = await window.KC.api.listSongs();
      if (items && items.length > 0) {
        // Sort by created_at desc just in case the API didn't.
        items.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
        window.location.replace("song.html?id=" + items[0].id);
        return true;
      }
    } catch (e) {
      console.warn("listSongs failed in resolveNoParam:", e);
    }
    return false;
  }

  /**
   * Load song by id. Kalau id numeric → API song by ID. Kalau alphanumeric
   * (YouTube ID) → API by youtube_id.
   *
   * Mock fallback ONLY when the API itself is unreachable. If the API is up
   * but a specific song is missing or fails, surface that — never substitute
   * a different song (Bug 0.3).
   */
  async function loadSong(idParam) {
    if (!window.KC.api) return getMockSong(idParam) || showNotFound(idParam);
    let available = false;
    try {
      available = await window.KC.api.isAvailable();
    } catch (probeErr) {
      console.warn("API health probe failed, using mock:", probeErr);
      return getMockSong(idParam) || showNotFound(idParam);
    }
    if (!available) {
      // API down — mock is the only data source.
      return getMockSong(idParam) || showNotFound(idParam);
    }
    // API is reachable. Don't fall back to mock for missing/erroring songs.
    if (/^\d+$/.test(idParam)) {
      try {
        return await window.KC.api.getSong(parseInt(idParam, 10));
      } catch (e) {
        if (e && e.status === 404) return showNotFound(idParam);
        console.error("API getSong failed:", e);
        return showError(idParam, e);
      }
    }
    // Treat as youtube_id
    try {
      return await window.KC.api.getSongByYoutube(idParam);
    } catch (e) {
      if (e && e.status === 404) return showNotFound(idParam);
      console.error("API getSongByYoutube failed:", e);
      return showError(idParam, e);
    }
  }

  /** Render a "not found" empty song so renderMeta/viewer degrade gracefully. */
  function showNotFound(idParam) {
    console.info("Song not found:", idParam);
    return {
      meta: { youtube_id: idParam || "", artist: "—", title: "Lagu tidak ditemukan",
              duration_sec: 0, bpm: null, key: "C major", capo: 0, time_sig: "4/4", language: null },
      beats: [], downbeats: [], sections: [], bars: [], lines: [],
      _notFound: true
    };
  }
  function showError(idParam, err) {
    return {
      meta: { youtube_id: idParam || "", artist: "—", title: "Gagal muat lagu",
              duration_sec: 0, bpm: null, key: "C major", capo: 0, time_sig: "4/4", language: null },
      beats: [], downbeats: [], sections: [], bars: [], lines: [],
      _loadError: err && err.message ? err.message : String(err)
    };
  }

  /* ====================== META ====================== */
  function renderMeta() {
    const m = state.transposed.meta;
    document.title = `${m.title} — ${m.artist} | KenshiChord`;
    document.getElementById("song-title").textContent = m.title;
    document.getElementById("song-artist").textContent = m.artist;
    document.getElementById("crumb-artist").textContent = m.artist;
    document.getElementById("tag-key").textContent  = window.KC.keyToDisplay(m.key).replace(" major","").replace(" minor","m");
    document.getElementById("tag-bpm").textContent  = m.bpm ? m.bpm.toFixed(0) : "—";
    document.getElementById("tag-time").textContent = m.time_sig || "4/4";
    document.getElementById("tag-lang").textContent = m.language ? m.language.toUpperCase() : "—";
    document.getElementById("tag-capo").textContent = "Capo 0";  // Capo dihapus dari UI
  }

  /* ====================== VIEWER ====================== */
  function renderViewer() {
    const viewer = document.getElementById("viewer");
    viewer.innerHTML = "";

    const song = state.transposed;
    const sections = song.sections || [];
    const lines = song.lines || [];
    const bars = song.bars || [];

    sections.forEach((sec) => {
      const head = document.createElement("div");
      head.className = "viewer-section-head";
      head.innerHTML = `<span class="name">${escapeHtml(sec.name)}</span>`;
      viewer.appendChild(head);

      if (!sec.has_lyrics) {
        const inBars = bars.filter(b => b.start >= sec.start && b.end <= sec.end + 0.01);
        if (inBars.length === 0) {
          const e = document.createElement("div");
          e.className = "empty-section";
          e.innerHTML = `<span class="icon" aria-hidden="true">♪</span><span>Bagian instrumental — chord di bar grid di bawah.</span>`;
          viewer.appendChild(e);
        } else {
          const grid = document.createElement("div");
          grid.className = "bar-grid";
          inBars.forEach((b) => {
            const barEl = document.createElement("div");
            barEl.className = "bar";
            barEl.dataset.start = b.start;
            barEl.dataset.end = b.end;
            const chord = b.chords && b.chords[0] ? b.chords[0].chord : "—";
            barEl.innerHTML = `
              <span class="num">${b.index + 1}</span>
              <div class="chord">${escapeHtml(chord)}</div>
              <div class="beats">
                <span class="beat down"></span>
                <span class="beat"></span>
                <span class="beat"></span>
                <span class="beat"></span>
              </div>
            `;
            barEl.addEventListener("click", () => {
              if (state.sync) state.sync.seek(b.start);
              showDiagram(chord);
            });
            grid.appendChild(barEl);
          });
          viewer.appendChild(grid);
        }
      } else {
        const secLines = lines.filter(l => l.start >= sec.start && l.start < sec.end);
        if (secLines.length === 0) {
          const e = document.createElement("div");
          e.className = "empty-section";
          e.innerHTML = `<span class="icon" aria-hidden="true">∅</span><span>Bagian ini instrumental, tapi ada penanda section "berlirik" di pipeline. Abaikan saja.</span>`;
          viewer.appendChild(e);
        } else {
          secLines.forEach((line) => viewer.appendChild(renderLine(line)));
        }
      }
    });

    applyRomajiMode();
  }

  function renderLine(line) {
    const el = document.createElement("div");
    el.className = "line";
    el.dataset.start = line.start;
    el.dataset.end = line.end;
    el.id = "line-" + line.line_index;

    const wordsEl = document.createElement("div");
    wordsEl.className = "words";

    (line.words || []).forEach((w) => {
      const wrap = document.createElement("span");
      wrap.className = "word";
      wrap.dataset.start = w.start;
      wrap.dataset.end = w.end;
      const jp = document.createElement("span");
      jp.className = "word-jp";
      jp.textContent = w.word;
      const ro = document.createElement("span");
      ro.className = "word-ro";
      ro.textContent = w.romaji || "";
      wrap.appendChild(jp);
      wrap.appendChild(ro);
      wordsEl.appendChild(wrap);
    });
    el.appendChild(wordsEl);

    (line.chords || []).forEach((c) => {
      const marker = document.createElement("span");
      marker.className = "chord-marker";
      marker.textContent = c.chord;
      marker.dataset.chord = c.chord;
      marker.dataset.start = c.start;
      marker.dataset.anchor = c.anchor_word_index || 0;
      marker.title = c.chord + " @ " + c.start.toFixed(1) + "s — klik untuk loncat";
      marker.addEventListener("click", () => {
        if (state.sync) state.sync.seek(c.start);
        showDiagram(c.chord);
      });
      el.appendChild(marker);
    });

    return el;
  }

  /** Position chord markers over their anchor words. Run after layout. */
  function positionChordMarkers() {
    document.querySelectorAll(".line").forEach((lineEl) => {
      const words = lineEl.querySelectorAll(".word");
      const lineRect = lineEl.getBoundingClientRect();

      // Group markers by anchor word index — kalau 2+ chord nempel di kata
      // yang sama (mis. G + Em di akhir baris), distribusi horizontal-nya
      // biar gak overlap.
      const markersByAnchor = new Map();
      lineEl.querySelectorAll(".chord-marker").forEach((m) => {
        const anchor = parseInt(m.dataset.anchor, 10) || 0;
        if (!markersByAnchor.has(anchor)) markersByAnchor.set(anchor, []);
        markersByAnchor.get(anchor).push(m);
      });

      const SPACING = 30;     // px antar marker kalau overlap
      const EDGE_SAFE = 4;
      const LINE_PAD_LEFT = 12;

      function place() {
        for (const [anchor, markers] of markersByAnchor) {
          const wordEl = words[Math.min(anchor, words.length - 1)];
          if (!wordEl) continue;
          const wRect = wordEl.getBoundingClientRect();
          const lr = lineEl.getBoundingClientRect();
          const wordCenter = wRect.left - lr.left + wRect.width / 2;
          const count = markers.length;
          const totalWidth = (count - 1) * SPACING;
          const startX = wordCenter - totalWidth / 2;
          markers.forEach((m, i) => {
            m.style.left = (startX + i * SPACING) + "px";
          });
        }
      }

      // Pass 1: reset indent + place markers
      lineEl.style.paddingLeft = "";
      lineEl.style.setProperty("justify-content", "");
      place();

      // Pass 2: kalau chord kiri nyembul keluar → INDENT lirik (biar chord muat)
      let minLeft = Infinity;
      let maxRight = -Infinity;
      const lr0 = lineEl.getBoundingClientRect();
      lineEl.querySelectorAll(".chord-marker").forEach((m) => {
        const r = m.getBoundingClientRect();
        minLeft = Math.min(minLeft, r.left - lr0.left);
        maxRight = Math.max(maxRight, r.right - lr0.left);
      });
      const lineWidth = lr0.width;

      if (minLeft !== Infinity && minLeft < EDGE_SAFE) {
        const indent = Math.ceil(EDGE_SAFE - minLeft);
        lineEl.style.paddingLeft = (LINE_PAD_LEFT + indent) + "px";
        place();
      }

      // Pass 3: kalau chord kanan masih nyembul keluar → kasih space antar
      // kata (justify-content: space-around) biar container melebar dan
      // chord muat. GPP kata agak renggang — yg penting chord ngepas.
      if (maxRight !== -Infinity) {
        const lr1 = lineEl.getBoundingClientRect();
        let realMaxRight = -Infinity;
        lineEl.querySelectorAll(".chord-marker").forEach((m) => {
          const r = m.getBoundingClientRect();
          realMaxRight = Math.max(realMaxRight, r.right - lr1.left);
        });
        if (realMaxRight > lr1.width - EDGE_SAFE) {
          lineEl.style.setProperty("justify-content", "space-between");
        }
      }
    });
  }

  function applyRomajiMode() {
    const viewer = document.getElementById("viewer");
    viewer.classList.remove("mode-romaji-off", "mode-romaji-ro", "mode-romaji-both");
    viewer.classList.add("mode-romaji-" + state.romajiMode);
    requestAnimationFrame(positionChordMarkers);
  }

  function applyChordVisibility() {
    const viewer = document.getElementById("viewer");
    viewer.classList.toggle("chord-hidden", !state.showChord);
  }

  function applyFontScale() {
    const viewer = document.getElementById("viewer");
    if (!viewer) return;
    const base = 17; // matches .viewer { font-size: 17px }
    viewer.style.fontSize = (base * state.fontScale) + "px";
    document.getElementById("fs-val").textContent = Math.round(state.fontScale * 100) + "%";
    requestAnimationFrame(positionChordMarkers);
  }

  function setFontScale(v) {
    state.fontScale = Math.max(0.8, Math.min(1.4, v));
    applyFontScale();
  }

  /* ====================== SYNC ====================== */
  function initSync() {
    state.sync = new window.KC.SyncEngine({
      song: state.transposed,
      onTick: handleTick,
      onEnd: () => {
        const ic = document.getElementById("play-icon");
        if (ic) ic.textContent = "▶";
        const lb = document.getElementById("play-label");
        if (lb) lb.textContent = "Play";
      }
    });
  }

  function handleTick({ time, activeLine, activeChord }) {
    // highlight updates
    document.querySelectorAll(".line.active").forEach(l => l.classList.remove("active"));
    document.querySelectorAll(".chord-marker.active").forEach(c => c.classList.remove("active"));
    document.querySelectorAll(".bar.active").forEach(b => b.classList.remove("active"));

    if (activeLine) {
      const lineEl = document.getElementById("line-" + activeLine.line_index);
      if (lineEl) {
        lineEl.classList.add("active");
        if (state.autoScroll) {
          const rect = lineEl.getBoundingClientRect();
          if (rect.top < 80 || rect.bottom > window.innerHeight - 80) {
            window.KC.scrollTo(lineEl, 140);
          }
        }
      }
      // Highlight ONLY the current chord (latest start <= time), bukan semua yang start <= time.
      // Sebelumnya: semua marker di-highlight merah, jadi layar penuh merah.
      const markers = lineEl ? lineEl.querySelectorAll(".chord-marker") : [];
      let currentMarker = null;
      for (const m of markers) {
        const s = parseFloat(m.dataset.start);
        if (s <= time && (!currentMarker || s > parseFloat(currentMarker.dataset.start))) {
          currentMarker = m;
        }
      }
      if (currentMarker) currentMarker.classList.add("active");
    }

    document.querySelectorAll(".bar").forEach((b) => {
      const s = parseFloat(b.dataset.start);
      const e = parseFloat(b.dataset.end);
      if (time >= s && time < e) b.classList.add("active");
    });

    // === AUTO-PLAY CHORD ON CHANGE (kalau sound on) ===
    if (activeChord && activeChord !== state.lastChordPlayed) {
      if (state.soundOn) playChordClick(activeChord);
      state.lastChordPlayed = activeChord;
    }

    // === METRONOME ON BEAT (kalau metronome on) ===
    // Pakai BPM-based calculation (bukan beats array) supaya metronome jalan
    // sampai akhir lagu, bukan cuma sampai beats array habis.
    if (state.metronomeOn) {
      const bpm = state.transposed.meta && state.transposed.meta.bpm;
      if (bpm) {
        const beatInterval = 60 / bpm;
        const beatIdx = Math.floor(time / beatInterval);
        if (beatIdx !== state.lastBeatIdx) {
          // Hanya fire kalau natural progression (+1) atau start.
          // Kalau user seek jauh, skip biar nggak bunyi 10 klik bareng.
          const isStart = state.lastBeatIdx < 0;
          const isNext = beatIdx - state.lastBeatIdx === 1;
          if (isStart || isNext) {
            const bpb = getBeatsPerBar(state.transposed);
            playMetronomeClick((beatIdx % bpb) === 0);
          }
          state.lastBeatIdx = beatIdx;
        }
      }
    }

    // === DIAGRAM update ===
    if (activeChord && activeChord !== state.activeChord) {
      state.activeChord = activeChord;
      showDiagram(activeChord);
    }
  }

  /* ====================== TOOLBAR ====================== */
  /** Set active button in a segmented control. Idempotent. */
  function setActive(selector, btn) {
    const group = document.querySelector(selector);
    if (!group) return;
    group.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
    if (btn) btn.classList.add("active");
  }

  function initToolbar() {
    document.getElementById("btn-play").addEventListener("click", () => {
      state.sync.toggle();
      const playing = state.sync.playing;
      document.getElementById("play-icon").textContent = playing ? "❚❚" : "▶";
      document.getElementById("play-label").textContent = playing ? "Pause" : "Play";
    });
    document.getElementById("btn-restart").addEventListener("click", () => {
      state.lastBeatIdx = -1;
      state.lastChordPlayed = null;
      state.sync.restart();
    });

    document.getElementById("tr-minus").addEventListener("click", () => transposeBy(-1));
    document.getElementById("tr-plus").addEventListener("click", () => transposeBy(+1));
    document.getElementById("tr-reset").addEventListener("click", () => {
      state.transpose = 0;
      state.transposed = state.song;
      state.sync.setSong(state.song);
      document.getElementById("tr-val").textContent = "0";
      renderMeta();
      renderViewer();
      renderSectionButtons();
      requestAnimationFrame(positionChordMarkers);
    });

    // === Speed segmented ===
    document.querySelectorAll("#speed-seg button").forEach(b => {
      b.addEventListener("click", () => {
        setActive("#speed-seg", b);
        state.speed = parseFloat(b.dataset.speed);
        state.sync.setSpeed(state.speed);
      });
    });

    // === Font size ===
    document.getElementById("fs-minus").addEventListener("click", () => setFontScale(state.fontScale - 0.1));
    document.getElementById("fs-plus").addEventListener("click", () => setFontScale(state.fontScale + 0.1));
    document.getElementById("fs-val").addEventListener("click", () => setFontScale(1.0));

    // === Instrument segmented ===
    document.querySelectorAll("#instr-seg button").forEach(b => {
      b.addEventListener("click", () => {
        setActive("#instr-seg", b);
        state.instrument = b.dataset.instr;
        document.getElementById("diagram-instr-label").textContent =
          state.instrument === "guitar" ? "Gitar" : state.instrument === "ukulele" ? "Ukulele" : "Piano";
        if (state.activeChord) showDiagram(state.activeChord);
      });
    });

    // === Romaji 3-mode segmented ===
    document.querySelectorAll("#romaji-seg button").forEach(b => {
      b.addEventListener("click", () => {
        setActive("#romaji-seg", b);
        state.romajiMode = b.dataset.romaji;
        applyRomajiMode();
      });
    });

    document.getElementById("opt-chord").addEventListener("change", (e) => {
      state.showChord = e.target.checked;
      applyChordVisibility();
    });

    document.getElementById("opt-sound").addEventListener("change", (e) => {
      state.soundOn = e.target.checked;
      if (e.target.checked) {
        ensureAudio();         // user gesture
        // Bunyiin sekali sebagai feedback
        playChordClick("C");
      }
    });

    document.getElementById("opt-metronome").addEventListener("change", (e) => {
      state.metronomeOn = e.target.checked;
      if (e.target.checked) {
        ensureAudio();
        // Test tick sekali
        playMetronomeClick(true);
      } else {
        state.lastBeatIdx = -1;
      }
    });

    document.getElementById("opt-autoscroll").addEventListener("change", (e) => {
      state.autoScroll = e.target.checked;
    });

    renderSectionButtons();
    window.addEventListener("resize", window.KC.debounce(positionChordMarkers, 100));
  }

  function transposeBy(delta) {
    state.transpose += delta;
    state.transposed = window.KC.transposeRender(state.song, state.transpose);
    state.sync.setSong(state.transposed);
    document.getElementById("tr-val").textContent = (state.transpose > 0 ? "+" : "") + state.transpose;
    renderMeta();
    renderViewer();
    renderSectionButtons();
    requestAnimationFrame(positionChordMarkers);
  }

  function renderSectionButtons() {
    const wrap = document.getElementById("section-buttons");
    wrap.innerHTML = "";
    (state.transposed.sections || []).forEach((s) => {
      const b = document.createElement("button");
      b.textContent = s.name;
      b.addEventListener("click", () => {
        if (state.sync) state.sync.seek(s.start);
      });
      wrap.appendChild(b);
    });
  }

  /* ====================== DIAGRAM ====================== */
  function showDiagram(chord) {
    const wrap = document.getElementById("diagram-svg");
    const name = document.getElementById("diagram-chord-name");
    name.textContent = chord;
    wrap.innerHTML = window.KC.renderChordDiagram(chord, state.instrument);
  }

  /* ====================== LOOP (removed) ====================== */
  // Loop feature removed per request.

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
  }

  document.addEventListener("DOMContentLoaded", () => {
    (async () => {
      await init();
      requestAnimationFrame(() => {
        positionChordMarkers();
        if (document.fonts && document.fonts.ready) {
          document.fonts.ready.then(() => requestAnimationFrame(positionChordMarkers));
        }
      });
    })();
  });
})();
