# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is KenshiChord

Chordify-style AI auto-chord platform with a Japanese-samurai theme. Paste a YouTube URL → backend downloads audio → ML pipeline (BTC-ISMIR19 + faster-whisper + torchaudio MMS_FA) detects chords, transcribes lyrics, and aligns them per-word → frontend renders chord-over-lyric with romaji, transpose, tempo, metronome, and chord-diagram support.

The project is built in **phases** (FASE 0–5, see `README.md`). FASE 0 (frontend mock mode) and FASE 1 (backend skeleton + scaffold pipeline) are done. **FASE 2 is the active work** — replacing the FASE 1 placeholder pipeline with real ML. Most of the active development happens in `backend/smartfix_auto.py` (and the deprecated `smartfix_v[1-4].py` / `smart_fix.py` / `fix_song.py` / `extract_one.py` siblings — prefer `smartfix_auto.py`).

## Commands

### Run the dev stack (two terminals)

**Backend (FastAPI on :8000):**
```bash
cd backend
# Windows:
run.bat
# Linux/macOS:
./run.sh
# Or manual:
python -m venv venv && source venv/bin/activate   # Linux/macOS  (Windows: venv\Scripts\activate)
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Swagger UI: `http://127.0.0.1:8000/docs`

**Frontend (static, :8080):**
```bash
cd frontend
python -m http.server 8080
# → http://127.0.0.1:8080/index.html
```
The frontend auto-detects the backend at `http://127.0.0.1:8000` and falls back to mock mode (`frontend/js/mock-data.js`) if the API is unreachable.

**Run the FASE 2 ML pipeline end-to-end on a single YouTube URL:**
```bash
cd backend
python smartfix_auto.py "https://www.youtube.com/watch?v=mG7lrRdm71A"   # audio only
python smartfix_auto.py "<url>" "path/to/reference_lyrics.txt"          # with ref lyrics
```
This writes audio to `data/audio/<youtube_id>.wav`, runs BTC-ISMIR19 (subprocess into `backend/BTC-ISMIR19/`), and persists a fully-aligned `render_json` to `data/kenshi.db`.

**Smoke-test endpoints:**
```bash
curl http://127.0.0.1:8000/api/health
curl -X POST -H "Content-Type: application/json" \
  -d '{"youtube_url":"https://youtu.be/mG7lrRdm71A"}' \
  http://127.0.0.1:8000/api/songs/generate
curl http://127.0.0.1:8000/api/jobs/<job_id>
curl http://127.0.0.1:8000/api/songs/<song_id>
```

### Lint / format / type-check

There is **no enforced lint or test suite** in this repo right now. When editing Python, follow the existing style (type hints on public functions, dataclasses, f-strings, `from __future__ import annotations`). When editing JS, follow the existing IIFE-module pattern in `frontend/js/*.js` (each file is `(function () { "use strict"; ... window.KC = ... })();`).

## Architecture

### Two-tier rendering contract

The whole app pivots on one JSON document: `render_json` stored in `songs.render_json` (TEXT column). It has this shape — see `frontend/js/mock-data.js` for the canonical example, and `SongDetail` in `backend/app/schemas.py` for the typed view:

```jsonc
{
  "meta":     { "youtube_id", "artist", "title", "duration_sec", "bpm", "key", "capo", "time_sig", "language" },
  "beats":    [0.0, 0.45, ...],          // seconds
  "downbeats":[0.0, 1.82, ...],
  "sections": [{ "name", "start", "end", "has_lyrics" }],
  "bars":     [{ "index", "start", "end", "chords": [{ "chord", "start", "end" }] }],
  "lines":    [{ "line_index", "start", "end", "text",
                 "words":  [{ "word", "start", "end", "romaji" }],
                 "chords": [{ "chord", "start", "anchor_word_index" }] }]
}
```

Anything that produces or consumes this document is the heart of the codebase. The frontend (`frontend/js/song-view.js`) renders it; the backend (`backend/app/pipeline.py` FASE 1 / `backend/smartfix_auto.py` FASE 2) produces it.

### Backend layering (`backend/app/`)

- **`main.py`** — FastAPI app, CORS, startup hook (`init_db`), mounts three routers.
- **`config.py`** — `pydantic-settings` `Settings`, reads `.env`. `data/` is auto-created. SQLite is default; MySQL via `DATABASE_URL` env var.
- **`db.py`** — `engine`, `SessionLocal`, `get_db` (FastAPI dep), `db_session` (context manager for scripts/BackgroundTasks), `init_db` calls `Base.metadata.create_all` (no Alembic migrations yet).
- **`models.py`** — Five tables: `songs` (cache key = `youtube_id` UNIQUE + composite UNIQUE on `(artist_norm, title_norm)`), `chords` (granular per-chord rows; cascade delete), `lyric_lines` + `lyric_words` (cascade delete), `jobs` (uuid PK, status enum: `queued|downloading|separating|detecting|transcribing|aligning|done|failed`).
- **`schemas.py`** — Pydantic v2 request/response shapes. `SongDetail` is the parsed `render_json`.
- **`cache.py`** — Artist/title normalization (lowercase, strip accents, drop "(Official Video)" / "[MV]" / "feat. X" suffixes). `find_existing_song` checks `youtube_id` first, then `(artist_norm, title_norm)`. Rapidfuzz fuzzy match wired in `requirements.txt` but not yet called.
- **`metadata.py`** — `yt-dlp` wrapper with 3-tier fallback (full extract → `process=False` raw → minimal placeholder). Never downloads audio — that's `pipeline.py`.
- **`pipeline.py`** — FASE 1 scaffold. Runs in `BackgroundTasks`; emits job status updates; calls `_build_scaffold_render` which produces a placeholder 1-bar, 1-chord document. **FASE 2 replaces this with `smartfix_auto.main`.**
- **`routes/generate.py`** — `POST /api/songs/generate`. Cache check by `youtube_id`, returns `{cached, song}` or enqueues `{job_id}`. `_safe_run_pipeline` catches exceptions so BackgroundTask can't kill the worker.
- **`routes/songs.py`** — `GET /api/songs` (list with `?search=&limit=&offset=`), `GET /api/songs/{id}`, `GET /api/songs/by-youtube/{youtube_id}`, `DELETE /api/songs/{id}`.
- **`routes/jobs.py`** — `GET /api/jobs/{job_id}` — frontend polls this every ~2s while loading.

### FASE 2 real-ML pipeline (`backend/smartfix_auto.py`)

This is the active development target. 8 stages, run sequentially in `main(url, reference_lyrics=None, save=True)`:

1. `download_audio` — yt-dlp → WAV via ffmpeg postprocessor.
2. `btc_detect` — shells out to `python test.py` in `backend/BTC-ISMIR19/` (subprocess, GPU if available). Returns `.lab` file with `(start, end, chord_root:modifier)` tuples. Parse via `parse_btc_lab`. Consolidate via `consolidate_btc` (root-merge adjacent identical chords, drop <0.8s noise).
3. `transcribe_whisper` — `faster_whisper.WhisperModel("base", device=cpu|cuda, compute_type="int8")` with `word_timestamps=True` and `vad_filter=True`. Returns `[{start, end, text, words:[{word, start, end}]}]`.
4. `try_fetch_reference_lyrics` — best-effort web fetch (kazelyrics → genius). Currently bypassed in `main()`.
5. **Forced alignment** — `forced_align(wav, romaji_list, window_ranges=...)` using `torchaudio.pipelines.MMS_FA` (MMS Force-Aligner, vocab is a-z + `-`). This is where the active debugging lives. The function:
   - Caps each window to 25s (MMS_FA's optimal range).
   - Distributes `romaji_list` tokens across windows proportional to window duration.
   - For each window: build a flat token list by mapping each romaji char through `bundle.get_dict()`, run `forced_align(emission, targets, blank=0)`, `merge_tokens`, then map back to `(start, end)` in absolute song time using `ratio = chunk_samples / emission_T`.
   - `romaji()` helper uses Janome for tokenization + pykakasi for kana→romaji. **Always returns `list[str]`** (one romaji per tokenized word). Critical: callers must pass a *flat* list of romaji strings, not a list of lists.
6. `interpolate_missing` — fills `None` slots from known neighbors (used when some windows fail).
7. `detect_sections` — auto-labels Intro/Interlude/Verse/Chorus/Outro from vocal gaps (`>2.5s`).
8. Bar grid + beats derived from consolidated BTC segments, then write to DB.

### Frontend (`frontend/`)

Vanilla HTML/CSS/JS, no build step. `index.html` (home) → `library.html` (browse) → `song.html` (player). All pages share the samurai design system in `frontend/css/tokens.css` (CSS variables — colors, fonts Shippori Mincho + Inter) and `frontend/css/components.css` (buttons, cards, tags).

State lives on `window.KC` (set by each module on load). Key entry points:
- `frontend/js/api.js` — `KC.api` (generate, getJob, getSong, listSongs, …). Caches the health-check result so the first 404 doesn't re-fire on every page.
- `frontend/js/song-view.js` — the big one. Loads a song by `?id=` (numeric) or `?y=` (YouTube ID), renders section headers + bar grid + lines, owns the Web Audio metronome/chord-click synth (`chordToMidiNotes` → triangle-wave chord, sine-wave downbeat tick), wires the toolbar (transpose, speed, romaji mode off/ro/both, font scale 80–140%, sound, metronome, auto-scroll).
- `frontend/js/sync-engine.js` — `requestAnimationFrame` loop that highlights the active line/chord given the audio playhead. Polled by `song-view.js` via `state.sync`.
- `frontend/js/chord-utils.js` + `chord-data.js` + `chord-diagram.js` — chord parsing (`parseChord` → `{root, modifier, kind:"parsed"}`), note math (`noteToIdx`), and SVG fretboard diagrams for guitar/ukulele/piano.
- `frontend/js/mock-data.js` — 2 public-domain Japanese songs (Soran Bushi, Sakura Sakura) + romaji injector (`kanaToRomaji` heuristic with manual overrides for the demo songs). Mock library has 8 entries.

### Key gotchas

- **Job state is updated by `_update_job` in `pipeline.py` / `smartfix_auto.main`** — each phase commits to the DB so the polled `/api/jobs/{id}` sees fresh `status` and `progress`.
- **Cache uniqueness**: `songs.youtube_id` is the primary cache key, but `(artist_norm, title_norm)` is also UNIQUE — uploading the same song from a re-uploaded video still hits cache.
- **`romaji()` always returns `list[str]`** (never a single string). Anywhere it feeds into `forced_align`, you must flatten any list-of-lists into a flat `list[str]` *after* `romaji()` runs, **not** before.
- **MMS_FA expects lowercase a-z + `-` only.** Kanji/hiragana must be converted to romaji first; chars not in `bundle.get_dict()` (29 entries) are silently skipped and produce `None` spans, which `interpolate_missing` then has to fill.
- **Windows are capped at 25s.** Anything longer gets split; the last window can be <25s and that's fine.
- **GPU vs CPU**: `_mms()` forces `.to("cpu")` because the aligner is small and CPU is fast enough; demucs/BTC still auto-select GPU when available.
- **Frontend mock fallback**: if `KC.api.isAvailable()` returns false (network error, CORS, 5xx), `song-view.js` calls `getMockSong(youtube_id)` from `mock-data.js`. So you can develop the UI without the backend running.
- **Deprecated pipeline files**: `smartfix_v[1-4].py`, `smart_fix.py`, `fix_song.py`, `extract_one.py` are snapshots from prior debugging sessions. Do not edit them — `smartfix_auto.py` is the canonical FASE 2 entry point.
