# 剣士コード KenshiChord

> **Listen. Detect. Play.** — AI auto-chord platform bergaya **Chordify** dengan tema **samurai Jepang**.
> Tempel link YouTube → sistem otomatis menghasilkan **chord + lirik ter-sync timing-nya**, lengkap dengan **romaji** untuk pengguna non-Jepang.

```
┌──────────────────────────────────────────────────────────┐
│  FASE 0  → Frontend samurai siap pakai (mock mode)      │
│  FASE 1  → Backend FastAPI + cache + scaffold pipeline  │
│  FASE 2  → Real ML (BTC + Whisper + MMS_FA) ✅ wired    │
│  FASE 3  → Integrasi & final polish                     │
│  FASE 4  → Production hardening                          │
│  FASE 5  → Tes nyata pertama                            │
└──────────────────────────────────────────────────────────┘
```

---

## 📦 Struktur Repo

```
chord/
├── frontend/                   ← HTML/CSS/JS samurai (FASE 0)
│   ├── index.html              Home / Landing
│   ├── song.html               Song View (Chordify-style)
│   ├── library.html            Library / Browse
│   ├── css/                    Design system samurai
│   ├── js/                     Mock data + sync + transposer + diagram + API wrapper
│   └── assets/                 Logo SVG, hinomaru, favicon
├── backend/                    ← Python FastAPI (FASE 1)
│   ├── app/
│   │   ├── main.py             FastAPI app + CORS + mount routes
│   │   ├── config.py           Settings (env + .env)
│   │   ├── db.py               SQLAlchemy engine + session
│   │   ├── models.py           ORM models (5 tables)
│   │   ├── schemas.py          Pydantic schemas
│   │   ├── cache.py            Normalize artist/title + cache lookup
│   │   ├── metadata.py         yt-dlp wrapper (resilient)
│   │   ├── pipeline.py         Pipeline orchestrator (placeholder FASE 2)
│   │   └── routes/
│   │       ├── songs.py        GET /api/songs, /api/songs/{id}, DELETE
│   │       ├── jobs.py         GET /api/jobs/{id}
│   │       └── generate.py     POST /api/songs/generate
│   ├── BTC-ISMIR19/            BTC chord detector (cloned, weights gitignored)
│   ├── smartfix_auto.py        FASE 2 real-ML pipeline (BTC + Whisper + MMS_FA)
│   ├── requirements.txt
│   ├── .env.example
│   ├── run.sh                  Linux/macOS dev script
│   └── run.bat                 Windows dev script
├── data/                       Created on first run (gitignored)
│   ├── kenshi.db               SQLite database
│   └── audio/                  Downloaded WAV files
├── .gitignore                  Ignores heavy models, venv, data, etc.
└── README.md
```

---

## 📥 Heavy models — what to download before running FASE 2

Repo keeps the code; the weights are too big to commit. The `.gitignore` blocks
them automatically. Download once per machine and place them in the paths
below.

### 1. BTC-ISMIR19 pretrained checkpoints (~12 MB each)
Needed by [backend/BTC-ISMIR19/](backend/BTC-ISMIR19/) (chord detector used in
FASE 2 via `backend/smartfix_auto.py`).

- **From the original repo's releases / drive** (paper authors' weights):
  - `btc_model.pt` — 12-class vocabulary (`--voca False`, default)
  - `btc_model_large_voca.pt` — 170-class vocabulary (`--voca True`)
- **Drop into:** `backend/BTC-ISMIR19/test/`
- **Verify:** files appear next to `test.py` in that folder, named exactly
  `btc_model.pt` and `btc_model_large_voca.pt`.

If a fresh clone of BTC-ISMIR19 doesn't include the weights (most mirrors
don't), grab them from the ISMIR19 paper repo's release page or the authors'
pretrained mirror and place them in `test/`.

### 2. faster-whisper (`base` int8) — ~150 MB
Auto-downloaded on first run by the `faster-whisper` package into your user
cache (e.g. `%LOCALAPPDATA%\faster-whisper\models\` on Windows,
`~/.cache/huggingface/hub/` on Linux/macOS). No manual step needed — just run
the pipeline once and the weights will be fetched and cached.

If you want to pre-fetch (saves ~2 min on first run), set
`HF_HOME=/path/to/cache` in `.env` and download with:
```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
```

### 3. torchaudio MMS_FA (force-aligner) — ~700 MB
Auto-downloaded by `torchaudio.pipelines.MMS_FA` into torch's hub cache the
first time `_mms()` is called. Same locations as faster-whisper. No manual
step needed.

### 4. demucs v4 (`htdemucs`) — ~80 MB per stem
Used implicitly if you wire demucs into the pipeline. Auto-downloaded on first
run into the same torch hub cache.

### Cache locations (for reference)
- **Windows:** `%LOCALAPPDATA%\torch\hub\`, `%LOCALAPPDATA%\huggingface\hub\`
- **Linux/macOS:** `~/.cache/torch/hub/`, `~/.cache/huggingface/hub/`

To reclaim disk space later, delete those folders — the pipeline will re-fetch
on the next run.

---

## 🗡️ FASE 0 — Frontend Samurai ✅

Mock mode (no backend needed). Tinggal buka `frontend/index.html` di browser atau serve via:

```bash
cd frontend && python -m http.server 8000
```

**Fitur lengkap** (lihat `frontend/_screens/` untuk screenshot):
- Hero samurai, library preview, CTA
- Song View ala Chordify (chord-over-lyric, bar grid, sync highlight, auto-scroll)
- Romaji 3-mode (Off / Romaji only / 両方)
- Font size adjuster (80%–140%)
- Transpose −/+
- Speed playback 0.5×–1.5×
- Sound toggle (chord click + auto-play on chord change)
- Metronome toggle (BPM-based, sampai akhir lagu)
- Chord diagram SVG (gitar/ukulele/piano)
- Section jump, auto-scroll, responsive

---

## ⚡ FASE 1 — Backend Skeleton ✅

**Stack:** FastAPI + SQLAlchemy 2.0 + Pydantic v2 + yt-dlp + BackgroundTasks
**DB:** SQLite (dev) — switch ke MySQL via `DATABASE_URL` di `.env`
**Cache key:** `youtube_id` primary, `(artist_norm, title_norm)` secondary

### Cara Menjalankan (FASE 1)

#### 1. Install dependencies
```bash
cd backend
python -m pip install -r requirements.txt
```

#### 2. Start backend
```bash
# Windows
run.bat

# Linux/macOS
./run.sh

# Atau manual
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend listen di `http://127.0.0.1:8000`. Swagger UI: `http://127.0.0.1:8000/docs`

#### 3. Start frontend (terminal lain)
```bash
cd frontend
python -m http.server 8080
# buka http://127.0.0.1:8080/index.html
```

Frontend auto-detect backend di `http://127.0.0.1:8000`. Kalau backend down, otomatis fallback ke mock mode (lihat frontend).

### API Endpoints

| Method | Path | Fungsi |
|---|---|---|
| `GET`  | `/api/health` | Health check |
| `POST` | `/api/songs/generate` | Submit YouTube URL → cache check, return cached atau enqueue job |
| `GET`  | `/api/jobs/{job_id}` | Job status (poll tiap ~2s pas loading) |
| `GET`  | `/api/songs` | Library list, query `?search=...&limit=50&offset=0` |
| `GET`  | `/api/songs/{id}` | Full render_json untuk song view |
| `GET`  | `/api/songs/by-youtube/{youtube_id}` | Lookup by YouTube ID |
| `DELETE` | `/api/songs/{id}` | Admin delete (cascade) |

### Database Schema (5 tables)

```sql
songs       ──── 1 row per lagu unik. Cache key: youtube_id (UNIQUE) + (artist_norm, title_norm) UNIQUE
              Fields: meta (artist/title/bpm/key/duration/etc), render_json (TEXT), status, source
chords      ──── Optional: granular chord-level data. Cascade delete dari song.
lyric_lines ──── Lirik per-baris. Cascade delete.
lyric_words ──── Per-kata dengan timestamp. Cascade delete dari line.
jobs        ──── Async job tracking. uuid PK. Status: queued|downloading|separating|detecting|transcribing|aligning|done|failed
```

Lihat [backend/app/models.py](backend/app/models.py) untuk full SQLAlchemy definitions.

### Pipeline (FASE 1 placeholder)

1. **Fetch metadata** (yt-dlp, dengan fallback 3-tier: full → raw → minimal)
2. **Cache check** (youtube_id first, then artist+title)
3. **Download audio** (best audio → WAV 44.1kHz via ffmpeg)
4. **Estimate BPM** (librosa kalau ada, else null)
5. **Detect language** (heuristic)
6. **Build scaffold render_json** (placeholder — FASE 2 replace)
7. **Persist ke DB**

FASE 2 akan replace step 6 dengan real ML:
- Demucs (source separation)
- madmom/allin1 (beat & downbeat)
- autochord/BTC (chord recognition)
- WhisperX (lyric transcription + alignment)

### Cache Logic

```python
# Normalization rules:
# - Lowercase
# - Strip accents (café → cafe)
# - Remove punctuation
# - Remove "(Official Video)", "[MV]", "(Lyrics)", etc
# - Remove "feat. X" di luar parens
# - Whitespace collapse

# Lookup:
# 1. by youtube_id (UNIQUE)
# 2. by (artist_norm, title_norm) UNIQUE
# 3. (FASE 2) rapidfuzz untuk fuzzy match dengan threshold configurable
```

---

## 🛣️ Build Order & Roadmap

- [x] **FASE 0** — Frontend samurai dengan mock data ← **selesai**
- [x] **FASE 1** — Backend skeleton (FastAPI + SQLAlchemy + cache + scaffold pipeline) ← **selesai**
- [x] **FASE 2** — Real ML pipeline (yt-dlp → BTC → Whisper → MMS_FA → DB) ← **wired, see [smartfix_auto.py](backend/smartfix_auto.py)**
- [ ] **FASE 3** — Integrasi polish: fuzzy cache match, transposer server-side, error states
- [ ] **FASE 4** — Production hardening: MySQL, RQ+Redis queue, auth, rate limit
- [ ] **FASE 5** — Tes nyata pertama dengan `https://youtu.be/mG7lrRdm71A` end-to-end

> **Engineering decision (FASE 1):** NO training from scratch. Pakai pretrained (BTC, faster-whisper, MMS_FA, demucs). Fine-tune HANYA kalau akurasi kurang di genre tertentu. Weights are downloaded once per machine — see [Heavy models](#-heavy-models--what-to-download-before-running-fase-2).

> **Pipeline selection:** default mode is FASE 2 (real ML). Set `PIPELINE_MODE=fase1` in `backend/.env` to fall back to the scaffold pipeline.

---

## 🧪 Test End-to-End (FASE 1 verified)

```bash
# 1. Health check
curl http://127.0.0.1:8000/api/health
# {"status":"ok","version":"0.1.0-fase1","fase":"1"}

# 2. Submit URL
curl -X POST -H "Content-Type: application/json" \
  -d '{"youtube_url":"https://youtu.be/mG7lrRdm71A"}' \
  http://127.0.0.1:8000/api/songs/generate
# {"cached":false,"job_id":"5737ca04-...","song_id":null}

# 3. Poll job (cache miss → pipeline runs → done in ~10-30s)
curl http://127.0.0.1:8000/api/jobs/5737ca04-...
# {"status":"done","progress":100,"song_id":1}

# 4. Get song
curl http://127.0.0.1:8000/api/songs/1
# {meta, beats, sections, bars, lines}

# 5. Cache hit (submit same URL again)
curl -X POST -H "Content-Type: application/json" \
  -d '{"youtube_url":"https://youtu.be/mG7lrRdm71A"}' \
  http://127.0.0.1:8000/api/songs/generate
# {"cached":true,"song_id":1,"song":{...}}
```

---

## 📂 File Penting untuk Dipelajari Dulu

Urutan bacaan kalau baru masuk codebase:

1. [frontend/css/tokens.css](frontend/css/tokens.css) — design system
2. [frontend/js/mock-data.js](frontend/js/mock-data.js) — struktur `render_json` (APPENDIX A)
3. [frontend/js/api.js](frontend/js/api.js) — backend wrapper (FASE 1)
4. [frontend/js/song-view.js](frontend/js/song-view.js) — render lyric + chord + bar grid, wire toolbar
5. [backend/app/main.py](backend/app/main.py) — FastAPI entrypoint
6. [backend/app/models.py](backend/app/models.py) — DB schema
7. [backend/app/pipeline.py](backend/app/pipeline.py) — pipeline orchestrator (FASE 2 target)
8. [backend/app/metadata.py](backend/app/metadata.py) — yt-dlp wrapper

---

## 📜 Lisensi Mock Data

Lagu di mock data:
- **Soran Bushi** — Japanese traditional, **public domain**
- **Sakura Sakura** — Japanese folk, **public domain**

Logo, design system, dan kode: © KenshiChord 2026.

---

## ⚔️ 一言

> *"剣士のように、コードを見つける。一音一音に、心を込めて。"*
> *Seperti seorang kenshi, menemukan chord. Di setiap nada, dengan segenap hati.*

⚔ **KenshiChord** — 耳で聴き、コードを見つけ、奏でる。
