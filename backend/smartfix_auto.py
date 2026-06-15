"""smartfix_auto — FULLY AUTOMATIC pipeline (no hardcoded lyrics).

Run on any YouTube URL → audio → ML → render_json → DB.

Steps:
  1. Download audio (yt-dlp)
  2. BTC-ISMIR19 chord detection (.lab)
  3. Whisper transcription (word_timestamps)
  4. Optional reference lyrics: try fetch from UG tab / kazelyrics / genius
  5. MMS_FA forced alignment (real per-word timing)
  6. Snap BTC chord ke real timing
  7. Section detection (intro/verse/chorus/outro + interludes)
  8. Save to DB
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import gc
import torchaudio.functional as AF
from torchaudio.functional import forced_align as _ta_forced_align, merge_tokens as _ta_merge_tokens
from torchaudio.pipelines import MMS_FA as bundle
import yt_dlp

# When imported as a library (e.g. by routes/generate.py or tests), don't override
# the host app's logging — only `__main__` block at the bottom calls basicConfig.
log = logging.getLogger("auto")


# ============ Helpers ============
_KK = None
_JT = None
def _kks():
    global _KK
    if _KK is None:
        import pykakasi as _p
        _KK = _p.kakasi()
    return _KK
def _jt():
    global _JT
    if _JT is None:
        from janome.tokenizer import Tokenizer
        _JT = Tokenizer()
    return _JT

def romaji(text):
    out = []
    for t in _jt().tokenize(text):
        s = t.surface
        if not s.strip(): continue
        if s.isascii():
            r = "".join(c for c in s.lower() if c.isalpha())
        else:
            reading = t.reading if t.reading and t.reading != "*" else s
            parts = _kks().convert(reading)
            r = "".join(p.get("hepburn","") for p in parts).lower()
            r = "".join(c for c in r if c.isalpha())
        if r: out.append(r)
    return out

def normalize_artist_title(artist: str, title: str) -> str:
    """For fuzzy match against fetched reference lyrics."""
    def clean(s):
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = re.sub(r"\(official[^)]*\)|\[official[^\]]*\]|\(lyric[^\)]*\)|\[lyric[^\]]*\]|\(music[^\)]*\)|\[mv\]|feat\.?[^\(]*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
        return re.sub(r"\s+", " ", s).strip()
    return f"{clean(artist)} {clean(title)}"


# ============ Step 1: Download audio ============
def download_audio(url, out_dir: Path, youtube_id_hint=None) -> tuple[Path, dict]:
    """Download best audio and return (wav_path, info_dict).

    `info_dict` is the raw yt-dlp `extract_info` result so the caller can pick
    out artist/title/thumbnail without a second network round-trip.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "quiet": True, "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    yt_id = info.get("id") or youtube_id_hint
    for ext in ("wav", "m4a", "webm", "mp4", "opus"):
        p = out_dir / f"{yt_id}.{ext}"
        if p.exists(): return p, info
    raise RuntimeError("WAV not found after download")


# ============ Step 2: BTC-ISMIR19 chord detection ============
def btc_detect(wav_path: Path, out_dir: Path, device="cuda" if torch.cuda.is_available() else "cpu") -> Path:
    """Run BTC-ISMIR19. Returns path to .lab file."""
    btc_dir = Path(__file__).resolve().parent / "BTC-ISMIR19"
    if not btc_dir.exists():
        raise RuntimeError(f"BTC-ISMIR19 not found at {btc_dir}. Run: git clone https://github.com/jayg996/BTC-ISMIR19")
    audio_in = out_dir / f"btc_in_{wav_path.stem}"
    audio_in.mkdir(parents=True, exist_ok=True)
    target = audio_in / wav_path.name
    if not target.exists():
        target.write_bytes(wav_path.read_bytes())
    btc_out = out_dir / f"btc_out_{wav_path.stem}"
    btc_out.mkdir(parents=True, exist_ok=True)
    cmd = ["python", "test.py", "--audio_dir", str(audio_in), "--save_dir", str(btc_out)]
    log.info(f"  Running BTC-ISMIR19 ({device})...")
    t0 = time.time()
    # Don't capture stderr — BTC's training script writes progress to stderr
    # and the real error trace when it fails. Let it stream so users can see
    # what went wrong.
    result = subprocess.run(cmd, cwd=btc_dir, capture_output=False, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"BTC test.py exited with code {result.returncode}")
    log.info(f"  BTC done in {time.time()-t0:.1f}s")
    # find .lab
    for p in btc_out.rglob("*.lab"):
        return p
    raise RuntimeError("BTC .lab not produced")


def parse_btc_lab(path: Path):
    segs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                try: segs.append((float(p[0]), float(p[1]), p[2]))
                except ValueError: pass
    return segs


# ============ Step 3: Whisper transcription ============
def transcribe_whisper(wav_path: Path, lang=None, initial_prompt=None):
    """Returns list of {start, end, text, words}."""
    import faster_whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = faster_whisper.WhisperModel("base", device=device, compute_type="int8")
    kwargs = dict(beam_size=5, vad_filter=True, word_timestamps=True)
    if lang: kwargs["language"] = lang
    if initial_prompt: kwargs["initial_prompt"] = initial_prompt
    segs, info = m.transcribe(str(wav_path), **kwargs)
    out = []
    for s in segs:
        words = []
        if s.words:
            for w in s.words:
                if w.start is None or w.end is None: continue
                words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
        if not words:
            text = s.text.strip()
            if text:
                words = [{"word": text, "start": float(s.start), "end": float(s.end)}]
        if not words: continue
        out.append({"start": float(s.start), "end": float(s.end),
                    "text": s.text.strip(), "words": words})
    return out, info.language


# ============ Step 4: Try fetch reference lyrics (UG/kazelyrics/genius) ============
def try_fetch_reference_lyrics(artist, title, timeout=8):
    """Try multiple sources. Returns list of (source, lines) or None."""
    q = f"{artist} {title}"
    # 1. Try kazelyrics (anime/J-pop often has it)
    try:
        import requests
        from bs4 import BeautifulSoup
        # Pseudo-search: try common URL patterns
        slug = re.sub(r"[^a-z0-9]+", "-", (artist + " " + title).lower()).strip("-")
        for base in [
            f"https://www.kazelyrics.com/2026/02/lirikterjemahan-{slug}.html",
            f"https://www.kazelyrics.com/2025/02/lirikterjemahan-{slug}.html",
        ]:
            r = requests.get(base, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and "lirik" in r.text.lower():
                soup = BeautifulSoup(r.text, "html.parser")
                lines = [p.get_text(" ", strip=True) for p in soup.select(".lirik p, .entry p")
                         if p.get_text(strip=True) and len(p.get_text(strip=True)) < 200]
                if lines: return ("kazelyrics", lines)
    except Exception as e:
        log.debug(f"  kazelyrics: {e}")
    # 2. Try genius (search)
    try:
        import requests
        r = requests.get("https://genius.com/api/search", params={"q": q}, timeout=timeout)
        if r.status_code == 200:
            hits = r.json().get("response", {}).get("sections", [{}])[0].get("hits", [])
            for hit in hits[:3]:
                title_match = (normalize_artist_title(artist, title) in
                                normalize_artist_title(hit["result"]["primary_artist"]["name"], hit["result"]["title"]))
                if title_match:
                    # fetch lyrics page
                    lyr_url = hit["result"]["url"]
                    r2 = requests.get(lyr_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
                    if r2.status_code == 200:
                        soup = BeautifulSoup(r2.text, "html.parser")
                        lines = [d.get_text(" ", strip=True) for d in soup.select("[data-lyrics-container] div, .lyrics p")
                                 if d.get_text(strip=True) and len(d.get_text(strip=True)) < 200]
                        if lines: return ("genius", lines)
    except Exception as e:
        log.debug(f"  genius: {e}")
    return None


# ============ Step 5: MMS_FA forced alignment ============
_MODEL = None
_MODEL_DEVICE = None
def _mms():
    """Lazy-load MMS_FA once. Prefer GPU only if a real forward pass works —
    `torch.cuda.is_available()` can lie (e.g. broken driver / no kernel)."""
    global _MODEL, _MODEL_DEVICE
    if _MODEL is not None:
        return _MODEL
    DICT = bundle.get_dict()
    model = bundle.get_model()
    device = "cpu"
    if torch.cuda.is_available():
        try:
            probe = torch.zeros(1, 16000)
            model.to("cuda").eval()(probe)
            device = "cuda"
            log.info(f"  MMS_FA: CUDA probe OK")
        except Exception as e:
            log.info(f"  MMS_FA: CUDA unusable ({type(e).__name__}), falling back to CPU")
    _MODEL = model.to(device).eval()
    _MODEL_DEVICE = device
    log.info(f"  MMS_FA loaded on {device} (vocab={len(DICT)})")
    return _MODEL

def _model_device() -> str:
    if _MODEL_DEVICE is None:
        _mms()
    return _MODEL_DEVICE or "cpu"

def forced_align(vocals_path: Path, romaji_list, window_ranges=None, word_intervals=None):
    """Align romaji tokens. If word_intervals given (list of (start_s, end_s)
    per token), align per-interval. Else use window_ranges or full song.
    """
    model = _mms()
    DICT = bundle.get_dict()
    sr = bundle.sample_rate
    data, fsr = sf.read(str(vocals_path))
    if data.ndim > 1: data = data.mean(axis=1)
    wav_full = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    wav16_full = AF.resample(wav_full, fsr, sr)
    full_len = wav16_full.shape[1]
    if word_intervals is not None:
        # Trust given per-token intervals (kalo reference lyrics + word times dari UG)
        return [(s, e) for s, e in word_intervals]
    if window_ranges is None:
        window_ranges = [(0.0, full_len / sr)]
    times = [None] * len(romaji_list)
    # Split tokens by window (sequential ranges, no overlap)
    win_idx = 0
    win_start, win_end = window_ranges[0]
    win_end_sample = int(win_end * sr)
    for ti, rom in enumerate(romaji_list):
        # Find which window this token belongs to (token = slice of romaji_list)
        # windows correspond to sequential chunks of token list
        # for simplicity: just align full song per-window with correct time offset
        pass
    n = len(romaji_list)
    if not n: return times
    # Distribute tokens roughly proportional to window duration
    total_win = sum(we - ws for ws, we in window_ranges)
    if total_win <= 0: return times
    n_per_win = []
    remaining = n
    for i, (ws, we) in enumerate(window_ranges):
        if i == len(window_ranges) - 1:
            n_per_win.append(remaining)
        else:
            share = max(1, int(round(n * (we - ws) / total_win)))
            share = min(share, remaining - (len(window_ranges) - 1 - i))
            n_per_win.append(share)
            remaining -= share
    ti = 0
    for wi, (ws, we) in enumerate(window_ranges):
        take = n_per_win[wi]
        ws_s = int(ws * sr); we_s = min(int(we * sr), full_len)
        if we_s <= ws_s:
            ti += take; continue
        chunk = wav16_full[:, ws_s:we_s]
        win_roms = romaji_list[ti:ti+take]
        if not win_roms:
            ti += take; continue
        tokens: list[int] = []
        spans: list[tuple[int, int, int] | None] = []
        for idx, rom in enumerate(win_roms):
            if not isinstance(rom, str):
                spans.append(None); continue
            clean = "".join(c for c in rom.lower() if c in DICT)
            if not clean:
                spans.append(None); continue
            st = len(tokens)
            for ch in clean:
                tokens.append(DICT[ch])
            spans.append((st, len(tokens), idx))
        if not tokens:
            log.debug(f"    win {wi}: all {len(win_roms)} tokens filtered (no DICT match)")
            ti += take; continue
        with torch.inference_mode():
            chunk_dev = chunk.to(_model_device())
            emission, _ = model(chunk_dev)
        ratio = chunk.shape[1] / emission.shape[1]
        targets = torch.tensor([tokens], dtype=torch.int32, device=_model_device())
        try:
            aligned, scores = _ta_forced_align(emission, targets, blank=0)
        except Exception as e:
            log.warning(f"    align window {wi} fail: {e}")
            ti += take; continue
        tspans = _ta_merge_tokens(aligned[0], scores[0].exp())
        nonblank = [ts for ts in tspans if ts.token != 0]
        offset = ws  # absolute time in song
        lti = 0
        for sp in spans:
            if sp is None: continue
            st, en, idx = sp
            wt = nonblank[lti:lti+(en-st)]
            if wt:
                times[ti + idx] = (wt[0].start*ratio/sr + offset, wt[-1].end*ratio/sr + offset)
            lti += (en - st)
        ti += take
        # Drop refs to big tensors so multi-window songs don't accumulate memory
        del emission, targets, aligned, scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            gc.collect()
    return times


def interpolate_missing(times, total):
    n = len(times); out = list(times)
    known = [(i, t) for i, t in enumerate(times) if t is not None]
    if not known:
        per = total / max(n, 1)
        return [(i*per, (i+1)*per) for i in range(n)]
    fi, (fs, fe) = known[0]
    for i in range(fi): out[i] = (fs, fs)
    for k in range(len(known) - 1):
        ai, (a_s, a_e) = known[k]; bi, (b_s, _) = known[k+1]
        gap = bi - ai
        if gap > 1:
            for j in range(1, gap):
                frac = j / gap; t = a_e + (b_s - a_e) * frac
                out[ai + j] = (t, t)
    li, (ls, le) = known[-1]
    for i in range(li+1, n): out[i] = (le, le)
    return out


# ============ Step 6: BTC consolidation + simplification ============
def simplify_chord(c):
    if c == "N": return "N"
    if ":" not in c: return c
    root, mod = c.split(":", 1)
    t = {"maj":"","min":"m","maj7":"maj7","min7":"m7","7":"7",
         "maj6":"6","min6":"m6","sus2":"sus2","sus4":"sus4",
         "dim":"dim","aug":"aug","hdim7":"m7b5","minmaj7":"mM7",
         "maj9":"maj7","min9":"m7","9":"7"}
    return root + t.get(mod, "")

def root_of(c):
    if c == "N": return "N"
    m = re.match(r"^([A-G][#b]?)", c); return m.group(1) if m else c

def consolidate_btc(btc_segs, min_dur=0.8):
    simp = [(s, e, simplify_chord(c)) for s, e, c in btc_segs]
    merged = []
    for s, e, c in simp:
        if merged and root_of(merged[-1][2]) == root_of(c):
            ps, pe, pc = merged[-1]
            merged[-1] = (ps, e, pc if len(pc) <= len(c) else c)
        else:
            merged.append((s, e, c))
    result = []
    for s, e, c in merged:
        if (e - s) < min_dur and result:
            ps, pe, pc = result[-1]
            result[-1] = (ps, e, pc)
        else: result.append((s, e, c))
    final = []
    for s, e, c in result:
        if final and root_of(final[-1][2]) == root_of(c):
            final[-1] = (final[-1][0], e, final[-1][2])
        else: final.append((s, e, c))
    return final


# ============ Section detection ============
def _estimate_bpm(wav_path):
    """Best-effort BPM via librosa. Returns int BPM, or None on any failure."""
    try:
        import librosa
        y, sr = librosa.load(str(wav_path), sr=None, mono=True, duration=60)
        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        if bpm is None:
            return None
        return int(round(float(bpm)))
    except Exception as e:
        log.info(f"  BPM estimation skipped: {e}")
        return None


def detect_sections(words_lines, btc, duration, gap_threshold=2.5):
    """Auto-detect intro, verse, chorus, outro from vocal gaps + position."""
    if not words_lines: return [{"name": "Intro", "start": 0.0, "end": duration, "has_lyrics": False}]
    first_vocal = words_lines[0]["start"]
    last_vocal = words_lines[-1]["end"]
    # Find vocal gaps
    gaps = []
    for i in range(len(words_lines) - 1):
        ge = words_lines[i]["end"]
        gs = words_lines[i+1]["start"]
        if gs - ge > gap_threshold:
            gaps.append((ge, gs))
    sections = []
    if first_vocal > 3.0:
        sections.append({"name": "Intro", "start": 0.0, "end": float(first_vocal), "has_lyrics": False})
    # naive: split by gaps into segments
    if not gaps:
        sections.append({"name": "Verse", "start": float(first_vocal), "end": float(last_vocal), "has_lyrics": True})
    else:
        prev_e = first_vocal
        for idx, (ge, gs) in enumerate(gaps):
            if ge > prev_e:
                # Alternating V/C pattern starting from Verse 1.
                #   idx=0 → Verse 1, idx=1 → Chorus 1, idx=2 → Verse 2, idx=3 → Chorus 2, …
                if idx % 2 == 0:
                    name = f"Verse {(idx // 2) + 1}"
                else:
                    name = f"Chorus {(idx // 2) + 1}"
                sections.append({"name": name, "start": float(prev_e), "end": float(ge), "has_lyrics": True})
            sections.append({"name": "Interlude", "start": float(ge), "end": float(gs), "has_lyrics": False})
            prev_e = gs
        if prev_e < last_vocal:
            # Trailing segment: pattern length is len(gaps)+1, so it's a Verse when
            # len(gaps) is even, Chorus when len(gaps) is odd.
            if len(gaps) % 2 == 0:
                name = f"Verse {(len(gaps) // 2) + 1}"
            else:
                name = f"Chorus {(len(gaps) // 2) + 1}"
            sections.append({"name": name, "start": float(prev_e), "end": float(last_vocal), "has_lyrics": True})
    if duration - last_vocal > 3.0:
        sections.append({"name": "Outro", "start": float(last_vocal), "end": float(duration), "has_lyrics": False})
    return sections


# ============ Main ============
def _update_job(job_id, **fields) -> None:
    """Update Job row when running under the FastAPI app. No-op if job_id is None
    (script mode) or the app's db is unavailable. Failures are swallowed — we
    never want a job-update error to mask a real pipeline failure."""
    if not job_id:
        return
    try:
        from app.db import db_session
        from app.models import Job
    except Exception as e:
        log.debug(f"  _update_job: app import failed ({e})")
        return
    try:
        with db_session() as db:
            job = db.get(Job, job_id)
            if not job:
                return
            for k, v in fields.items():
                setattr(job, k, v)
            db.commit()
    except Exception as e:
        log.warning(f"  _update_job({job_id}, {fields.keys()}) failed: {e}")


def _parse_artist_title(info: dict) -> tuple[str, str, str | None]:
    """Pull (artist, title, thumbnail) out of yt-dlp info dict. Best-effort.
    Mirrors the logic in `app/metadata._parse_artist_title`."""
    artist = info.get("artist") or info.get("creator") or info.get("uploader") or info.get("channel") or ""
    track = info.get("track") or info.get("title") or ""
    # "Artist - Title" pattern in title
    if track and " - " in track and not info.get("track"):
        parts = track.split(" - ", 1)
        if artist and parts[0].strip().lower() == str(artist).strip().lower():
            track = parts[1].strip()
        elif not info.get("artist"):
            artist, track = parts[0].strip(), parts[1].strip()
    return (str(artist).strip() if artist else "",
            str(track).strip() if track else "",
            info.get("thumbnail"))


def main(url: str, reference_lyrics: str = None, save: bool = True, job_id: str = None):
    log.info(f"=== Processing {url} ===")
    out_dir = Path(__file__).resolve().parent.parent / "data" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    _update_job(job_id, status="downloading", progress=5, message="Nge-tap metadata…")

    # 1. Download
    log.info("1/8 Download audio")
    wav, info = download_audio(url, out_dir)
    yt_id = wav.stem
    duration = float(sf.info(str(wav)).duration)
    artist, title, thumbnail_url = _parse_artist_title(info)
    log.info(f"  → {wav.name}, {duration:.0f}s, artist={artist!r}, title={title!r}")
    _update_job(job_id, status="downloading", progress=15, message="Audio downloaded ✓")

    # 2. BTC
    log.info("2/8 BTC-ISMIR19 chord detection")
    _update_job(job_id, status="detecting", progress=25, message="Nge-detect chord…")
    btc_lab = btc_detect(wav, out_dir)
    btc = consolidate_btc(parse_btc_lab(btc_lab), min_dur=0.8)
    log.info(f"  → {len(btc)} consolidated chord segments")
    _update_job(job_id, status="detecting", progress=45, message=f"Chord detected ✓ ({len(btc)} segs)")

    # 3. Whisper transcription
    log.info("3/8 Whisper transcription")
    _update_job(job_id, status="transcribing", progress=55, message="Nyalin lirik…")
    lyrics_lines, lang = transcribe_whisper(wav, initial_prompt=reference_lyrics)
    flat = [w for ln in lyrics_lines for w in ln["words"]]
    log.info(f"  → {len(lyrics_lines)} segments, {len(flat)} words, lang={lang}")
    _update_job(job_id, status="transcribing", progress=70, message=f"Lirik transcribed ✓ ({len(flat)} kata)")

    # 4. Reference lyrics (optional)
    if reference_lyrics:
        log.info("4/8 Using provided reference lyrics")
        # tokenize into lines using newline + romanize
        ref_lines = [l.strip() for l in reference_lyrics.split("\n") if l.strip()]
        ref_romaji_lines = [romaji(l) for l in ref_lines]
    else:
        # Try fetch from URL
        log.info("4/8 Try fetch reference lyrics (skip for now)")
        ref_lines = None
        ref_romaji_lines = None

    # 5. MMS_FA forced alignment (real per-word timing) — per-window per whisper seg
    log.info("5/8 Forced alignment (CPU, ~60s for 4min song)")
    _update_job(job_id, status="aligning", progress=80, message="Nge-align kata ↔ audio…")
    # Build a flat list[str] of romaji tokens regardless of source:
    #   - ref_romaji_lines path: list[list[str]]     → flatten one level
    #   - whisper path:         list[str] (kanji/kana) → run each through romaji() to
    #     get romaji chars, then flatten. Skipping this is the historical bug — DICT
    #     is a-z + '-' only, so raw kanji words produce 0/117 alignment.
    if ref_romaji_lines:
        use_lyrics = ref_romaji_lines
    else:
        use_lyrics = [w["word"] for w in flat]
    romaji_list: list[str] = []
    if use_lyrics and isinstance(use_lyrics[0], list):
        for line in use_lyrics:
            if isinstance(line, list):
                romaji_list.extend(r for r in line if isinstance(r, str) and r)
            elif isinstance(line, str) and line:
                romaji_list.extend(r for r in romaji(line) if r)
    elif use_lyrics:
        for word in use_lyrics:
            if not isinstance(word, str) or not word.strip():
                continue
            try:
                rs = romaji(word)
            except Exception as e:
                log.debug(f"    romaji({word!r}) failed: {e}")
                continue
            romaji_list.extend(rs)
    # Filter out any leftover empty strings just in case
    romaji_list = [r for r in romaji_list if r]
    log.info(f"  → {len(romaji_list)} romaji tokens prepared for alignment")
    # Group tokens by whisper segments → window_ranges
    if ref_lines:
        # map ref tokens to whisper segments proportionally
        n_total = len(romaji_list)
        n_win = max(1, len(lyrics_lines))
        win_ranges = []
        for i, ln in enumerate(lyrics_lines):
            ws = ln["start"]; we = ln["end"]
            # extend windows sedikit to avoid cut
            win_ranges.append((max(0, ws-0.5), min(duration, we+0.5)))
    else:
        win_ranges = [(max(0, ln["start"]-0.5), min(duration, ln["end"]+0.5))
                      for ln in lyrics_lines]
    # Merge overlapping windows
    merged = []
    for ws, we in sorted(win_ranges):
        if merged and ws <= merged[-1][1] + 0.1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], we))
        else:
            merged.append((ws, we))
    # Cap each window to 25s (MMS_FA optimal range)
    final_ranges = []
    for ws, we in merged:
        while we - ws > 25:
            final_ranges.append((ws, ws + 25))
            ws = ws + 25
        final_ranges.append((ws, we))
    log.info(f"  {len(final_ranges)} alignment windows (max 25s each)")
    raw_times = forced_align(wav, romaji_list, window_ranges=final_ranges)
    aligned = sum(1 for t in raw_times if t is not None)
    log.info(f"  → aligned {aligned}/{len(romaji_list)} tokens")
    token_times = interpolate_missing(raw_times, duration)

    # 6. Build lines with real timing + multi-chord
    log.info("6/8 Build lines with BTC chord anchoring")
    if ref_lines:
        # Use ref_lines as source of truth for text
        ti = 0
        lines_out = []
        for li, line_text in enumerate(ref_lines):
            rws = ref_romaji_lines[li]
            n = len(rws)
            tt = token_times[ti:ti+n]
            if not tt or tt[0] is None: continue
            line_start = tt[0][0]; line_end = max(tt[-1][1], line_start + 0.3)
            words_out = [{"word": rws[j], "start": tt[j][0], "end": tt[j][1],
                          "romaji": rws[j]} for j in range(n) if tt[j] is not None]
            ti += n
            # chord anchor
            line_chords = []
            for cs, ce, cc in btc:
                if line_start <= cs < line_end:
                    sc = simplify_chord(cc)
                    if sc == "N": continue
                    bi, bd = 0, 1e9
                    for k, w in enumerate(words_out):
                        d = abs((w["start"]+w["end"])/2 - cs)
                        if d < bd: bd, bi = d, k
                    line_chords.append({"chord": sc, "start": float(cs), "anchor_word_index": bi})
            lines_out.append({"line_index": len(lines_out), "start": line_start, "end": line_end,
                              "text": line_text, "words": words_out, "chords": line_chords})
    else:
        # Use whisper lines
        ti = 0
        lines_out = []
        for ln in lyrics_lines:
            n = len(ln["words"])
            tt = token_times[ti:ti+n]
            ti += n
            if not tt or tt[0] is None: continue
            line_start = tt[0][0]; line_end = max(tt[-1][1], line_start + 0.3)
            words_out = [{"word": ln["words"][j]["word"], "start": tt[j][0], "end": tt[j][1],
                          "romaji": romaji(ln["words"][j]["word"])[0] if romaji(ln["words"][j]["word"]) else ""}
                         for j in range(n) if tt[j] is not None]
            line_chords = []
            for cs, ce, cc in btc:
                if line_start <= cs < line_end:
                    sc = simplify_chord(cc)
                    if sc == "N": continue
                    bi, bd = 0, 1e9
                    for k, w in enumerate(words_out):
                        d = abs((w["start"]+w["end"])/2 - cs)
                        if d < bd: bd, bi = d, k
                    line_chords.append({"chord": sc, "start": float(cs), "anchor_word_index": bi})
            lines_out.append({"line_index": len(lines_out), "start": line_start, "end": line_end,
                              "text": ln["text"], "words": words_out, "chords": line_chords})

    first_vocal = lines_out[0]["start"] if lines_out else 0.0
    last_vocal = lines_out[-1]["end"] if lines_out else duration

    # 7. Sections + bars
    log.info("7/8 Detect sections + build bar grid")
    sections = detect_sections(lines_out, btc, duration)
    bars = []
    for cs, ce, cc in btc:
        sc = simplify_chord(cc)
        if sc == "N": continue
        if cs < first_vocal - 0.3 or cs >= last_vocal + 0.3: continue
        if ce - cs < 0.3: continue
        if bars and root_of(bars[-1]["chords"][0]["chord"]) == root_of(sc):
            bars[-1]["end"] = float(ce); bars[-1]["chords"][0]["end"] = float(ce)
        else:
            bars.append({"index": len(bars), "start": float(cs), "end": float(ce),
                         "chords": [{"chord": sc, "start": float(cs), "end": float(ce)}]})

    # Beats from BTC midpoints
    beats_set = set()
    for cs, ce, _ in btc:
        beats_set.add(round(cs * 4) / 4); beats_set.add(round(ce * 4) / 4)
    beats = sorted(b for b in beats_set if 0 < b < duration)
    downbeats = beats[::4] if beats else []

    # 8. Save
    log.info("8/8 Save to DB")
    _update_job(job_id, status="saving", progress=95, message="Nge-save ke DB…")
    bpm = _estimate_bpm(wav)
    # BPM is always an integer in display — never trust raw float from librosa.
    if bpm is not None:
        bpm = int(round(bpm))
    # Build render_json with the SAME meta as the Song row, so /api/songs/{id}
    # response stays self-consistent.
    render = {
        "meta": {
            "youtube_id":   yt_id,
            "artist":       artist or "Unknown",
            "title":        title or yt_id,
            "duration_sec": int(duration),
            "bpm":          bpm,
            "key":          "C major",
            "capo":         0,
            "time_sig":     "4/4",
            "language":     lang,
        },
        "beats":     beats,
        "downbeats": downbeats,
        "sections":  sections,
        "bars":      bars,
        "lines":     lines_out,
    }
    if save:
        # Imports live inside the function so the module is importable without
        # the FastAPI app context (e.g. for unit tests / CLI runs).
        from app.db import db_session
        from app.models import Song
        from app.cache import normalize_artist, normalize_title
        with db_session() as db:
            existing = db.query(Song).filter(Song.youtube_id == yt_id).one_or_none()
            if existing:
                existing.artist       = artist or existing.artist
                existing.title        = title or existing.title
                existing.artist_norm  = normalize_artist(artist) if artist else existing.artist_norm
                existing.title_norm   = normalize_title(title) if title else existing.title_norm
                existing.duration_sec = int(duration)
                existing.bpm          = bpm
                existing.language     = lang or existing.language
                existing.render_json  = json.dumps(render, ensure_ascii=False)
                existing.audio_path   = str(wav)
                db.commit()
                song_id = existing.id
                log.info(f"  → updated song id={song_id}")
            else:
                s = Song(
                    youtube_id=yt_id,
                    artist=artist or "Unknown",
                    title=title or yt_id,
                    artist_norm=normalize_artist(artist),
                    title_norm=normalize_title(title),
                    duration_sec=int(duration),
                    bpm=bpm,
                    music_key="C major",
                    capo=0,
                    time_sig="4/4",
                    language=lang,
                    thumbnail_url=thumbnail_url,
                    status="ready",
                    source="ai",
                    render_json=json.dumps(render, ensure_ascii=False),
                    audio_path=str(wav),
                )
                db.add(s); db.commit()
                song_id = s.id
                log.info(f"  → saved song id={song_id}")
        _update_job(job_id, status="done", progress=100, message="Disimpen ✓",
                    song_id=song_id)
    else:
        song_id = None
    log.info(f"=== DONE: {len(sections)} sections, {len(lines_out)} lines, {len(bars)} bars ===")
    return render


if __name__ == "__main__":
    # Standalone CLI mode: own the root logger so logs go to stderr at INFO.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=mG7lrRdm71A"
    ref_path = sys.argv[2] if len(sys.argv) > 2 else None
    ref = Path(ref_path).read_text(encoding="utf-8") if ref_path else None
    main(url, reference_lyrics=ref)
