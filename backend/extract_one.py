"""One-shot extraction: URL → audio → ML → render_json → DB.

Run: python extract_one.py <youtube_url>

Pipeline:
  1. yt-dlp download audio
  2. demucs source separation (vocals / other)
  3. librosa: BPM, beat tracking, key (Krumhansl-Schmuckler)
  4. chroma template matching: chord detection on mix non-vokal
  5. whisperx: lyric transcription + word-level alignment on vocals
  6. alignment: snap chord ke beat, anchor ke word timing
  7. save to DB
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# === Config ===
URL = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=mG7lrRdm71A"
AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
USE_DEMUCS = False  # segfault di Windows + flash-attention. FASE 2: fix nanti, skip dulu.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract")


# === Step 1: yt-dlp download ===
log.info("=== Step 1: Download audio dari YouTube ===")
import yt_dlp

ydl_opts = {
    "quiet": True,
    "no_warnings": True,
    "format": "bestaudio/best",
    "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
    "noplaylist": True,
    "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(URL, download=True)
    youtube_id = info.get("id")
    title = info.get("title", "Untitled")
    uploader = info.get("uploader", "Unknown")
    duration = info.get("duration", 60)
    artist = info.get("artist") or info.get("creator") or uploader
log.info(f"  Got: {title} by {artist} ({duration}s)")

# Resolve wav path
for ext in ("wav", "m4a", "webm", "mp4", "opus"):
    cand = AUDIO_DIR / f"{youtube_id}.{ext}"
    if cand.exists():
        wav_path = cand
        break
else:
    raise RuntimeError("WAV not found after download")
log.info(f"  Audio: {wav_path} ({wav_path.stat().st_size/1e6:.1f} MB)")


# === Step 2: demucs source separation ===
vocals_path = None
other_path = None
if USE_DEMUCS:
    log.info("=== Step 2: Demucs source separation ===")
    t0 = time.time()
    import soundfile as sf
    import torchaudio
    from demucs.pretrained import get_model as demucs_get_model
    from demucs.apply import apply_model

    # Load model
    model = demucs_get_model("htdemucs")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model_name = getattr(model, "name", None) or type(model).__name__
    log.info(f"  Demucs model loaded: {model_name} on {next(model.parameters()).device}")

    # Load audio — htdemucs butuh stereo (2 channel)
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # mono → stereo
    wav = wav.to(next(model.parameters()).device)

    # Separate in chunks (segment=5s, shifts=0 untuk hemat VRAM di 8GB)
    try:
        with torch.no_grad():
            sources = apply_model(model, wav[None], segment=5.0, overlap=0.1,
                                  shifts=0, num_workers=0, progress=False)[0]
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "memory" in str(e).lower():
            log.warning(f"  GPU OOM ({e}), retrying on CPU...")
            torch.cuda.empty_cache()
            model.to("cpu")
            wav_cpu = wav.cpu()
            with torch.no_grad():
                sources = apply_model(model, wav_cpu[None], segment=5.0, overlap=0.1,
                                      shifts=0, num_workers=0, progress=False)[0]
            model.to("cuda")  # restore
        else:
            raise
    # sources: [num_sources, channels, samples] — order: drums, bass, other, vocals
    sr_out = model.samplerate
    sources_dict = {name: sources[i].cpu().numpy() for i, name in enumerate(model.sources)}
    log.info(f"  Sources: {list(sources_dict.keys())}, sr={sr_out}")

    vocals_path = AUDIO_DIR / f"{youtube_id}.vocals.wav"
    other_path = AUDIO_DIR / f"{youtube_id}.other.wav"
    bass_path = AUDIO_DIR / f"{youtube_id}.bass.wav"
    drums_path = AUDIO_DIR / f"{youtube_id}.drums.wav"

    # Save ke int16 PCM (langsung, biar numpy gak allocate float32 dulu yg 2x lebih gede)
    def save_wav(path, audio, sr):
        # audio: (channels, samples) float32 in [-1, 1]
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        sf.write(str(path), audio_int16.T, sr, subtype='PCM_16')

    save_wav(vocals_path, sources_dict["vocals"], sr_out)
    save_wav(other_path, sources_dict["other"], sr_out)
    save_wav(bass_path, sources_dict["bass"], sr_out)
    save_wav(drums_path, sources_dict["drums"], sr_out)
    log.info(f"  Demucs done in {time.time()-t0:.1f}s")
    del model, sources
    torch.cuda.empty_cache()
else:
    log.info("=== Step 2: Demucs SKIPPED (using full mix) ===")
    vocals_path = wav_path
    other_path = wav_path


# === Step 3: librosa — BPM, beat tracking, key ===
log.info("=== Step 3: librosa BPM/beat/key ===")
import librosa
t0 = time.time()
y_full, sr = librosa.load(str(wav_path), sr=None, mono=True, duration=min(duration, 120))
tempo, beat_frames = librosa.beat.beat_track(y=y_full, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
log.info(f"  BPM: {bpm:.1f}  ({len(beat_times)} beats in first 120s)")

# Key detection (Krumhansl-Schmuckler)
CHROMA_PROFILE_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
CHROMA_PROFILE_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

chroma = librosa.feature.chroma_cqt(y=y_full, sr=sr)
chroma_mean = chroma.mean(axis=1)
chroma_corrs = []
for shift in range(12):
    rotated = np.roll(chroma_mean, -shift)
    major_corr = np.corrcoef(rotated, CHROMA_PROFILE_MAJOR)[0, 1]
    minor_corr = np.corrcoef(rotated, CHROMA_PROFILE_MINOR)[0, 1]
    chroma_corrs.append((major_corr, minor_corr))
best = max(range(12), key=lambda i: max(chroma_corrs[i]))
major_score, minor_score = chroma_corrs[best]
key_name = NOTE_NAMES[best]
is_minor = minor_score > major_score
music_key = f"{key_name} {'minor' if is_minor else 'major'}"
log.info(f"  Key: {music_key} (took {time.time()-t0:.1f}s)")


# === Step 4: Chord detection via chroma template matching ===
log.info("=== Step 4: Chord detection (chroma template matching) ===")
t0 = time.time()
y_other, sr_other = librosa.load(str(other_path), sr=22050, mono=True)
chroma_other = librosa.feature.chroma_cqt(y=y_other, sr=sr_other, hop_length=512)
chroma_times = librosa.frames_to_time(np.arange(chroma_other.shape[1]), sr=sr_other, hop_length=512)

# Chord templates (root + quality) — simple but works for major/minor
def make_chord_template(root, quality="maj"):
    template = np.zeros(12)
    template[root] = 1.0
    if quality == "maj":
        template[(root + 4) % 12] = 0.8   # major third
        template[(root + 7) % 12] = 0.7   # perfect fifth
    else:
        template[(root + 3) % 12] = 0.8   # minor third
        template[(root + 7) % 12] = 0.7   # perfect fifth
    return template

chord_templates = {}
for r in range(12):
    chord_templates[NOTE_NAMES[r]] = make_chord_template(r, "maj")
    chord_templates[NOTE_NAMES[r] + "m"] = make_chord_template(r, "min")

# Detect chord per frame
frame_chords = []
prev_chord = None
for i in range(chroma_other.shape[1]):
    frame = chroma_other[:, i]
    if frame.sum() < 0.01:
        frame_chords.append("N")
        continue
    # Find best match
    best_name, best_score = "N", -1
    for name, tmpl in chord_templates.items():
        score = np.dot(frame, tmpl) / (np.linalg.norm(frame) * np.linalg.norm(tmpl) + 1e-9)
        if score > best_score:
            best_score = score
            best_name = name
    frame_chords.append(best_name if best_score > 0.5 else "N")

# Merge consecutive same chords + min duration
chord_segments = []
N = len(frame_chords)
i = 0
while i < N:
    j = i
    while j < N and frame_chords[j] == frame_chords[i]:
        j += 1
    if j - i >= 4:  # min ~250ms at hop=512/sr=22050
        end_idx = min(j, N - 1)
        chord_segments.append((chroma_times[i], chroma_times[end_idx], frame_chords[i]))
    i = j if j > i else i + 1
log.info(f"  Chord segments: {len(chord_segments)} (took {time.time()-t0:.1f}s)")


# === Step 5: WhisperX/faster-whisper lyric transcription + alignment ===
log.info("=== Step 5: Lyric transcription (faster-whisper) ===")
t0 = time.time()
import faster_whisper
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model (use 'base' untuk dev — bisa upgrade ke 'small'/'medium' untuk akurasi)
whisper_model = faster_whisper.WhisperModel("base", device=device, compute_type="int8")
segments_iter, info = whisper_model.transcribe(
    str(vocals_path), beam_size=5, vad_filter=True, word_timestamps=True,
    language=None,  # auto-detect
)
log.info(f"  Whisper: lang={info.language} (prob={info.language_probability:.2f})")

# Collect segments + words
lines_data = []
for seg in segments_iter:
    words = []
    if seg.words:
        for w in seg.words:
            if w.start is None or w.end is None:
                continue
            words.append({
                "word": w.word.strip(),
                "start": float(w.start),
                "end": float(w.end),
            })
    if not words:
        # Fallback: 1 word = seluruh text
        words = [{"word": seg.text.strip(), "start": float(seg.start), "end": float(seg.end)}]
    if not words or not words[0]["word"]:
        continue
    text = " ".join(w["word"] for w in words)
    lines_data.append({
        "line_index": len(lines_data),
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "text": text,
        "words": words,
    })
log.info(f"  Lines: {len(lines_data)}, words: {sum(len(l['words']) for l in lines_data)} (whisper took {time.time()-t0:.1f}s)")
language = info.language if info.language else "en"
del whisper_model
if device == "cuda": torch.cuda.empty_cache()


# === Step 6: Alignment engine — snap chord to closest line, anchor ke word ===
log.info("=== Step 6: Alignment engine ===")
t0 = time.time()

def snap_to_beat(t, beats):
    """Return beat time closest to t."""
    if not beats: return t
    return min(beats, key=lambda b: abs(b - t))

# Build chords list per line
def find_line_at(t):
    for l in lines_data:
        if l["start"] <= t < l["end"]:
            return l
    return None

# For each chord segment, find line + anchor word
for seg_start, seg_end, chord in chord_segments:
    # Snap to nearest beat (in extended beat array)
    line = find_line_at(seg_start) or find_line_at(seg_end) or (lines_data[0] if lines_data else None)
    if not line:
        continue
    # Anchor to word whose center is closest to seg_start
    best_idx, best_dist = 0, float("inf")
    for i, w in enumerate(line["words"]):
        center = (w["start"] + w["end"]) / 2
        d = abs(center - seg_start)
        if d < best_dist:
            best_dist = d
            best_idx = i
    line.setdefault("chords", []).append({
        "chord": chord,
        "start": seg_start,
        "anchor_word_index": best_idx,
    })

log.info(f"  Alignment done in {time.time()-t0:.1f}s")

# Add empty chords list to lines that don't have any
for l in lines_data:
    l.setdefault("chords", [])


# === Step 7: Build render_json + save to DB ===
log.info("=== Step 7: Build render_json + save to DB ===")
# Detect language: pakai hasil whisper kalau ada, fallback heuristic
def _guess_language(text):
    text = (text or "").lower()
    if any("぀" <= c <= "ゟ" or "゠" <= c <= "ヿ" for c in text):
        return "ja"
    id_words = ["yang", "aku", "kamu", "cinta", "hati", "indonesia", "jakarta"]
    if any(w in text.split() for w in id_words):
        return "id"
    return "en"
language = language if language in ("ja", "id", "en") else _guess_language(title + " " + artist)

# Simple sections: just put everything in "Verse 1"
sections = [{"name": "Verse 1", "start": 0.0, "end": float(duration), "has_lyrics": len(lines_data) > 0}]

# Build bars from chord segments (group every 4 chord segments OR per beat)
bars = []
if beat_times:
    bar_idx = 0
    for i in range(0, len(beat_times), 4):
        if i + 4 > len(beat_times) and i > 0:
            break
        b_start = float(beat_times[i])
        b_end = float(beat_times[i + 4]) if i + 4 < len(beat_times) else float(duration)
        if b_end - b_start < 0.1: break
        # Find dominant chord in this bar
        bar_chords = [c for s, e, c in chord_segments if s >= b_start and s < b_end]
        chord = bar_chords[0] if bar_chords else "N"
        bars.append({"index": bar_idx, "start": b_start, "end": b_end,
                    "chords": [{"chord": chord, "start": b_start, "end": b_end}]})
        bar_idx += 1

render = {
    "meta": {
        "youtube_id": youtube_id,
        "artist": artist,
        "title": title,
        "duration_sec": duration,
        "bpm": bpm,
        "key": music_key,
        "capo": 0,
        "time_sig": "4/4",
        "language": language,
    },
    "beats": beat_times[:200],   # first 200 beats to keep it sane
    "downbeats": beat_times[::4][:50],
    "sections": sections,
    "bars": bars[:60],
    "lines": lines_data,
}

# Save to DB
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.db import db_session, init_db
from app.cache import normalize_artist, normalize_title
from app.models import Song

init_db()
with db_session() as db:
    existing = db.query(Song).filter(Song.youtube_id == youtube_id).one_or_none()
    if existing:
        log.info(f"  Updating existing song id={existing.id}")
        existing.render_json = json.dumps(render, ensure_ascii=False)
        existing.artist = artist
        existing.title = title
        existing.artist_norm = normalize_artist(artist)
        existing.title_norm = normalize_title(title)
        existing.duration_sec = duration
        existing.bpm = bpm
        existing.music_key = music_key
        existing.language = language
        existing.status = "ready"
        existing.source = "ai"
        db.commit()
        song_id = existing.id
    else:
        song = Song(
            youtube_id=youtube_id, artist=artist, title=title,
            artist_norm=normalize_artist(artist),
            title_norm=normalize_title(title),
            duration_sec=duration, bpm=bpm, music_key=music_key, capo=0,
            time_sig="4/4", language=language, status="ready", source="ai",
            render_json=json.dumps(render, ensure_ascii=False),
            audio_path=str(wav_path),
        )
        db.add(song); db.commit()
        song_id = song.id

log.info(f"=== DONE: song_id={song_id} ===")
log.info(f"  Artist: {artist}")
log.info(f"  Title: {title}")
log.info(f"  Key: {music_key}")
log.info(f"  BPM: {bpm:.1f}")
log.info(f"  Language: {language}")
log.info(f"  Duration: {duration}s")
log.info(f"  Beats: {len(beat_times)}")
log.info(f"  Bars: {len(bars)}")
log.info(f"  Chord segments: {len(chord_segments)}")
log.info(f"  Lines: {len(lines_data)}, words: {sum(len(l['words']) for l in lines_data)}")
