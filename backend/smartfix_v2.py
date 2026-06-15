"""SMART FIX v2: integrate BTC-ISMIR19 chord detection + proper section detection.

Workflow:
1. Read BTC-ISMIR19 .lab output → 147 chord segments
2. Read hardcoded lyrics with section structure (Verse 1, Pre-Chorus, Chorus 1, ...)
3. Detect intro from BTC (N at start) → add Intro section
4. Distribute timing proportional to word count per section
5. Use user's explicit chord markers (G, Am, C) at specific lines, else BTC chord
6. Generate romaji via pykakasi
7. Save to DB

Run: python smartfix_v2.py
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import librosa
import torch
import faster_whisper
import pykakasi
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smart2")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.db import db_session, init_db
from app.models import Song

init_db()

# === Lyrics with section structure (user-specified) ===
# Format: {section_name: [(inline_chord_or_None, line_text), ...]}
LYRICS = {
    "Verse 1": [
        (None, "肩を濡らす雨粒で"),
    ],
    "Pre-Chorus": [
        ("G",  "傘を忘れたことも忘れてた"),
        (None, "Ah"),
        (None, "眩しさは虚しさ照らして"),
        ("G",  "寂しさは悔しさ溶かして"),
        (None, "君と出会ったあの日も"),
        ("G",  "2人してずぶ濡れの日も"),
        (None, "僕が知ってる"),
        (None, "雨はこんなに"),
        (None, "冷たかっただろうか"),
        (None, "今更だけど"),
        (None, "降り出す"),
        (None, "rain"),
        (None, "何もかもを流して"),
        (None, "どうしてあの時"),
        (None, "気づけなかった"),
        ("Am", "君のいない朝"),
        (None, "無神経に晴れわたる空"),
    ],
    "Chorus 1": [
        (None, "止まない"),
        (None, "rain"),
        (None, "駆け出した"),
        (None, "空の下"),
        (None, "何もかも忘れて君との日を"),
        (None, "胸にしまって歩き出す今"),
        (None, "そこにいてくれて"),
        (None, "ありがとう"),
    ],
    "Verse 2": [
        (None, "失って気づく安らいだ日々たちは"),
        ("C",  "どこか遠くへと"),
        (None, "そう"),
        (None, "雨"),
        (None, "が今"),
        (None, "君が好きだと言った"),
        (None, "雨"),
        (None, "が今降る"),
        (None, "ほら"),
        (None, "降り出す"),
        (None, "rain"),
        (None, "何もかもを流して"),
        (None, "溢れ出す"),
        (None, "涙を隠して"),
    ],
    "Chorus 2": [
        (None, "また"),
        (None, "誰もいない部屋"),
        (None, "雨音だけが"),
        (None, "僕を包む"),
        (None, "oh"),
        (None, "輝く"),
        (None, "rain"),
        (None, "駆け出した"),
        (None, "空の下"),
        (None, "君という"),
        (None, "rain"),
        (None, "浴びて蘇る日々"),
        (None, "君のいない世界歩き出す今"),
    ],
    "Outro": [
        ("Am", "出逢ってくれて"),
    ],
}

# === Helpers ===
KAKASI = pykakasi.kakasi()
# Common Japanese particles + punctuation — split on these buat word boundary
PARTICLES = set("はがをにでとからまでのもやねよかなんです".split())
PUNCT = set("。、!?！？「」『』・…ー".split())

def to_romaji(text: str) -> str:
    """Segment Japanese (split on particles) → per-word pykakasi → joined with spaces.
    Hasil lebih readable dari concat langsung.
    """
    if not text or not text.strip():
        return ""
    # Manual segment: split on particles (preserve them as separate words)
    segments = []
    current = ""
    for ch in text:
        current += ch
        if ch in PARTICLES or ch in PUNCT:
            segments.append(current)
            current = ""
    if current:
        segments.append(current)
    # Run pykakasi per segment (concat small segments kalau partikel terlalu pendek)
    out_words = []
    for seg in segments:
        if not seg.strip(): continue
        # pykakasi per segment
        parts = KAKASI.convert(seg)
        romaji = "".join(p.get("hepburn", "") for p in parts)
        if romaji.strip():
            out_words.append(romaji)
    return " ".join(out_words)

def parse_btc_lab(lab_path: Path):
    """Parse BTC-ISMIR19 .lab file → list of (start, end, chord)."""
    log.info(f"Parsing BTC .lab: {lab_path}")
    segments = []
    with open(lab_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                start, end, chord = float(parts[0]), float(parts[1]), parts[2]
                segments.append((start, end, chord))
            except (ValueError, IndexError):
                continue
    log.info(f"  Got {len(segments)} BTC segments")
    return segments

def btc_chord_at(t: float, btc_segments, default="C") -> str:
    """Get BTC chord aktif di time t. Simplify extended chord ke basic."""
    for s, e, chord in btc_segments:
        if s <= t < e:
            return _simplify_chord(chord)
    return default

def _simplify_chord(btc_chord: str) -> str:
    """BTC output: 'A:min7', 'G:sus4', 'F:maj7', 'C', 'N'.
    Simplify ke bentuk basic yang frontend support (major/minor doang)."""
    if btc_chord == "N":
        return "N"
    # Pisahkan root:modifier (BTC pakai ':', frontend pakai 'm' untuk minor)
    if ":" in btc_chord:
        root, mod = btc_chord.split(":", 1)
        # Major dengan qualifier (maj7, maj6, sus2, sus4) → keep as major
        if mod.startswith("maj") or mod.startswith("sus"):
            return root
        # Minor → 'm' suffix
        if mod.startswith("min") or mod == "m":
            return root + "m"
        # 7th etc → keep root + "7"
        if "7" in mod:
            return root + "7"
        return root
    return btc_chord

def detect_intro_duration(btc_segments, audio_duration, min_intro=1.0):
    """Detect intro: first 'N' chord yang panjang di awal = intro."""
    intro_end = 0.0
    for s, e, chord in btc_segments:
        if chord == "N":
            intro_end = e
        else:
            break
    if intro_end < min_intro:
        intro_end = 0.0
    log.info(f"  Intro detected: {intro_end:.1f}s")
    return intro_end


# === Whisper for word-level timing (with initial_prompt) ===
def whisper_word_timings(audio_path, lyrics_text):
    """Use faster-whisper + initial_prompt for word-level timing."""
    log.info("Running faster-whisper with initial_prompt...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = faster_whisper.WhisperModel("base", device=device, compute_type="int8")
    segs, info = model.transcribe(
        str(audio_path), beam_size=5, vad_filter=True, word_timestamps=True,
        language="ja", initial_prompt=lyrics_text,
    )
    words = []
    for s in segs:
        if not s.words: continue
        for w in s.words:
            if w.start is None or w.end is None: continue
            words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
    log.info(f"  Whisper: {len(words)} words")
    del model
    if device == "cuda": torch.cuda.empty_cache()
    return words


def main():
    with db_session() as db:
        song = db.get(Song, 1)
        if not song:
            log.error("Song #1 not found"); return
        wav_path = Path(song.audio_path) if song.audio_path else None
        if not wav_path or not wav_path.exists():
            wav_path = Path(__file__).resolve().parent.parent / "data" / "audio" / "mG7lrRdm71A.wav"
        if not wav_path.exists():
            log.error(f"Audio not found: {wav_path}"); return
        song.audio_path = str(wav_path)
        old_render = json.loads(song.render_json)
        duration = song.duration_sec
        log.info(f"Song: {song.title} | {duration}s | {wav_path.name}")

        # === Step 1: Read BTC output ===
        # BTC saves to /tmp/btc_out by default. Try multiple locations.
        btc_lab = None
        for cand in [
            Path("/tmp/btc_out/rain.lab"),
            Path("C:/Users/lenov/AppData/Local/Temp/btc_out/rain.lab"),
            Path(__file__).resolve().parent / "BTC-ISMIR19" / "test" / "rain.lab",
            Path(__file__).resolve().parent / "btc_out" / "rain.lab",
        ]:
            if cand.exists():
                btc_lab = cand
                break
        if btc_lab is None:
            log.error("BTC .lab not found. Run BTC-ISMIR19 first:")
            log.error("  cd BTC-ISMIR19 && python test.py --audio_dir /tmp/btc_audio --save_dir /tmp/btc_out")
            return
        btc_segments = parse_btc_lab(btc_lab)

        # === Step 2: Detect intro ===
        intro_end = detect_intro_duration(btc_segments, duration)
        song_duration = duration - intro_end
        if intro_end > 0:
            log.info(f"  Intro: 0 → {intro_end:.1f}s (will create 'Intro' section)")

        # === Step 3: Count total words + per section ===
        # Flatten lyrics to token list
        all_lines = []  # (sec_name, chord, text, word_count)
        for sec, lines in LYRICS.items():
            for chord, text in lines:
                wc = len(text.split()) if text.strip() else 0
                all_lines.append((sec, chord, text, wc))
        total_words = sum(w[3] for w in all_lines)
        if total_words == 0:
            log.error("No words in lyrics"); return
        log.info(f"  Total: {total_words} words across {len(all_lines)} lines")

        # === Step 4: Whisper for word-level timing ===
        # Build initial_prompt from lyrics text
        prompt_text = " ".join(w[2] for w in all_lines if w[2].strip())
        whisper_words = whisper_word_timings(wav_path, prompt_text)

        # === Step 5: Build sections + lines with timing ===
        # Distribute timing: sequential from intro_end
        # Use word count ratio
        section_starts = {}
        running_time = intro_end
        cumulative_words = 0

        # First pass: calculate section start times based on word count
        for sec_name in LYRICS.keys():
            section_starts[sec_name] = running_time
            sec_words = sum(w[3] for w in all_lines if w[0] == sec_name)
            sec_duration = (sec_words / total_words) * song_duration
            running_time += sec_duration
        log.info(f"  Section starts: {section_starts}")

        # Build render_json
        lines_out = []
        sections_out = []
        line_idx = 0
        cumulative_time = intro_end

        # Add Intro section
        if intro_end > 0:
            sections_out.append({
                "name": "Intro", "start": 0.0, "end": float(intro_end),
                "has_lyrics": False,
            })

        chord_segments_for_bars = []  # for building bar grid

        for sec_name, sec_lines in LYRICS.items():
            sec_start = section_starts[sec_name]
            sec_words = sum(1 for w in all_lines if w[0] == sec_name and w[2].strip())
            sec_dur = (sec_words / total_words) * song_duration if total_words else 30
            sec_end = sec_start + sec_dur

            # Section header
            sections_out.append({
                "name": sec_name, "start": float(sec_start), "end": float(sec_end),
                "has_lyrics": True,
            })

            for chord_in, text in sec_lines:
                if not text.strip():
                    line_idx += 1
                    continue
                words = text.split()
                wc = len(words)
                if wc == 0:
                    line_idx += 1
                    continue
                # Distribute line duration within section
                line_dur = sec_dur / max(sec_words, 1)
                line_start = cumulative_time
                line_end = cumulative_time + line_dur
                cumulative_time = line_end

                # Word-level timing
                word_list = []
                for j, w in enumerate(words):
                    wd = line_dur / wc
                    word_list.append({
                        "word": w,
                        "start": float(line_start + j * wd),
                        "end": float(line_start + (j + 1) * wd),
                        "romaji": to_romaji(w),
                    })

                # Chord: prefer explicit user marker, else BTC at line_start
                if chord_in:
                    line_chord = chord_in
                else:
                    line_chord = btc_chord_at(line_start, btc_segments, default="C")
                line_chords = [{
                    "chord": line_chord,
                    "start": float(line_start),
                    "anchor_word_index": 0,
                }]
                chord_segments_for_bars.append((line_start, line_end, line_chord))

                lines_out.append({
                    "line_index": line_idx,
                    "start": float(line_start),
                    "end": float(line_end),
                    "text": text,
                    "words": word_list,
                    "chords": line_chords,
                })
                line_idx += 1

        # === Step 6: Build bar grid (4 beats/bar) from beat array ===
        beats = old_render.get("beats", [])
        if not beats:
            # Fallback: detect from audio
            log.info("  Detecting beats from audio...")
            y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
            _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        downbeats = beats[::4]
        bars = []
        for i, db_t in enumerate(downbeats[:-1]):
            b_start = float(db_t)
            b_end = float(downbeats[i+1]) if i+1 < len(downbeats) else float(duration)
            # Find dominant chord in this bar
            active = "N"
            for cs, ce, cn in chord_segments_for_bars:
                if cs <= b_start < ce:
                    active = cn; break
            if active == "N":
                # fallback: BTC
                active = btc_chord_at(b_start, btc_segments, default="C")
            bars.append({
                "index": i, "start": b_start, "end": b_end,
                "chords": [{"chord": active, "start": b_start, "end": b_end}]
            })

        # === Step 7: Build new render_json ===
        new_render = {
            "meta": old_render["meta"],
            "beats": beats,
            "downbeats": downbeats,
            "sections": sections_out,
            "bars": bars,
            "lines": lines_out,
        }

        # === Step 8: Save ===
        song.render_json = json.dumps(new_render, ensure_ascii=False)
        db.commit()
        log.info(f"=== DONE ===")
        log.info(f"  Sections: {[s['name'] for s in sections_out]}")
        log.info(f"  Lines: {len(lines_out)}")
        log.info(f"  Chord markers (G/Am/C): {sum(1 for w in all_lines if w[1])}")
        log.info(f"  BTC segments used: {len(btc_segments)}")
        log.info(f"  Bars: {len(bars)}")
        if lines_out:
            for i in [0, 5, 10, 15]:
                if i < len(lines_out):
                    l = lines_out[i]
                    log.info(f"  Line {i}: [{l['start']:.1f}s-{l['end']:.1f}s] \"{l['text']}\" (chord={l['chords'][0]['chord'] if l['chords'] else 'N'}, romaji[0]={l['words'][0]['romaji'] if l['words'] else ''})")


if __name__ == "__main__":
    main()
