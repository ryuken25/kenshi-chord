"""SMART fix for song #1: parse full lyrics (semua section), generate romaji,
anchor chord dari lirik, word-level timing via Whisper + initial_prompt.

Run: python smart_fix.py
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
import librosa
import torch
import faster_whisper
import pykakasi
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smart")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.db import db_session, init_db
from app.models import Song

init_db()

# === Full lyrics dengan section markers (dari user-paste) ===
# Format: "SECTION_NAME": [(chord_or_None, line_text), ...]
# Chord di-embed inline di line text sebagai token pertama; di-parse terpisah.

LYRICS = {
    "Verse 1": [
        (None,  "肩を濡らす雨粒で"),
        ("G",   "傘を忘れたことも忘れてた"),
    ],
    "Pre-Chorus": [
        (None,  "Ah 眩しさは虚しさ照らして"),
        ("G",   "寂しさは悔しさ溶かして"),
        (None,  "君と出会ったあの日も"),
        ("G",   "2人してずぶ濡れの日も"),
        (None,  "僕が知ってる"),
        (None,  "雨はこんなに"),
        (None,  "冷たかっただろうか"),
        (None,  "今更だけど"),
    ],
    "Chorus 1": [
        (None,    "降り出す"),
        (None,    "rain"),
        (None,    "何もかもを流して"),
        (None,    "どうしてあの時"),
        (None,    "気づけなかった"),
        ("Am",   "君のいない朝"),
        (None,    "無神経に晴れわたる空"),
        (None,    "止まない"),
        (None,    "rain"),
        (None,    "駆け出した"),
        (None,    "空の下"),
        (None,    "何もかも忘れて君との日を"),
        (None,    "胸にしまって歩き出す今"),
        (None,    "そこにいてくれて"),
        (None,    "ありがとう"),
    ],
    "Verse 2": [
        (None,  "失って気づく安らいだ日々たちは"),
        ("C",   "どこか遠くへと"),
        (None,  "そう"),
        (None,  "雨"),
        (None,  "が今"),
        (None,  "君が好きだと言った"),
        (None,  "雨"),
        (None,  "が今降る"),
        (None,  "ほら"),
    ],
    "Chorus 2": [
        (None,  "降り出す"),
        (None,  "rain"),
        (None,  "何もかもを流して"),
        (None,  "溢れ出す"),
        (None,  "涙を隠して"),
        (None,  "また"),
        (None,  "誰もいない部屋"),
        (None,  "雨音だけが"),
        (None,  "僕を包む"),
        (None,  "oh"),
    ],
    "Outro": [
        (None,    "輝く"),
        (None,    "rain"),
        (None,    "駆け出した"),
        (None,    "空の下"),
        (None,    "君という"),
        (None,    "rain"),
        (None,    "浴びて蘇る日々"),
        (None,    "君のいない世界歩き出す今"),
        ("Am",   "出逢ってくれて"),
    ],
}

# Section durations estimasi (seconds, total 212)
SECTION_DURATIONS = {
    "Verse 1":     24,
    "Pre-Chorus":  32,
    "Chorus 1":    52,
    "Verse 2":     28,
    "Chorus 2":    38,
    "Outro":       38,
}

# === Kanji → romaji via pykakasi ===
KAKASI = pykakasi.kakasi()
def to_romaji(text: str) -> str:
    """Convert Japanese (kanji+kana) → romaji (hepburn)."""
    if not text or not text.strip():
        return ""
    parts = KAKASI.convert(text)
    return "".join(p.get("hepburn", "") for p in parts)


# === Whisper: word-level timing with initial_prompt ===
def whisper_words(audio_path: Path, lyrics_text: str):
    """Transcribe pakai faster-whisper dgn initial_prompt = known lyrics.
    Returns: list of {word, start, end} sorted by start.
    """
    log.info(f"  Loading Whisper...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = faster_whisper.WhisperModel("base", device=device, compute_type="int8")
    log.info(f"  Transcribing (with initial_prompt)...")
    segs, info = model.transcribe(
        str(audio_path),
        beam_size=5, vad_filter=True, word_timestamps=True,
        language="ja",
        initial_prompt=lyrics_text,  # bias ke lirik yang kita punya
    )
    words = []
    for s in segs:
        if not s.words:
            continue
        for w in s.words:
            if w.start is None or w.end is None: continue
            words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
    log.info(f"  Got {len(words)} words from whisper (lang={info.language}, prob={info.language_probability:.2f})")
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
        log.info(f"Song: {song.title}")
        log.info(f"Audio: {wav_path.name} ({wav_path.stat().st_size/1e6:.1f} MB, {song.duration_sec}s)")

        # === Step 1: Flatten hardcoded lyrics → text prompt + token list ===
        all_lines = []  # list of (section_name, chord_at_start, line_text)
        for sec_name, lines in LYRICS.items():
            for chord, text in lines:
                all_lines.append((sec_name, chord, text))

        # Build plain text for initial_prompt
        prompt_text = " ".join(t for _, _, t in all_lines)
        log.info(f"  Built prompt: {len(prompt_text)} chars from {len(all_lines)} lines")

        # === Step 2: Whisper for word-level timing ===
        whisper_out = whisper_words(wav_path, prompt_text)

        # === Step 3: Map whisper words → hardcoded lines (in order) ===
        # Whisper returns ~word count, our lines have multiple words. Simple matching:
        # iterate all words, count total, then split by expected word counts per line.
        all_hardcoded_words = []  # list of (sec_name, chord_at_start, line_text, word_idx, word, romaji)
        for sec_name, chord, text in all_lines:
            words = text.split()
            for wi, w in enumerate(words):
                all_hardcoded_words.append((sec_name, chord, text, wi, w, to_romaji(w)))

        # Align whisper words to hardcoded words (in order)
        # Whisper may have slight differences in tokenization; we use start times
        # of whisper words and split proportionally.
        n_hard = len(all_hardcoded_words)
        n_whisper = len(whisper_out)
        log.info(f"  Hardcoded: {n_hard} words | Whisper: {n_whisper} words")

        # Strategy: each hardcoded word gets time = avg position in whisper
        def interp_time(i, n_hard, n_whisper, w_words):
            if n_whisper == 0 or n_hard == 0:
                return 0.0, 0.1
            # Map hardcoded word index → whisper word index (proportional)
            pos = (i / max(n_hard - 1, 1)) * (n_whisper - 1)
            idx = int(pos)
            # Start time from this whisper word; end from next
            if idx >= n_whisper - 1:
                start = w_words[idx]["start"]
                end = w_words[idx]["end"]
            else:
                start = w_words[idx]["start"]
                # Interpolate to next
                frac = pos - idx
                end = start + frac * (w_words[idx + 1]["start"] - start) + (w_words[idx + 1]["end"] - w_words[idx + 1]["start"])
            return start, max(end, start + 0.05)

        # Build render_json
        sections = []
        lines_out = []
        current_section = None
        current_section_start = 0.0
        line_idx = 0
        cumulative_word_idx = 0
        chord_segments = []  # (start, end, chord_name)

        # Section starts cumulative
        running_time = 0.0
        for sec_name, lines in LYRICS.items():
            sec_dur = SECTION_DURATIONS.get(sec_name, 30)
            sec_start = running_time
            sec_end = running_time + sec_dur
            sec_lines = []

            for chord, text in lines:
                words = text.split()
                if not words:
                    line_idx += 1; continue
                word_list = []
                for wi, w in enumerate(words):
                    cs, ce = interp_time(cumulative_word_idx, n_hard, n_whisper, whisper_out)
                    romaji = to_romaji(w)
                    word_list.append({
                        "word": w,
                        "start": float(cs),
                        "end": float(ce),
                        "romaji": romaji,
                    })
                    cumulative_word_idx += 1
                line_start = word_list[0]["start"]
                line_end = word_list[-1]["end"]
                # Anchor chord (kalau ada) ke word pertama
                line_chords = []
                if chord:
                    line_chords.append({
                        "chord": chord,
                        "start": float(line_start),
                        "anchor_word_index": 0,
                    })
                    chord_segments.append((line_start, line_end, chord))
                sec_lines.append({
                    "line_index": line_idx,
                    "start": float(line_start),
                    "end": float(line_end),
                    "text": text,
                    "words": word_list,
                    "chords": line_chords,
                })
                line_idx += 1

            # Close section
            sections.append({
                "name": sec_name, "start": float(sec_start), "end": float(sec_end),
                "has_lyrics": len(sec_lines) > 0,
            })
            lines_out.extend(sec_lines)
            running_time = sec_end

        log.info(f"  Built {len(sections)} sections, {len(lines_out)} lines")

        # === Step 4: Re-detect chord dari audio (untuk instrumental sections) ===
        # Skip — kita udah punya chord dari lyrics. Atau tambahin auto-detect
        # buat bagian yang gak ada chord marker.
        # Untuk sekarang, pakai chord dari lyrics aja.

        # === Step 5: Build bars (4 beats/bar) dari beat array ===
        old_render = json.loads(song.render_json)
        beats = old_render.get("beats", [])
        bars = []
        if beats:
            downbeats = beats[::4]
            for i, db_time in enumerate(downbeats[:-1]):
                b_start = float(db_time)
                b_end = float(downbeats[i+1]) if i+1 < len(downbeats) else float(song.duration_sec)
                # find chord aktif di bar ini
                active_chord = "N"
                for cs, ce, cn in chord_segments:
                    if cs <= b_start < ce:
                        active_chord = cn; break
                bars.append({
                    "index": i, "start": b_start, "end": b_end,
                    "chords": [{"chord": active_chord, "start": b_start, "end": b_end}]
                })

        # === Step 6: Build render_json ===
        new_render = {
            "meta": old_render["meta"],
            "beats": old_render.get("beats", []),
            "downbeats": old_render.get("downbeats", []),
            "sections": sections,
            "bars": bars,
            "lines": lines_out,
        }

        # === Step 7: Update DB ===
        song.render_json = json.dumps(new_render, ensure_ascii=False)
        db.commit()
        log.info(f"=== DONE: song #1 updated ===")
        log.info(f"  Sections: {len(sections)} ({', '.join(LYRICS.keys())})")
        log.info(f"  Lines: {len(lines_out)}")
        log.info(f"  Word count: {n_hard}")
        log.info(f"  Chord markers from lyrics: {len(chord_segments)}")
        log.info(f"  Bars: {len(bars)}")
        # Sample romaji
        if lines_out:
            sample = lines_out[0]
            log.info(f"  Sample line 0: \"{sample['text']}\" → romaji: \"{sample['words'][0]['romaji'] if sample['words'] else ''}\"")


if __name__ == "__main__":
    main()
