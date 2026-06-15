"""Final fix for song #1:
1. Set audio_path di DB (ketinggalan dari extract_one.py)
2. Fetch lirik dari URL baru (lyricfind) + hardcoded fallback (user-pasted)
3. Re-detect chord dengan filter lebih ketat (min 1.5s)
4. Distribute lyrics evenly across song
5. Update DB
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
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fix")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.db import db_session, init_db
from app.models import Song

init_db()

# === Hardcoded lyrics (user-provided ground truth) ===
HARDCODED_LYRICS = [
    "肩を濡らす雨粒で",
    "傘を忘れたことも忘れてた",
    "",
    "Ah 眩しさは虚しさ照らして",
    "寂しさは悔しさ溶かして",
    "君と出会ったあの日も",
    "2人してずぶ濡れの日も",
    "",
    "僕が知ってる",
    "雨はこんなに",
    "冷たかっただろうか",
    "今更だけど",
    "",
    "降り出す rain",
    "何もかもを流して",
    "どうしてあの時 気づけなかった",
    "君のいない朝 無神経に晴れ渡る空",
    "",
    "止まない rain",
    "駆け出した 空の下",
    "何もかも忘れて君との日を",
    "胸にしまって歩き出す今",
    "そこにいてくれて ありがとう",
    "",
    "失って気づく安らいだ日々たちは",
    "どこか遠くへと",
    "そう 雨 が今",
    "君が好きだと言った",
    " 雨 が今降る ほら",
    "",
    "降り出す rain",
    "何もかもを流して",
    "溢れ出す 涙を隠して また",
    "誰もいない部屋 雨音だけが",
    "僕を包む oh",
    "",
    "輝く rain",
    "駆け出した 空の下",
    "君という rain",
    "浴びて蘇る日々",
    "君のいない世界歩き出す今",
    "出逢ってくれて",
    "",
]

def fetch_from_lyricfind(url):
    """Try fetch dari lyricfind (umumnya punya lirik Jepang/Inggris/Indonesia)."""
    log.info(f"Trying: {url}")
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 KenshiChord/0.1"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # LyricFind biasanya pakai <div class="lyrics"> atau <p class="lyrics">
        container = (
            soup.find("div", class_=re.compile("lyric", re.I)) or
            soup.find("p", class_=re.compile("lyric", re.I)) or
            soup.find(id=re.compile("lyric", re.I))
        )
        if not container:
            # Fallback: cari semua <p> yang ada teks panjang
            ps = soup.find_all("p")
            for p in ps:
                t = p.get_text(separator="\n", strip=True)
                if any("぀" <= c <= "ゟ" or "゠" <= c <= "ヿ" for c in t) and len(t) > 50:
                    return [ln.strip() for ln in t.split("\n") if ln.strip()]
            return []
        text = container.get_text(separator="\n", strip=True)
        return [ln.strip() for ln in text.split("\n") if ln.strip()]
    except Exception as e:
        log.warning(f"  Failed: {e}")
        return []


def detect_chords_strict(audio_path, min_duration=1.5, sr_target=22050):
    """Re-detect chord dengan filter lebih ketat. Real song ~10-20 chord."""
    log.info(f"Re-detecting chords (min_duration={min_duration}s)")
    audio_path = Path(audio_path)
    if not audio_path.exists():
        log.error(f"  Audio not found: {audio_path}")
        return []
    y, sr = librosa.load(str(audio_path), sr=sr_target, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    N = chroma.shape[1]
    times = librosa.frames_to_time(np.arange(N), sr=sr, hop_length=512)

    NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    templates = {}
    for r in range(12):
        t_maj = np.zeros(12); t_maj[r] = 1; t_maj[(r+4)%12] = 0.8; t_maj[(r+7)%12] = 0.7
        t_min = np.zeros(12); t_min[r] = 1; t_min[(r+3)%12] = 0.8; t_min[(r+7)%12] = 0.7
        templates[NOTE[r]] = t_maj
        templates[NOTE[r]+"m"] = t_min

    frame_chords = []
    for i in range(N):
        frame = chroma[:, i]
        if frame.sum() < 0.01:
            frame_chords.append("N"); continue
        best, score = "N", -1
        for name, tmpl in templates.items():
            s = np.dot(frame, tmpl) / (np.linalg.norm(frame) * np.linalg.norm(tmpl) + 1e-9)
            if s > score: score, best = s, name
        frame_chords.append(best if score > 0.65 else "N")

    # Merge consecutive same chords
    raw_segs = []
    i = 0
    while i < N:
        j = i
        while j < N and frame_chords[j] == frame_chords[i]:
            j += 1
        raw_segs.append((times[i], times[min(j, N-1)], frame_chords[i]))
        i = j if j > i else i + 1

    # Filter: drop N, drop < min_duration
    filtered = [s for s in raw_segs if s[2] != "N" and (s[1] - s[0]) >= min_duration]
    log.info(f"  Raw: {len(raw_segs)} → Filtered: {len(filtered)}")
    return filtered


def make_lines(lyrics, duration):
    """Distribute lyric lines evenly across song duration, with rough word timing."""
    # Filter empty lines for display, but track positions for empty lines
    non_empty = [(i, ln) for i, ln in enumerate(lyrics) if ln.strip()]
    if not non_empty:
        return []

    n = len(non_empty)
    # 5s intro, 5s outro buffer
    usable = max(0, duration - 10)
    seg_dur = usable / n
    start_offset = 5.0

    lines = []
    for idx, (orig_idx, text) in enumerate(non_empty):
        start = start_offset + idx * seg_dur
        end = start_offset + (idx + 1) * seg_dur
        # Rough word timing
        words = text.split()
        if not words:
            continue
        wd = (end - start) / len(words)
        word_list = [{
            "word": w,
            "start": float(start + j * wd),
            "end": float(start + (j + 1) * wd),
        } for j, w in enumerate(words)]
        lines.append({
            "line_index": len(lines),
            "start": float(start),
            "end": float(end),
            "text": text,
            "words": word_list,
        })
    return lines


def main():
    with db_session() as db:
        song = db.get(Song, 1)
        if not song:
            log.error("Song #1 not found")
            return
        log.info(f"Song: {song.title} ({song.artist})")

        # === Step 1: Set audio_path (was None) ===
        wav_path = Path(__file__).resolve().parent.parent / "data" / "audio" / "mG7lrRdm71A.wav"
        if wav_path.exists():
            song.audio_path = str(wav_path)
            log.info(f"  Set audio_path: {wav_path}")
        else:
            log.error(f"  Audio not on disk: {wav_path}")
            return

        # === Step 2: Fetch lyrics (try URL first, fallback ke hardcoded) ===
        old_render = json.loads(song.render_json)
        duration = song.duration_sec
        new_lyrics = None

        # Try URL
        url = "https://lyrics.lyricfind.com/lyrics/f-ace-rain-1"
        fetched = fetch_from_lyricfind(url)
        if fetched and len(fetched) > 10:
            new_lyrics = fetched
            log.info(f"  Got {len(fetched)} lines from URL")
        else:
            log.info("  URL fetch insufficient, using hardcoded lyrics")
            new_lyrics = HARDCODED_LYRICS

        # === Step 3: Build lines (with timing) ===
        new_lines = make_lines(new_lyrics, duration)
        log.info(f"  → {len(new_lines)} lines distributed across {duration}s song")

        # === Step 4: Re-detect chords (strict) ===
        new_chord_segs = detect_chords_strict(wav_path, min_duration=1.5)
        log.info(f"  Chord segments: {len(new_chord_segs)}")

        # === Step 5: Anchor chords to words ===
        for l in new_lines:
            l["chords"] = []
        anchored = 0
        for seg in new_chord_segs:
            cs, ce, chord = seg
            # find target line (where seg starts)
            target = None
            for l in new_lines:
                if l["start"] <= cs < l["end"]:
                    target = l; break
            if not target and new_lines:
                target = new_lines[-1]
            if not target: continue

            # anchor to closest word
            best_i, best_d = 0, float("inf")
            for i, w in enumerate(target["words"]):
                center = (w["start"] + w["end"]) / 2
                d = abs(center - cs)
                if d < best_d: best_d, best_i = d, i
            target["chords"].append({
                "chord": chord, "start": float(cs), "anchor_word_index": best_i
            })
            anchored += 1
        log.info(f"  Anchored {anchored} chords")

        # === Step 6: Build sections + bars ===
        new_sections = [{
            "name": "Verse 1", "start": 0.0, "end": float(duration),
            "has_lyrics": len(new_lines) > 0
        }]
        bars = []
        beats = old_render.get("beats", [])
        if beats:
            downbeats = beats[::4]
            for i, db_time in enumerate(downbeats[:-1]):
                b_start = float(db_time)
                b_end = float(downbeats[i+1]) if i+1 < len(downbeats) else float(duration)
                bar_chords = [s for s in new_chord_segs if s[0] >= b_start and s[0] < b_end]
                chord_name = bar_chords[0][2] if bar_chords else "N"
                bars.append({
                    "index": i, "start": b_start, "end": b_end,
                    "chords": [{"chord": chord_name, "start": b_start, "end": b_end}]
                })

        new_render = {
            "meta": old_render["meta"],
            "beats": old_render.get("beats", []),
            "downbeats": old_render.get("downbeats", []),
            "sections": new_sections,
            "bars": bars,
            "lines": new_lines,
        }

        # === Step 7: Update DB ===
        song.render_json = json.dumps(new_render, ensure_ascii=False)
        song.audio_path = str(wav_path)
        db.commit()
        log.info(f"=== DONE: Updated song #{song.id} ===")
        log.info(f"  Lines: {len(new_lines)}")
        log.info(f"  Chord segments: {len(new_chord_segs)} (was 572)")
        log.info(f"  Audio: {wav_path.name} ({wav_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
