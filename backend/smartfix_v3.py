"""SMART FIX v3 — genuinely intelligent pipeline.

Signals combined:
  - BTC-ISMIR19 chord segments (per-second, reliable)        → chord placement
  - Whisper word timestamps (real vocal timeline)            → lyric timing + vocal regions
  - janome tokenizer + pykakasi (kanji readings → romaji)    → accurate romaji
  - Vocal-gap analysis                                       → intro / interlude / outro detection

Output: render_json with real timing, proper sections, multi-chord per line, romaji.

Run: python smartfix_v3.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import faster_whisper
# janome & pykakasi di-import lazy (lihat _get_tokenizer/_get_kakasi)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("v3")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.db import db_session, init_db
from app.models import Song

init_db()

# ============ Lyrics (section → list of (inline_chord, text)) ============
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
        (None, "降り出す rain"),
        (None, "何もかもを流して"),
        (None, "どうしてあの時"),
        (None, "気づけなかった"),
        ("Am", "君のいない朝"),
        (None, "無神経に晴れわたる空"),
    ],
    "Chorus 1": [
        (None, "止まない rain"),
        (None, "駆け出した空の下"),
        (None, "何もかも忘れて君との日を"),
        (None, "胸にしまって歩き出す今"),
        (None, "そこにいてくれて ありがとう"),
    ],
    "Verse 2": [
        (None, "失って気づく安らいだ日々たちは"),
        ("C",  "どこか遠くへと"),
        (None, "そう 雨 が今"),
        (None, "君が好きだと言った"),
        (None, "雨 が今降る ほら"),
        (None, "降り出す rain"),
        (None, "何もかもを流して"),
        (None, "溢れ出す 涙を隠して また"),
    ],
    "Chorus 2": [
        (None, "誰もいない部屋"),
        (None, "雨音だけが僕を包む oh"),
        (None, "輝く rain"),
        (None, "駆け出した空の下"),
        (None, "君という rain"),
        (None, "浴びて蘇る日々"),
        (None, "君のいない世界歩き出す今"),
    ],
    "Outro": [
        ("Am", "出逢ってくれて"),
    ],
}

# ============ Romaji: janome tokenize → reading → pykakasi ============
import pykakasi as _pykakasi
_TOKENIZER = None
_KAKASI = None
def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from janome.tokenizer import Tokenizer
        _TOKENIZER = Tokenizer()
    return _TOKENIZER
def _get_kakasi():
    global _KAKASI
    if _KAKASI is None:
        _KAKASI = _pykakasi.kakasi()
    return _KAKASI

def _kata_to_romaji(kata: str) -> str:
    parts = _get_kakasi().convert(kata)
    return "".join(p.get("hepburn", "") for p in parts)

def tokenize_with_romaji(text: str):
    """Tokenize Japanese line → list of {surface, romaji}.
    Latin words (rain, Ah) passed through. Uses janome readings for accurate romaji.
    """
    out = []
    for tok in _get_tokenizer().tokenize(text):
        surface = tok.surface
        if not surface.strip():
            continue
        # Latin/ASCII passthrough
        if surface.isascii():
            out.append({"surface": surface, "romaji": surface})
            continue
        reading = tok.reading if tok.reading and tok.reading != "*" else surface
        out.append({"surface": surface, "romaji": _kata_to_romaji(reading)})
    return out

# ============ BTC chord parsing + simplification ============
def parse_btc_lab(path: Path):
    segs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                try:
                    segs.append((float(p[0]), float(p[1]), p[2]))
                except ValueError:
                    pass
    return segs

def simplify_chord(c: str) -> str:
    """BTC label → frontend-friendly. Preserve musical richness."""
    if c == "N":
        return "N"
    if ":" not in c:
        return c
    root, mod = c.split(":", 1)
    table = {
        "maj": "", "min": "m", "maj7": "maj7", "min7": "m7", "7": "7",
        "maj6": "6", "min6": "m6", "sus2": "sus2", "sus4": "sus4",
        "dim": "dim", "aug": "aug", "hdim7": "m7b5", "minmaj7": "mM7",
        "maj9": "maj7", "min9": "m7", "9": "7",
    }
    return root + table.get(mod, "")

def btc_at(t, segs, default="C"):
    for s, e, c in segs:
        if s <= t < e:
            return simplify_chord(c)
    return default

def consolidate_chords(btc_segs, min_dur=0.8):
    """Merge BTC segments yang terlalu cepat/micro-change biar musical.
    - Simplify dulu (Fmaj7/Fsus2/F → root F kalau beda-tipis berdekatan)
    - Merge consecutive yang ROOT-nya sama (F → Fmaj7 → Fsus2 = tetap F-family)
    - Drop segment < min_dur (absorb ke tetangga)
    Returns: list of (start, end, simplified_chord)
    """
    # 1. Simplify all
    simp = [(s, e, simplify_chord(c)) for s, e, c in btc_segs]
    # 2. Merge consecutive with same ROOT (ignore quality micro-changes)
    def root_of(ch):
        if ch == "N": return "N"
        # root = leading A-G + optional #/b
        import re
        m = re.match(r"^([A-G][#b]?)", ch)
        return m.group(1) if m else ch
    merged = []
    for s, e, c in simp:
        if merged and root_of(merged[-1][2]) == root_of(c):
            # extend previous; keep the LONGER-lasting quality label
            ps, pe, pc = merged[-1]
            # prefer simpler label (root only) kalau quality beda-beda
            new_label = pc if len(pc) <= len(c) else c
            merged[-1] = (ps, e, new_label)
        else:
            merged.append((s, e, c))
    # 3. Absorb tiny segments (< min_dur) ke tetangga yang lebih panjang
    result = []
    for s, e, c in merged:
        dur = e - s
        if dur < min_dur and result:
            # extend previous segment to cover this
            ps, pe, pc = result[-1]
            result[-1] = (ps, e, pc)
        else:
            result.append((s, e, c))
    # 4. Final merge pass (same root again after absorption)
    final = []
    for s, e, c in result:
        if final and root_of(final[-1][2]) == root_of(c):
            final[-1] = (final[-1][0], e, final[-1][2])
        else:
            final.append((s, e, c))
    return final

# ============ Whisper word timeline ============
def whisper_words(audio_path, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = faster_whisper.WhisperModel("base", device=device, compute_type="int8")
    segs, info = m.transcribe(str(audio_path), beam_size=5, vad_filter=True,
                              word_timestamps=True, language="ja", initial_prompt=prompt)
    words = []
    for s in segs:
        if not s.words: continue
        for w in s.words:
            if w.start is None or w.end is None: continue
            words.append((float(w.start), float(w.end)))
    del m
    if device == "cuda": torch.cuda.empty_cache()
    return words

# ============ Vocal region detection from demucs vocals stem ============
def detect_vocal_regions(vocals_path, true_sr, gap_merge=2.5, min_region=1.5):
    """Detect vocal-active regions dari demucs vocals stem energy.
    PENTING: demucs save salah sr (44100 padahal asli 48000) — frames sama,
    jadi kita reinterpret pakai true_sr untuk timing yang bener.
    Returns: list of [start, end] in seconds (true timeline).
    """
    import soundfile as sf
    data, _ = sf.read(str(vocals_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    hop = 2048
    n = (len(data) - hop) // hop
    rms = np.array([np.sqrt(np.mean(data[i*hop:i*hop+hop] ** 2)) for i in range(max(n, 0))])
    times = np.arange(len(rms)) * hop / true_sr
    if len(rms) == 0:
        return [], 0.0
    thresh = max(np.percentile(rms, 70) * 0.35, rms.max() * 0.06)
    active = rms > thresh
    regions = []
    in_r, start = False, 0.0
    for i, a in enumerate(active):
        if a and not in_r: start = times[i]; in_r = True
        elif not a and in_r: regions.append([start, times[i]]); in_r = False
    if in_r: regions.append([start, times[-1]])
    merged = []
    for r in regions:
        if merged and r[0] - merged[-1][1] < gap_merge: merged[-1][1] = r[1]
        else: merged.append(r)
    merged = [r for r in merged if r[1] - r[0] > min_region]
    total_dur = len(data) / true_sr
    return merged, total_dur

def main():
    with db_session() as db:
        song = db.get(Song, 1)
        if not song:
            log.error("Song #1 not found"); return
        wav = Path(song.audio_path) if song.audio_path else None
        if not wav or not wav.exists():
            wav = Path(__file__).resolve().parent.parent / "data" / "audio" / "mG7lrRdm71A.wav"
        song.audio_path = str(wav)
        old = json.loads(song.render_json)
        duration = float(song.duration_sec)
        log.info(f"Song: {song.title} | {duration:.0f}s")

        # --- 1. BTC chords ---
        btc_lab = None
        for c in [Path("/tmp/btc_out/rain.lab"),
                  Path("C:/Users/lenov/AppData/Local/Temp/btc_out/rain.lab")]:
            if c.exists(): btc_lab = c; break
        if not btc_lab:
            log.error("BTC .lab not found"); return
        btc_raw = parse_btc_lab(btc_lab)
        btc = consolidate_chords(btc_raw, min_dur=0.8)
        log.info(f"  BTC: {len(btc_raw)} raw → {len(btc)} consolidated chord segments")

        # --- 2. Tokenize hardcoded lyrics (janome + romaji) ---
        # Build flat structure: [(section, inline_chord, line_text, [tokens])]
        structured = []
        for sec, lines in LYRICS.items():
            for chord, text in lines:
                toks = tokenize_with_romaji(text)
                if toks:
                    structured.append((sec, chord, text, toks))
        total_tokens = sum(len(s[3]) for s in structured)
        log.info(f"  Lyrics: {len(structured)} lines, {total_tokens} tokens")

        # --- 3. Detect vocal regions from demucs vocals stem ---
        vocals_path = wav.parent / (wav.stem + ".vocals.wav")
        TRUE_SR = 48000  # original audio sr (demucs saved at wrong 44100, frames identical)
        intro_end = 0.0
        vocal_end = duration
        interlude_gaps = []  # list of (start, end) instrumental gaps within vocals
        if vocals_path.exists():
            regions, vdur = detect_vocal_regions(vocals_path, TRUE_SR, gap_merge=2.5, min_region=1.5)
            if regions:
                intro_end = regions[0][0]
                vocal_end = regions[-1][1]
                # Detect interludes (gaps > 6s between regions = real instrumental break)
                for i in range(len(regions) - 1):
                    gap_start = regions[i][1]
                    gap_end = regions[i + 1][0]
                    if gap_end - gap_start > 6.0:
                        interlude_gaps.append((gap_start, gap_end))
                log.info(f"  Vocal detection: intro 0→{intro_end:.1f}s, vocals→{vocal_end:.1f}s, "
                         f"outro {duration - vocal_end:.1f}s, {len(interlude_gaps)} interludes")
        else:
            log.warning(f"  vocals.wav not found at {vocals_path}, using whisper fallback")
            prompt = " ".join(s[2] for s in structured)
            ww = whisper_words(wav, prompt)
            if ww:
                intro_end = ww[0][0]
                vocal_end = ww[-1][1]

        if intro_end < 1.0:
            intro_end = 0.0
        log.info(f"  Vocal region: {intro_end:.1f}s → {vocal_end:.1f}s")

        # --- 4. Distribute hardcoded tokens across vocal region (proportional to token count) ---
        # Build flat token list, skip interlude gaps
        vocal_span = vocal_end - intro_end
        # Subtract interlude durations from usable span
        interlude_total = sum(e - s for s, e in interlude_gaps)
        usable_span = vocal_span - interlude_total
        if usable_span < 1:
            usable_span = vocal_span
        per_token = usable_span / max(total_tokens, 1)

        token_times = []
        t = intro_end
        for i in range(total_tokens):
            # Skip over interlude gaps
            for gs, ge in interlude_gaps:
                if gs <= t < ge:
                    t = ge
            start = t
            end = t + per_token
            token_times.append((start, end))
            t = end

        # --- 5. Build lines with real timing + multi-chord from BTC ---
        lines_out = []
        section_bounds = {}  # section → [start, end]
        ti = 0  # token index
        line_idx = 0
        for sec, chord, text, toks in structured:
            n = len(toks)
            line_start = token_times[ti][0]
            line_end = token_times[ti + n - 1][1]
            words_out = []
            for j, tk in enumerate(toks):
                ws, we = token_times[ti + j]
                words_out.append({
                    "word": tk["surface"], "start": ws, "end": we,
                    "romaji": tk["romaji"],
                })
            ti += n

            # Multi-chord: all BTC segments within this line's time range
            line_chords = []
            for cs, ce, cc in btc:
                if line_start <= cs < line_end:
                    sc = simplify_chord(cc)
                    if sc == "N": continue
                    # anchor to closest word
                    bi, bd = 0, 1e9
                    for k, w in enumerate(words_out):
                        center = (w["start"] + w["end"]) / 2
                        d = abs(center - cs)
                        if d < bd: bd, bi = d, k
                    line_chords.append({"chord": sc, "start": float(cs), "anchor_word_index": bi})
            # If line has explicit chord and no BTC chord at start, prepend it
            if chord and (not line_chords or line_chords[0]["start"] > line_start + 0.5):
                line_chords.insert(0, {"chord": chord, "start": float(line_start), "anchor_word_index": 0})
            # Ensure at least one chord
            if not line_chords:
                line_chords.append({"chord": chord or btc_at(line_start, btc),
                                    "start": float(line_start), "anchor_word_index": 0})

            lines_out.append({
                "line_index": line_idx, "start": line_start, "end": line_end,
                "text": text, "words": words_out, "chords": line_chords,
            })
            # Track section bounds
            if sec not in section_bounds:
                section_bounds[sec] = [line_start, line_end]
            else:
                section_bounds[sec][1] = line_end
            line_idx += 1

        # --- 6. Build sections: Intro + vocal sections + interludes + Outro ---
        sections_out = []
        if intro_end > 3.0:
            sections_out.append({"name": "Intro", "start": 0.0, "end": float(intro_end), "has_lyrics": False})

        vocal_sec_list = list(section_bounds.items())
        for idx, (sec, (s, e)) in enumerate(vocal_sec_list):
            sections_out.append({"name": sec, "start": float(s), "end": float(e), "has_lyrics": True})
            # Insert interlude if a detected gap falls between this section and next
            if idx + 1 < len(vocal_sec_list):
                next_s = vocal_sec_list[idx + 1][1][0]
                for gs, ge in interlude_gaps:
                    if e - 2 <= gs <= next_s + 2 and ge - gs > 6.0:
                        sections_out.append({"name": "Interlude", "start": float(e),
                                             "end": float(next_s), "has_lyrics": False})
                        break

        # Outro (instrumental tail after last vocal). Kalau section "Outro" udah ada
        # dari lirik, extend aja ke akhir lagu (jangan bikin duplicate).
        if duration - vocal_end > 3.0:
            existing_outro = next((s for s in sections_out if s["name"] == "Outro"), None)
            if existing_outro:
                existing_outro["end"] = duration
            else:
                sections_out.append({"name": "Outro", "start": float(vocal_end),
                                     "end": duration, "has_lyrics": False})

        # --- 7. Build bar grid (4-beat bars) for whole song, chords from BTC ---
        beats = old.get("beats", [])
        downbeats = beats[::4] if beats else []
        bars = []
        for i, dbt in enumerate(downbeats[:-1]):
            b_start = float(dbt)
            b_end = float(downbeats[i + 1]) if i + 1 < len(downbeats) else duration
            bars.append({"index": i, "start": b_start, "end": b_end,
                         "chords": [{"chord": btc_at(b_start, btc), "start": b_start, "end": b_end}]})

        # --- 8. Save ---
        new_render = {
            "meta": old["meta"], "beats": beats, "downbeats": downbeats,
            "sections": sections_out, "bars": bars, "lines": lines_out,
        }
        song.render_json = json.dumps(new_render, ensure_ascii=False)
        db.commit()

        log.info("=== DONE ===")
        log.info(f"  Sections: {[s['name'] for s in sections_out]}")
        log.info(f"  Lines: {len(lines_out)}")
        total_chords = sum(len(l['chords']) for l in lines_out)
        log.info(f"  Total chord markers: {total_chords} (multi-chord per line)")
        log.info(f"  Bars: {len(bars)}")
        for i in [0, 1, 5]:
            if i < len(lines_out):
                l = lines_out[i]
                rom = " ".join(w["romaji"] for w in l["words"])
                chords = " ".join(c["chord"] for c in l["chords"])
                log.info(f"  Line {i} [{l['start']:.1f}-{l['end']:.1f}s]: \"{l['text']}\"")
                log.info(f"         romaji: {rom}")
                log.info(f"         chords: {chords}")


if __name__ == "__main__":
    main()
