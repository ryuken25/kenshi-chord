"""smartfix_v4 — FORCED ALIGNMENT edition.

Beda dari v3: timing kata BUKAN distribusi rata, tapi dari MMS_FA forced
alignment (timing vokal asli). Chord BTC di-anchor ke kata pakai timing real,
jadi chord ngepas sama lirik.

Run di CPU (MMS_FA segfault di GPU Windows, sama kayak demucs flash-attn).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torchaudio.functional import forced_align, merge_tokens
from torchaudio.pipelines import MMS_FA as bundle

from app.db import db_session
from app.models import Song
from smartfix_v3 import (
    LYRICS, tokenize_with_romaji, parse_btc_lab, simplify_chord,
    btc_at, consolidate_chords,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("v4")


# ============ Forced alignment (CPU) ============
_MODEL = None
def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = bundle.get_model().to("cpu")
    return _MODEL


def forced_align_tokens(vocals_path: Path, romaji_list: list[str]):
    """Align each token's romaji to vocals. Returns list of (start, end) or None.

    romaji_list: 1 romaji string per token (urutannya sama dengan token di lagu).
    Token dengan romaji kosong (tanda baca) → None (nanti di-interpolate).
    """
    model = _get_model()
    DICT = bundle.get_dict()
    sr_model = bundle.sample_rate

    data, sr = sf.read(str(vocals_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    wav16 = AF.resample(wav, sr, sr_model)

    # Build char-token stream; remember which romaji index each span belongs to
    tokens: list[int] = []
    spans: list[tuple[int, int, int] | None] = []
    for idx, rom in enumerate(romaji_list):
        clean = "".join(c for c in rom.lower() if c in DICT)
        if not clean:
            spans.append(None)
            continue
        st = len(tokens)
        for ch in clean:
            tokens.append(DICT[ch])
        spans.append((st, len(tokens), idx))

    if not tokens:
        return [None] * len(romaji_list)

    with torch.inference_mode():
        emission, _ = model(wav16)
    targets = torch.tensor([tokens], dtype=torch.int32)
    ratio = wav16.shape[1] / emission.shape[1]
    aligned, scores = forced_align(emission, targets, blank=0)
    tspans = merge_tokens(aligned[0], scores[0].exp())
    nonblank = [ts for ts in tspans if ts.token != 0]

    times: list[tuple[float, float] | None] = [None] * len(romaji_list)
    ti = 0
    for sp in spans:
        if sp is None:
            continue
        st, en, idx = sp
        n = en - st
        wt = nonblank[ti:ti + n]
        if wt:
            s = wt[0].start * ratio / sr_model
            e = wt[-1].end * ratio / sr_model
            times[idx] = (float(s), float(e))
        ti += n
    return times


def interpolate_missing(times, total_dur):
    """Isi None (token tanpa romaji) dengan interpolasi linear antar tetangga."""
    n = len(times)
    out = list(times)
    # Forward fill anchors
    known = [(i, t) for i, t in enumerate(times) if t is not None]
    if not known:
        # fallback: spread uniform
        per = total_dur / max(n, 1)
        return [(i * per, (i + 1) * per) for i in range(n)]
    # before first known
    first_i, (fs, fe) = known[0]
    for i in range(first_i):
        out[i] = (fs, fs)
    # between knowns
    for k in range(len(known) - 1):
        ai, (a_s, a_e) = known[k]
        bi, (b_s, b_e) = known[k + 1]
        gap = bi - ai
        if gap > 1:
            for j in range(1, gap):
                frac = j / gap
                t = a_e + (b_s - a_e) * frac
                out[ai + j] = (t, t)
    # after last known
    last_i, (ls, le) = known[-1]
    for i in range(last_i + 1, n):
        out[i] = (le, le)
    return out


def main():
    with db_session() as db:
        song = db.get(Song, 1)
        wav = Path(song.audio_path)
        if not wav.exists():
            wav = Path(__file__).resolve().parent.parent / "data" / "audio" / "mG7lrRdm71A.wav"
        vocals = wav.parent / (wav.stem + ".vocals.wav")
        old = json.loads(song.render_json)
        duration = float(song.duration_sec)
        log.info(f"Song: {song.title} | {duration:.0f}s")

        # 1. BTC chords (consolidated)
        btc_lab = next((c for c in [Path("/tmp/btc_out/rain.lab"),
                        Path("C:/Users/lenov/AppData/Local/Temp/btc_out/rain.lab")] if c.exists()), None)
        btc = consolidate_chords(parse_btc_lab(btc_lab), min_dur=0.8)
        log.info(f"  BTC: {len(btc)} consolidated chords")

        # 2. Tokenize lyrics
        structured = []
        for sec, lines in LYRICS.items():
            for chord, text in lines:
                toks = tokenize_with_romaji(text)
                if toks:
                    structured.append((sec, chord, text, toks))
        flat = [tk for _, _, _, toks in structured for tk in toks]
        romaji_list = [tk["romaji"] for tk in flat]
        log.info(f"  Lyrics: {len(structured)} lines, {len(flat)} tokens")

        # 3. FORCED ALIGNMENT (the fix!)
        log.info("  Running forced alignment (CPU)...")
        raw_times = forced_align_tokens(vocals, romaji_list)
        aligned_n = sum(1 for t in raw_times if t is not None)
        log.info(f"  Aligned {aligned_n}/{len(flat)} tokens to real vocal timing")
        token_times = interpolate_missing(raw_times, duration)

        # 4. Build lines with REAL timing + multi-chord
        lines_out = []
        section_bounds = {}
        ti = 0
        for li, (sec, chord, text, toks) in enumerate(structured):
            n = len(toks)
            tt = token_times[ti:ti + n]
            line_start = tt[0][0]
            line_end = max(tt[-1][1], line_start + 0.3)
            words_out = [{"word": toks[j]["surface"], "start": tt[j][0],
                          "end": tt[j][1], "romaji": toks[j]["romaji"]} for j in range(n)]
            ti += n

            line_chords = []
            for cs, ce, cc in btc:
                if line_start <= cs < line_end:
                    sc = simplify_chord(cc)
                    if sc == "N":
                        continue
                    bi, bd = 0, 1e9
                    for k, w in enumerate(words_out):
                        d = abs((w["start"] + w["end"]) / 2 - cs)
                        if d < bd:
                            bd, bi = d, k
                    line_chords.append({"chord": sc, "start": float(cs), "anchor_word_index": bi})
            if chord and (not line_chords or line_chords[0]["start"] > line_start + 0.5):
                line_chords.insert(0, {"chord": chord, "start": float(line_start), "anchor_word_index": 0})
            if not line_chords:
                line_chords.append({"chord": chord or btc_at(line_start, btc),
                                    "start": float(line_start), "anchor_word_index": 0})

            lines_out.append({"line_index": li, "start": line_start, "end": line_end,
                              "text": text, "words": words_out, "chords": line_chords})
            if sec not in section_bounds:
                section_bounds[sec] = [line_start, line_end]
            else:
                section_bounds[sec][1] = line_end

        # 5. Sections: Intro + vocal sections + Outro
        first_vocal = lines_out[0]["start"]
        last_vocal = lines_out[-1]["end"]
        sections_out = []
        if first_vocal > 3.0:
            sections_out.append({"name": "Intro", "start": 0.0, "end": float(first_vocal), "has_lyrics": False})
        sec_items = list(section_bounds.items())
        for idx, (sec, (s, e)) in enumerate(sec_items):
            sections_out.append({"name": sec, "start": float(s), "end": float(e), "has_lyrics": True})
            # interlude jika gap > 6s ke section berikutnya
            if idx + 1 < len(sec_items):
                ns = sec_items[idx + 1][1][0]
                if ns - e > 6.0:
                    sections_out.append({"name": "Interlude", "start": float(e), "end": float(ns), "has_lyrics": False})
        if duration - last_vocal > 3.0:
            outro = next((s for s in sections_out if s["name"] == "Outro"), None)
            if outro:
                outro["end"] = duration
            else:
                sections_out.append({"name": "Outro", "start": float(last_vocal), "end": duration, "has_lyrics": False})

        # 6. Bars — pakai BTC chord change boundaries, FILTER:
        #    - skip kalau di area intro/outro (< first_vocal - 0.3 atau > last_vocal + 0.3)
        #    - skip kalau BTC bilang "N"
        #    - skip kalau durasinya < 0.3s (micro-segment)
        #    - merge consecutive kalau root-nya sama (konsolidasi anti micro-change)
        #    - merge ke "user-specified" kalau line di LYRICS punya inline chord
        def root_of(ch):
            if ch == "N": return "N"
            import re as _re
            m = _re.match(r"^([A-G][#b]?)", ch)
            return m.group(1) if m else ch
        bars = []
        prev_root = None
        for cs, ce, cc in btc:
            sc = simplify_chord(cc)
            if sc == "N":
                continue
            # Skip bar di range intro (0-16.4s) dan outro
            if cs < first_vocal - 0.3:
                continue
            if cs >= last_vocal + 0.3:
                continue
            dur = ce - cs
            if dur < 0.3:
                continue
            # Konsolidasi: kalau root sama dengan sebelumnya, extend
            r = root_of(sc)
            if bars and root_of(bars[-1]["chords"][0]["chord"]) == r:
                bars[-1]["end"] = float(ce)
                bars[-1]["chords"][0]["end"] = float(ce)
            else:
                bars.append({"index": len(bars), "start": float(cs), "end": float(ce),
                             "chords": [{"chord": sc, "start": float(cs), "end": float(ce)}]})

        # 7. Save (regenerate beats from BTC timings biar intro/outro gak punya beat)
        # Beats = sorted midpoints of bar boundaries (gives a sensible beat array)
        beats_set = set()
        for cs, ce, _ in btc:
            beats_set.add(round(cs * 4) / 4)  # quantize to 0.25s
            beats_set.add(round(ce * 4) / 4)
        beats = sorted(b for b in beats_set if 0 < b < duration)
        downbeats = beats[::4] if beats else []
        song.render_json = json.dumps({
            "meta": old["meta"], "beats": beats, "downbeats": downbeats,
            "sections": sections_out, "bars": bars, "lines": lines_out,
        }, ensure_ascii=False)
        db.commit()

        log.info("=== DONE (forced-aligned) ===")
        log.info(f"  Sections: {[s['name'] for s in sections_out]}")
        log.info(f"  Lines: {len(lines_out)}, chord markers: {sum(len(l['chords']) for l in lines_out)}")
        for i in [0, 1, 2, 5]:
            if i < len(lines_out):
                l = lines_out[i]
                rom = " ".join(w["romaji"] for w in l["words"])
                ch = " ".join(f"{c['chord']}@{c['start']:.1f}" for c in l["chords"])
                log.info(f"  L{i} [{l['start']:.1f}-{l['end']:.1f}] {l['text']}")
                log.info(f"      rom: {rom}")
                log.info(f"      chords: {ch}")


if __name__ == "__main__":
    main()
