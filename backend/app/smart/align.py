"""WS-4: Per-line MMS_FA forced alignment at syllable level.

Approach:
  - Each lyric line gets its OWN MMS_FA pass over the slice
    `[line_start-0.3, line_end+0.3]` (cap 25 s).
  - Token sequence is built from that line's syllables (romaji chars ∈ MMS_FA dict).
  - Result: per-syllable spans → per-word spans → line start/end.
  - Lines with low confidence are interpolated within-line only.

Optional vocals-stem path: wire demucs to isolate vocals first (config flag).
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torchaudio.functional import forced_align as ta_forced_align, merge_tokens as ta_merge_tokens
from torchaudio.pipelines import MMS_FA as bundle

from .romaji import LineRomaji, Word, Syllable

log = logging.getLogger(__name__)

_MODEL = None
_MODEL_DEVICE: str = "cpu"
_DICT = None


def _get_model():
    """Lazy-load MMS_FA model + vocab. Pin to CPU (aligner is small)."""
    global _MODEL, _MODEL_DEVICE, _DICT
    if _MODEL is not None:
        return _MODEL, _MODEL_DEVICE, _DICT

    _DICT = bundle.get_dict()
    model = bundle.get_model()
    device = "cpu"
    # Probe CUDA — fall back if not usable
    if torch.cuda.is_available():
        try:
            probe = torch.zeros(1, 16000)
            model.to("cuda").eval()(probe)
            device = "cuda"
        except Exception as e:
            log.info("MMS_FA: CUDA unusable (%s), using CPU", type(e).__name__)
    _MODEL = model.to(device).eval()
    _MODEL_DEVICE = device
    log.info("MMS_FA loaded on %s (vocab=%d)", device, len(_DICT))
    return _MODEL, _MODEL_DEVICE, _DICT


def _load_wav_16k(path: Path) -> Tuple[torch.Tensor, int]:
    data, fsr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0), int(fsr)


def _build_token_seq(romaji_items: List[str]):
    """Build integer token sequence + span map from romaji list."""
    DICT = bundle.get_dict()
    tokens = []
    spans = []  # list of (start_idx, end_idx, src_idx) — src_idx = index in romaji_items
    for src_idx, rom in enumerate(romaji_items):
        clean = "".join(c for c in rom.lower() if c in DICT)
        if not clean:
            spans.append(None)
            continue
        st = len(tokens)
        for ch in clean:
            tokens.append(DICT[ch])
        spans.append((st, len(tokens), src_idx))
    return tokens, spans


def _align_slice(wav16: torch.Tensor, start_s: float, end_s: float,
                 romaji_list: List[str]) -> List[Optional[Tuple[float, float]]]:
    """Align romaji_list against audio slice [start_s, end_s].

    Returns list[Optional[(start, end)]] in absolute seconds (offset by start_s).
    """
    model, device, DICT = _get_model()
    sr = bundle.sample_rate

    total_samples = wav16.shape[1]
    s0 = max(0, int(start_s * sr))
    s1 = min(total_samples, int(end_s * sr))
    if s1 <= s0:
        return [None] * len(romaji_list)

    chunk = wav16[:, s0:s1]
    tokens, spans = _build_token_seq(romaji_list)
    if not tokens:
        return [None] * len(romaji_list)

    with torch.inference_mode():
        emission, _ = model(chunk.to(device))
    ratio = chunk.shape[1] / emission.shape[1]
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    try:
        aligned, scores = ta_forced_align(emission, targets, blank=0)
    except Exception as e:
        log.warning("forced_align failed on slice [%.2f,%.2f]: %s", start_s, end_s, e)
        return [None] * len(romaji_list)
    tspans = ta_merge_tokens(aligned[0], scores[0].exp())
    nonblank = [ts for ts in tspans if ts.token != 0]

    times: List[Optional[Tuple[float, float]]] = [None] * len(romaji_list)
    lti = 0
    for sp in spans:
        if sp is None:
            continue
        st, en, idx = sp
        wt = nonblank[lti:lti + (en - st)]
        if wt:
            t0 = wt[0].start * ratio / sr + start_s
            t1 = wt[-1].end * ratio / sr + start_s
            times[idx] = (float(t0), float(t1))
        lti += en - st

    del emission, targets, aligned, scores
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        gc.collect()
    return times


def align_lines(
    vocals_path,
    lines: List[LineRomaji],
    line_windows: List[Tuple[float, float]],
    pad: float = 0.3,
    max_slice: float = 25.0,
) -> list:
    """Per-line MMS_FA alignment, syllable-level.

    Args:
      vocals_path: path to WAV (vocals or mix).
      lines: list of LineRomaji from `romanize_line`.
      line_windows: list of (start, end) per line — rough initial windows
        (from Whisper VAD). We extend by `pad` on each side.
      pad: seconds to pad each side of a line window.
      max_slice: split a single line's slice if it exceeds this (seconds).

    Returns:
      list of dicts: same length as `lines`. Each entry is an aligned-line dict:
      {
        "line_index": int,
        "start": float, "end": float,
        "text": str,                # original line
        "display": str,             # spaced romaji
        "confidence": "high" | "low",
        "words": [{ "surface","romaji","start","end",
                    "syllables": [{"romaji","start","end"}, ...] }]
      }
    """
    wav_full, fsr = _load_wav_16k(Path(vocals_path))
    if fsr != bundle.sample_rate:
        wav16 = AF.resample(wav_full, fsr, bundle.sample_rate)
    else:
        wav16 = wav_full
    full_dur = wav16.shape[1] / bundle.sample_rate

    out = []
    for li, line in enumerate(lines):
        if li >= len(line_windows):
            s_s, e_s = 0.0, full_dur
        else:
            s_s, e_s = line_windows[li]
        # Pad + cap
        win_s = max(0.0, s_s - pad)
        win_e = min(full_dur, e_s + pad)
        if win_e - win_s > max_slice:
            # Split into chunks of max_slice seconds, align, concatenate
            all_times: List[Optional[Tuple[float, float]]] = []
            cursor = win_s
            syms_all = [s for w in line.words for s in w.syllables]
            syl_roms = [s.romaji for s in syms_all]
            # Chunk both time and syllable list proportionally
            n_syl = len(syl_roms)
            cursor_idx = 0
            while cursor < win_e:
                chunk_end = min(win_e, cursor + max_slice)
                n_take = max(1, int(round(n_syl * (chunk_end - cursor) / (win_e - win_s))))
                n_take = min(n_take, n_syl - cursor_idx)
                chunk_times = _align_slice(wav16, cursor, chunk_end,
                                           syl_roms[cursor_idx:cursor_idx + n_take])
                all_times.extend(chunk_times)
                cursor = chunk_end
                cursor_idx += n_take
            times = all_times
        else:
            syms_all = [s for w in line.words for s in w.syllables]
            syl_roms = [s.romaji for s in syms_all]
            times = _align_slice(wav16, win_s, win_e, syl_roms)

        # Compute confidence (fraction of non-None)
        hit = sum(1 for t in times if t is not None)
        confidence = "high" if hit / max(1, len(times)) >= 0.5 else "low"

        # Fill in missing times by simple linear interpolation within line only
        times = _linear_interp_within(times)

        # Aggregate syllable spans → word spans → line span
        word_starts, word_ends = [], []
        syl_idx = 0
        for w in line.words:
            n = len(w.syllables)
            chunk = times[syl_idx:syl_idx + n]
            syl_idx += n
            nonnull = [c for c in chunk if c is not None]
            if nonnull:
                word_starts.append(nonnull[0][0])
                word_ends.append(nonnull[-1][1])
            else:
                word_starts.append(None)
                word_ends.append(None)

        valid_ws = [s for s in word_starts if s is not None]
        valid_we = [e for e in word_ends if e is not None]
        if not valid_ws:
            log.warning("line %d: all syllables failed alignment", li)
            continue

        line_start = min(valid_ws)
        line_end = max(valid_we)

        # Build word dicts (copy times back onto WS-1 Word/Syllable objects)
        syl_idx = 0
        out_words = []
        for wi, w in enumerate(line.words):
            n = len(w.syllables)
            syl_spans = []
            for k in range(n):
                t = times[syl_idx + k] if (syl_idx + k) < len(times) else None
                if t is not None:
                    syl_spans.append({"romaji": w.syllables[k].romaji,
                                      "start": t[0], "end": t[1]})
                    w.syllables[k].start = t[0]
                    w.syllables[k].end = t[1]
                    w.syllables[k].score = 1.0
                else:
                    syl_spans.append({"romaji": w.syllables[k].romaji,
                                      "start": None, "end": None})
                # Extend Syllable dataclass at runtime with start/end/score
                if not hasattr(w.syllables[k], "start"):
                    w.syllables[k].start = None
                    w.syllables[k].end = None
                    w.syllables[k].score = None
            syl_idx += n
            ws = word_starts[wi]
            we = word_ends[wi]
            if ws is None or we is None:
                continue
            w.start = ws
            w.end = we
            out_words.append({
                "surface": w.surface,
                "romaji": w.romaji,
                "start": ws, "end": we,
                "syllables": syl_spans,
            })

        out.append({
            "line_index": li,
            "start": line_start,
            "end": line_end,
            "text": line.text,
            "display": line.display,
            "confidence": confidence,
            "words": out_words,
        })

    return out


def _linear_interp_within(times: List[Optional[Tuple[float, float]]]) -> List[Optional[Tuple[float, float]]]:
    """Linear interpolation over None gaps within a line."""
    out = list(times)
    # Left fill
    first_nonnull = next((i for i, t in enumerate(out) if t is not None), None)
    if first_nonnull is None:
        return out
    fs = out[first_nonnull][0]
    for i in range(first_nonnull):
        out[i] = (fs, fs)
    # Mid fill
    for i in range(len(out) - 1):
        if out[i] is not None and out[i + 1] is None:
            left_end = out[i][1]
            # Find next non-null on the right
            r = i + 2
            while r < len(out) and out[r] is None:
                r += 1
            if r < len(out):
                right_start = out[r][0]
                gap = r - i
                for j, k in enumerate(range(i + 1, r)):
                    frac = (j + 1) / gap
                    t = left_end + frac * (right_start - left_end)
                    out[k] = (t, t)
    # Right fill
    last_nonnull = next((i for i in range(len(out) - 1, -1, -1) if out[i] is not None), None)
    if last_nonnull is not None:
        le = out[last_nonnull][1]
        for i in range(last_nonnull + 1, len(out)):
            out[i] = (le, le)
    return out


__all__ = ["align_lines"]
