"""Test MMS_FA forced alignment on vocals stem."""
import torch, torchaudio
import soundfile as sf
import numpy as np
from torchaudio.pipelines import MMS_FA as bundle
from janome.tokenizer import Tokenizer
import pykakasi

device = "cpu"
model = bundle.get_model().to(device)
DICT = bundle.get_dict()

TOK = Tokenizer()
KKS = pykakasi.kakasi()

def romaji_of(text):
    out = []
    for t in TOK.tokenize(text):
        s = t.surface
        if not s.strip(): continue
        if s.isascii():
            out.append(s.lower())
            continue
        reading = t.reading if t.reading and t.reading != "*" else s
        parts = KKS.convert(reading)
        rom = "".join(p.get("hepburn", "") for p in parts).lower()
        rom = "".join(c for c in rom if c.isalpha())  # only a-z
        if rom:
            out.append(rom)
    return out

# First 3 lines
lines = ["肩を濡らす雨粒で", "傘を忘れたことも忘れてた", "眩しさは虚しさ照らして"]
all_words = []
for ln in lines:
    all_words.extend(romaji_of(ln))
print("Romaji words:", all_words)

# Load vocals @ correct sr
data, _ = sf.read("../data/audio/mG7lrRdm71A.vocals.wav")
if data.ndim > 1: data = data.mean(axis=1)
TRUE_SR = 48000
# Resample to model sr (16000)
import torchaudio.functional as F
wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
wav16 = F.resample(wav, TRUE_SR, bundle.sample_rate)
# Only first 35s (where these lines should be — intro ends 15s)
seg = wav16[:, :int(35 * bundle.sample_rate)].to(device)

# Tokenize words → token ids
def tokenize(words):
    ids = []
    spans = []  # (start_idx, end_idx) per word
    for w in words:
        start = len(ids)
        for ch in w:
            if ch in DICT:
                ids.append(DICT[ch])
        spans.append((start, len(ids)))
    return ids, spans

with torch.inference_mode():
    emission, _ = model(seg)

from torchaudio.functional import forced_align, merge_tokens
tokens, spans = tokenize(all_words)
targets = torch.tensor([tokens], dtype=torch.int32, device=device)
ratio = seg.shape[1] / emission.shape[1]

aligned, scores = forced_align(emission, targets, blank=0)
token_spans = merge_tokens(aligned[0], scores[0].exp())

# Map token spans → word times
print("\nWord alignment (first 35s window):")
ti = 0
for wi, w in enumerate(all_words):
    ws, we = spans[wi]
    nchar = we - ws
    if nchar == 0: continue
    word_tokens = [ts for ts in token_spans if ts.token != 0][ti:ti+nchar]
    if not word_tokens:
        ti += nchar; continue
    start_t = word_tokens[0].start * ratio / bundle.sample_rate
    end_t = word_tokens[-1].end * ratio / bundle.sample_rate
    print(f"  {w:12s} {start_t:6.2f}s - {end_t:6.2f}s")
    ti += nchar
