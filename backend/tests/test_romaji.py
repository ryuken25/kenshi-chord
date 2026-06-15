"""WS-1 romaji test — RAIN golden test.

Validates against Appendix A of the Phase 2 megaprompt:
  出逢った → "deatta"
  眩しさは虚しさ → "mabushisa wa munashisa"
  肩を濡らす雨粒で → "kata wo nurasu amatsubu de"
  知ってる → "shitteru"
"""
from __future__ import annotations
import sys
sys.path.insert(0, r"d:\CodePaid\chord\backend")
from app.smart.romaji import romanize, romanize_line, ROMAJI_AVAILABLE


def test_display():
    assert ROMAJI_AVAILABLE, "cutlet not installed"

    cases = [
        ("出逢った",           "deatta"),
        ("眩しさは虚しさ",     "mabushisa wa munashisa"),
        ("肩を濡らす雨粒で",   "kata wo nurasu amatsubu de"),
        ("知ってる",           "shitteru"),
        ("傘を忘れた",         "kasa wo wasureta"),     # を → wo (Appendix A confirms)
        ("冷たかった",         "tsumetakatta"),         # sokuon doubled
        ("2人してずぶ濡れの日も", None),                 # complex, just don't crash
    ]

    print("WS-1 romaji display test:")
    all_pass = True
    for kanji, expected in cases:
        got = romanize(kanji)
        if expected is None:
            print(f"  [info] {kanji!r:30s} → {got!r}")
            continue
        ok = got == expected
        tag = "OK" if ok else "FAIL"
        print(f"  [{tag:4s}] {kanji!r:30s} → {got!r}")
        if not ok:
            print(f"         expected {expected!r}")
            all_pass = False
    return all_pass


def test_line_romaji():
    """Verify word segmentation for '濡らす' gives 3 syllables."""
    line = romanize_line("濡らす")
    print("\nWS-1 LineRomaji test on '濡らす':")
    print(f"  display = {line.display!r}")
    print(f"  words   = {[(w.surface, w.romaji) for w in line.words]}")

    # Find the word containing '濡らす'
    nurasu = [w for w in line.words if "濡ら" in w.surface]
    assert nurasu, f"No word contains '濡ら': {[w.surface for w in line.words]}"
    w = nurasu[0]
    print(f"  word syls = {[s.romaji for s in w.syllables]}")

    # Should have 3 syllables
    syl_roms = [s.romaji for s in w.syllables]
    assert syl_roms == ["nu", "ra", "su"], f"Expected [nu, ra, su], got {syl_roms}"
    print("  OK — 3 syllables")
    return True


if __name__ == "__main__":
    ok1 = test_display()
    ok2 = test_line_romaji()
    if ok1 and ok2:
        print("\nALL WS-1 TESTS PASS")
    else:
        print("\nSOME WS-1 TESTS FAIL")
        sys.exit(1)
