"""WS-6/7/8 smoke tests (no audio required)."""
import sys
sys.path.insert(0, r"d:\CodePaid\chord\backend")
from app.smart.sections import detect_sections
from app.smart.contract import validate_render, build_render
from app.smart.anchor import parse_override, marks_to_anchors


def test_parse_override_v2():
    raw = "[Am]肩を[G]濡らす[Fmaj7]雨粒で"
    clean, marks = parse_override(raw)
    assert clean == "肩を濡らす雨粒で", clean
    assert len(marks) == 3
    # Am sits at char 0 (before 肩)
    assert marks[0] == (0, "Am")
    # G sits at char 2 (after 濡, before 雨) — no, actually at "before 濡" = char 2
    # [Am]肩を[G]濡らす[Fmaj7]雨粒で
    #   0: before 肩
    #   2: before 濡 (position after "肩を")
    #   5: before 雨 (position after "濡らす")
    assert marks[1] == (2, "G")
    assert marks[2] == (5, "Fmaj7")
    print("T1 parse_override v2 OK")


def test_marks_to_anchors():
    # Simulated aligned line with 3 words + syllables
    words = [
        {"surface": "肩", "romaji": "kata", "_char_start": 0, "_char_end": 1,
         "start": 12.3, "end": 12.8, "syllables": [{"romaji": "ka", "char_start": 0, "char_end": 1}]},
        {"surface": "濡らす", "romaji": "nurasu", "_char_start": 2, "_char_end": 5,
         "start": 13.0, "end": 13.6,
         "syllables": [{"romaji": "nu", "char_start": 0, "char_end": 1},
                       {"romaji": "ra", "char_start": 1, "char_end": 2},
                       {"romaji": "su", "char_start": 2, "char_end": 3}]},
        {"surface": "雨粒で", "romaji": "amatsubu de", "_char_start": 5, "_char_end": 8,
         "start": 14.5, "end": 15.1,
         "syllables": [{"romaji": "a", "char_start": 0, "char_end": 1},
                       {"romaji": "ma", "char_start": 1, "char_end": 2}]},
    ]
    clean, marks = parse_override("[Am]肩を[G]濡らす[Fmaj7]雨粒で")
    anchors = marks_to_anchors(marks, clean, words)
    assert anchors[0]["chord"] == "Am"
    assert anchors[0]["anchor_word_index"] == 0  # attaches to 肩
    assert anchors[1]["chord"] == "G"
    assert anchors[1]["anchor_word_index"] == 1  # attaches to 濡らす
    assert anchors[2]["chord"] == "Fmaj7"
    assert anchors[2]["anchor_word_index"] == 2  # attaches to 雨粒で
    print("T2 marks_to_anchors OK")


def test_validate_render_passes_good():
    r = build_render(
        meta={"bpm": 165, "key": "B minor", "beats_per_bar": 4, "time_sig": "4/4"},
        beats=[1.0, 1.36, 1.72],
        downbeats=[1.0],
        sections=[{"name": "Intro", "start": 0.0, "end": 5.0}],
        bars=[{"start": 0.0, "end": 1.36, "index": 0}],
        lines=[{
            "start": 12.0, "end": 15.0, "text": "肩を濡らす雨粒で",
            "display": "kata wo nurasu amatsubu de",
            "confidence": "high",
            "words": [
                {"surface": "肩", "romaji": "kata", "start": 12.0, "end": 12.8},
                {"surface": "濡らす", "romaji": "nurasu", "start": 13.0, "end": 13.6},
            ],
            "chords": [
                {"chord": "Am", "start": 12.3, "anchor_word_index": 0}
            ],
        }],
    )
    validate_render(r)
    print("T3 validate_render (good) OK")


def test_validate_render_rejects_bad_anchor():
    r = build_render(
        meta={"bpm": 165, "key": "B minor"},
        beats=[1.0], downbeats=[],
        sections=[], bars=[],
        lines=[{
            "start": 12.0, "end": 15.0,
            "words": [{"start": 12.0, "end": 12.8}],
            "chords": [{"chord": "Am", "start": 12.3, "anchor_word_index": 50}],
        }],
    )
    try:
        validate_render(r)
    except ValueError as e:
        if "anchor_word_index" in str(e):
            print("T4 validate_render (bad anchor) OK")
            return
    raise AssertionError("expected ValueError on bad anchor")


def test_sections_repeats_chorus():
    """Repeated progression should be labeled Chorus 1 each time."""
    # 4 vocal segments, 3 gaps
    lines = [
        {"start":  5.0, "end": 10.0},
        {"start": 15.0, "end": 20.0},
        {"start": 30.0, "end": 35.0},
        {"start": 45.0, "end": 50.0},
    ]
    # BTC has same Am-G-F progression in each segment (repeated chorus)
    btc = [
        (5,  7.5, "A:min"), (7.5, 10, "G:maj"), (10, 14.5, "F:maj"),
        (15, 17.5, "A:min"), (17.5, 20, "G:maj"), (20, 24.5, "F:maj"),
        (30, 32.5, "A:min"), (32.5, 35, "G:maj"), (35, 39.5, "F:maj"),
        (45, 47.5, "A:min"), (47.5, 50, "G:maj"), (50, 54.5, "F:maj"),
    ]
    secs = detect_sections(lines, btc, duration=60.0, gap_threshold=4.0)
    names = [s["name"] for s in secs]
    print("T5 sections names:", names)
    chorus_occurrences = [s["name"] for s in secs if s["name"].startswith("Chorus")]
    assert len(chorus_occurrences) >= 2, f"expected repeated Chorus, got {chorus_occurrences}"
    # All Chorus occurrences should share the SAME number (Chorus 1)
    assert len(set(chorus_occurrences)) == 1, f"Chorus labels must be consistent: {chorus_occurrences}"
    print("T5 detect_sections OK — 4 repeated segments all labeled Chorus 1")


if __name__ == "__main__":
    test_parse_override_v2()
    test_marks_to_anchors()
    test_validate_render_passes_good()
    test_validate_render_rejects_bad_anchor()
    test_sections_repeats_chorus()
    print("\nALL WS-6/7/8 UNIT TESTS PASS")
