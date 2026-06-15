/* =================================================================
   KENSHICHORD — CHORD UTILS
   Transposer, parser
   ================================================================= */

(function () {
  "use strict";

  const SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const FLAT  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

  /** Parse chord name → {root, modifier, bass?} */
  function parseChord(name) {
    if (!name) return null;
    const str = String(name).trim();
    if (str === "N" || str === "—" || str === "-") return { kind: "none" };

    // Slash chord: G/B, F#m/A#, C7/G
    const slash = str.split("/");
    const main = slash[0];
    const bass = slash[1] || null;

    // Root (C, C#, Db, D, ...)
    const m = main.match(/^([A-Ga-g])([#♯b♭]?)(.*)$/);
    if (!m) return { kind: "raw", raw: str };

    const rootLetter = m[1].toUpperCase();
    let accidental = m[2].replace("♯","#").replace("♭","b");
    const modifier = m[3] || "";

    return { kind: "parsed", root: rootLetter + accidental, modifier, bass, raw: str };
  }

  /** Get semitone index of a note (uses sharp) */
  function noteToIdx(note) {
    const map = { "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11 };
    return map[note] ?? null;
  }

  /** Idx → preferred name (sharp by default, with b preference) */
  function idxToNote(idx, preferFlat = false) {
    const arr = preferFlat ? FLAT : SHARP;
    return arr[((idx % 12) + 12) % 12];
  }

  /** Transpose a single chord name by N semitones */
  function transposeChord(name, semitones, preferFlat = false) {
    const p = parseChord(name);
    if (!p) return name;
    if (p.kind === "none") return name;
    if (p.kind === "raw") return name;

    const rIdx = noteToIdx(p.root);
    if (rIdx === null) return name;
    const newRoot = idxToNote(rIdx + semitones, preferFlat);
    let out = newRoot + p.modifier;
    if (p.bass) {
      const bIdx = noteToIdx(p.bass);
      if (bIdx !== null) out += "/" + idxToNote(bIdx + semitones, preferFlat);
    }
    return out;
  }

  /** Transpose all chord labels in a render_json (in place safe) */
  function transposeRender(song, semitones) {
    if (!semitones) return song;
    const out = JSON.parse(JSON.stringify(song));
    // Section chord labels in bars
    if (out.bars) {
      out.bars.forEach(b => (b.chords || []).forEach(c => { c.chord = transposeChord(c.chord, semitones); }));
    }
    if (out.lines) {
      out.lines.forEach(l => (l.chords || []).forEach(c => { c.chord = transposeChord(c.chord, semitones); }));
    }
    if (out.meta) {
      // If key is known, also shift key
      const kp = parseChord(out.meta.key);
      if (kp && kp.kind === "parsed") {
        const rIdx = noteToIdx(kp.root);
        const newRoot = idxToNote(rIdx + semitones);
        out.meta.key = newRoot + " " + (kp.modifier || "").replace("m","minor") + (kp.modifier ? "" : "major");
        out.meta.key = out.meta.key.replace("major major","major").replace("minor major","minor");
      }
    }
    return out;
  }

  /** Format a key string nicely: "Am" → "A minor" */
  function keyToDisplay(key) {
    if (!key) return "—";
    const p = parseChord(key);
    if (!p || p.kind !== "parsed") return key;
    if (p.modifier.startsWith("m") && !p.modifier.startsWith("maj")) return p.root + " minor";
    return p.root + " major";
  }

  /** Unique chord names used in a song (in order) */
  function uniqueChords(song) {
    const set = new Set();
    (song.lines || []).forEach(l => (l.chords || []).forEach(c => set.add(c.chord)));
    (song.bars || []).forEach(b => (b.chords || []).forEach(c => set.add(c.chord)));
    return Array.from(set);
  }

  /** Generate a "Capo N → X" suggestion from original key + capo */
  function capoHint(key, capo) {
    if (!capo) return null;
    const p = parseChord(key);
    if (!p || p.kind !== "parsed") return null;
    const shape = transposeChord(key, -capo);
    return `Capo ${capo} → main pakai shape ${shape}`;
  }

  window.KC = window.KC || {};
  Object.assign(window.KC, { parseChord, transposeChord, transposeRender, keyToDisplay, uniqueChords, capoHint, noteToIdx, idxToNote });
})();
