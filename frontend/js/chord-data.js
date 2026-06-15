/* =================================================================
   KENSHICHORD — CHORD DATA
   Chord fingering untuk gitar, ukulele, piano.
   Format: { frets: [E,A,D,G,B,E] (gitar) | [G,C,E,A] (ukulele) | null piano, fingers: [..], baseFret: 1, barres: [] }
   "x" = muted, "0" = open, integer = fret, null = same string
   ================================================================= */

const NOTES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const NOTES_FLAT  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

/* ============== GUITAR ============== */
const GUITAR_CHORDS = {
  // Major (open positions)
  "C":  { frets: ["x", 3, 2, 0, 1, 0], fingers: [0,3,2,0,1,0], baseFret: 1 },
  "D":  { frets: ["x", "x", 0, 2, 3, 2], fingers: [0,0,0,1,3,2], baseFret: 1 },
  "E":  { frets: [0, 2, 2, 1, 0, 0], fingers: [0,2,3,1,0,0], baseFret: 1 },
  "F":  { frets: [1, 3, 3, 2, 1, 1], fingers: [1,3,4,2,1,1], baseFret: 1, barres: [{fret:1, from:0, to:5}] },
  "G":  { frets: [3, 2, 0, 0, 0, 3], fingers: [2,1,0,0,0,3], baseFret: 1 },
  "A":  { frets: ["x", 0, 2, 2, 2, 0], fingers: [0,0,1,2,3,0], baseFret: 1 },
  "B":  { frets: ["x", 2, 4, 4, 4, 2], fingers: [0,1,2,3,4,1], baseFret: 2, barres: [{fret:2, from:1, to:5}] },

  // Minor
  "Cm": { frets: ["x", 3, 5, 5, 4, 3], fingers: [0,1,3,4,2,1], baseFret: 3, barres:[{fret:3, from:1, to:5}] },
  "Dm": { frets: ["x", "x", 0, 2, 3, 1], fingers: [0,0,0,2,3,1], baseFret: 1 },
  "Em": { frets: [0, 2, 2, 0, 0, 0], fingers: [0,2,3,0,0,0], baseFret: 1 },
  "Fm": { frets: [1, 3, 3, 1, 1, 1], fingers: [1,3,4,1,1,1], baseFret: 1, barres:[{fret:1, from:0, to:5}] },
  "Gm": { frets: [3, 5, 5, 3, 3, 3], fingers: [1,3,4,1,1,1], baseFret: 3, barres:[{fret:3, from:0, to:5}] },
  "Am": { frets: ["x", 0, 2, 2, 1, 0], fingers: [0,0,2,3,1,0], baseFret: 1 },
  "Bm": { frets: ["x", 2, 4, 4, 3, 2], fingers: [0,1,3,4,2,1], baseFret: 2, barres:[{fret:2, from:1, to:5}] },

  // 7
  "C7": { frets: ["x", 3, 2, 3, 1, 0], fingers: [0,3,2,4,1,0], baseFret: 1 },
  "D7": { frets: ["x", "x", 0, 2, 1, 2], fingers: [0,0,0,2,1,3], baseFret: 1 },
  "E7": { frets: [0, 2, 0, 1, 0, 0], fingers: [0,2,0,1,0,0], baseFret: 1 },
  "F7":  { frets: [1, 3, 1, 2, 1, 1], fingers: [1,3,1,2,1,1], baseFret: 1, barres:[{fret:1, from:0, to:5}] },
  "G7": { frets: [3, 2, 0, 0, 0, 1], fingers: [3,2,0,0,0,1], baseFret: 1 },
  "A7": { frets: ["x", 0, 2, 0, 2, 0], fingers: [0,0,2,0,3,0], baseFret: 1 },
  "B7": { frets: ["x", 2, 1, 2, 0, 2], fingers: [0,2,1,3,0,4], baseFret: 1 },

  // maj7
  "Cmaj7": { frets: ["x", 3, 2, 0, 0, 0], fingers: [0,3,2,0,0,0], baseFret: 1 },
  "Dmaj7": { frets: ["x", "x", 0, 2, 2, 2], fingers: [0,0,0,1,1,1], baseFret: 1 },
  "Em7":   { frets: [0, 2, 0, 0, 0, 0], fingers: [0,2,0,0,0,0], baseFret: 1 },
  "Am7":   { frets: ["x", 0, 2, 0, 1, 0], fingers: [0,0,2,0,1,0], baseFret: 1 },

  // Sus
  "Dsus2": { frets: ["x", "x", 0, 2, 3, 0], fingers: [0,0,0,1,2,0], baseFret: 1 },
  "Dsus4": { frets: ["x", "x", 0, 2, 3, 3], fingers: [0,0,0,1,2,3], baseFret: 1 },

  // Sharp keys (commonly used)
  "F#m": { frets: [2, 4, 4, 2, 2, 2], fingers: [1,3,4,1,1,1], baseFret: 2, barres:[{fret:2, from:0, to:5}] },
  "C#m": { frets: ["x", 4, 6, 6, 5, 4], fingers: [0,1,3,4,2,1], baseFret: 4, barres:[{fret:4, from:1, to:5}] },
  "G#m": { frets: [4, 6, 6, 4, 4, 4], fingers: [1,3,4,1,1,1], baseFret: 4, barres:[{fret:4, from:0, to:5}] },
  "Bbm": { frets: ["x", 1, 3, 3, 2, 1], fingers: [0,1,3,4,2,1], baseFret: 1, barres:[{fret:1, from:1, to:5}] }
};

/* ============== UKULELE (G-C-E-A) ============== */
const UKULELE_CHORDS = {
  "C":  { frets: [0, 0, 0, 3], fingers: [0,0,0,3], baseFret: 1 },
  "D":  { frets: [2, 2, 0, 0], fingers: [1,2,0,0], baseFret: 1 },
  "E":  { frets: [1, 4, 0, 2], fingers: [1,4,0,2], baseFret: 1 },
  "F":  { frets: [2, 0, 1, 0], fingers: [2,0,1,0], baseFret: 1 },
  "G":  { frets: [0, 2, 3, 2], fingers: [0,1,3,2], baseFret: 1 },
  "A":  { frets: [2, 1, 0, 0], fingers: [2,1,0,0], baseFret: 1 },
  "B":  { frets: [2, 4, 4, 4], fingers: [1,2,3,4], baseFret: 4 },
  "Cm": { frets: [0, 3, 3, 3], fingers: [0,1,2,3], baseFret: 3 },
  "Dm": { frets: [2, 2, 1, 0], fingers: [2,3,1,0], baseFret: 1 },
  "Em": { frets: [0, 4, 3, 2], fingers: [0,4,3,2], baseFret: 1 },
  "Am": { frets: [2, 0, 0, 0], fingers: [1,0,0,0], baseFret: 1 },
  "G7": { frets: [0, 2, 1, 2], fingers: [0,2,1,3], baseFret: 1 },
  "C7": { frets: [0, 0, 0, 1], fingers: [0,0,0,1], baseFret: 1 },
  "F7": { frets: [2, 3, 1, 0], fingers: [2,3,1,0], baseFret: 1 },
  "E7": { frets: [1, 2, 0, 2], fingers: [1,2,0,3], baseFret: 1 },
  "D7": { frets: [2, 2, 0, 2], fingers: [1,2,0,3], baseFret: 1 },
  "A7": { frets: [0, 1, 0, 0], fingers: [0,1,0,0], baseFret: 1 }
};

/* ============== PIANO (note names only) ============== */
const PIANO_KEYS = {
  white: ["C", "D", "E", "F", "G", "A", "B"],
  black: { "C#":1, "D#":3, "F#":6, "G#":8, "A#":10 }
};

/** Lookup chord shape in an instrument; transpose to find enharmonic */
function lookupChord(name, instrument) {
  const lib = instrument === "ukulele" ? UKULELE_CHORDS : GUITAR_CHORDS;
  if (lib[name]) return { name, ...lib[name] };
  // Try enharmonic
  const norm = normalizeChordName(name);
  for (const k of Object.keys(lib)) {
    if (normalizeChordName(k) === norm) return { name: k, ...lib[k] };
  }
  return null;
}

function normalizeChordName(name) {
  if (!name) return "";
  return name.replace("♯","#").replace("♭","b").toLowerCase();
}

window.KC = window.KC || {};
Object.assign(window.KC, {
  GUITAR_CHORDS, UKULELE_CHORDS, PIANO_KEYS, NOTES_SHARP, NOTES_FLAT,
  lookupChord, normalizeChordName
});
