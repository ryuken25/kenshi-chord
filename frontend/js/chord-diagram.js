/* =================================================================
   KENSHICHORD — CHORD DIAGRAM
   Renderer SVG untuk gitar / ukulele / piano.
   ================================================================= */

(function () {
  "use strict";

  /** Render a guitar or ukulele fretboard diagram */
  function renderFretboard(chordName, instrument) {
    const lookup = window.KC.lookupChord(chordName, instrument);
    if (!lookup) {
      return `<div style="color: var(--text-muted); text-align: center; padding: 20px; font-size: 12px;">
        Diagram untuk <span class="mono" style="color: var(--gold)">${escapeHtml(chordName || "—")}</span> belum tersedia.<br/>
        <span style="font-size: 11px;">(Coba transpose atau ganti instrumen)</span>
      </div>`;
    }

    const { frets, fingers, baseFret, barres } = lookup;
    const isUke = instrument === "ukulele";
    const numStrings = isUke ? 4 : 6;
    const numFrets = 4;

    const width = 160, height = 200;
    const padX = 24, padY = 50;
    const stringSpacing = (width - padX * 2) / (numStrings - 1);
    const fretSpacing = (height - padY - 24) / numFrets;

    const dots = [];
    const fingerDots = [];
    const barreEls = [];
    const stringMutes = [];

    // Mute / open markers above nut
    for (let s = 0; s < numStrings; s++) {
      const x = padX + s * stringSpacing;
      const f = frets[s];
      if (f === "x" || f === null) {
        stringMutes.push(`<text x="${x}" y="34" text-anchor="middle" font-size="14" font-weight="700" fill="#8A8A93">✕</text>`);
      } else if (f === 0) {
        stringMutes.push(`<circle cx="${x}" cy="28" r="6" fill="none" stroke="#F4F1EA" stroke-width="1.2"/>`);
      } else {
        const fretOffset = f - baseFret;
        if (fretOffset >= 1 && fretOffset <= numFrets) {
          const cy = padY + (fretOffset - 0.5) * fretSpacing;
          dots.push(`<circle cx="${x}" cy="${cy}" r="9" fill="url(#dotGrad)" stroke="#D4AF37" stroke-width="1.4"/>`);
          if (fingers && fingers[s]) {
            fingerDots.push(`<text x="${x}" y="${cy + 4}" text-anchor="middle" font-size="10" font-weight="700" fill="#0B0B0D">${fingers[s]}</text>`);
          }
        }
      }
    }

    // Barres
    if (barres) {
      barres.forEach(b => {
        const offset = b.fret - baseFret;
        if (offset >= 1 && offset <= numFrets) {
          const cy = padY + (offset - 0.5) * fretSpacing;
          const x1 = padX + b.from * stringSpacing;
          const x2 = padX + b.to * stringSpacing;
          barreEls.push(`<rect x="${x1}" y="${cy - 7}" width="${x2 - x1}" height="14" rx="7" fill="url(#barreGrad)" opacity=".95"/>`);
        }
      });
    }

    // Fret lines
    const fretLines = [];
    for (let f = 0; f <= numFrets; f++) {
      const y = padY + f * fretSpacing;
      const isNut = f === 0;
      fretLines.push(`<line x1="${padX}" y1="${y}" x2="${padX + (numStrings - 1) * stringSpacing}" y2="${y}" stroke="${isNut ? '#D4AF37' : '#5A5A65'}" stroke-width="${isNut ? 3 : 1}"/>`);
    }
    // String lines
    const stringLines = [];
    for (let s = 0; s < numStrings; s++) {
      const x = padX + s * stringSpacing;
      stringLines.push(`<line x1="${x}" y1="${padY}" x2="${x}" y2="${padY + numFrets * fretSpacing}" stroke="#8A8A93" stroke-width="1"/>`);
    }

    const baseFretLabel = baseFret > 1 ? `<text x="${width - 8}" y="${padY + 14}" text-anchor="end" font-size="10" fill="#8A8A93" font-family="JetBrains Mono">${baseFret}fr</text>` : "";

    return `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" style="display:block; max-width: 100%; height: auto;">
      <defs>
        <radialGradient id="dotGrad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#F2C94C"/>
          <stop offset="100%" stop-color="#D4AF37"/>
        </radialGradient>
        <linearGradient id="barreGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#F2C94C"/>
          <stop offset="100%" stop-color="#D4AF37"/>
        </linearGradient>
      </defs>
      ${stringMutes.join("")}
      ${fretLines.join("")}
      ${stringLines.join("")}
      ${baseFretLabel}
      ${barreEls.join("")}
      ${dots.join("")}
      ${fingerDots.join("")}
      <text x="8" y="${padY + numFrets * fretSpacing + 18}" font-size="9" fill="#5A5A65" font-family="Inter" letter-spacing="1.5">${isUke ? "UKULELE" : "GUITAR"}</text>
    </svg>`;
  }

  /** Render piano keyboard chord */
  function renderPianoChord(chordName) {
    const p = window.KC.parseChord(chordName);
    if (!p || p.kind !== "parsed") {
      return `<div style="color: var(--text-muted); text-align: center; padding: 20px; font-size: 12px;">
        Diagram piano untuk <span class="mono" style="color: var(--gold)">${escapeHtml(chordName || "—")}</span> belum tersedia.
      </div>`;
    }
    const rootIdx = window.KC.noteToIdx(p.root);
    // Build notes: root, +3 for min, +4 for maj, +7, +modifier bonus
    const notes = [rootIdx];
    if (p.modifier.startsWith("m") && !p.modifier.startsWith("maj")) {
      notes.push(rootIdx + 3);
    } else {
      notes.push(rootIdx + 4);
    }
    notes.push(rootIdx + 7);
    if (p.modifier.includes("7")) notes.push(rootIdx + (p.modifier.startsWith("maj") ? 11 : 10));
    if (p.bass) {
      const bIdx = window.KC.noteToIdx(p.bass);
      if (bIdx !== null) notes.push(bIdx);
    }

    const width = 220, height = 130;
    const whiteKeys = ["C", "D", "E", "F", "G", "A", "B"];
    const whiteW = width / 14; // 2 octaves
    const whiteH = height - 30;
    const blackW = whiteW * 0.6;
    const blackH = whiteH * 0.6;

    const whiteNotes = [];
    for (let o = 0; o < 2; o++) {
      whiteKeys.forEach((k, i) => {
        const x = (o * 7 + i) * whiteW;
        const num = window.KC.noteToIdx(k) + o * 12;
        const isOn = notes.some(n => n === num);
        whiteNotes.push(`<rect x="${x}" y="0" width="${whiteW}" height="${whiteH}" fill="${isOn ? 'url(#pianoKeyOn)' : '#F4F1EA'}" stroke="#0B0B0D" stroke-width="1"/>`);
        if (isOn) whiteNotes.push(`<text x="${x + whiteW / 2}" y="${whiteH - 8}" text-anchor="middle" font-size="8" font-weight="700" fill="#0B0B0D">${k}${o === 1 ? "2" : ""}</text>`);
      });
    }

    const blackNotes = [];
    const blackOffsets = { "C#": 0, "D#": 1, "F#": 3, "G#": 4, "A#": 5 };
    for (let o = 0; o < 2; o++) {
      Object.entries(blackOffsets).forEach(([k, idx]) => {
        const num = window.KC.noteToIdx(k) + o * 12;
        const isOn = notes.some(n => n === num);
        const x = (o * 7 + idx) * whiteW + whiteW - blackW / 2;
        blackNotes.push(`<rect x="${x}" y="0" width="${blackW}" height="${blackH}" fill="${isOn ? 'url(#pianoBlackOn)' : '#16161A'}" stroke="#0B0B0D" stroke-width="1" rx="2"/>`);
        if (isOn) blackNotes.push(`<text x="${x + blackW / 2}" y="${blackH - 6}" text-anchor="middle" font-size="7" font-weight="700" fill="#F4F1EA">${k}</text>`);
      });
    }

    return `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" style="display:block; max-width: 100%;">
      <defs>
        <linearGradient id="pianoKeyOn" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#F2C94C"/>
          <stop offset="100%" stop-color="#D4AF37"/>
        </linearGradient>
        <linearGradient id="pianoBlackOn" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#E01E2B"/>
          <stop offset="100%" stop-color="#7A0E14"/>
        </linearGradient>
      </defs>
      ${whiteNotes.join("")}
      ${blackNotes.join("")}
      <text x="8" y="${height - 6}" font-size="9" fill="#5A5A65" font-family="Inter" letter-spacing="1.5">PIANO</text>
    </svg>`;
  }

  function render(chordName, instrument) {
    if (instrument === "piano") return renderPianoChord(chordName);
    return renderFretboard(chordName, instrument);
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
  }

  window.KC = window.KC || {};
  Object.assign(window.KC, { renderChordDiagram: render });
})();
