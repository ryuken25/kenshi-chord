/* =================================================================
   KENSHICHORD — MOCK DATA
   Sample render_json structures untuk FASE 0 (mock mode).
   Data: 2 lagu Jepang public-domain + 1 lagu Indonesia fiktif.
   Bentuk persis seperti APPENDIX A di megaprompt.
   ================================================================= */

const MOCK_SONGS = {
  soranBushi: {
    meta: {
      youtube_id: "mockSoranBushi01",
      artist: "Traditional",
      title: "Soran Bushi",
      duration_sec: 78,
      bpm: 132,
      key: "G major",
      capo: 0,
      time_sig: "4/4",
      language: "ja"
    },
    beats:  [0.00, 0.45, 0.91, 1.36, 1.82, 2.27, 2.73, 3.18, 3.64, 4.09, 4.55, 5.00, 5.45, 5.91, 6.36, 6.82, 7.27, 7.73, 8.18, 8.64, 9.09, 9.55, 10.00, 10.45, 10.91, 11.36, 11.82, 12.27, 12.73, 13.18, 13.64, 14.09, 14.55, 15.00],
    downbeats: [0.00, 1.82, 3.64, 5.45, 7.27, 9.09, 10.91, 12.73, 14.55],
    sections: [
      { name: "Intro",        start: 0.0,  end: 8.0,  has_lyrics: false },
      { name: "Verse 1",      start: 8.0,  end: 24.0, has_lyrics: true },
      { name: "Interlude",    start: 24.0, end: 30.0, has_lyrics: false },
      { name: "Verse 2",      start: 30.0, end: 46.0, has_lyrics: true },
      { name: "Interlude 2",  start: 46.0, end: 54.0, has_lyrics: false },
      { name: "Verse 3",      start: 54.0, end: 70.0, has_lyrics: true },
      { name: "Outro",        start: 70.0, end: 78.0, has_lyrics: false }
    ],
    bars: [
      { index: 0, start: 0.00, end: 1.82, chords: [{ chord: "Em", start: 0.00, end: 1.82 }] },
      { index: 1, start: 1.82, end: 3.64, chords: [{ chord: "C",  start: 1.82, end: 3.64 }] },
      { index: 2, start: 3.64, end: 5.45, chords: [{ chord: "G",  start: 3.64, end: 5.45 }] },
      { index: 3, start: 5.45, end: 7.27, chords: [{ chord: "D",  start: 5.45, end: 7.27 }] }
    ],
    lines: [
      // Verse 1
      {
        line_index: 0, start: 8.5, end: 12.0,
        text: "どど どど いがっぺ いがんない",
        words: [
          { word: "どど", start: 8.5,  end: 9.0 },
          { word: "どど", start: 9.1,  end: 9.6 },
          { word: "いがっぺ", start: 9.7,  end: 10.6 },
          { word: "いがんない", start: 10.7, end: 12.0 }
        ],
        chords: [
          { chord: "Em", start: 8.5,  anchor_word_index: 0 },
          { chord: "C",  start: 9.7,  anchor_word_index: 2 },
          { chord: "G",  start: 10.7, anchor_word_index: 3 }
        ]
      },
      {
        line_index: 1, start: 12.3, end: 16.1,
        text: "おら ほろ ほろ ほろいわ",
        words: [
          { word: "おら",   start: 12.3, end: 12.8 },
          { word: "ほろ",   start: 12.9, end: 13.4 },
          { word: "ほろ",   start: 13.5, end: 14.0 },
          { word: "ほろいわ", start: 14.1, end: 16.1 }
        ],
        chords: [
          { chord: "D",  start: 12.3, anchor_word_index: 0 },
          { chord: "G",  start: 14.1, anchor_word_index: 3 },
          { chord: "Em", start: 15.4, anchor_word_index: 3 }
        ]
      },
      {
        line_index: 2, start: 16.5, end: 20.3,
        text: "いがんない べ いのぼる",
        words: [
          { word: "いがんない", start: 16.5, end: 17.7 },
          { word: "べ",        start: 17.8, end: 18.2 },
          { word: "いのぼる",   start: 18.3, end: 20.3 }
        ],
        chords: [
          { chord: "C",  start: 16.5, anchor_word_index: 0 },
          { chord: "G",  start: 18.3, anchor_word_index: 2 },
          { chord: "D",  start: 19.5, anchor_word_index: 2 }
        ]
      },
      // Verse 2
      {
        line_index: 3, start: 30.5, end: 34.1,
        text: "ま つ だ い て つ ち え す",
        words: [
          { word: "ま",  start: 30.5, end: 30.9 },
          { word: "つ",  start: 31.0, end: 31.4 },
          { word: "だい", start: 31.5, end: 32.0 },
          { word: "て",  start: 32.1, end: 32.5 },
          { word: "つち", start: 32.6, end: 33.1 },
          { word: "えす", start: 33.2, end: 34.1 }
        ],
        chords: [
          { chord: "Em", start: 30.5, anchor_word_index: 0 },
          { chord: "C",  start: 31.5, anchor_word_index: 2 },
          { chord: "G",  start: 32.6, anchor_word_index: 4 }
        ]
      },
      {
        line_index: 4, start: 34.5, end: 38.2,
        text: "ま つ だ い ま た は ら す",
        words: [
          { word: "ま",  start: 34.5, end: 34.9 },
          { word: "つ",  start: 35.0, end: 35.4 },
          { word: "だい", start: 35.5, end: 36.0 },
          { word: "ま",  start: 36.1, end: 36.5 },
          { word: "たはらす", start: 36.6, end: 38.2 }
        ],
        chords: [
          { chord: "D",  start: 34.5, anchor_word_index: 0 },
          { chord: "G",  start: 36.6, anchor_word_index: 4 }
        ]
      },
      {
        line_index: 5, start: 38.5, end: 42.2,
        text: "は ら す ち に ま つ だ",
        words: [
          { word: "はらす", start: 38.5, end: 39.4 },
          { word: "ちに",   start: 39.5, end: 40.0 },
          { word: "まつだ", start: 40.1, end: 42.2 }
        ],
        chords: [
          { chord: "Em", start: 38.5, anchor_word_index: 0 },
          { chord: "C",  start: 39.5, anchor_word_index: 1 },
          { chord: "G",  start: 40.1, anchor_word_index: 2 }
        ]
      },
      // Verse 3
      {
        line_index: 6, start: 54.5, end: 58.1,
        text: "こ ろ も え が き な ま",
        words: [
          { word: "ころも", start: 54.5, end: 55.4 },
          { word: "えが",   start: 55.5, end: 56.0 },
          { word: "きなま", start: 56.1, end: 58.1 }
        ],
        chords: [
          { chord: "Em", start: 54.5, anchor_word_index: 0 },
          { chord: "C",  start: 55.5, anchor_word_index: 1 },
          { chord: "G",  start: 56.1, anchor_word_index: 2 }
        ]
      },
      {
        line_index: 7, start: 58.4, end: 62.2,
        text: "き な ま の ま ん ま が",
        words: [
          { word: "きなま", start: 58.4, end: 59.1 },
          { word: "の",     start: 59.2, end: 59.5 },
          { word: "まんま", start: 59.6, end: 60.3 },
          { word: "が",     start: 60.4, end: 62.2 }
        ],
        chords: [
          { chord: "D",  start: 58.4, anchor_word_index: 0 },
          { chord: "G",  start: 60.4, anchor_word_index: 3 }
        ]
      },
      {
        line_index: 8, start: 62.5, end: 66.0,
        text: "き な ま の ま ん ま が",
        words: [
          { word: "きなま", start: 62.5, end: 63.2 },
          { word: "の",     start: 63.3, end: 63.6 },
          { word: "まんま", start: 63.7, end: 64.4 },
          { word: "が",     start: 64.5, end: 66.0 }
        ],
        chords: [
          { chord: "Em", start: 62.5, anchor_word_index: 0 },
          { chord: "C",  start: 64.5, anchor_word_index: 3 }
        ]
      },
      {
        line_index: 9, start: 66.4, end: 70.0,
        text: "だ い す き な ま だ よ",
        words: [
          { word: "だいすき", start: 66.4, end: 67.5 },
          { word: "なま",     start: 67.6, end: 68.1 },
          { word: "だよ",     start: 68.2, end: 70.0 }
        ],
        chords: [
          { chord: "G",  start: 66.4, anchor_word_index: 0 },
          { chord: "D",  start: 67.6, anchor_word_index: 1 }
        ]
      }
    ]
  },

  sakuraSakura: {
    meta: {
      youtube_id: "mockSakura01",
      artist: "Folk Jepang",
      title: "Sakura Sakura",
      duration_sec: 52,
      bpm: 88,
      key: "C major",
      capo: 0,
      time_sig: "4/4",
      language: "ja"
    },
    beats:  [0.00, 0.68, 1.36, 2.05, 2.73, 3.41, 4.09, 4.77, 5.45, 6.13, 6.81, 7.50, 8.18, 8.86, 9.54, 10.22, 10.90, 11.59, 12.27, 12.95, 13.63, 14.31, 15.00],
    downbeats: [0.00, 2.73, 5.45, 8.18, 10.90, 13.63],
    sections: [
      { name: "Intro",    start: 0.0,  end: 6.0, has_lyrics: false },
      { name: "Verse",    start: 6.0,  end: 26.0, has_lyrics: true },
      { name: "Interlude",start: 26.0, end: 32.0, has_lyrics: false },
      { name: "Verse 2",  start: 32.0, end: 48.0, has_lyrics: true },
      { name: "Outro",    start: 48.0, end: 52.0, has_lyrics: false }
    ],
    bars: [
      { index: 0, start: 0.00, end: 2.73, chords: [{ chord: "C", start: 0.00, end: 2.73 }] },
      { index: 1, start: 2.73, end: 5.45, chords: [{ chord: "G7", start: 2.73, end: 5.45 }] },
      { index: 2, start: 5.45, end: 8.18, chords: [{ chord: "F", start: 5.45, end: 8.18 }] },
      { index: 3, start: 8.18, end: 10.90, chords: [{ chord: "C", start: 8.18, end: 10.90 }] }
    ],
    lines: [
      {
        line_index: 0, start: 6.5, end: 12.0,
        text: "さ く ら さ く ら",
        words: [
          { word: "さくら", start: 6.5, end: 7.4 },
          { word: "さくら", start: 7.5, end: 8.4 }
        ],
        chords: [
          { chord: "C", start: 6.5, anchor_word_index: 0 }
        ]
      },
      {
        line_index: 1, start: 12.4, end: 18.0,
        text: "の や ま も と さ と も",
        words: [
          { word: "の",   start: 12.4, end: 12.6 },
          { word: "やま", start: 12.7, end: 13.5 },
          { word: "もと", start: 13.6, end: 14.2 },
          { word: "さと", start: 14.3, end: 14.9 },
          { word: "も",   start: 15.0, end: 18.0 }
        ],
        chords: [
          { chord: "G7", start: 12.4, anchor_word_index: 0 },
          { chord: "F",  start: 14.3, anchor_word_index: 3 }
        ]
      },
      {
        line_index: 2, start: 18.4, end: 24.0,
        text: "み ず う き の か な",
        words: [
          { word: "みずうき", start: 18.4, end: 19.4 },
          { word: "の",       start: 19.5, end: 19.7 },
          { word: "かな",     start: 19.8, end: 24.0 }
        ],
        chords: [
          { chord: "C",  start: 18.4, anchor_word_index: 0 },
          { chord: "G7", start: 19.8, anchor_word_index: 2 }
        ]
      },
      // Verse 2
      {
        line_index: 3, start: 32.5, end: 38.0,
        text: "さ く ら さ く ら",
        words: [
          { word: "さくら", start: 32.5, end: 33.4 },
          { word: "さくら", start: 33.5, end: 34.4 }
        ],
        chords: [
          { chord: "F", start: 32.5, anchor_word_index: 0 }
        ]
      },
      {
        line_index: 4, start: 38.4, end: 44.0,
        text: "は な の か お り",
        words: [
          { word: "はな",   start: 38.4, end: 39.1 },
          { word: "の",     start: 39.2, end: 39.4 },
          { word: "かおり", start: 39.5, end: 44.0 }
        ],
        chords: [
          { chord: "C",  start: 38.4, anchor_word_index: 0 },
          { chord: "G7", start: 39.5, anchor_word_index: 2 }
        ]
      },
      {
        line_index: 5, start: 44.4, end: 48.0,
        text: "お う か の か ぜ",
        words: [
          { word: "おうか", start: 44.4, end: 45.2 },
          { word: "の",     start: 45.3, end: 45.5 },
          { word: "かぜ",   start: 45.6, end: 48.0 }
        ],
        chords: [
          { chord: "F",  start: 44.4, anchor_word_index: 0 },
          { chord: "C",  start: 45.6, anchor_word_index: 2 }
        ]
      }
    ]
  }
};

// === Mock data extended: tambah romaji per-kata & per-baris
// Supaya user non-Jepang bisa ikut nyanyi.

// Mapping hiragana/katakana → romaji (lookup sederhana)
const ROMAJI_MAP = {
  // Hiragana
  "あ":"a","い":"i","う":"u","え":"e","お":"o",
  "か":"ka","き":"ki","く":"ku","け":"ke","こ":"ko",
  "が":"ga","ぎ":"gi","ぐ":"gu","げ":"ge","ご":"go",
  "さ":"sa","し":"shi","す":"su","せ":"se","そ":"so",
  "ざ":"za","じ":"ji","ず":"zu","ぜ":"ze","ぞ":"zo",
  "た":"ta","ち":"chi","つ":"tsu","て":"te","と":"to",
  "だ":"da","ぢ":"ji","づ":"zu","で":"de","ど":"do",
  "な":"na","に":"ni","ぬ":"nu","ね":"ne","の":"no",
  "は":"ha","ひ":"hi","ふ":"fu","へ":"he","ほ":"ho",
  "ば":"ba","び":"bi","ぶ":"bu","べ":"be","ぼ":"bo",
  "ぱ":"pa","ぴ":"pi","ぷ":"pu","ぺ":"pe","ぽ":"po",
  "ま":"ma","み":"mi","む":"mu","め":"me","も":"mo",
  "や":"ya","ゆ":"yu","よ":"yo",
  "ら":"ra","り":"ri","る":"ru","れ":"re","ろ":"ro",
  "わ":"wa","ゐ":"wi","ゑ":"we","を":"wo",
  "ん":"n",
  "ゔ":"vu",
  // Dakuten/handakuten combos (less common ones)
  "きゃ":"kya","きゅ":"kyu","きょ":"kyo",
  "しゃ":"sha","しゅ":"shu","しょ":"sho",
  "ちゃ":"cha","ちゅ":"chu","ちょ":"cho",
  "にゃ":"nya","にゅ":"nyu","にょ":"nyo",
  "ひゃ":"hya","ひゅ":"hyu","ひょ":"hyo",
  "みゃ":"mya","みゅ":"myu","みょ":"myo",
  "りゃ":"rya","りゅ":"ryu","りょ":"ryo",
  "ぎゃ":"gya","ぎゅ":"gyu","ぎょ":"gyo",
  "じゃ":"ja","じゅ":"ju","じょ":"jo",
  "びゃ":"bya","びゅ":"byu","びょ":"byo",
  "ぴゃ":"pya","ぴゅ":"pyu","ぴょ":"pyo"
};

// Heuristic romaji converter (kasar tapi cukup untuk mock/learning).
// Approach: longest-match greedy dari ROMAJI_MAP; sisa karakter → approximate
// vokal. Hasilnya dipakai hanya untuk display, BUKAN data pipeline.
function kanaToRomaji(str) {
  if (!str) return "";
  const out = [];
  let i = 0;
  const s = Array.from(str); // Unicode-safe
  while (i < s.length) {
    // Try 2-char combo first
    const two = s[i] + (s[i+1] || "");
    if (ROMAJI_MAP[two]) {
      out.push(ROMAJI_MAP[two]);
      i += 2;
      continue;
    }
    if (ROMAJI_MAP[s[i]]) {
      out.push(ROMAJI_MAP[s[i]]);
    } else {
      // Katakana fallback (sama ke romaji dengan mapping yang sama)
      const k = s[i];
      const fromKana = {"ア":"a","イ":"i","ウ":"u","エ":"e","オ":"o",
        "カ":"ka","キ":"ki","ク":"ku","ケ":"ke","コ":"ko",
        "サ":"sa","シ":"shi","ス":"su","セ":"se","ソ":"so",
        "タ":"ta","チ":"chi","ツ":"tsu","テ":"te","ト":"to",
        "ナ":"na","ニ":"ni","ヌ":"nu","ネ":"ne","ノ":"no",
        "ハ":"ha","ヒ":"hi","フ":"fu","ヘ":"he","ホ":"ho",
        "マ":"ma","ミ":"mi","ム":"mu","メ":"me","モ":"mo",
        "ヤ":"ya","ユ":"yu","ヨ":"yo",
        "ラ":"ra","リ":"ri","ル":"ru","レ":"re","ロ":"ro",
        "ワ":"wa","ヲ":"wo","ン":"n"};
      out.push(fromKana[k] || k);
    }
    i++;
  }
  // Tambah hyphen untuk keterbacaan (sakura-no → "sakura no")
  return out.join("").replace(/([aeiou])(n)(?=[bcdfghjklmnpqrstvwz])/g, "$1n ");
}

/** Inject romaji ke sebuah song (in-place). */
function injectRomaji(song) {
  if (!song || !song.lines) return song;
  song.lines.forEach((line) => {
    let romajiText = "";
    (line.words || []).forEach((w) => {
      w.romaji = kanaToRomaji(w.word);
      romajiText += (romajiText && !romajiText.endsWith(" ") ? " " : "") + w.romaji;
    });
    line.romaji_text = romajiText.trim();
  });
  return song;
}
injectRomaji(MOCK_SONGS.soranBushi);
injectRomaji(MOCK_SONGS.sakuraSakura);

// Override beberapa romaji yang kasar heuristic-nya (override manual untuk
// frasa khas lagu ini biar natural)
(function overrideRomaji() {
  const ovr = {
    soranBushi: {
      "いがっぺ": "igappe",
      "いがんない": "igannai",
      "ほろいわ": "horoiwa",
      "いのぼる": "inoboru",
      "えす": "esu",
      "たはらす": "taharasu",
      "はらす": "harasu",
      "まつだ": "matsuda",
      "ころも": "koromo",
      "えが": "ega",
      "きなま": "kinama",
      "まんま": "manma",
      "だいすき": "daisuki",
      "だよ": "dayo"
    },
    sakuraSakura: {
      "さくら": "sakura",
      "やま": "yama",
      "もと": "moto",
      "さと": "sato",
      "みずうき": "mizuki",
      "かな": "kana",
      "はな": "hana",
      "かおり": "kaori",
      "おうか": "ouka",
      "かぜ": "kaze"
    }
  };
  for (const key in ovr) {
    const lines = MOCK_SONGS[key].lines;
    lines.forEach((line) => {
      (line.words || []).forEach((w) => {
        if (ovr[key][w.word]) {
          w.romaji = ovr[key][w.word];
        }
      });
      // rebuild romaji_text
      line.romaji_text = line.words.map(w => w.romaji).join(" ");
    });
  }
})();

const MOCK_LIBRARY = [
  {
    id: 1, song: MOCK_SONGS.soranBushi, thumbnail: null,
    cache_hit: false, processed_at: "2026-06-12T14:20:00Z"
  },
  {
    id: 2, song: MOCK_SONGS.sakuraSakura, thumbnail: null,
    cache_hit: true, processed_at: "2026-06-11T09:14:00Z"
  },
  {
    id: 3, song: {
      meta: {
        youtube_id: "mockRindu01", artist: "Rumah Sasendo",
        title: "Rindu Senja", duration_sec: 215, bpm: 96,
        key: "Am", capo: 3, time_sig: "4/4", language: "id"
      },
      sections: [{name:"Verse 1", start:0, end:30, has_lyrics:true}],
      lines: []
    },
    thumbnail: null, cache_hit: false, processed_at: "2026-06-10T20:00:00Z"
  },
  {
    id: 4, song: {
      meta: {
        youtube_id: "mockEvening01", artist: "Tokyo Lights",
        title: "Evening Glow", duration_sec: 192, bpm: 84,
        key: "G major", capo: 0, time_sig: "4/4", language: "en"
      },
      sections: [], lines: []
    },
    thumbnail: null, cache_hit: true, processed_at: "2026-06-09T18:45:00Z"
  },
  {
    id: 5, song: {
      meta: {
        youtube_id: "mockHana01", artist: "Hana Mori",
        title: "Haru no Hikari", duration_sec: 240, bpm: 100,
        key: "D major", capo: 2, time_sig: "4/4", language: "ja"
      },
      sections: [], lines: []
    },
    thumbnail: null, cache_hit: false, processed_at: "2026-06-08T11:30:00Z"
  },
  {
    id: 6, song: {
      meta: {
        youtube_id: "mockBali01", artist: "Genta Bayu",
        title: "Pulang ke Bali", duration_sec: 198, bpm: 110,
        key: "Dm", capo: 1, time_sig: "4/4", language: "id"
      },
      sections: [], lines: []
    },
    thumbnail: null, cache_hit: false, processed_at: "2026-06-07T15:20:00Z"
  },
  {
    id: 7, song: {
      meta: {
        youtube_id: "mockKaerimichi01", artist: "Folk Jepang",
        title: "帰り道 (Kaerimichi)", duration_sec: 165, bpm: 78,
        key: "C major", capo: 0, time_sig: "3/4", language: "ja"
      },
      sections: [], lines: []
    },
    thumbnail: null, cache_hit: true, processed_at: "2026-06-06T08:00:00Z"
  },
  {
    id: 8, song: {
      meta: {
        youtube_id: "mockRays01", artist: "Bali Sunset Crew",
        title: "Sunset Rays", duration_sec: 220, bpm: 92,
        key: "Em", capo: 0, time_sig: "4/4", language: "en"
      },
      sections: [], lines: []
    },
    thumbnail: null, cache_hit: false, processed_at: "2026-06-05T17:10:00Z"
  }
];

// Helper: get default song for FASE 0
const DEFAULT_SONG_ID = "mockSoranBushi01";
function getMockSong(youtube_id) {
  if (youtube_id === MOCK_SONGS.soranBushi.meta.youtube_id)  return MOCK_SONGS.soranBushi;
  if (youtube_id === MOCK_SONGS.sakuraSakura.meta.youtube_id) return MOCK_SONGS.sakuraSakura;
  // Bug 0.3: don't silently substitute a different song when the user asked
  // for something we don't know. Return null so the caller can show a real
  // "not found" state instead of a wrong song.
  return null;
}

// Thumbnail generator (SVG placeholder, no external requests)
function getSongThumb(seed, title) {
  const palette = [
    ["#1a3a52", "#0B0B0D", "#D4AF37"],
    ["#521a2a", "#0B0B0D", "#C1121F"],
    ["#2a4a1a", "#0B0B0D", "#D4AF37"],
    ["#3a1a52", "#0B0B0D", "#D4AF37"],
    ["#521a3a", "#0B0B0D", "#E01E2B"],
    ["#1a2a4a", "#0B0B0D", "#D4AF37"]
  ];
  const [c1, c2, accent] = palette[seed % palette.length];
  const initial = (title || "♪").charAt(0);
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 320 180'>
    <defs>
      <linearGradient id='g${seed}' x1='0' y1='0' x2='1' y2='1'>
        <stop offset='0%' stop-color='${c1}'/>
        <stop offset='100%' stop-color='${c2}'/>
      </linearGradient>
      <filter id='n${seed}'><feTurbulence baseFrequency='0.7' numOctaves='2'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.18 0'/></filter>
    </defs>
    <rect width='320' height='180' fill='url(#g${seed})'/>
    <rect width='320' height='180' filter='url(#n${seed})' opacity='.4'/>
    <circle cx='240' cy='50' r='40' fill='${accent}' opacity='.5'/>
    <circle cx='240' cy='50' r='55' fill='none' stroke='${accent}' stroke-width='0.6' opacity='.4'/>
    <text x='30' y='130' font-family='Shippori Mincho, serif' font-size='80' font-weight='900' fill='${accent}' opacity='.9'>${initial}</text>
    <text x='30' y='160' font-family='Inter, sans-serif' font-size='10' letter-spacing='2' fill='#F4F1EA' opacity='.7'>KENSHICHORD</text>
  </svg>`;
  return "data:image/svg+xml;utf8," + encodeURIComponent(svg);
}
