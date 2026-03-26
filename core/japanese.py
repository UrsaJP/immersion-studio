"""AIST Japanese NLP: tokenization, furigana, morphology, frequency, grammar, analytics.

Single responsibility: Japanese text analysis — tokenization, morphology,
furigana, frequency ranking, grammar pattern detection, analytics.
Used by core/subtitle.py (detect_language_per_line), core/pipeline.py.

STATUS: extended
DIVERGES_FROM_AIST: True
Changes: added MorphologicalEngine ABC, FugashiTokenizer, RegexTokenizer,
         JapaneseNLP façade (Phase 2 Step 2).  All legacy module-level globals
         and functions preserved unchanged for backward compatibility.
"""

from __future__ import annotations

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: extended
# DIVERGES_FROM_AIST: True
# Changes: see module docstring.
# ─────────────────────────────────────────────────────────────────────────────

import csv
import os
import re
from abc import ABC, abstractmethod

from .config import DEFAULT_GRAMMAR_PATTERNS
from .utils import srt_time_to_ms
from .subtitle import parse_srt


# ══════════════════════════════════════════════════════════════════════════════
# MORPHOLOGICAL ENGINE ABSTRACTION  (Phase 2 additions)
# ══════════════════════════════════════════════════════════════════════════════

# Token dict keys returned by all MorphologicalEngine implementations:
#   surface  (str)  — the original text form as it appears in the input
#   lemma    (str)  — dictionary / base form (食べた → 食べる)
#   pos      (str)  — part-of-speech tag (MeCab style, e.g. "動詞", "名詞")
#   pos2     (str)  — secondary POS category (e.g. "固有名詞")
#   reading  (str)  — kana reading (hiragana), empty string when unavailable

# Japanese unicode ranges used throughout this module
_JP_KANJI_KANA_RE: re.Pattern = re.compile(r"[぀-ヿ一-鿿]")
_JP_KANA_RE:       re.Pattern = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


class MorphologicalEngine(ABC):
    """Abstract base for Japanese morphological analysis engines.

    All implementations must return tokens as ``list[dict]`` where each dict
    contains the keys: ``surface``, ``lemma``, ``pos``, ``pos2``, ``reading``.
    """

    @abstractmethod
    def tokenize(self, text: str) -> list[dict]:
        """Tokenize *text* and return morphological token dicts.

        Args:
            text: Raw Japanese text.

        Returns:
            List of token dicts — see module-level comment for key names.
        """
        ...

    def get_lemmas(self, text: str) -> list[str]:
        """Convenience wrapper: return only the lemma strings for Japanese tokens.

        Filters to tokens whose lemma contains at least one kanji/kana character.

        Args:
            text: Raw Japanese text.

        Returns:
            List of lemma strings (may be empty).
        """
        return [
            t["lemma"]
            for t in self.tokenize(text)
            if _JP_KANJI_KANA_RE.search(t["lemma"])
        ]

    @property
    def name(self) -> str:
        """Human-readable engine identifier."""
        return type(self).__name__


class FugashiTokenizer(MorphologicalEngine):
    """MorphologicalEngine backed by fugashi + unidic-lite (ipadic-style features).

    The tagger is initialised lazily on the first call to ``tokenize()`` so that
    importing this module never raises even when fugashi is not installed.

    Args:
        tagger_args: Extra arguments forwarded to ``fugashi.Tagger``.
    """

    # ipadic feature column indices
    _LEMMA_IDX:   int = 6
    _READING_IDX: int = 7

    def __init__(self, tagger_args: str = "") -> None:
        self._tagger_args = tagger_args
        self._tagger = None  # type: ignore[assignment]

    def _ensure_tagger(self) -> None:
        """Lazy-initialise the fugashi Tagger (idempotent)."""
        if self._tagger is not None:
            return
        import fugashi  # intentional late import — keeps module importable without fugashi
        self._tagger = fugashi.Tagger(self._tagger_args)

    @property
    def name(self) -> str:
        return "fugashi"

    @staticmethod
    def _extract_feature(word) -> tuple[str, str, str, str]:
        """Extract (pos, pos2, lemma, reading) from a fugashi word.

        Handles two dictionary formats transparently:
        * **unidic / unidic-lite** — ``word.feature`` is a namedtuple with
          named fields (``pos1``, ``lemma``, ``pron``, etc.).
        * **ipadic** — ``word.feature`` is a plain comma-separated string.

        Returns:
            Tuple of (pos, pos2, lemma, reading).
        """
        surface = word.surface
        feat = word.feature if hasattr(word, "feature") else None

        # ── unidic namedtuple path ────────────────────────────────────────────
        if feat is not None and not isinstance(feat, str):
            try:
                pos     = getattr(feat, "pos1",  "") or ""
                pos2    = getattr(feat, "pos2",  "") or ""
                lemma   = getattr(feat, "lemma", "") or ""
                reading = getattr(feat, "pron",  "") or getattr(feat, "kana", "") or ""
                if not lemma or lemma == "*":
                    lemma = surface
                return pos, pos2, lemma, reading
            except Exception:
                return "", "", surface, ""

        # ── ipadic string path ────────────────────────────────────────────────
        if isinstance(feat, str):
            parts = feat.split(",")
            pos     = parts[0] if len(parts) > 0 else ""
            pos2    = parts[1] if len(parts) > 1 else ""
            raw_lemma = parts[6] if len(parts) > 6 else ""
            lemma   = raw_lemma if (raw_lemma and raw_lemma != "*") else surface
            reading = parts[7] if len(parts) > 7 else ""
            return pos, pos2, lemma, reading

        return "", "", surface, ""

    def tokenize(self, text: str) -> list[dict]:
        """Tokenize with fugashi/MeCab.  Falls back to empty list on any error.

        Args:
            text: Raw Japanese text.

        Returns:
            List of token dicts.  Only tokens with a kanji/kana surface are
            included; punctuation and ASCII tokens are dropped.
        """
        self._ensure_tagger()
        tokens: list[dict] = []
        try:
            for word in self._tagger(text):
                pos, pos2, lemma, reading = self._extract_feature(word)
                tokens.append({
                    "surface": word.surface,
                    "lemma":   lemma,
                    "pos":     pos,
                    "pos2":    pos2,
                    "reading": reading,
                })
        except Exception:
            pass
        return tokens


class RegexTokenizer(MorphologicalEngine):
    """Lightweight regex-based fallback engine (no external dependencies).

    Splits text on kanji+kana runs. Lemma == surface (no de-conjugation).
    POS is always ``"名詞"`` (noun) as a conservative placeholder.
    """

    _TOKEN_RE: re.Pattern = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]+")

    @property
    def name(self) -> str:
        return "regex"

    def tokenize(self, text: str) -> list[dict]:
        """Tokenize *text* using a simple kanji/kana regex.

        Args:
            text: Raw Japanese text.

        Returns:
            List of token dicts.
        """
        return [
            {
                "surface": m,
                "lemma":   m,
                "pos":     "名詞",
                "pos2":    "",
                "reading": "",
            }
            for m in self._TOKEN_RE.findall(text)
        ]


class JapaneseNLP:
    """High-level façade for Japanese NLP operations.

    Wraps a ``MorphologicalEngine`` and exposes convenience methods.
    If no engine is supplied the constructor tries ``FugashiTokenizer`` first;
    on failure it silently falls back to ``RegexTokenizer``.

    Args:
        engine: Explicit engine to use.  Pass ``None`` to auto-detect.
    """

    def __init__(self, engine: MorphologicalEngine | None = None) -> None:
        if engine is not None:
            self._engine = engine
        else:
            self._engine = self._auto_detect()

    @staticmethod
    def _auto_detect() -> MorphologicalEngine:
        try:
            ft = FugashiTokenizer()
            ft._ensure_tagger()   # will raise if fugashi/mecab unavailable
            return ft
        except Exception:
            return RegexTokenizer()

    # ── Delegation ────────────────────────────────────────────────────────────

    def tokenize(self, text: str) -> list[dict]:
        """Return morphological token dicts for *text*.

        Args:
            text: Raw Japanese text.

        Returns:
            List of token dicts (see module-level comment for keys).
        """
        return self._engine.tokenize(text)

    def get_lemmas(self, text: str) -> list[str]:
        """Return lemma strings for Japanese tokens in *text*.

        Args:
            text: Raw Japanese text.

        Returns:
            List of lemma strings containing kanji/kana.
        """
        return self._engine.get_lemmas(text)

    @property
    def engine_name(self) -> str:
        """Name of the active morphological engine."""
        return self._engine.name

    @property
    def engine(self) -> MorphologicalEngine:
        """The active ``MorphologicalEngine`` instance."""
        return self._engine

# ══════════════════════════════════════════════════════════════════════════════
# JAPANESE NLP
# ══════════════════════════════════════════════════════════════════════════════

def has_japanese(text: str) -> bool:
    return bool(re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text))


def is_english(text: str) -> bool:
    if has_japanese(text):
        return False
    return sum(1 for c in text if ord(c) < 128) / max(len(text), 1) > 0.85


def tokenize_jp(text: str) -> list:
    """Simple regex tokenizer — kanji+kana sequences. Good enough for word lookup."""
    return re.findall(r'[\u3040-\u30ff\u4e00-\u9fff]+', text)


_kakasi = None
def _get_kakasi():
    global _kakasi
    if _kakasi is None:
        try:
            import pykakasi
            _kakasi = pykakasi.kakasi()
        except ImportError:
            _kakasi = False
    return _kakasi


def add_furigana(text: str) -> str:
    kks = _get_kakasi()
    if not kks:
        return text
    try:
        out = ""
        for item in kks.convert(text):
            orig, hira = item["orig"], item["hira"]
            if orig != hira and re.search(r'[\u4e00-\u9fff]', orig):
                out += f"{{{orig}|{hira}}}"
            else:
                out += orig
        return out
    except Exception:
        return text


_freq_cache: dict = {}
_freq_path_loaded: str = ""

def load_freq_list(path: str) -> dict:
    global _freq_cache, _freq_path_loaded
    if path == _freq_path_loaded and _freq_cache:
        return _freq_cache
    freq = {}
    _JP_RE = re.compile(r'[\u3040-\u30ff\u4e00-\u9fff\uff65-\uff9f]')
    try:
        # Auto-detect delimiter (tab for some lists, comma for others)
        with open(path, encoding="utf-8", errors="replace") as f:
            sample = f.read(2048)
        delim = "\t" if sample.count("\t") > sample.count(",") else ","
        rank = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            for row in csv.reader(f, delimiter=delim):
                if not row:
                    continue
                word = row[0].strip()
                # Skip header rows and non-Japanese tokens
                if not word or not _JP_RE.search(word):
                    continue
                rank += 1
                freq[word] = rank
        _freq_cache = freq
        _freq_path_loaded = path
    except Exception:
        pass
    return freq



# ── Morphological analysis (optional — fugashi or sudachipy) ─────────────────
# If either library is installed, we use it for proper lemmatization:
# 食べた → 食べる, 走っている → 走る etc.
# Falls back to the raw regex tokenizer if neither is available.

_morph_engine   = None   # "fugashi" | "sudachi" | None
_fugashi_tagger = None
_sudachi_dict   = None

def _init_morph():
    global _morph_engine, _fugashi_tagger, _sudachi_dict
    if _morph_engine is not None:
        return
    # Try fugashi first (smaller install, faster)
    try:
        import fugashi
        _fugashi_tagger = fugashi.Tagger()
        _morph_engine   = "fugashi"
        return
    except Exception:
        pass
    # Try sudachipy
    try:
        import sudachipy
        import sudachipy.dictionary
        _sudachi_dict = sudachipy.dictionary.Dictionary()
        _morph_engine = "sudachi"
        return
    except Exception:
        pass
    _morph_engine = None   # neither available


def tokenize_jp_morphs(text: str) -> list:
    """Return list of dictionary-form (lemma) tokens from Japanese text.
    Uses fugashi or sudachipy if available, falls back to regex tokenizer."""
    _init_morph()
    if _morph_engine == "fugashi":
        try:
            tokens = []
            for word in _fugashi_tagger(text):
                # MeCab feature format: surface, pos, pos2, ..., reading, lemma
                feature = word.feature
                # Try to get dictionary form (lemma) from feature[6] (ipadic)
                try:
                    lemma = feature.split(",")[6]
                    if lemma and lemma != "*":
                        tokens.append(lemma)
                    else:
                        tokens.append(word.surface)
                except (IndexError, AttributeError):
                    tokens.append(word.surface)
            # Filter to JP characters only
            return [t for t in tokens if re.search(r'[぀-ヿ一-鿿]', t)]
        except Exception:
            pass
    elif _morph_engine == "sudachi":
        try:
            tokenizer = _sudachi_dict.create()
            morphs    = tokenizer.tokenize(text)
            tokens    = [m.dictionary_form() for m in morphs]
            return [t for t in tokens if re.search(r'[぀-ヿ一-鿿]', t)]
        except Exception:
            pass
    # Fallback: raw regex
    return tokenize_jp(text)


def morph_engine_name() -> str:
    _init_morph()
    return _morph_engine or "regex (install fugashi or sudachipy for better accuracy)"


# ── Inline grammar gloss for translation prompt ───────────────────────────────

# MeCab/unidic-lite part-of-speech tags → readable labels
_POS_LABELS = {
    "名詞":     None,          # noun — skip, obvious
    "代名詞":   "pronoun",
    "動詞":     None,          # verb — handled via conjugation
    "形容詞":   "adj",
    "形容動詞": "adj",
    "副詞":     "adv",
    "助詞":     None,          # particle — handled below
    "助動詞":   None,          # aux verb — handled via conjugation
    "接続詞":   "conj",
    "感動詞":   "intj",
}

_PARTICLE_LABELS = {
    "は":  "topic",
    "が":  "subj",
    "を":  "obj",
    "に":  "to/at",
    "へ":  "toward",
    "で":  "at/by",
    "から":"from",
    "まで":"until",
    "と":  "with/and",
    "も":  "also",
    "の":  "poss",
    "か":  "?",
    "ね":  "ne",
    "よ":  "yo",
    "な":  "na",
    "ぞ":  "zo",
    "ぜ":  "ze",
    "さ":  "sa",
    "わ":  "wa",
    "ばかり": "only",
    "だけ":   "only",
    "しか":   "only(neg)",
    "くらい": "about",
    "ほど":   "extent",
}

_CONJ_LABELS = {
    "連用形": "masu-stem",
    "連体形": "attr",
    "仮定形": "cond",
    "命令形": "imperative",
    "未然形": "neg-stem",
    "終止形": None,    # plain/dict form — skip
    "基本形": None,
    "テ形":   "te-form",
    "タ形":   "PAST",
    "タ-連用": "PAST",
    "タ形-連用": "PAST",
}

_AUX_LABELS = {
    "ない":    "NEG",
    "ぬ":      "NEG",
    "ず":      "NEG",
    "た":      "PAST",
    "だ":      "COP",
    "です":    "COP-POL",
    "ます":    "POL",
    "ている":  "PROG",
    "てる":    "PROG",
    "てある":  "resultant",
    "ておく":  "prep-action",
    "てしまう":"regret/complete",
    "しまう":  "regret/complete",
    "たい":    "want-to",
    "たがる":  "want-to(3p)",
    "られる":  "PASS/POT",
    "れる":    "PASS/POT",
    "させる":  "CAUS",
    "せる":    "CAUS",
    "かもしれない": "maybe",
    "だろう":  "probably",
    "でしょう":"probably(pol)",
    "はず":    "expected",
    "べき":    "should",
    "そう":    "looks-like",
    "ようだ":  "seems",
    "みたい":  "like/seems",
}


def gloss_jp_line(text: str) -> str:
    """Return an inline-glossed version of a JP line using fugashi.
    e.g. '彼女は学校に行った' → '彼女は(topic) 学校に(to/at) 行った(go-PAST)'
    Falls back to plain text if fugashi not available."""
    _init_morph()
    if _morph_engine != "fugashi" or not _fugashi_tagger:
        return text  # no-op if fugashi not installed

    try:
        words   = list(_fugashi_tagger(text))
        glossed = []
        i       = 0
        while i < len(words):
            word    = words[i]
            surface = word.surface
            feature = word.feature if hasattr(word, "feature") else ""
            parts   = feature.split(",") if feature else []

            pos   = parts[0] if len(parts) > 0 else ""
            pos2  = parts[1] if len(parts) > 1 else ""
            conj  = parts[5] if len(parts) > 5 else ""
            lemma = parts[6] if len(parts) > 6 else surface

            label = None

            # Particles — most important for learners
            if pos == "助詞":
                label = _PARTICLE_LABELS.get(surface) or _PARTICLE_LABELS.get(lemma)

            # Auxiliary verbs — conjugation meaning
            elif pos == "助動詞":
                label = _AUX_LABELS.get(surface) or _AUX_LABELS.get(lemma)
                # Also try multi-token aux (e.g. て+いる = PROG)
                if not label and i + 1 < len(words):
                    combined = surface + words[i+1].surface
                    label = _AUX_LABELS.get(combined)

            # Verbs — show conjugation form when non-plain
            elif pos == "動詞":
                conj_label = _CONJ_LABELS.get(conj)
                if conj_label:
                    # Get lemma for readability: 行った → go-PAST
                    base = lemma if lemma and lemma != "*" else surface
                    # Transliterate common verbs to EN for clarity
                    label = conj_label

            # Named particles that are really nouns (は、が are separate pos)
            elif pos == "名詞" and pos2 in ("代名詞", "固有名詞"):
                label = None  # keep clean

            # Append surface ± label
            if label:
                glossed.append(f"{surface}({label})")
            else:
                glossed.append(surface)

            i += 1

        return " ".join(glossed)
    except Exception:
        return text


def tag_line_frequency(text: str, freq: dict) -> str:
    if not freq:
        return ""
    best = None
    for w in tokenize_jp_morphs(text):
        r = freq.get(w)
        if r and (best is None or r < best):
            best = r
    return f"[freq:{best}]" if best else ""


_known_cache: set = set()
_known_path_loaded: str = ""

def load_known_words(path: str) -> set:
    global _known_cache, _known_path_loaded
    if path == _known_path_loaded and _known_cache:
        return _known_cache
    words = set()
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                w = line.strip()
                if w:
                    words.add(w)
        _known_cache = words
        _known_path_loaded = path
    except Exception:
        pass
    return words


def line_is_known(text: str, known: set) -> bool:
    if not known:
        return False
    words = tokenize_jp_morphs(text)
    return bool(words) and all(w in known for w in words)


def complexity_score(text: str, grammar_patterns: list) -> tuple:
    """Returns (score 0.0–5.0, level str N5–N1)."""
    if not text.strip():
        return 0.0, "N5"
    kanji  = re.findall(r'[\u4e00-\u9fff]', text)
    chars  = [c for c in text if not c.isspace()]
    # use morph count for length scoring when available
    morph_tokens = tokenize_jp_morphs(text)
    score  = 0.0
    # Length (max 1.5)
    score += min(1.5, len(chars) / 20)
    # Kanji density (max 2.0)
    score += min(2.0, (len(kanji) / max(len(chars), 1)) * 4)
    # Grammar patterns (max 1.5)
    matched = sum(1 for _, pat in grammar_patterns if re.search(pat, text))
    score += min(1.5, matched * 0.5)
    score = min(5.0, score)
    level = ["N5","N5","N4","N3","N2","N1"][int(score)]
    return round(score, 2), level


def detect_grammar_patterns(text: str, patterns: list) -> list:
    """Return list of (name, match_str) for patterns found in text."""
    found = []
    for name, pat in patterns:
        m = re.search(pat, text)
        if m:
            found.append((name, m.group(0)))
    return found


_pitch_cache: dict = {}
_pitch_path_loaded: str = ""

def load_pitch_accent(path: str) -> dict:
    global _pitch_cache, _pitch_path_loaded
    if path == _pitch_path_loaded and _pitch_cache:
        return _pitch_cache
    data = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    data[row[0].strip()] = row[1].strip()
        _pitch_cache = data
        _pitch_path_loaded = path
    except Exception:
        pass
    return data


def tag_pitch_accent(text: str, pitch_db: dict) -> str:
    """Append pitch patterns for known words. e.g. '行く[LHL]'"""
    if not pitch_db:
        return text
    words = tokenize_jp(text)
    tags  = [f"{w}[{pitch_db[w]}]" for w in words if w in pitch_db]
    return text + (" " + " ".join(tags) if tags else "")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_srt(entries: list, known: set, grammar_patterns: list) -> dict:
    """Compute all immersion metrics from a list of SRT entries."""
    if not entries:
        return {}

    total       = len(entries)
    known_lines = 0
    i1_lines    = 0   # exactly 1 unknown word
    complexity_scores = []
    pattern_counts = {}
    all_unknowns = []

    for _, timing, text in entries:
        words   = tokenize_jp(text)
        unknown = [w for w in words if w not in known] if known else words

        if known:
            if not unknown:
                known_lines += 1
            elif len(unknown) == 1:
                i1_lines += 1
                all_unknowns.extend(unknown)
            else:
                all_unknowns.extend(unknown)

        score, level = complexity_score(text, grammar_patterns)
        complexity_scores.append((score, level))

        for name, pat in grammar_patterns:
            if re.search(pat, text):
                pattern_counts[name] = pattern_counts.get(name, 0) + 1

    # Sentence density
    if total >= 2:
        first_ms = srt_time_to_ms(entries[0][1].split("-->")[0].strip())
        last_ms  = srt_time_to_ms(entries[-1][1].split("-->")[1].strip())
        duration_min = (last_ms - first_ms) / 60000
        density = round(total / max(duration_min, 0.1), 1)
        duration_s = (last_ms - first_ms) / 1000
    else:
        density    = 0
        duration_s = 0

    avg_complexity = round(sum(s for s, _ in complexity_scores) / max(total, 1), 2)
    level_dist = {}
    for _, lv in complexity_scores:
        level_dist[lv] = level_dist.get(lv, 0) + 1

    # Top unknown words by frequency
    from collections import Counter
    top_unknowns = Counter(all_unknowns).most_common(20)

    return {
        "total":            total,
        "known_lines":      known_lines,
        "i1_lines":         i1_lines,
        "comprehensibility": round(known_lines / total * 100, 1) if known and total else None,
        "mining_yield":     round(i1_lines / total * 100, 1) if known and total else None,
        "avg_complexity":   avg_complexity,
        "level_dist":       level_dist,
        "density":          density,
        "duration_s":       duration_s,
        "pattern_counts":   dict(sorted(pattern_counts.items(), key=lambda x: -x[1])),
        "top_unknowns":     top_unknowns,
    }


def rank_folder_difficulty(srt_paths: list, grammar_patterns: list) -> list:
    """Return list of (path, avg_complexity, level) sorted hardest first."""
    results = []
    for path in srt_paths:
        try:
            entries = parse_srt(path)
            scores  = [complexity_score(e[2], grammar_patterns)[0] for e in entries]
            avg     = round(sum(scores) / max(len(scores), 1), 2)
            level   = ["N5","N5","N4","N3","N2","N1"][min(5, int(avg))]
            results.append((path, avg, level))
        except Exception:
            pass
    results.sort(key=lambda x: -x[1])
    return results
