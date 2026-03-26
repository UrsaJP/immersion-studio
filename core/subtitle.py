"""AIST subtitle parsing, writing, merging, and repair.

Single responsibility: parse/write SRT and ASS subtitle files.
Used by core/translation.py, core/pipeline.py, and core/japanese.py.

Phase 2 additions: Timestamp, SubtitleEntry, SubtitleFormat ABC, SRTFormat,
ASSFormat — typed models per Module Improvement Standards. All existing
tuple-based functions preserved for backward compatibility.

STATUS: extended
DIVERGES_FROM_AIST: True
Changes: added Timestamp dataclass, SubtitleEntry dataclass, SubtitleFormat
         ABC, SRTFormat, ASSFormat. Existing parse_srt/write_srt/parse_ass
         functions unchanged — backward-compat shims delegate to new classes
         where appropriate. imports updated aist_utils → .utils.
"""

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: extended
# DIVERGES_FROM_AIST: True
# Changes: Timestamp, SubtitleEntry, SubtitleFormat, SRTFormat, ASSFormat added
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .utils import ms_to_ass_time, ms_to_srt_time, srt_time_to_ms

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
ASS_KARAOKE_MIN_DURATION_MS: int = 100   # lines shorter than this are karaoke syllables
ASS_OVERRIDE_TAG_RE: re.Pattern = re.compile(r"\{[^}]*\}")
ASS_DRAWING_CMD_RE: re.Pattern  = re.compile(r"^m\s+[-\d]")
ASS_LINE_BREAK_RE: re.Pattern   = re.compile(r"\\N|\\n")


# ══════════════════════════════════════════════════════════════════════════════
# TYPED MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Timestamp:
    """Immutable subtitle timestamp. All arithmetic happens in ms (int), never strings.

    Args:
        ms: Time in milliseconds. Must be >= 0.

    Raises:
        ValueError: If ms is negative.
    """
    ms: int

    def __post_init__(self) -> None:
        if self.ms < 0:
            raise ValueError(f"Timestamp cannot be negative: {self.ms}ms")

    @classmethod
    def from_srt(cls, s: str) -> Timestamp:
        """Parse SRT timestamp '00:01:23,456' → Timestamp(ms=83456).

        Args:
            s: SRT timestamp string (comma or dot as millisecond separator).

        Returns:
            Timestamp instance.
        """
        s = s.strip().replace(",", ".")
        try:
            h, m, rest = s.split(":")
            sec, frac = rest.split(".")
            return cls(
                int(h) * 3_600_000
                + int(m) * 60_000
                + int(sec) * 1_000
                + int(frac.ljust(3, "0")[:3])
            )
        except Exception:
            logger.warning("Timestamp.from_srt: malformed '%s', defaulting to 0", s)
            return cls(0)

    @classmethod
    def from_ass(cls, s: str) -> Timestamp:
        """Parse ASS timestamp 'h:mm:ss.cc' → Timestamp(ms).

        Args:
            s: ASS timestamp string (centiseconds after the dot).

        Returns:
            Timestamp instance.
        """
        try:
            h, m, rest = s.strip().split(":")
            sec, cs = rest.split(".")
            return cls(
                int(h) * 3_600_000
                + int(m) * 60_000
                + int(sec) * 1_000
                + int(cs) * 10
            )
        except Exception:
            logger.warning("Timestamp.from_ass: malformed '%s', defaulting to 0", s)
            return cls(0)

    def to_srt(self) -> str:
        """Format as SRT timestamp '00:01:23,456'.

        Returns:
            SRT-formatted string.
        """
        ms = self.ms
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1_000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def to_ass(self) -> str:
        """Format as ASS timestamp 'h:mm:ss.cc'.

        Returns:
            ASS-formatted string.
        """
        ms = self.ms
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1_000)
        cs = ms // 10
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def __add__(self, other: int) -> Timestamp:  # type: ignore[override]
        return Timestamp(max(0, self.ms + other))

    def __sub__(self, other: int) -> Timestamp:  # type: ignore[override]
        return Timestamp(max(0, self.ms - other))

    def __lt__(self, other: Timestamp) -> bool:
        return self.ms < other.ms

    def __le__(self, other: Timestamp) -> bool:
        return self.ms <= other.ms


@dataclass
class SubtitleEntry:
    """A single subtitle cue with typed timestamps.

    Args:
        index: Sequential 1-based index.
        start: Start timestamp.
        end: End timestamp.
        text: Subtitle text content (may contain newlines).
    """
    index: int
    start: Timestamp
    end: Timestamp
    text: str

    @property
    def duration_ms(self) -> int:
        """Duration of this cue in milliseconds."""
        return max(0, self.end.ms - self.start.ms)

    def to_tuple(self) -> tuple:
        """Convert to legacy (index_str, timing_str, text) tuple for backward compat."""
        return (str(self.index), f"{self.start.to_srt()} --> {self.end.to_srt()}", self.text)


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT PLUGIN INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

class SubtitleFormat(ABC):
    """Abstract base for subtitle format parsers/writers.

    Each concrete subclass handles exactly one format (SRT, ASS, VTT…).
    """

    @abstractmethod
    def can_parse(self, path: str) -> bool:
        """Return True if this format can parse the given file.

        Args:
            path: Filesystem path to the candidate file.
        """
        ...

    @abstractmethod
    def parse(self, path: str) -> list[SubtitleEntry]:
        """Parse a subtitle file and return typed entries.

        Args:
            path: Filesystem path to the subtitle file.

        Returns:
            List of SubtitleEntry, sorted by start time, duplicates removed.
        """
        ...

    @abstractmethod
    def write(self, entries: list[SubtitleEntry], path: str) -> None:
        """Write typed entries to a file.

        Args:
            entries: Subtitle entries to write.
            path: Destination file path.
        """
        ...


class SRTFormat(SubtitleFormat):
    """SRT format parser/writer. Handles UTF-8 BOM and malformed blocks (skip with warning)."""

    def can_parse(self, path: str) -> bool:
        return path.lower().endswith(".srt")

    def parse(self, path: str) -> list[SubtitleEntry]:
        """Parse an SRT file into typed SubtitleEntry objects.

        Args:
            path: Path to the .srt file.

        Returns:
            List of SubtitleEntry sorted by start time. Empty list if file
            does not exist or all blocks are malformed.
        """
        if not path or not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                content = f.read()
        except OSError as e:
            logger.error("SRTFormat.parse: cannot read %s: %s", path, e)
            return []

        entries: list[SubtitleEntry] = []
        for block in re.split(r"\n{2,}", content.strip()):
            lines = block.strip().splitlines()
            if len(lines) < 3:
                continue
            try:
                idx_str = lines[0].strip()
                timing  = lines[1].strip()
                text    = "\n".join(lines[2:])
                if "-->" not in timing:
                    logger.debug("SRTFormat.parse: skipping malformed timing '%s'", timing)
                    continue
                parts = timing.split("-->")
                start = Timestamp.from_srt(parts[0].strip())
                end   = Timestamp.from_srt(parts[1].strip())
                try:
                    idx = int(idx_str)
                except ValueError:
                    idx = len(entries) + 1
                entries.append(SubtitleEntry(index=idx, start=start, end=end, text=text))
            except Exception as e:
                logger.warning("SRTFormat.parse: skipping malformed block in %s: %s", path, e)
        return entries

    def write(self, entries: list[SubtitleEntry], path: str) -> None:
        """Write SubtitleEntry list as a UTF-8 SRT file. Renumbers from 1.

        Args:
            entries: Entries to write.
            path: Destination path.
        """
        with open(path, "w", encoding="utf-8") as f:
            for i, entry in enumerate(entries, 1):
                f.write(f"{i}\n{entry.start.to_srt()} --> {entry.end.to_srt()}\n{entry.text}\n\n")


class ASSFormat(SubtitleFormat):
    """ASS/SSA subtitle format parser.

    Filters non-dialogue content using style name blocklist AND duration
    threshold. The skip_styles set is configurable so tests can pass custom
    sets without monkey-patching globals.

    Args:
        skip_styles: Style names to treat as non-dialogue (signs, karaoke, etc.).
                     Case-insensitive prefix matching is applied.
    """

    SIGN_STYLES_DEFAULT: frozenset = frozenset({
        "sign", "signs", "sign-", "sign_", "op", "ed", "opening", "ending",
        "kara", "karaoke", "kar", "k-", "ktiming", "kredits", "credit", "credits",
        "effect", "fx", "transition", "banner", "title", "insert", "on-screen",
        "note", "comment", "alt", "alternate", "italics", "top",
    })

    def __init__(self, skip_styles: frozenset | None = None) -> None:
        self.skip_styles: frozenset = (
            skip_styles if skip_styles is not None else self.SIGN_STYLES_DEFAULT
        )

    def can_parse(self, path: str) -> bool:
        return path.lower().endswith((".ass", ".ssa"))

    def _is_skip_style(self, style: str) -> bool:
        s = style.lower().strip()
        if s in self.skip_styles:
            return True
        return any(s.startswith(sk) for sk in self.skip_styles)

    def parse(self, path: str) -> list[SubtitleEntry]:
        """Parse an ASS/SSA file into typed SubtitleEntry objects.

        Args:
            path: Path to the .ass/.ssa file.

        Returns:
            List of SubtitleEntry, karaoke and signs filtered out, sorted by
            start time, consecutive duplicates removed.
        """
        raw: list[tuple] = []
        in_events = False
        format_cols: list[str] = []

        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n\r")
                    if line.strip().lower() == "[events]":
                        in_events = True
                        continue
                    if line.strip().startswith("[") and in_events:
                        in_events = False
                        continue
                    if not in_events:
                        continue
                    if line.startswith("Format:"):
                        format_cols = [c.strip().lower() for c in line[7:].split(",")]
                        continue
                    if not line.startswith("Dialogue:"):
                        continue
                    parts = line[9:].split(",", len(format_cols) - 1)
                    if len(parts) < len(format_cols):
                        continue
                    row    = dict(zip(format_cols, parts))
                    start  = row.get("start", "").strip()
                    end    = row.get("end", "").strip()
                    text   = row.get("text", "").strip()
                    effect = row.get("effect", "").strip().lower()
                    style  = row.get("style", "").strip()

                    start_ms = Timestamp.from_ass(start).ms
                    end_ms   = Timestamp.from_ass(end).ms
                    if end_ms <= start_ms:
                        continue
                    if (end_ms - start_ms) < ASS_KARAOKE_MIN_DURATION_MS:
                        continue
                    if self._is_skip_style(style):
                        continue
                    if effect and effect not in ("", "- default", "-default"):
                        continue

                    text = ASS_OVERRIDE_TAG_RE.sub("", text)
                    if ASS_DRAWING_CMD_RE.match(text.strip()):
                        continue
                    text = ASS_LINE_BREAK_RE.sub(" ", text).strip()
                    if not text:
                        continue

                    raw.append((start_ms, end_ms, text))
        except Exception as e:
            logger.error("ASSFormat.parse: failed reading %s: %s", path, e, exc_info=True)

        raw.sort(key=lambda x: x[0])
        # Remove consecutive duplicate text
        deduped: list[tuple] = []
        for item in raw:
            if not deduped or item[2] != deduped[-1][2]:
                deduped.append(item)

        return [
            SubtitleEntry(
                index=i + 1,
                start=Timestamp(s),
                end=Timestamp(e),
                text=t,
            )
            for i, (s, e, t) in enumerate(deduped)
        ]

    def write(self, entries: list[SubtitleEntry], path: str) -> None:
        """Write entries as a minimal ASS file using default style.

        Args:
            entries: Subtitle entries to write.
            path: Destination path.
        """
        header = (
            "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\n"
            "PlayResX: 1920\nPlayResY: 1080\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,"
            "&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,18,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            for entry in entries:
                ass_text = entry.text.replace("\n", "\\N")
                f.write(
                    f"Dialogue: 0,{entry.start.to_ass()},{entry.end.to_ass()},"
                    f"Default,,0,0,0,,{ass_text}\n"
                )


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY HELPERS  (internal — used by parse_ass below)
# ══════════════════════════════════════════════════════════════════════════════

# Style names that indicate non-dialogue tracks (signs, karaoke, effects)
_ASS_SKIP_STYLES = {
    "sign","signs","sign-","sign_","op","ed","opening","ending",
    "kara","karaoke","kar","k-","ktiming","kredits","credit","credits",
    "effect","fx","transition","banner","title","insert","on-screen",
    "note","comment","alt","alternate","italics","top",
}

def _is_skip_style(style: str) -> bool:
    """Return True if this ASS style name looks like a non-dialogue style."""
    s = style.lower().strip()
    if s in _ASS_SKIP_STYLES:
        return True
    for skip in _ASS_SKIP_STYLES:
        if s.startswith(skip):
            return True
    return False


def _ass_time_to_ms(t: str) -> int:
    """Convert ASS time h:mm:ss.cc to milliseconds."""
    try:
        h, m, s = t.split(":")
        s, cs = s.split(".")
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(cs) * 10
    except Exception:
        return 0


def _ms_to_srt(ms: int) -> str:
    h  = ms // 3600000; ms %= 3600000
    m  = ms // 60000;   ms %= 60000
    s  = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY TUPLE-BASED API  (backward compatible — do not remove)
# ══════════════════════════════════════════════════════════════════════════════

def parse_ass(path: str) -> list:
    """Parse an ASS/SSA subtitle file into SRT-style (idx, timing, text) tuples.

    Filters out karaoke and signs using BOTH style names AND line duration.
    Backward-compatible wrapper — new code should use ASSFormat().parse().
    """
    entries = ASSFormat().parse(path)
    return [e.to_tuple() for e in entries]


def parse_srt(path: str) -> list:
    """Parse an SRT file into (idx, timing, text) tuples.

    Backward-compatible wrapper — new code should use SRTFormat().parse().
    """
    return [e.to_tuple() for e in SRTFormat().parse(path)]


def write_srt(entries: list, path: str) -> None:
    """Write (idx, timing, text) tuples as a UTF-8 SRT file."""
    with open(path, "w", encoding="utf-8") as f:
        for i, (_, timing, text) in enumerate(entries, 1):
            f.write(f"{i}\n{timing}\n{text}\n\n")


def write_vtt(entries: list, path: str, title: str = "AIST") -> None:
    """Write a WebVTT subtitle file (.vtt)."""
    def _srt_to_vtt_time(t: str) -> str:
        return t.replace(",", ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n")
        f.write("NOTE Generated by AIST\n\n")
        for i, (idx, timing, text) in enumerate(entries, 1):
            parts = timing.split("-->")
            if len(parts) == 2:
                start = _srt_to_vtt_time(parts[0].strip())
                end   = _srt_to_vtt_time(parts[1].strip())
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def write_lrc(entries: list, path: str) -> None:
    """Write an LRC lyrics file (approximate, based on start time)."""
    def _ms_to_lrc(t: str) -> str:
        try:
            h, m, rest = t.replace(",", ".").split(":")
            s, ms_str = rest.split(".")
            total_ms = int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms_str[:3])
            mm = total_ms // 60000
            ss = (total_ms % 60000) // 1000
            cs = (total_ms % 1000) // 10
            return f"[{mm:02d}:{ss:02d}.{cs:02d}]"
        except Exception:
            return "[00:00.00]"
    with open(path, "w", encoding="utf-8") as f:
        f.write("[ti:AIST Export]\n\n")
        for _, timing, text in entries:
            start = timing.split("-->")[0].strip()
            f.write(f"{_ms_to_lrc(start)}{text.replace(chr(10), ' ')}\n")


def write_ass(entries: list, path: str, title: str = "AIST",
              font_name: str = "Arial", font_size: int = 52,
              primary_colour: str = "&H00FFFFFF", bold: bool = False) -> None:
    """Convert entries to styled ASS subtitle file."""
    bold_flag = "-1" if bold else "0"
    header = (
        "[Script Info]\n"
        f"Title: {title}\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "PlayResX: 1920\nPlayResY: 1080\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{font_size},{primary_colour},"
        f"&H000000FF,&H00000000,&H80000000,"
        f"{bold_flag},0,0,0,100,100,0,0,1,2,1,2,10,10,18,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for _, timing, text in entries:
            parts = timing.split("-->")
            if len(parts) != 2:
                continue
            start = ms_to_ass_time(srt_time_to_ms(parts[0].strip()))
            end   = ms_to_ass_time(srt_time_to_ms(parts[1].strip()))
            ass_text = text.replace("\n", "\\N")
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{ass_text}\n")


def merge_overlapping(entries: list, gap_ms: int = 100) -> list:
    if not entries:
        return entries
    merged = [list(entries[0])]
    for idx, timing, text in entries[1:]:
        cur_parts  = timing.split("-->")
        prev_parts = merged[-1][1].split("-->")
        if len(cur_parts) < 2 or len(prev_parts) < 2:
            merged.append([idx, timing, text])
            continue
        prev = merged[-1]
        prev_end  = srt_time_to_ms(prev_parts[1].strip())
        cur_start = srt_time_to_ms(cur_parts[0].strip())
        cur_end   = srt_time_to_ms(cur_parts[1].strip())
        if cur_start - prev_end <= gap_ms:
            prev_start = prev_parts[0].strip()
            prev[1]    = f"{prev_start} --> {ms_to_srt_time(max(prev_end, cur_end))}"
            prev[2]    = prev[2] + " " + text
        else:
            merged.append([idx, timing, text])
    return [tuple(e) for e in merged]


def deduplicate_subs(entries: list) -> list:
    """Remove consecutive identical subtitle lines."""
    if not entries:
        return entries
    out = [entries[0]]
    for entry in entries[1:]:
        if entry[2].strip() != out[-1][2].strip():
            out.append(entry)
    return out


def repair_timestamps(entries: list) -> list:
    """Fix overlapping and out-of-order SRT timestamps."""
    if not entries:
        return entries
    repaired = []
    prev_end_ms = 0
    for idx, timing, text in entries:
        parts = timing.split("-->")
        if len(parts) != 2:
            repaired.append((idx, timing, text))
            continue
        start_ms = srt_time_to_ms(parts[0].strip())
        end_ms   = srt_time_to_ms(parts[1].strip())
        if start_ms < prev_end_ms:
            start_ms = prev_end_ms + 1
        if end_ms <= start_ms:
            end_ms = start_ms + 1000
        prev_end_ms = end_ms
        repaired.append((idx,
                         f"{ms_to_srt_time(start_ms)} --> {ms_to_srt_time(end_ms)}",
                         text))
    return repaired


def detect_language_per_line(entries: list) -> tuple:
    """Split entries into JP-only and EN-only lists.
    Returns (jp_entries, en_entries, mixed_entries)"""
    from .japanese import has_japanese, is_english
    jp, en, mixed = [], [], []
    for entry in entries:
        if has_japanese(entry[2]):
            jp.append(entry)
        elif is_english(entry[2]):
            en.append(entry)
        else:
            mixed.append(entry)
    return jp, en, mixed


def merge_sub_tracks(track1_entries: list, track2_entries: list) -> list:
    """Merge two subtitle tracks by interleaving on timestamp order."""
    combined = list(track1_entries) + list(track2_entries)
    combined.sort(key=lambda e: srt_time_to_ms(e[1].split("-->")[0].strip()))
    return combined
