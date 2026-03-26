"""AIST core utilities: subprocess wrapper, file discovery, time conversion, notifications.

Single responsibility: low-level helpers shared across all core modules.
No dependencies on other core/ modules.

STATUS: verbatim_copy
DIVERGES_FROM_AIST: False
"""

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: verbatim_copy
# DIVERGES_FROM_AIST: False
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import subprocess


def run(cmd: list, timeout: int = 600, allow_warning_exit=False, **kw) -> subprocess.CompletedProcess:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, **kw)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Timed out after {timeout}s: {' '.join(str(c) for c in cmd)}\n"
            "If file is in iCloud, right-click → Download Now then retry."
        )
    # mkvmerge exit 1 = warnings only, not failure
    allowed = (0, 1) if allow_warning_exit else (0,)
    if r.returncode not in allowed:
        stderr = r.stderr.decode(errors="replace").strip()
        stdout = r.stdout.decode(errors="replace").strip()
        raise RuntimeError(
            f"Command failed (exit {r.returncode}): {' '.join(str(c) for c in cmd)}\n"
            f"stderr: {stderr or '(none)'}\nstdout: {stdout[:600] or '(none)'}"
        )
    return r


def is_icloud_placeholder(path: str) -> bool:
    # FIX: removed unreliable 1MB size heuristic — only trust .icloud twin
    d, f = os.path.dirname(path), os.path.basename(path)
    return os.path.exists(os.path.join(d, "." + f + ".icloud"))


def find_mkv_files(folder: str) -> list:
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in sorted(files):
            if f.lower().endswith(".mkv") and not f.startswith("."):
                out.append(os.path.join(root, f))
    return out


def find_srt_files(folder: str) -> list:
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in sorted(files):
            if f.lower().endswith((".srt",".ass")) and not f.startswith("."):
                out.append(os.path.join(root, f))
    return out


def guess_show_name(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    # Strip bracketed/parenthesised group tags and metadata
    stem = re.sub(r'\[.*?\]|\(.*?\)', '', stem)
    # Strip episode identifiers
    stem = re.sub(r'(?i)(S\d+E\d+|E\d+|\d+x\d+)', '', stem)
    # Strip common release metadata words
    stem = re.sub(
        r'(?i)\b(1080p|720p|480p|2160p|4k|BluRay|BDRip|BDRemux|WEBRip|WEB-DL|'
        r'HDTV|x264|x265|HEVC|AVC|AAC|FLAC|DD|DTS|Opus|'
        r'Dual.Audio|Dual|Multi|Repack|Proper|EXTENDED|v2|v3|v4|'
        r'copy|[Cc]opy|COPY|sample|SAMPLE)\b', '', stem)
    # Collapse separators and trim
    stem = re.sub(r'[-_.]+', ' ', stem)
    stem = re.sub(r'\s{2,}', ' ', stem).strip()
    return stem[:40] if stem else "Unknown"


def parse_episode(filename: str) -> tuple:
    m = re.search(r'[Ss](\d+)[Ee](\d+)', filename)
    if m: return int(m.group(1)), int(m.group(2))
    m = re.search(r'[Ee](\d+)', filename)
    if m: return 1, int(m.group(1))
    m = re.search(r'(\d{2,3})', os.path.basename(filename))
    if m: return 1, int(m.group(1))
    return 1, 0


def apply_rename_pattern(pattern: str, mkv_path: str, suffix: str) -> str:
    stem = os.path.splitext(os.path.basename(mkv_path))[0]
    show = guess_show_name(mkv_path)
    s, e = parse_episode(stem)
    r = pattern
    for k, v in [("{stem}", stem), ("{show}", show), ("{s}", f"{s:02d}"),
                 ("{e}", f"{e:02d}"), ("{season}", str(s)), ("{ep}", str(e)),
                 ("{type}", suffix.lstrip("_"))]:
        r = r.replace(k, v)
    return r


def srt_time_to_ms(t: str) -> int:
    try:
        t = t.replace(",", ".")
        h, m, rest = t.split(":")
        s, ms = rest.split(".")
        return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms[:3])
    except Exception:
        return 0


def ms_to_srt_time(ms: int) -> str:
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ms_to_ass_time(ms: int) -> str:
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    cs = ms // 10
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def notify(title: str, message: str):
    try:
        subprocess.run(["osascript", "-e",
                        f'display notification "{message}" with title "{title}"'],
                       capture_output=True, timeout=5)
    except Exception:
        pass


def mask_key(key: str) -> str:
    """Return a display-safe masked version of an API key."""
    key = key.strip()
    if not key:
        return ""
    if len(key) <= 8:
        return "••••••••"
    return "•" * max(4, len(key) - 8) + key[-4:]


def open_in_finder(path: str):
    try:
        subprocess.Popen(["open", path] if os.path.isdir(path) else ["open", "-R", path])
    except Exception:
        pass
