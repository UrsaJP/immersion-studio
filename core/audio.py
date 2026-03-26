"""AIST audio processing: condensed audio, per-card clips, screenshots.

Single responsibility: ffmpeg-based audio/screenshot extraction from MKV files.
Used by core/pipeline.py for condensed audio generation.

STATUS: verbatim_copy
DIVERGES_FROM_AIST: False

# DECISION: imports updated from `aist_utils` → `.utils` (relative within core
# package). No logic changed.
"""

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: verbatim_copy
# DIVERGES_FROM_AIST: False
# ─────────────────────────────────────────────────────────────────────────────

import os
import subprocess
import wave
import struct

from .utils import srt_time_to_ms


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO
# ══════════════════════════════════════════════════════════════════════════════

def build_condensed_audio(mkv_path: str, entries: list, output: str, tmpdir: str,
                          pad_before_ms: int = 200, pad_after_ms: int = 400,
                          exclude_low_conf: bool = True):
    """Single ffmpeg concat pass with silence padding and optional low-conf exclusion.

    pad_before_ms / pad_after_ms: silence inserted around each clip (ms).
    exclude_low_conf: skip lines whose text starts with '[?]' (Whisper low-confidence).
    Normalize to -16 LUFS in-place after concat.
    """
    clips      = []
    concat_txt = os.path.join(tmpdir, "concat.txt")

    # Write a short silence file used for padding between clips
    pad_total_s  = (pad_before_ms + pad_after_ms) / 1000
    silence_clip = os.path.join(tmpdir, "silence.mp3")
    if pad_total_s > 0:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
             "-t", f"{pad_total_s:.3f}", "-q:a", "9", silence_clip],
            capture_output=True, timeout=10
        )

    for i, (_, timing, text) in enumerate(entries):
        # Skip low-confidence Whisper lines from the listening file
        if exclude_low_conf and text.strip().startswith("[?]"):
            continue
        parts = timing.split("-->")
        if len(parts) != 2:
            continue
        # Expand clip slightly for natural listening — add pad_before from source audio
        raw_start_ms = srt_time_to_ms(parts[0].strip())
        raw_end_ms   = srt_time_to_ms(parts[1].strip())
        start_s = max(0, raw_start_ms - pad_before_ms) / 1000
        end_s   = (raw_end_ms + pad_after_ms) / 1000
        dur     = end_s - start_s
        if dur <= 0.05:
            continue
        clip = os.path.join(tmpdir, f"clip_{i:04d}.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-t", f"{dur:.3f}",
             "-i", mkv_path, "-q:a", "3", "-vn", clip],
            capture_output=True, timeout=30
        )
        if os.path.exists(clip):
            clips.append(clip)

    if not clips:
        return

    with open(concat_txt, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")

    # First pass: concat all clips
    raw_concat = output.replace(".mp3", "_raw.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", concat_txt, "-c", "copy", raw_concat],
        capture_output=True, timeout=300
    )
    if not os.path.exists(raw_concat):
        return

    # Second pass: loudnorm to -16 LUFS in-place → final output file
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_concat,
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
         output],
        capture_output=True, timeout=300
    )
    # Remove raw intermediate
    try:
        os.remove(raw_concat)
    except OSError:
        pass


def build_per_card_audio(mkv_path: str, entries: list, out_dir: str):
    """Export individual .mp3 per subtitle line for Anki audio fields."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, (_, timing, text) in enumerate(entries):
        parts = timing.split("-->")
        if len(parts) != 2:
            paths.append(None)
            continue
        start_s = srt_time_to_ms(parts[0].strip()) / 1000
        end_s   = srt_time_to_ms(parts[1].strip()) / 1000
        dur     = end_s - start_s
        if dur <= 0.05:
            paths.append(None)
            continue
        out = os.path.join(out_dir, f"line_{i:04d}.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-t", f"{dur:.3f}",
             "-i", mkv_path, "-q:a", "3", "-vn", out],
            capture_output=True, timeout=30
        )
        paths.append(out if os.path.exists(out) else None)
    return paths


def extract_screenshot(mkv_path: str, timing_str: str, output: str):
    """Extract video frame at subtitle start timestamp."""
    parts = timing_str.split("-->")
    if len(parts) < 2:
        return
    start_s = srt_time_to_ms(parts[0].strip()) / 1000
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", mkv_path,
         "-frames:v", "1", "-q:v", "2", output],
        capture_output=True, timeout=15
    )


def make_silent_wav(path: str, duration_s: float = 3.0, sample_rate: int = 16000):
    """Generate a short silent WAV for Whisper benchmark warm-up."""
    n_samples = int(duration_s * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0]*n_samples)))
