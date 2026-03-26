"""AIST Show Glossary — per-show terminology, character names, custom notes.

Each show has its own JSON file in ~/.config/aist/glossaries/.
The glossary is injected into every translation prompt so character names
and show-specific terms stay consistent across episodes.

STATUS: verbatim_copy
DIVERGES_FROM_AIST: False

# DECISION: imports updated from `aist_config` → `.config` (relative within core
# package). No logic changed.
"""

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: verbatim_copy
# DIVERGES_FROM_AIST: False
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import re
from pathlib import Path

from .config import CONFIG_DIR

GLOSSARY_DIR = CONFIG_DIR / "glossaries"
GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)


# ── File I/O ──────────────────────────────────────────────────────────────────

def _normalize_name(show_name: str) -> str:
    """Normalize show name to a safe filename slug."""
    return re.sub(r'[^\w\u3040-\u9fff-]', '_', show_name.strip())[:80].strip("_") or "unknown"


def get_glossary_path(show_name: str) -> Path:
    return GLOSSARY_DIR / f"{_normalize_name(show_name)}.json"


def load_glossary(show_name: str) -> dict:
    """Load glossary for a show. Returns default structure if not found."""
    path = get_glossary_path(show_name)
    default = {
        "show_name":      show_name,
        "terms":          {},   # {jp_text: en_text}
        "notes":          "",   # free-form show-specific instructions
        "auto_extracted": False,
    }
    if not path.exists():
        return default
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for k, v in default.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return default


def save_glossary(show_name: str, data: dict):
    """Save glossary atomically."""
    path = get_glossary_path(show_name)
    tmp  = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def glossary_exists(show_name: str) -> bool:
    return get_glossary_path(show_name).exists()


def list_glossaries() -> list:
    """Return list of (show_name, term_count) for all saved glossaries."""
    result = []
    for p in sorted(GLOSSARY_DIR.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            result.append((data.get("show_name", p.stem), len(data.get("terms", {}))))
        except Exception:
            pass
    return result


def delete_glossary(show_name: str):
    path = get_glossary_path(show_name)
    if path.exists():
        path.unlink()


# ── Prompt helpers ────────────────────────────────────────────────────────────

def build_glossary_block(glossary: dict) -> str:
    """Build the glossary block to inject into translation prompts.
    Returns empty string if glossary has no terms and no notes."""
    terms = glossary.get("terms", {})
    notes = glossary.get("notes", "").strip()
    if not terms and not notes:
        return ""

    show = glossary.get("show_name", "")
    parts = []
    if show:
        parts.append(f"SHOW GLOSSARY — {show}")
    else:
        parts.append("SHOW GLOSSARY")
    parts.append("Translate these terms EXACTLY as listed. Never vary these translations.")
    parts.append("")

    if terms:
        parts.append("Term consistency (JP → EN):")
        for jp, en in sorted(terms.items()):
            parts.append(f"  {jp}  →  {en}")

    if notes:
        if terms:
            parts.append("")
        parts.append("Show-specific notes:")
        for line in notes.splitlines():
            if line.strip():
                parts.append(f"  {line.strip()}")

    return "\n".join(parts) + "\n\n"


def auto_extract_prompt(jp_texts: list, en_texts: list, show_name: str) -> str:
    """Build a prompt to auto-extract key glossary terms from a translation sample."""
    jp_block = "\n".join(f"JP: {t}" for t in jp_texts[:40])
    en_block = "\n".join(f"EN: {t}" for t in en_texts[:40])
    return (
        f"You just translated subtitle lines from the anime '{show_name}'.\n"
        "Extract a GLOSSARY of important terms that should be translated consistently "
        "in ALL future episodes.\n\n"
        "Include ONLY:\n"
        "  - Character names (and how they address each other)\n"
        "  - Unique show-specific terms, abilities, items, factions, locations\n"
        "  - Any recurring JP phrases with a specific preferred EN translation\n\n"
        "EXCLUDE:\n"
        "  - Common Japanese words (はい, ありがとう, etc.)\n"
        "  - Generic grammar or particles\n"
        "  - Anything with an obvious standard translation\n\n"
        "Return ONLY a valid JSON object: {\"JP_term\": \"EN_translation\", ...}\n"
        "Maximum 40 terms. Return {} if nothing important was found.\n\n"
        f"JP subtitles:\n{jp_block}\n\n"
        f"EN translations:\n{en_block}"
    )
