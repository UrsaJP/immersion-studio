"""AIST configuration: paths, constants, providers, settings, themes.

Single responsibility: app-level constants, AI provider registry, default
settings dict, AIConfig dataclass, Keychain helpers, ModelFetchWorker.

STATUS: extended
DIVERGES_FROM_AIST: True
Changes: added AIConfig, KEYCHAIN_SERVICE, save_api_key, load_api_key,
         MODEL_CACHE_TTL, _is_cache_fresh, fetch_model_list,
         ModelFetchSignals, ModelFetchWorker, watch_ollama_models,
         PROVIDERS dict (18 providers), IS_SETTINGS_PATH, IS_MODEL_CACHE_PATH.
"""

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: extended
# DIVERGES_FROM_AIST: True
# Changes: added AIConfig, Keychain helpers, ModelFetchWorker, 18-provider
#          PROVIDERS dict
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import os
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

VERSION = "4.9.21"

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_DIR   = Path.home() / ".config" / "aist"
CONFIG_FILE  = CONFIG_DIR / "settings.json"
DB_FILE      = CONFIG_DIR / "aist.db"
RESUME_FILE  = CONFIG_DIR / "resume.json"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

BASIC    = 1
ADVANCED = 2
MAX_RETRY = 3

DEFAULT_GRAMMAR_PATTERNS = [
    ["ている (progressive/resultant)", r"ていr?[るたらない]|てる"],
    ["てしまう (regret/completion)",   r"てしまう|ちゃう|じゃう"],
    ["てみる (try doing)",             r"てみ[るたら]"],
    ["てあげる/くれる/もらう (giving)", r"てあげ|てくれ|てもら"],
    ["ようにする/なる (change/effort)", r"ようにす[るr]|ようにな[るr]"],
    ["なければならない (obligation)",   r"なければならない|ないといけない|なきゃ"],
    ["かもしれない (possibility)",      r"かもしれない|かもしれません"],
    ["はずだ/はずがない (expectation)", r"はずだ|はずです|はずが[ないn]"],
    ["と思う (thought/opinion)",        r"と思[うった]"],
    ["ために (purpose/reason)",         r"ために"],
    ["ながら (simultaneous action)",    r"ながら"],
    ["ば conditional",                  r"[^なk]ば"],
    ["たら conditional",                r"たら"],
    ["なら conditional",                r"なら"],
    ["ことがある/できる",               r"ことが[あできa]"],
    ["そうだ/らしい/ようだ (hearsay)",  r"そうだ|らしい|ようだ|みたいだ"],
]

# ── Provider registry ─────────────────────────────────────────────────────────
# Single source of truth for all AI backends.
# (id, display_name, base_url, default_model, suggested_models, note)
# base_url=None      → Ollama (custom handler)
# base_url="gemini"  → Gemini (custom REST handler)
# otherwise          → OpenAI-compatible /v1/chat/completions
AI_PROVIDERS = [
    ("ollama",      "Ollama (local)",   None,
     "qwen2.5:7b",
     ["qwen2.5:7b","qwen2.5:14b","qwen2.5:32b","llama3.3:70b","llama3.1:8b",
      "mistral","gemma3:4b","gemma3:12b","phi4","deepseek-r1:8b"],
     "Free — runs on your Mac"),
    ("openai",      "OpenAI",           "https://api.openai.com/v1",
     "gpt-4o-mini",
     ["gpt-4o-mini","gpt-4o","gpt-4.1","gpt-4.1-mini","gpt-4.1-nano",
      "o4-mini","o3-mini","o1-mini"],
     "~$0.02/ep (gpt-4.1-mini)"),
    ("anthropic",   "Anthropic",        "anthropic",
     "claude-haiku-4-5-20251001",
     ["claude-haiku-4-5-20251001","claude-sonnet-4-6","claude-opus-4-6",
      "claude-3-5-haiku-20241022","claude-3-5-sonnet-20241022"],
     "~$0.05/ep (Haiku) — excellent JP"),
    ("gemini",      "Google Gemini",    "gemini",
     "gemini-2.0-flash",
     ["gemini-2.0-flash","gemini-2.5-flash-preview-04-17","gemini-2.5-pro-preview-03-25",
      "gemini-1.5-flash","gemini-1.5-pro"],
     "Free tier: 1500 req/day"),
    ("groq",        "Groq",             "https://api.groq.com/openai/v1",
     "llama-3.3-70b-versatile",
     ["llama-3.3-70b-versatile","llama-3.1-8b-instant","llama-3.1-70b-versatile",
      "deepseek-r1-distill-llama-70b","mixtral-8x7b-32768","gemma2-9b-it"],
     "Free tier — very fast"),
    ("deepseek",    "DeepSeek",         "https://api.deepseek.com/v1",
     "deepseek-chat",
     ["deepseek-chat","deepseek-reasoner"],
     "~$0.01/ep — very cheap"),
    ("mistral",     "Mistral",          "https://api.mistral.ai/v1",
     "mistral-small-latest",
     ["mistral-small-latest","mistral-medium-latest","mistral-large-latest",
      "open-mistral-7b","open-mixtral-8x7b"],
     "~$0.02/ep"),
    ("kimi",        "Kimi (Moonshot)",  "https://api.moonshot.cn/v1",
     "moonshot-v1-8k",
     ["moonshot-v1-8k","moonshot-v1-32k","moonshot-v1-128k"],
     "Good JP support"),
    ("together",    "Together AI",      "https://api.together.xyz/v1",
     "Qwen/Qwen2.5-72B-Instruct-Turbo",
     ["Qwen/Qwen2.5-72B-Instruct-Turbo","Qwen/Qwen2.5-7B-Instruct-Turbo",
      "meta-llama/Llama-3.3-70B-Instruct-Turbo","meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
      "mistralai/Mixtral-8x7B-Instruct-v0.1"],
     "Pay-per-token, many models"),
    ("openrouter",  "OpenRouter",       "https://openrouter.ai/api/v1",
     "anthropic/claude-haiku-4-5",
     ["anthropic/claude-haiku-4-5","anthropic/claude-sonnet-4-6","anthropic/claude-opus-4-6",
      "google/gemini-2.0-flash-exp:free","google/gemini-2.5-pro-exp-03-25:free",
      "deepseek/deepseek-chat-v3-0324:free","deepseek/deepseek-r1",
      "meta-llama/llama-3.3-70b-instruct:free","qwen/qwen-2.5-72b-instruct:free"],
     "200+ models, free tier options"),
    ("perplexity",  "Perplexity",       "https://api.perplexity.ai",
     "sonar",
     ["sonar","sonar-pro","sonar-reasoning","llama-3.1-sonar-large-128k-online"],
     "Online models with web access"),
    ("cohere",      "Cohere",           "https://api.cohere.com/v1",
     "command-r-plus",
     ["command-r-plus","command-r","command-a-03-2025","command-light"],
     "Good multilingual support"),
    ("xai",         "xAI (Grok)",       "https://api.x.ai/v1",
     "grok-3-mini",
     ["grok-3-mini","grok-3","grok-2-1212","grok-beta"],
     "Strong reasoning, competitive pricing"),
    ("cerebras",    "Cerebras",         "https://api.cerebras.ai/v1",
     "llama-3.3-70b",
     ["llama-3.3-70b","llama3.1-70b","llama3.1-8b"],
     "Extremely fast inference"),
]

def get_provider(pid: str) -> dict:
    for p in AI_PROVIDERS:
        if p[0] == pid:
            return {"id":p[0],"name":p[1],"base_url":p[2],"default_model":p[3],
                    "models":p[4],"note":p[5]}
    return get_provider("ollama")


DEFAULT_SETTINGS = {
    # Models
    "ollama_model":          "llama3",
    "whisper_model":         "base",
    "whisper_mode":          "local",  # "local" = openai-whisper CLI, "cloud" = OpenAI API
    "whisper_gpu":           False,
    # Output
    "mkv_out_dir":           "",
    "srt_out_dir":           "",
    "strip_orig":            False,
    "strip_eng_audio":       False,   # remove English audio tracks from output MKV
    "force_whisper":         False,   # always use Whisper, ignore embedded subs
    "allow_en_backtranslate": False,  # allow back-translating EN subs (not recommended — loses JP meaning)
    # Online JP subtitle lookup
    "online_sub_lookup":      True,   # try online sources before falling back to Whisper
    "jimaku_api_key":         "",     # Jimaku.cc API key (free at jimaku.cc)
    "opensubs_api_key":       "",     # OpenSubtitles API key (free at opensubtitles.com)
    "online_sub_lang":        "ja",   # subtitle language to search for
    "ai_title_lookup":        False,  # use AI to resolve show title from filename before sub search
    # Episode context for translation
    "use_mal_context":        True,   # fetch show/episode/character info from Jikan (MAL)
    "use_ai_ep_summary":      False,  # ask AI to summarise the JP subs before translating
    # OP/ED skip
    "skip_music_lines":       True,   # filter lines containing ♪ ♩ ♫ ♬ (song lyrics)
    "skip_op_secs":           0,      # skip first N seconds from translation (OP) — 0=off
    "skip_ed_secs":           0,      # skip last N seconds from translation (ED) — 0=off
    # Output extras
    "export_plain_txt":       False,  # also export a .txt file alongside the .srt
    "save_raw_jp_srt":        False,  # save raw JP source .srt for comparison view
    "ai_quality_note":        False,  # AI self-rates translation quality at the end
    # Subtitle style (ASS baked track)
    "sub_font_name":          "Arial",
    "sub_font_size":          52,
    "sub_primary_colour":     "&H00FFFFFF",  # white ABGR hex
    "sub_bold":               False,
    # Anki extras
    "anki_jlpt_tags":         True,   # tag cards jlpt::n1…n5 based on complexity
    "anki_check_dupes":       True,   # warn + skip if card already exists in deck
    # Named presets
    "settings_presets":       {},     # {preset_name: {setting_key: value, …}}
    # Source compression
    "slim_source":           False,   # lossless: strip extra tracks/fonts from source after processing
    "reencode_source":       False,   # lossy: re-encode video to smaller CRF
    "reencode_crf":          28,      # CRF value (18=high quality, 28=good balance, 35=small file)
    "reencode_preset":       "medium",# ffmpeg preset: ultrafast/fast/medium/slow/veryslow
    "reencode_replace":      False,   # replace source file with re-encoded version
    "skip_existing":         False,
    "skip_extras":           False,   # skip files whose path/name suggests bonus/extra content
    "auto_organize":         False,
    "rename_pattern":        "{stem}_literal",
    "open_finder_after":     True,
    "output_ass":            False,
    "output_vtt":            False,
    "output_lrc":            False,
    # Processing
    "dry_run":               False,
    "verbosity":             BASIC,
    "workers":               1,
    "filter_duration":       False,
    "min_dur_ms":            500,
    "max_dur_ms":            10000,
    "merge_subs":            False,
    "merge_gap_ms":          100,
    "context_window":        2,
    "condensed_audio":         True,
    "processing_mode":         "video",   # "video" | "audio"
    "silence_pad_before_ms":   200,       # ms of silence before each clip in condensed audio
    "silence_pad_after_ms":    400,       # ms of silence after each clip
    "exclude_low_conf_audio":  True,      # skip [?] flagged lines from condensed audio
    "home_folder_delete":      False,
    "watch_folder_path":       "",
    "watch_folder_active":     False,
    "detect_language_lines":   False,
    "whisper_conf_threshold":  -0.5,
    # Review
    "review_before_save":    False,
    # Translation
    "custom_prompt":         "",
    # Japanese
    "furigana":              False,
    "freq_list_path":        "",
    "freq_overlay":          False,
    "known_words_path":      "",
    "pitch_accent_path":     "",
    "grammar_patterns":      DEFAULT_GRAMMAR_PATTERNS,
    # Anki
    "anki_enabled":          False,
    "anki_url":              "http://localhost:8765",
    "anki_deck":             "Anime Mining",
    "anki_note_type":        "MpvMining",
    "anki_sentence_fld":     "Sentence",
    "anki_meaning_fld":      "Meaning",
    "anki_audio_fld":        "Audio",
    "anki_img_fld":          "Picture",
    "anki_def_fld":          "Definition",  # AI-generated abstract definition
    "anki_ai_definition":    False,         # generate AI definition for each card
    "anki_def_batch":        True,          # True = one batch prompt per episode; False = one prompt per card
    "anki_tag_episodes":     True,
    "anki_screenshot":       False,
    "anki_per_card_audio":   False,
    "anki_morph_deck":       "",
    "anki_morph_field":      "Expression",
    # Deduplication / repair
    "dedup_subs":            False,
    "repair_timestamps":     False,
    "cross_ep_dedup":        False,   # skip lines already seen in previous episodes this session
    "anki_max_cards":        0,       # 0 = all lines; N = top N cards by frequency rank
    "anki_skip_known":       False,   # skip fully-known lines when exporting to Anki
    # UI
    "theme":                 "dark",
    # Updates
    "update_url":            "",   # raw URL to version.json on your Gist
    "auto_check_updates":    True,
    # AI backend
    "ai_backend":            "ollama",
    "translation_workers":   1,
    "grammar_gloss":         True,
    "ai_temperature":       0.3,    # 0.0=deterministic, 1.0=creative
    "ai_max_tokens":        3000,   # max tokens per AI response
    # Generic provider key + model store
    "api_keys":              {},   # {"groq": "sk-...", "gemini": "AI...", ...}
    "api_models":            {},   # {"groq": "llama-3.3-70b-versatile", ...}
    # Per-action backend overrides — each is {"pid","key","model"}, "" = use default
    "action_overrides":      {
        "translation":   {"pid": "", "key": "", "model": ""},
        "anki_def":      {"pid": "", "key": "", "model": ""},
        "ep_summary":    {"pid": "", "key": "", "model": ""},
        "ai_title":      {"pid": "", "key": "", "model": ""},
        "quality_check": {"pid": "", "key": "", "model": ""},
    },
    # Whether to include EN localised subs as reference context when available
    "en_ref_subs":           True,
    # Translation memory — cache JP→EN and reuse on repeated files
    "use_translation_memory": True,
    # Show glossary — inject per-show term consistency block into prompts
    "use_glossary":           True,
    # Auto-extract glossary from the first episode of a new show
    "auto_extract_glossary":  True,
    # Optional: consistency pass after translation (fixes name variations)
    "consistency_pass":       False,
    # Legacy — kept for migration from older settings files
    "openai_api_key":        "",
    "anthropic_api_key":     "",
    "gemini_api_key":        "",
}

# ══════════════════════════════════════════════════════════════════════════════
# THEMES
# ══════════════════════════════════════════════════════════════════════════════

THEMES = {
    "dark": dict(
        BG="#0d1117", PANEL="#161b22", PANEL2="#21262d", BORDER="#30363d",
        ACCENT="#e05c2a", ACCENT2="#b94a1e", TEXT="#e6edf3", MUTED="#7d8590",
        SUCCESS="#3fb950", ERROR="#f85149", WARN="#d29922",
        ENTRY_BG="#0d1117", LISTBOX_BG="#0d1117",
        BTN_WHITE="#21262d", BTN_WHITE_FG="#e6edf3",
        HIST_ODD="#161b22", HIST_EVEN="#1c2128",
    ),
    "light": dict(
        BG="#ffffff", PANEL="#f6f8fa", PANEL2="#eaeef2", BORDER="#d0d7de",
        ACCENT="#e05c2a", ACCENT2="#b94a1e", TEXT="#1f2328", MUTED="#656d76",
        SUCCESS="#1a7f37", ERROR="#d1242f", WARN="#9a6700",
        ENTRY_BG="#ffffff", LISTBOX_BG="#f6f8fa",
        BTN_WHITE="#f6f8fa", BTN_WHITE_FG="#1f2328",
        HIST_ODD="#f6f8fa", HIST_EVEN="#ffffff",
    ),
}

# ── Fonts ─────────────────────────────────────────────────────────────────────

FONT      = ("Helvetica Neue", 13)
FONT_SM   = ("Helvetica Neue", 12)
FONT_XS   = ("Helvetica Neue", 11)
FONT_MONO = ("Menlo", 11)   # log, code, raw text areas


# ══════════════════════════════════════════════════════════════════════════════
# IMMERSION STUDIO — AI PROVIDER LAYER (Phase 1 extension)
# ══════════════════════════════════════════════════════════════════════════════

# ── Immersion Studio paths ────────────────────────────────────────────────────
IS_APP_SUPPORT_DIR: Path = (
    Path.home() / "Library" / "Application Support" / "ImmersionStudio"
)
IS_SETTINGS_PATH: Path = IS_APP_SUPPORT_DIR / "settings.json"
IS_MODEL_CACHE_PATH: Path = IS_APP_SUPPORT_DIR / "model_cache.json"

IS_APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Keychain ──────────────────────────────────────────────────────────────────
KEYCHAIN_SERVICE: str = "ImmersionStudio"

# ── Model cache TTL (days; 0 = always refetch) ────────────────────────────────
MODEL_CACHE_TTL: dict[str, int] = {
    "ollama":     0,   # always fresh — reflects `ollama pull` immediately
    "lmstudio":   0,
    "openrouter": 7,
    # all others default to 14 (see _is_cache_fresh)
}

# ── 18-provider PROVIDERS dict ─────────────────────────────────────────────────
# Keys match provider_id used throughout the app (same as AI_PROVIDERS tuples).
# Additional entries beyond the 15 in AI_PROVIDERS: fireworks, lepton, anyscale.
PROVIDERS: dict[str, dict] = {
    "ollama": {
        "name": "Ollama (local)", "base_url": None,
        "default_model": "qwen2.5:7b",
        "models_api": "http://localhost:11434/api/tags",
        "key_env": None,
        "note": "Free — runs on your Mac",
    },
    "lmstudio": {
        "name": "LM Studio", "base_url": "http://localhost:1234/v1",
        "default_model": "",
        "models_api": "http://localhost:1234/v1/models",
        "key_env": None,
        "note": "Free local — LM Studio app",
    },
    "openai": {
        "name": "OpenAI", "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models_api": "https://api.openai.com/v1/models",
        "key_env": "OPENAI_API_KEY",
        "note": "~$0.02/ep (gpt-4.1-mini)",
    },
    "anthropic": {
        "name": "Anthropic", "base_url": "anthropic",
        "default_model": "claude-haiku-4-5-20251001",
        "models_api": "https://api.anthropic.com/v1/models",
        "key_env": "ANTHROPIC_API_KEY",
        "note": "~$0.05/ep (Haiku) — excellent JP",
    },
    "gemini": {
        "name": "Google Gemini", "base_url": "gemini",
        "default_model": "gemini-2.0-flash",
        "models_api": "https://generativelanguage.googleapis.com/v1beta/models",
        "key_env": "GOOGLE_API_KEY",
        "note": "Free tier: 1500 req/day",
    },
    "groq": {
        "name": "Groq", "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "models_api": "https://api.groq.com/openai/v1/models",
        "key_env": "GROQ_API_KEY",
        "note": "Free tier — very fast",
    },
    "deepseek": {
        "name": "DeepSeek", "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "models_api": "https://api.deepseek.com/v1/models",
        "key_env": "DEEPSEEK_API_KEY",
        "note": "~$0.01/ep — very cheap",
    },
    "openrouter": {
        "name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
        "default_model": "anthropic/claude-haiku-4-5",
        "models_api": "https://openrouter.ai/api/v1/models",
        "key_env": "OPENROUTER_API_KEY",
        "note": "200+ models, free tier options",
    },
    "mistral": {
        "name": "Mistral", "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-small-latest",
        "models_api": "https://api.mistral.ai/v1/models",
        "key_env": "MISTRAL_API_KEY",
        "note": "~$0.02/ep",
    },
    "together": {
        "name": "Together AI", "base_url": "https://api.together.xyz/v1",
        "default_model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "models_api": "https://api.together.xyz/v1/models",
        "key_env": "TOGETHER_API_KEY",
        "note": "Pay-per-token, many models",
    },
    "xai": {
        "name": "xAI (Grok)", "base_url": "https://api.x.ai/v1",
        "default_model": "grok-3-mini",
        "models_api": "https://api.x.ai/v1/models",
        "key_env": "XAI_API_KEY",
        "note": "Strong reasoning, competitive pricing",
    },
    "cerebras": {
        "name": "Cerebras", "base_url": "https://api.cerebras.ai/v1",
        "default_model": "llama-3.3-70b",
        "models_api": "https://api.cerebras.ai/v1/models",
        "key_env": "CEREBRAS_API_KEY",
        "note": "Extremely fast inference",
    },
    "cohere": {
        "name": "Cohere", "base_url": "https://api.cohere.com/v1",
        "default_model": "command-r-plus",
        "models_api": "https://api.cohere.com/v1/models",
        "key_env": "COHERE_API_KEY",
        "note": "Good multilingual support",
    },
    "fireworks": {
        "name": "Fireworks AI", "base_url": "https://api.fireworks.ai/inference/v1",
        "default_model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "models_api": "https://api.fireworks.ai/inference/v1/models",
        "key_env": "FIREWORKS_API_KEY",
        "note": "Fast open-source models",
    },
    "kimi": {
        "name": "Kimi (Moonshot)", "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
        "models_api": "https://api.moonshot.cn/v1/models",
        "key_env": "MOONSHOT_API_KEY",
        "note": "Good JP support",
    },
    "perplexity": {
        "name": "Perplexity", "base_url": "https://api.perplexity.ai",
        "default_model": "sonar",
        "models_api": None,  # static list
        "static_models": ["sonar", "sonar-pro", "sonar-reasoning",
                          "llama-3.1-sonar-large-128k-online"],
        "key_env": "PERPLEXITY_API_KEY",
        "note": "Online models with web access",
    },
    "lepton": {
        "name": "Lepton AI", "base_url": "https://llama3-3-70b.lepton.run/api/v1",
        "default_model": "llama3.3-70b",
        "models_api": None,  # static list
        "static_models": ["llama3.3-70b", "mistral-7b"],
        "key_env": "LEPTON_API_KEY",
        "note": "Affordable cloud inference",
    },
    "anyscale": {
        "name": "Anyscale", "base_url": "https://api.endpoints.anyscale.com/v1",
        "default_model": "meta-llama/Llama-3-70b-chat-hf",
        "models_api": None,  # static list
        "static_models": ["meta-llama/Llama-3-70b-chat-hf",
                          "mistralai/Mixtral-8x7B-Instruct-v0.1"],
        "key_env": "ANYSCALE_API_KEY",
        "note": "Managed Ray compute",
    },
}


# ── AIConfig dataclass ────────────────────────────────────────────────────────

@dataclass
class AIConfig:
    """Immersion Studio AI feature configuration.

    Non-secret settings; persisted to settings.json via atomic write.
    API keys go to Keychain only (never serialized here).
    """
    default_provider: str = "ollama"
    api_models: dict[str, str] = field(default_factory=dict)
    action_overrides: dict[str, dict] = field(default_factory=lambda: {
        "subtitle_translation": {"provider": "ollama", "model": "qwen2.5:7b"},
        "anki_concept_def":     {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    })
    enable_grammar_explain: bool = False  # off by default; never auto-called


# ── Keychain helpers ──────────────────────────────────────────────────────────

def save_api_key(provider_id: str, key: str) -> None:
    """Store an API key in macOS Keychain.

    Args:
        provider_id: Provider ID string (e.g. 'anthropic').
        key: The API key string to store.

    Raises:
        keyring.errors.KeyringError: If Keychain access is denied.
    """
    import keyring
    keyring.set_password(KEYCHAIN_SERVICE, provider_id, key)


def load_api_key(provider_id: str) -> str | None:
    """Load an API key from macOS Keychain.

    Args:
        provider_id: Provider ID string.

    Returns:
        API key string if found, None if not set.
        Returns None (not raises) if Keychain is locked or key not found.
    """
    try:
        import keyring
        return keyring.get_password(KEYCHAIN_SERVICE, provider_id)
    except Exception as e:
        logger.error("load_api_key(%s) failed: %s", provider_id, e, exc_info=True)
        return None


def delete_api_key(provider_id: str) -> None:
    """Delete an API key from macOS Keychain.

    Args:
        provider_id: Provider ID string.
    """
    try:
        import keyring
        keyring.delete_password(KEYCHAIN_SERVICE, provider_id)
    except Exception as e:
        logger.error("delete_api_key(%s) failed: %s", provider_id, e, exc_info=True)


# ── Model cache helpers ───────────────────────────────────────────────────────

def _is_cache_fresh(provider_id: str, fetched_at: str) -> bool:
    """Return True if the cached model list is still within TTL.

    Args:
        provider_id: Provider ID string.
        fetched_at: ISO-format datetime string of when the cache was populated.

    Returns:
        True if cache should be used; False if a fresh fetch is needed.
    """
    ttl_days = MODEL_CACHE_TTL.get(provider_id, 14)
    if ttl_days == 0:
        return False
    try:
        fetched = datetime.fromisoformat(fetched_at)
        return datetime.now() - fetched < timedelta(days=ttl_days)
    except Exception as e:
        logger.error("_is_cache_fresh: bad fetched_at value '%s': %s", fetched_at, e, exc_info=True)
        return False


def fetch_model_list(provider_id: str, api_key: str | None, base_url: str | None) -> list[str]:
    """Fetch live model list from a provider's API.

    Args:
        provider_id: Provider ID string.
        api_key: API key (None for local providers).
        base_url: Provider base URL (None for Ollama).

    Returns:
        List of model name strings.

    Raises:
        RuntimeError: If the request fails or returns an unexpected format.
    """
    import requests

    provider = PROVIDERS.get(provider_id, {})
    static_models = provider.get("static_models")
    if static_models:
        return list(static_models)

    models_api = provider.get("models_api")
    if not models_api:
        return []

    headers: dict[str, str] = {}
    if api_key:
        if provider_id == "anthropic":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {api_key}"

    try:
        params: dict[str, str] = {}
        if provider_id == "gemini" and api_key:
            params["key"] = api_key

        resp = requests.get(models_api, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if provider_id == "ollama":
            return [m["name"] for m in data.get("models", [])]
        if provider_id == "gemini":
            return [m.get("name", "").split("/")[-1] for m in data.get("models", [])]
        # OpenAI-compatible format
        models_list = data.get("data", data.get("models", []))
        return [m.get("id", "") for m in models_list if m.get("id")]
    except Exception as e:
        logger.error("fetch_model_list(%s) failed: %s", provider_id, e, exc_info=True)
        raise RuntimeError(f"fetch_model_list({provider_id}): {e}") from e


# ── Qt model fetch worker ──────────────────────────────────────────────────────

try:
    from PySide6.QtCore import QObject, QRunnable, Signal

    class ModelFetchSignals(QObject):
        """Signals emitted by ModelFetchWorker.

        Emits:
            finished(provider_id, model_list): On successful fetch.
            error(provider_id, error_message): On failure.
        """
        finished = Signal(str, list)   # (provider_id, model_list)
        error    = Signal(str, str)    # (provider_id, error_message)

    class ModelFetchWorker(QRunnable):
        """Fetch model list for one provider on a background thread.

        Uses setAutoDelete(True) — do not hold a reference after
        QThreadPool.globalInstance().start(worker).

        Args:
            provider_id: Provider ID string.
            api_key: API key (None for local providers).
            base_url: Provider base URL (None for Ollama).
        """

        def __init__(self, provider_id: str, api_key: str | None, base_url: str | None):
            super().__init__()
            self.setAutoDelete(True)
            self.signals = ModelFetchSignals()
            self.provider_id = provider_id
            self.api_key = api_key
            self.base_url = base_url

        # ⚠️ WORKER THREAD — runs on QThreadPool, never touch Qt widgets directly.
        def run(self) -> None:
            """Fetch model list and emit signals."""
            try:
                models = fetch_model_list(self.provider_id, self.api_key, self.base_url)
                self.signals.finished.emit(self.provider_id, models)
            except Exception as e:
                logger.error(
                    "ModelFetchWorker(%s) failed: %s", self.provider_id, e, exc_info=True
                )
                self.signals.error.emit(self.provider_id, str(e))

    def watch_ollama_models(on_change) -> None:
        """Watch ~/.ollama/models/ for changes and refresh model list on change.

        Uses a QFileSystemWatcher. The watcher is stored on the calling object
        so it stays alive. on_change is stored via weakref to avoid preventing
        GC of the owning widget.

        Args:
            on_change: Callable to invoke when Ollama models directory changes.
                       Typically a bound method on a widget — weakref is used
                       internally.
        """
        from PySide6.QtCore import QFileSystemWatcher
        from pathlib import Path

        ollama_models_dir = Path.home() / ".ollama" / "models"
        if not ollama_models_dir.exists():
            return

        watcher = QFileSystemWatcher([str(ollama_models_dir)])
        on_change_ref = (
            weakref.ref(on_change)
            if hasattr(on_change, "__self__")
            else None
        )

        def _safe_refresh(path: str) -> None:
            cb = on_change_ref() if on_change_ref else on_change
            if cb is not None:
                api_key = load_api_key("ollama")
                worker = ModelFetchWorker("ollama", api_key, None)
                worker.signals.finished.connect(
                    lambda pid, models: cb(pid, models)
                )
                from PySide6.QtCore import QThreadPool
                QThreadPool.globalInstance().start(worker)

        watcher.directoryChanged.connect(_safe_refresh)
        # Return watcher so caller can keep it alive as self._ollama_watcher
        return watcher

except ImportError:
    # PySide6 not available — stubs so the module can be imported in tests
    class ModelFetchSignals:  # type: ignore[no-redef]
        pass

    class ModelFetchWorker:  # type: ignore[no-redef]
        pass

    def watch_ollama_models(on_change):  # type: ignore[no-redef]
        pass


# ── Settings I/O with atomic write ────────────────────────────────────────────

def load_is_settings() -> dict:
    """Load Immersion Studio settings.json.

    Returns:
        Settings dict. Returns minimal defaults if file does not exist.
    """
    try:
        with open(IS_SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.error("load_is_settings failed: %s", e, exc_info=True)
        return {}


def save_is_settings(settings: dict) -> None:
    """Save Immersion Studio settings atomically (write .tmp then os.replace).

    Args:
        settings: Settings dict. Must not contain API keys.

    Raises:
        OSError: If the write fails.
    """
    IS_APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = IS_SETTINGS_PATH.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp, IS_SETTINGS_PATH)
    except Exception as e:
        logger.error("save_is_settings failed: %s", e, exc_info=True)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_model_cache() -> dict:
    """Load cached model lists from model_cache.json.

    Returns:
        Dict mapping provider_id → {models: [...], fetched_at: ISO str}.
        Returns empty dict if file does not exist.
    """
    try:
        with open(IS_MODEL_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.error("load_model_cache failed: %s", e, exc_info=True)
        return {}


def save_model_cache(cache: dict) -> None:
    """Atomically write model_cache.json.

    Args:
        cache: Dict mapping provider_id → {models: [...], fetched_at: ISO str}.
    """
    IS_APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = IS_MODEL_CACHE_PATH.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, IS_MODEL_CACHE_PATH)
    except Exception as e:
        logger.error("save_model_cache failed: %s", e, exc_info=True)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ══════════════════════════════════════════════════════════════════════════════
# RESOURCE CACHE  (Phase 2 Step 4)
# ══════════════════════════════════════════════════════════════════════════════

import threading  # noqa: E402 — late import keeps top-of-file imports clean


class ResourceCache:
    """Thread-safe, lazily populated key→value cache.

    Uses a ``threading.Lock`` for all check-then-set operations so the GIL
    alone cannot cause TOCTOU races in the populate callback.

    Example::

        freq_cache = ResourceCache()
        def _load(path):
            return load_freq_list(path)
        data = freq_cache.get("freq_list", _load)   # first call: populates
        data = freq_cache.get("freq_list", _load)   # subsequent: from cache

    Args:
        None  — instantiate per-resource or as a module-level singleton.
    """

    def __init__(self) -> None:
        self._store:  dict = {}
        self._lock:   threading.Lock = threading.Lock()

    def get(self, key: str, populate: object) -> object:
        """Return the cached value for *key*, calling *populate(key)* to fill it.

        ``populate`` is only called once per key even if multiple threads race
        to call ``get()`` simultaneously.

        Args:
            key:      Cache key string.
            populate: Callable(key) → value.  Called exactly once per key.

        Returns:
            Cached value.
        """
        with self._lock:
            if key not in self._store:
                self._store[key] = populate(key)
            return self._store[key]

    def invalidate(self, key: str) -> None:
        """Remove *key* from the cache so the next ``get()`` repopulates it.

        Args:
            key: Cache key to evict.
        """
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
