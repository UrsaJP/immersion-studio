# Immersion Studio — Claude Code Context

## What this app is
Companion to the user's existing standalone MPV setup. No MPV embedded.
Provides: batch MKV subtitle translation, chorusing practice,
HVPT pitch training (minimal-pairs), NWD scoring, immersion tracking dashboard.

## Architecture
- `core/` — backend (mostly copied from AIST; see Source Material)
- `ui/` — PySide6 widgets; main_window.py is the entry hub

## Key Relationships
- translation_widget.py → core/translation.py (ffprobe/ffmpeg + AI provider)
- settings_widget.py → writes settings.json (provider/model config)
- core/nwd.py → core/japanese.py (MeCab) + core/db.py (known_vocab from AnkiMorphs)
- chorusing_widget.py → core/chorusing.py (DTW scoring in QThreadPool)
- tracker_widget.py → core/session.py + core/db.py (sessions, media_info)
- pitch_widget.py → resources/minimal-pairs/index.html (QWebEngineView, local)

## Copied from AIST (do not modify logic — only extend):
core/db.py, core/audio.py, core/anki.py, core/japanese.py
core/subtitle.py, core/utils.py, core/config.py, core/glossary.py

## Run / Test
python main.py                   — launch app
pytest tests/ -q --tb=short      — run tests
Log: ~/Library/Application Support/ImmersionStudio/logs/immersion_studio.log

## DB
~/Library/Application Support/ImmersionStudio/immersion_studio.db

## Key Gotchas
- AI config: settings.json stores provider/model (non-secret). API keys go in Keychain only.
  Use load_api_key(provider_id) — never read settings.json for keys.
  PyInstaller rebuild changes codesig → re-prompt for Keychain "Always Allow".
- known_vocab is sourced from AnkiMorphs TSV export (import_ankimorph_vocab()).
  NOT from AnkiConnect mature card sync. Re-sync button in Settings → Data.
- librosa.pyin() returns NaN for unvoiced frames. Always filter with ~np.isnan() before DTW.
- All chorusing audio processing (pyin, DTW, MFCC) runs in ScoringWorker(QRunnable).
  Never on Qt main thread — blocks 1–3 seconds. Use Qt.QueuedConnection for signal.
- All MKV translation runs in TranslationWorker(QRunnable). Never on Qt main thread.
- ResourceCache uses threading.Lock for TOCTOU safety — GIL alone does NOT protect check-then-set.
- chorusing scores are relative vs user's own baseline — not calibrated absolute values.
  Do not label "84/100" as meaningful until 50+ recordings collected.
- Atomic writes for settings.json and model_cache.json: write .tmp then os.replace().
  A crash during non-atomic write corrupts user config.
- minimal-pairs is a static local site loaded via QUrl.fromLocalFile — no network needed.
  Clone https://github.com/Kuuuube/minimal-pairs into resources/minimal-pairs/ and commit.

## Module Responsibilities
- core/pipeline.py — Whisper transcription (optional); precompute_subtitle_tokens() NWD pass
- core/translation.py — list_subtitle_tracks(), translate_srt(), mux_subtitle_into_mkv(), TranslationWorker
- core/nwd.py — calculate_nwd(), get_frequent_unknowns(), import_ankimorph_vocab()
- core/chorusing.py — extract_pitch(), normalize_f0(), compute_closeness(), DTWScoringEngine
- core/session.py — manual session start/stop; active/passive minute tracking
- core/backup.py — auto_backup_if_needed(), create_backup(), restore_backup()
