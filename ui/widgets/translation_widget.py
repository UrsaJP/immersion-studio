"""Immersion Studio — Batch MKV Translation panel.

Single responsibility: let the user pick an MKV, choose a subtitle track and
AI provider, then run TranslationWorker in a QThreadPool while reporting
progress line-by-line.  The original file is never modified.

Used by: ui/main_window.py (PANEL_DASHBOARD slot for Translation tab).
Reads:   core/config.py (load_is_settings, load_model_cache, load_api_key, AI_PROVIDERS)
         core/translation.py (list_subtitle_tracks, TranslationWorker)
"""

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: new
# DIVERGES_FROM_AIST: True
# Changes: written from scratch per Feature 4 / Step 10 spec.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import weakref

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.config import AI_PROVIDERS, load_api_key, load_is_settings, load_model_cache
from core.translation import TranslationWorker, list_subtitle_tracks

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MKV_FILE_FILTER: str = "MKV files (*.mkv);;All files (*)"
TRACK_COMBO_NO_FILE: str = "(pick a file first)"
TRACK_COMBO_LOADING: str = "Loading tracks…"
TRACK_COMBO_NONE: str = "(no subtitle tracks found)"
STATUS_IDLE: str = "Ready."
STATUS_TRANSLATING: str = "Translating…"
STATUS_DONE_PREFIX: str = "Done → "
STATUS_ERROR_PREFIX: str = "Error: "
PROGRESS_BAR_MIN: int = 0
PROGRESS_BAR_INITIAL_MAX: int = 1   # avoids divide-by-zero before first signal
TRANSLATE_BTN_LABEL: str = "Translate"
BROWSE_BTN_LABEL: str = "Browse…"
REFRESH_MODELS_BTN_LABEL: str = "↻ Refresh models"
PROVIDER_COMBO_PLACEHOLDER: str = "— select provider —"


class TranslationWidget(QWidget):
    """Batch MKV subtitle translation panel.

    Layout (top → bottom):
        File row:    [path label (read-only)]  [Browse button]
        Track row:   [subtitle track combo]
        Provider row:[provider combo]           [model combo]  [↻ button]
        Action row:  [Translate button]
        Progress:    [QProgressBar]
        Status:      [status label]

    Worker dispatch:
        TranslationWorker runs on QThreadPool.globalInstance().
        All signal connections use weakref + Qt.QueuedConnection to prevent
        accessing a destroyed widget from a background thread.

    Args:
        parent: Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.TabFocus)

        self._mkv_path: str = ""
        self._tracks: list[dict] = []        # output of list_subtitle_tracks()
        self._job_running: bool = False

        self._build_ui()
        self._populate_providers()

    # ══════════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        """Construct and lay out all child widgets."""
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # ── File picker row ───────────────────────────────────────────────────
        file_row = QHBoxLayout()
        self._path_label = QLabel("(no file selected)")
        self._path_label.setFocusPolicy(Qt.TabFocus)
        self._path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._path_label.setToolTip("Path to source MKV file")

        self._browse_btn = QPushButton(BROWSE_BTN_LABEL)
        self._browse_btn.setFocusPolicy(Qt.TabFocus)
        self._browse_btn.clicked.connect(self._on_browse)

        file_row.addWidget(self._path_label, stretch=1)
        file_row.addWidget(self._browse_btn)

        file_container = QWidget()
        file_container.setLayout(file_row)
        form.addRow("MKV file:", file_container)

        # ── Subtitle track combo ──────────────────────────────────────────────
        self._track_combo = QComboBox()
        self._track_combo.setFocusPolicy(Qt.TabFocus)
        self._track_combo.addItem(TRACK_COMBO_NO_FILE)
        self._track_combo.setEnabled(False)
        self._track_combo.setToolTip("Subtitle track to translate")
        form.addRow("Subtitle track:", self._track_combo)

        # ── Provider + model row ──────────────────────────────────────────────
        prov_row = QHBoxLayout()

        self._provider_combo = QComboBox()
        self._provider_combo.setFocusPolicy(Qt.TabFocus)
        self._provider_combo.setToolTip("AI provider for translation")
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)

        self._model_combo = QComboBox()
        self._model_combo.setFocusPolicy(Qt.TabFocus)
        self._model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._model_combo.setToolTip("Model to use for translation")

        self._refresh_btn = QPushButton(REFRESH_MODELS_BTN_LABEL)
        self._refresh_btn.setFocusPolicy(Qt.TabFocus)
        self._refresh_btn.setToolTip("Reload model list from cache")
        self._refresh_btn.clicked.connect(self._on_refresh_models)

        prov_row.addWidget(self._provider_combo, stretch=1)
        prov_row.addWidget(self._model_combo, stretch=2)
        prov_row.addWidget(self._refresh_btn)

        prov_container = QWidget()
        prov_container.setLayout(prov_row)
        form.addRow("Provider / Model:", prov_container)

        root.addLayout(form)

        # ── Translate button ──────────────────────────────────────────────────
        self._translate_btn = QPushButton(TRANSLATE_BTN_LABEL)
        self._translate_btn.setFocusPolicy(Qt.TabFocus)
        self._translate_btn.setEnabled(False)
        self._translate_btn.clicked.connect(self._on_translate)
        root.addWidget(self._translate_btn)

        # ── Progress bar ──────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setFocusPolicy(Qt.TabFocus)
        self._progress.setMinimum(PROGRESS_BAR_MIN)
        self._progress.setMaximum(PROGRESS_BAR_INITIAL_MAX)
        self._progress.setValue(PROGRESS_BAR_MIN)
        self._progress.setFormat("%v / %m lines")
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        # ── Status label ──────────────────────────────────────────────────────
        self._status_label = QLabel(STATUS_IDLE)
        self._status_label.setFocusPolicy(Qt.TabFocus)
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        root.addWidget(self._status_label)

        root.addStretch()

    # ══════════════════════════════════════════════════════════════════════════
    # PROVIDERS / MODELS
    # ══════════════════════════════════════════════════════════════════════════

    def _populate_providers(self) -> None:
        """Fill the provider combo from AI_PROVIDERS and restore saved selection."""
        settings = load_is_settings()
        saved_provider = settings.get("subtitle_translation_provider", "")

        self._provider_combo.blockSignals(True)
        self._provider_combo.clear()
        for pid, pname, *_ in AI_PROVIDERS:
            self._provider_combo.addItem(pname, userData=pid)

        # Restore saved provider, fall back to first entry
        idx = 0
        if saved_provider:
            for i in range(self._provider_combo.count()):
                if self._provider_combo.itemData(i) == saved_provider:
                    idx = i
                    break
        self._provider_combo.setCurrentIndex(idx)
        self._provider_combo.blockSignals(False)

        self._populate_models()

    def _populate_models(self) -> None:
        """Fill the model combo for the currently selected provider."""
        provider_id: str = self._provider_combo.currentData() or ""
        if not provider_id:
            return

        # Try live model cache first
        cache = load_model_cache()
        cached_models: list[str] = cache.get(provider_id, [])

        # Fall back to static suggested models from AI_PROVIDERS
        static_models: list[str] = []
        default_model: str = ""
        for pid, _pname, _base, def_model, suggested, *_ in AI_PROVIDERS:
            if pid == provider_id:
                static_models = suggested
                default_model = def_model
                break

        models = cached_models if cached_models else static_models

        settings = load_is_settings()
        saved_model = settings.get("subtitle_translation_model", default_model)

        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in models:
            self._model_combo.addItem(m)

        # Restore saved model or default
        target = saved_model if saved_model in models else (default_model if default_model in models else "")
        if target:
            idx = self._model_combo.findText(target)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
        self._model_combo.blockSignals(False)

    # ══════════════════════════════════════════════════════════════════════════
    # FILE + TRACK HANDLING
    # ══════════════════════════════════════════════════════════════════════════

    def _on_browse(self) -> None:
        """Open a file dialog for the user to choose an MKV."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MKV file", "", MKV_FILE_FILTER
        )
        if not path:
            return
        self._mkv_path = path
        self._path_label.setText(path)
        self._path_label.setToolTip(path)
        self._load_tracks()

    def _load_tracks(self) -> None:
        """Populate the subtitle track combo from the selected MKV."""
        self._track_combo.clear()
        self._track_combo.addItem(TRACK_COMBO_LOADING)
        self._track_combo.setEnabled(False)
        self._translate_btn.setEnabled(False)

        try:
            self._tracks = list_subtitle_tracks(self._mkv_path)
        except RuntimeError as exc:
            logger.error("list_subtitle_tracks failed: %s", exc)
            self._tracks = []
            self._track_combo.clear()
            self._track_combo.addItem(f"Error: {exc}")
            self._status_label.setText(f"{STATUS_ERROR_PREFIX}{exc}")
            return

        self._track_combo.clear()
        if not self._tracks:
            self._track_combo.addItem(TRACK_COMBO_NONE)
            self._track_combo.setEnabled(False)
            self._translate_btn.setEnabled(False)
            self._status_label.setText(STATUS_IDLE)
            return

        for t in self._tracks:
            lang = t.get("language", "und")
            title = t.get("title", "")
            codec = t.get("codec", "")
            index = t.get("index", "?")
            label = f"Track {index} [{lang}] {codec}"
            if title:
                label += f" — {title}"
            self._track_combo.addItem(label)

        self._track_combo.setEnabled(True)
        self._translate_btn.setEnabled(True)
        self._status_label.setText(STATUS_IDLE)

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL SLOTS
    # ══════════════════════════════════════════════════════════════════════════

    def _on_provider_changed(self, _index: int) -> None:
        """Reload model combo when the provider selection changes."""
        self._populate_models()

    def _on_refresh_models(self) -> None:
        """Re-read model cache and repopulate model combo."""
        self._populate_models()

    def _on_translate(self) -> None:
        """Validate inputs and start a TranslationWorker."""
        if self._job_running:
            return
        if not self._mkv_path:
            self._status_label.setText(f"{STATUS_ERROR_PREFIX}No MKV file selected.")
            return
        if not self._tracks:
            self._status_label.setText(f"{STATUS_ERROR_PREFIX}No subtitle tracks available.")
            return

        track_idx_in_combo = self._track_combo.currentIndex()
        if track_idx_in_combo < 0 or track_idx_in_combo >= len(self._tracks):
            self._status_label.setText(f"{STATUS_ERROR_PREFIX}Invalid track selection.")
            return

        track_stream_index: int = self._tracks[track_idx_in_combo]["index"]
        provider_id: str = self._provider_combo.currentData() or ""
        model: str = self._model_combo.currentText().strip()

        if not provider_id or not model:
            self._status_label.setText(
                f"{STATUS_ERROR_PREFIX}Select a provider and model first."
            )
            return

        api_key = load_api_key(provider_id)

        worker = TranslationWorker(
            mkv_path=self._mkv_path,
            track_index=track_stream_index,
            provider_id=provider_id,
            model=model,
            api_key=api_key,
        )

        # ── weakref connections (Qt.QueuedConnection) ─────────────────────────
        weak_self = weakref.ref(self)

        def _on_progress(current: int, total: int) -> None:
            w = weak_self()
            if w is not None:
                w._handle_progress(current, total)

        def _on_finished(output_path: str) -> None:
            w = weak_self()
            if w is not None:
                w._handle_finished(output_path)

        def _on_error(message: str) -> None:
            w = weak_self()
            if w is not None:
                w._handle_error(message)

        worker.signals.progress.connect(_on_progress, Qt.QueuedConnection)
        worker.signals.finished.connect(_on_finished, Qt.QueuedConnection)
        worker.signals.error.connect(_on_error, Qt.QueuedConnection)

        # ── Disable UI while running ──────────────────────────────────────────
        self._job_running = True
        self._translate_btn.setEnabled(False)
        self._browse_btn.setEnabled(False)
        self._progress.setValue(PROGRESS_BAR_MIN)
        self._progress.setMaximum(PROGRESS_BAR_INITIAL_MAX)
        self._progress.setVisible(True)
        self._status_label.setText(STATUS_TRANSLATING)

        QThreadPool.globalInstance().start(worker)
        logger.info(
            "TranslationWorker started: file=%s track=%d provider=%s model=%s",
            self._mkv_path, track_stream_index, provider_id, model,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # WORKER CALLBACKS  (called from QueuedConnection — runs on main thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_progress(self, current: int, total: int) -> None:
        """Update progress bar as each subtitle entry is translated.

        Args:
            current: Number of lines translated so far.
            total: Total number of lines in the subtitle file.
        """
        self._progress.setMaximum(total)
        self._progress.setValue(current)

    def _handle_finished(self, output_path: str) -> None:
        """React to successful job completion.

        Args:
            output_path: Absolute path to the newly created MKV file.
        """
        self._job_running = False
        self._translate_btn.setEnabled(True)
        self._browse_btn.setEnabled(True)
        self._progress.setValue(self._progress.maximum())
        self._status_label.setText(f"{STATUS_DONE_PREFIX}{output_path}")
        logger.info("TranslationWorker finished: output=%s", output_path)

    def _handle_error(self, message: str) -> None:
        """React to a job failure.

        Args:
            message: Human-readable error description from TranslationWorker.
        """
        self._job_running = False
        self._translate_btn.setEnabled(True)
        self._browse_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"{STATUS_ERROR_PREFIX}{message}")
        logger.error("TranslationWorker error: %s", message)
