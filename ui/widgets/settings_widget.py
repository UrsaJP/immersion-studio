"""Immersion Studio Settings panel.

Single responsibility: AI Providers tab (provider selection, model list,
API key management, per-task overrides) and System Health tab (dependency
status checks with pull-on-open pattern).

Imports from: core/config.py (PROVIDERS, load/save, Keychain, ModelFetchWorker),
              core/db.py (DatabaseManager for AnkiMorphs row count).
Used by: ui/main_window.py (_create_panel).

Signals emitted: none (self-contained settings panel).
"""

# ── Constants ─────────────────────────────────────────────────────────────────
import json
import logging
import shutil
import weakref
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QGroupBox, QFormLayout,
    QScrollArea, QFrame, QFileDialog, QSizePolicy,
    QAbstractItemView,
)

from core.config import (
    PROVIDERS, load_api_key, save_api_key, delete_api_key,
    load_is_settings, save_is_settings, load_model_cache, save_model_cache,
    ModelFetchWorker, _is_cache_fresh,
)

logger = logging.getLogger(__name__)

# ── Appearance ─────────────────────────────────────────────────────────────────
_DARK_BG:     str = "#0d1117"
_DARK_PANEL:  str = "#161b22"
_DARK_BORDER: str = "#30363d"
_DARK_TEXT:   str = "#e6edf3"
_DARK_MUTED:  str = "#7d8590"
_DARK_ACCENT: str = "#e05c2a"
_DARK_GREEN:  str = "#3fb950"
_DARK_AMBER:  str = "#d29922"
_DARK_RED:    str = "#f85149"

# ── Status indicator strings ───────────────────────────────────────────────────
STATUS_OK:      str = "● OK"
STATUS_OFFLINE: str = "○ Offline"
STATUS_AMBER:   str = "◐ Degraded"

# ── Action override keys ───────────────────────────────────────────────────────
ACTION_SUBTITLE_TRANSLATION: str = "subtitle_translation"
ACTION_ANKI_CONCEPT_DEF:     str = "anki_concept_def"
ACTION_OVERRIDE_KEYS: list[str] = [ACTION_SUBTITLE_TRANSLATION, ACTION_ANKI_CONCEPT_DEF]


class SettingsWidget(QWidget):
    """Settings panel with AI Providers, Data, and System Health tabs.

    Tabs:
        Tab 1 — AI Providers: provider selection, model list, API key table,
                per-task overrides, Save / Reset buttons.
        Tab 2 — Data: AnkiMorphs vocab sync, JLPT seed import, backup/restore.
        Tab 3 — System Health: pull-on-open dependency status checks.

    Focus policy: Qt.TabFocus (keyboard accessible).
    """

    def __init__(self, db_path: Path, parent: Optional[QWidget] = None):
        """Initialize SettingsWidget.

        Args:
            db_path: Path to the SQLite database (for AnkiMorphs row count check).
            parent: Optional Qt parent.
        """
        super().__init__(parent)
        self.setFocusPolicy(Qt.TabFocus)
        self._db_path = db_path
        self._model_cache: dict = load_model_cache()
        # Strong refs to keep ModelFetchWorker.signals alive until callbacks fire
        self._pending_signals: list = []

        self._build_ui()
        self._load_settings_into_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct the full settings panel."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        self._tabs = QTabWidget()
        self._tabs.setFocusPolicy(Qt.TabFocus)
        layout.addWidget(self._tabs)

        self._tabs.addTab(self._build_ai_tab(), "AI Providers")
        self._tabs.addTab(self._build_data_tab(), "Data")
        self._tabs.addTab(self._build_health_tab(), "System Health")

        # Load health checks when tab becomes visible
        self._tabs.currentChanged.connect(self._on_tab_changed)

    # ── AI Providers tab ──────────────────────────────────────────────────────

    def _build_ai_tab(self) -> QWidget:
        """Build the AI Providers tab.

        Returns:
            QWidget containing the full AI providers form.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFocusPolicy(Qt.TabFocus)

        container = QWidget()
        container.setFocusPolicy(Qt.TabFocus)
        layout = QVBoxLayout(container)
        layout.setSpacing(16)

        # ── Default provider + model row ──────────────────────────────────────
        default_group = QGroupBox("Default Provider")
        default_group.setFocusPolicy(Qt.TabFocus)
        default_layout = QHBoxLayout(default_group)

        default_layout.addWidget(QLabel("Provider:"))
        self._provider_combo = QComboBox()
        self._provider_combo.setFocusPolicy(Qt.TabFocus)
        for pid, pdata in PROVIDERS.items():
            self._provider_combo.addItem(pdata["name"], pid)
        default_layout.addWidget(self._provider_combo)

        default_layout.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.setFocusPolicy(Qt.TabFocus)
        self._model_combo.setMinimumWidth(200)
        default_layout.addWidget(self._model_combo)

        self._refresh_btn = QPushButton("↻ Refresh")
        self._refresh_btn.setFocusPolicy(Qt.TabFocus)
        self._refresh_btn.setToolTip("Fetch latest model list from provider API")
        self._refresh_btn.clicked.connect(self._on_refresh_models)
        default_layout.addWidget(self._refresh_btn)
        default_layout.addStretch()

        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        layout.addWidget(default_group)

        # ── API Keys table ─────────────────────────────────────────────────────
        keys_group = QGroupBox("API Keys")
        keys_group.setFocusPolicy(Qt.TabFocus)
        keys_layout = QVBoxLayout(keys_group)

        self._keys_table = QTableWidget(0, 3)
        self._keys_table.setFocusPolicy(Qt.TabFocus)
        self._keys_table.setHorizontalHeaderLabels(["Provider", "Key (masked)", "Action"])
        self._keys_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._keys_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._keys_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._keys_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._keys_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._keys_table.setAlternatingRowColors(True)
        keys_layout.addWidget(self._keys_table)

        add_key_btn = QPushButton("+ Add Key")
        add_key_btn.setFocusPolicy(Qt.TabFocus)
        add_key_btn.clicked.connect(self._on_add_key)
        keys_layout.addWidget(add_key_btn, alignment=Qt.AlignRight)
        layout.addWidget(keys_group)

        # ── Per-task overrides ─────────────────────────────────────────────────
        overrides_group = QGroupBox("Per-Task Overrides")
        overrides_group.setFocusPolicy(Qt.TabFocus)
        overrides_layout = QFormLayout(overrides_group)

        self._override_widgets: dict[str, tuple[QComboBox, QComboBox]] = {}
        for action_key in ACTION_OVERRIDE_KEYS:
            label_text = action_key.replace("_", " ").title()
            p_combo = QComboBox()
            p_combo.setFocusPolicy(Qt.TabFocus)
            m_combo = QComboBox()
            m_combo.setFocusPolicy(Qt.TabFocus)
            m_combo.setMinimumWidth(160)
            for pid, pdata in PROVIDERS.items():
                p_combo.addItem(pdata["name"], pid)
            row_widget = QWidget()
            row_widget.setFocusPolicy(Qt.TabFocus)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(p_combo)
            row_layout.addWidget(m_combo)
            overrides_layout.addRow(label_text + ":", row_widget)
            self._override_widgets[action_key] = (p_combo, m_combo)
            p_combo.currentIndexChanged.connect(
                lambda idx, ak=action_key: self._on_override_provider_changed(ak)
            )
        layout.addWidget(overrides_group)

        # ── Save / Reset buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._save_btn = QPushButton("Save")
        self._save_btn.setFocusPolicy(Qt.TabFocus)
        self._save_btn.setDefault(True)
        self._save_btn.clicked.connect(self._on_save)
        self._reset_btn = QPushButton("Reset to Defaults")
        self._reset_btn.setFocusPolicy(Qt.TabFocus)
        self._reset_btn.clicked.connect(self._on_reset)
        btn_row.addStretch()
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(self._reset_btn)
        layout.addLayout(btn_row)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    # ── System Health tab ─────────────────────────────────────────────────────

    # ── Data tab ──────────────────────────────────────────────────────────────

    def _build_data_tab(self) -> QWidget:
        """Build the Data tab.

        Sections:
            1. AnkiMorphs Vocab Sync — pick TSV, import into known_vocab.
            2. JLPT Seed Vocab — pick plain-text / TSV seed file, import.
            3. Database Backup / Restore — manual backup, restore from file.

        Returns:
            QWidget containing all Data section controls.
        """
        w = QWidget()
        w.setFocusPolicy(Qt.TabFocus)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        # ── 1. AnkiMorphs Vocab Sync ──────────────────────────────────────────
        am_group = QGroupBox("AnkiMorphs Vocab Sync")
        am_group.setFocusPolicy(Qt.TabFocus)
        am_layout = QVBoxLayout(am_group)
        am_layout.setSpacing(8)

        am_desc = QLabel(
            "Sync known morphs directly from the AnkiMorphs database. "
            "Run AnkiMorphs recalc in Anki first, then click Sync."
        )
        am_desc.setWordWrap(True)
        am_desc.setStyleSheet(f"color: {_DARK_MUTED};")
        am_layout.addWidget(am_desc)

        # Auto-detect DB path and show it
        from core.nwd import find_ankimorph_db
        detected = find_ankimorph_db()
        self._am_db_path: str = str(detected) if detected else ""
        db_path_display = self._am_db_path or "Not found — run AnkiMorphs recalc in Anki first"
        self._am_path_label = QLabel(db_path_display)
        self._am_path_label.setFocusPolicy(Qt.TabFocus)
        self._am_path_label.setWordWrap(True)
        self._am_path_label.setStyleSheet(
            f"color: {_DARK_TEXT};" if detected else f"color: {_DARK_AMBER};"
        )
        am_layout.addWidget(self._am_path_label)

        # Threshold selector
        threshold_row = QHBoxLayout()
        threshold_label = QLabel("Known = interval ≥")
        threshold_label.setFocusPolicy(Qt.TabFocus)
        self._am_threshold_combo = QComboBox()
        self._am_threshold_combo.setFocusPolicy(Qt.TabFocus)
        self._am_threshold_combo.addItem("0 days  (all morphs AnkiMorphs has seen)", 0)
        self._am_threshold_combo.addItem("1 day   (reviewed at least once)", 1)
        self._am_threshold_combo.addItem("7 days  (learning)", 7)
        self._am_threshold_combo.addItem("21 days (mature)", 21)
        self._am_threshold_combo.setCurrentIndex(3)   # default: mature (≥21 days)
        self._am_threshold_combo.setToolTip(
            "Only morphs whose highest review interval meets this threshold "
            "are counted as 'known' for NWD scoring."
        )
        threshold_row.addWidget(threshold_label)
        threshold_row.addWidget(self._am_threshold_combo, stretch=1)
        am_layout.addLayout(threshold_row)

        am_action_row = QHBoxLayout()
        self._am_sync_btn = QPushButton("Sync from AnkiMorphs DB")
        self._am_sync_btn.setFocusPolicy(Qt.TabFocus)
        self._am_sync_btn.setEnabled(bool(detected))
        self._am_sync_btn.clicked.connect(self._on_am_sync)

        am_browse_btn = QPushButton("Browse…")
        am_browse_btn.setFocusPolicy(Qt.TabFocus)
        am_browse_btn.clicked.connect(self._on_am_browse)
        am_browse_btn.setToolTip("Manually locate ankimorphs.db")

        self._am_status_label = QLabel("")
        self._am_status_label.setFocusPolicy(Qt.TabFocus)
        self._am_status_label.setWordWrap(True)
        am_action_row.addWidget(self._am_sync_btn)
        am_action_row.addWidget(am_browse_btn)
        am_action_row.addWidget(self._am_status_label, stretch=1)
        am_layout.addLayout(am_action_row)

        outer.addWidget(am_group)

        # ── 2. JLPT Seed Vocab ────────────────────────────────────────────────
        seed_group = QGroupBox("JLPT Seed Vocab")
        seed_group.setFocusPolicy(Qt.TabFocus)
        seed_layout = QVBoxLayout(seed_group)
        seed_layout.setSpacing(8)

        seed_desc = QLabel(
            "Bootstrap known_vocab with a JLPT word list or any plain-text "
            "file (one word per line or tab-separated word + reading)."
        )
        seed_desc.setWordWrap(True)
        seed_desc.setStyleSheet(f"color: {_DARK_MUTED};")
        seed_layout.addWidget(seed_desc)

        seed_file_row = QHBoxLayout()
        self._seed_path_label = QLabel("(no file selected)")
        self._seed_path_label.setFocusPolicy(Qt.TabFocus)
        self._seed_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        seed_browse_btn = QPushButton("Browse…")
        seed_browse_btn.setFocusPolicy(Qt.TabFocus)
        seed_browse_btn.clicked.connect(self._on_seed_browse)
        seed_file_row.addWidget(self._seed_path_label, stretch=1)
        seed_file_row.addWidget(seed_browse_btn)
        seed_layout.addLayout(seed_file_row)

        seed_action_row = QHBoxLayout()
        self._seed_import_btn = QPushButton("Import Seed Vocab")
        self._seed_import_btn.setFocusPolicy(Qt.TabFocus)
        self._seed_import_btn.setEnabled(False)
        self._seed_import_btn.clicked.connect(self._on_seed_import)
        self._seed_status_label = QLabel("")
        self._seed_status_label.setFocusPolicy(Qt.TabFocus)
        self._seed_status_label.setWordWrap(True)
        seed_action_row.addWidget(self._seed_import_btn)
        seed_action_row.addWidget(self._seed_status_label, stretch=1)
        seed_layout.addLayout(seed_action_row)

        outer.addWidget(seed_group)

        # ── 3. Database Backup / Restore ──────────────────────────────────────
        bk_group = QGroupBox("Database Backup / Restore")
        bk_group.setFocusPolicy(Qt.TabFocus)
        bk_layout = QVBoxLayout(bk_group)
        bk_layout.setSpacing(8)

        bk_desc = QLabel(
            "Create a manual backup of the Immersion Studio database, or "
            "restore from a previous backup file. Automatic backups run on "
            "launch (kept for 7 days)."
        )
        bk_desc.setWordWrap(True)
        bk_desc.setStyleSheet(f"color: {_DARK_MUTED};")
        bk_layout.addWidget(bk_desc)

        bk_btn_row = QHBoxLayout()
        backup_btn = QPushButton("Create Backup Now")
        backup_btn.setFocusPolicy(Qt.TabFocus)
        backup_btn.clicked.connect(self._on_create_backup)
        restore_btn = QPushButton("Restore from File…")
        restore_btn.setFocusPolicy(Qt.TabFocus)
        restore_btn.clicked.connect(self._on_restore_backup)
        bk_btn_row.addWidget(backup_btn)
        bk_btn_row.addWidget(restore_btn)
        bk_btn_row.addStretch()
        bk_layout.addLayout(bk_btn_row)

        self._backup_status_label = QLabel("")
        self._backup_status_label.setFocusPolicy(Qt.TabFocus)
        self._backup_status_label.setWordWrap(True)
        bk_layout.addWidget(self._backup_status_label)

        outer.addWidget(bk_group)
        outer.addStretch()
        return w

    # ── Data tab slots ─────────────────────────────────────────────────────────

    def _on_am_browse(self) -> None:
        """Manually locate ankimorphs.db."""
        from pathlib import Path as _Path
        default_dir = str(_Path.home() / "Library" / "Application Support" / "Anki2")
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate ankimorphs.db", default_dir,
            "AnkiMorphs DB (ankimorphs.db);;All files (*)",
        )
        if path:
            self._am_db_path = path
            self._am_path_label.setText(path)
            self._am_path_label.setStyleSheet(f"color: {_DARK_TEXT};")
            self._am_sync_btn.setEnabled(True)

    def _on_am_sync(self) -> None:
        """Import morphs directly from the AnkiMorphs SQLite database."""
        if not self._am_db_path:
            return
        min_interval: int = self._am_threshold_combo.currentData() or 0
        try:
            from core.db import DatabaseManager
            from core.nwd import import_ankimorph_db
            db = DatabaseManager(path=self._db_path)
            count = import_ankimorph_db(
                db,
                ankimorph_db_path=self._am_db_path,
                min_interval=min_interval,
            )
            if count == 0:
                self._am_status_label.setText(
                    "⚠  0 morphs at this threshold — try a lower interval or run recalc"
                )
                self._am_status_label.setStyleSheet(f"color: {_DARK_AMBER};")
            else:
                self._am_status_label.setText(f"✔  Synced {count:,} morphs")
                self._am_status_label.setStyleSheet(f"color: {_DARK_GREEN};")
            logger.info(
                "AnkiMorphs DB sync: %d morphs (min_interval=%d) from %s",
                count, min_interval, self._am_db_path,
            )
        except Exception as exc:
            self._am_status_label.setText(f"✖  {exc}")
            self._am_status_label.setStyleSheet(f"color: {_DARK_RED};")
            logger.error("AnkiMorphs DB sync failed: %s", exc, exc_info=True)

    def _on_seed_browse(self) -> None:
        """Open a file picker for the seed vocab file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Seed Vocab File", "",
            "Text files (*.txt *.tsv *.csv);;All files (*)",
        )
        if path:
            self._seed_vocab_path: str = path
            self._seed_path_label.setText(path)
            self._seed_path_label.setToolTip(path)
            self._seed_import_btn.setEnabled(True)

    def _on_seed_import(self) -> None:
        """Import the selected seed vocab file into known_vocab."""
        seed_path = getattr(self, "_seed_vocab_path", "")
        if not seed_path:
            return
        try:
            from core.db import DatabaseManager
            from core.nwd import import_seed_vocab
            db = DatabaseManager(path=self._db_path)
            count = import_seed_vocab(seed_path, db)
            self._seed_status_label.setText(f"✔  Imported {count:,} words")
            self._seed_status_label.setStyleSheet(f"color: {_DARK_GREEN};")
            logger.info("Seed vocab import: %d words from %s", count, seed_path)
        except Exception as exc:
            self._seed_status_label.setText(f"✖  {exc}")
            self._seed_status_label.setStyleSheet(f"color: {_DARK_RED};")
            logger.error("Seed vocab import failed: %s", exc, exc_info=True)

    def _on_create_backup(self) -> None:
        """Trigger a manual database backup."""
        try:
            from core.backup import create_backup
            from core.config import IS_APP_SUPPORT_DIR
            backup_dir = IS_APP_SUPPORT_DIR / "backups"
            dest = create_backup(self._db_path, backup_dir)
            self._backup_status_label.setText(f"✔  Backup created: {dest.name}")
            self._backup_status_label.setStyleSheet(f"color: {_DARK_GREEN};")
            logger.info("Manual backup created: %s", dest)
        except Exception as exc:
            self._backup_status_label.setText(f"✖  Backup failed: {exc}")
            self._backup_status_label.setStyleSheet(f"color: {_DARK_RED};")
            logger.error("Manual backup failed: %s", exc, exc_info=True)

    def _on_restore_backup(self) -> None:
        """Open a file picker and restore selected backup."""
        from PySide6.QtWidgets import QMessageBox
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Backup File", "",
            "Backup files (*.bak.db);;All files (*)",
        )
        if not path:
            return
        confirm = QMessageBox.question(
            self,
            "Confirm Restore",
            f"Restore database from:\n{path}\n\n"
            "The current database will be saved as a .pre_restore.db file. "
            "The app may need to be restarted after restore. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            from core.backup import restore_backup
            restore_backup(path, self._db_path)
            self._backup_status_label.setText(
                "✔  Restore complete. Restart the app to apply changes."
            )
            self._backup_status_label.setStyleSheet(f"color: {_DARK_GREEN};")
            logger.info("Database restored from %s", path)
        except Exception as exc:
            self._backup_status_label.setText(f"✖  Restore failed: {exc}")
            self._backup_status_label.setStyleSheet(f"color: {_DARK_RED};")
            logger.error("Restore failed: %s", exc, exc_info=True)

    def _build_health_tab(self) -> QWidget:
        """Build the System Health tab.

        Returns:
            QWidget with status rows for all dependencies.
        """
        w = QWidget()
        w.setFocusPolicy(Qt.TabFocus)
        layout = QVBoxLayout(w)
        layout.setSpacing(8)

        self._health_rows: dict[str, tuple[QLabel, QLabel]] = {}

        health_items = [
            ("ffmpeg",     "ffmpeg"),
            ("anki",       "Anki"),
            ("ollama",     "Ollama"),
            ("mecab",      "MeCab / fugashi"),
            ("keychain",   "Keychain"),
            ("ankimorphs", "AnkiMorphs"),
        ]
        for key, name in health_items:
            row_widget = QWidget()
            row_widget.setFocusPolicy(Qt.TabFocus)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 2, 4, 2)
            name_label = QLabel(name)
            name_label.setFixedWidth(140)
            status_label = QLabel("…")
            status_label.setStyleSheet(f"color: {_DARK_MUTED};")
            row_layout.addWidget(name_label)
            row_layout.addWidget(status_label)
            row_layout.addStretch()
            self._health_rows[key] = (name_label, status_label)
            layout.addWidget(row_widget)

        # Re-check button
        recheck_btn = QPushButton("Re-check")
        recheck_btn.setFocusPolicy(Qt.TabFocus)
        recheck_btn.clicked.connect(self._run_health_checks)
        layout.addWidget(recheck_btn, alignment=Qt.AlignLeft)
        layout.addStretch()
        return w

    # ── Settings load/save ────────────────────────────────────────────────────

    def _load_settings_into_ui(self) -> None:
        """Populate UI controls from settings.json and model cache."""
        settings = load_is_settings()
        ai = settings.get("ai", {})

        # Default provider
        default_provider = ai.get("default_provider", "ollama")
        idx = self._provider_combo.findData(default_provider)
        if idx >= 0:
            self._provider_combo.setCurrentIndex(idx)
        self._populate_model_combo(self._model_combo, default_provider,
                                   ai.get("api_models", {}).get(default_provider, ""))

        # Per-task overrides
        overrides = ai.get("action_overrides", {})
        for action_key, (p_combo, m_combo) in self._override_widgets.items():
            ov = overrides.get(action_key, {})
            pid = ov.get("provider", "ollama")
            model = ov.get("model", "")
            pidx = p_combo.findData(pid)
            if pidx >= 0:
                p_combo.setCurrentIndex(pidx)
            self._populate_model_combo(m_combo, pid, model)

        # API keys table
        self._refresh_keys_table()

    def _on_save(self) -> None:
        """Save AI provider settings to settings.json (never writes keys)."""
        settings = load_is_settings()
        ai = settings.setdefault("ai", {})

        provider_pid = self._provider_combo.currentData()
        ai["default_provider"] = provider_pid
        ai.setdefault("api_models", {})[provider_pid] = self._model_combo.currentText()

        overrides: dict[str, dict] = {}
        for action_key, (p_combo, m_combo) in self._override_widgets.items():
            overrides[action_key] = {
                "provider": p_combo.currentData(),
                "model":    m_combo.currentText(),
            }
        ai["action_overrides"] = overrides

        try:
            save_is_settings(settings)
            self._save_btn.setText("✔ Saved")
            QTimer.singleShot(2000, lambda: self._save_btn.setText("Save"))
            logger.info("Settings saved")
        except Exception as e:
            logger.error("Failed to save settings: %s", e, exc_info=True)
            self._save_btn.setText("✖ Error")
            QTimer.singleShot(3000, lambda: self._save_btn.setText("Save"))

    def _on_reset(self) -> None:
        """Reset AI provider settings to defaults."""
        idx = self._provider_combo.findData("ollama")
        if idx >= 0:
            self._provider_combo.setCurrentIndex(idx)
        for action_key, (p_combo, m_combo) in self._override_widgets.items():
            pidx = p_combo.findData("ollama")
            if pidx >= 0:
                p_combo.setCurrentIndex(pidx)
        logger.info("Settings reset to defaults")

    # ── Model list helpers ────────────────────────────────────────────────────

    def _populate_model_combo(
        self, combo: QComboBox, provider_id: str, selected: str = ""
    ) -> None:
        """Populate a model combo from cache or provider static list.

        Args:
            combo: The QComboBox to populate.
            provider_id: Provider ID to look up.
            selected: Model name to pre-select if present.
        """
        combo.blockSignals(True)
        combo.clear()

        cached = self._model_cache.get(provider_id, {})
        fetched_at = cached.get("fetched_at", "")
        models = cached.get("models", [])

        if not models or not _is_cache_fresh(provider_id, fetched_at):
            # Fall back to static suggested models from PROVIDERS
            pdata = PROVIDERS.get(provider_id, {})
            models = pdata.get("static_models", [pdata.get("default_model", "")])
            models = [m for m in models if m]

        for m in models:
            combo.addItem(m)

        if selected:
            idx = combo.findText(selected)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif selected:
                combo.insertItem(0, selected)
                combo.setCurrentIndex(0)

        combo.blockSignals(False)

    def _on_provider_changed(self) -> None:
        """Refresh model combo when default provider changes."""
        pid = self._provider_combo.currentData()
        settings = load_is_settings()
        current_model = settings.get("ai", {}).get("api_models", {}).get(pid, "")
        self._populate_model_combo(self._model_combo, pid, current_model)

    def _on_override_provider_changed(self, action_key: str) -> None:
        """Refresh model combo when per-task provider override changes."""
        p_combo, m_combo = self._override_widgets[action_key]
        pid = p_combo.currentData()
        self._populate_model_combo(m_combo, pid, "")

    def _on_refresh_models(self) -> None:
        """Dispatch ModelFetchWorker for the currently selected provider."""
        pid = self._provider_combo.currentData()
        api_key = load_api_key(pid)
        pdata = PROVIDERS.get(pid, {})
        base_url = pdata.get("base_url")

        worker = ModelFetchWorker(pid, api_key, base_url)
        widget_ref = weakref.ref(self)

        # Keep a strong Python reference to signals until the callback fires.
        # Without this, worker goes out of scope when this function returns,
        # worker.signals gets GC'd, and Qt's queued signal delivery crashes
        # (EXC_ARM_DA_ALIGN / SIGBUS) accessing the deleted QObject.
        signals = worker.signals
        self._pending_signals.append(signals)

        def _release_signals() -> None:
            try:
                self._pending_signals.remove(signals)
            except ValueError:
                pass

        def _on_finished(provider_id: str, models: list) -> None:
            _release_signals()
            w = widget_ref()
            if w is None:
                return
            # Update cache
            cache = load_model_cache()
            from datetime import datetime
            cache[provider_id] = {
                "models": models,
                "fetched_at": datetime.now().isoformat(),
            }
            try:
                save_model_cache(cache)
            except Exception as e:
                logger.error("save_model_cache failed: %s", e, exc_info=True)
            w._model_cache = cache
            w._populate_model_combo(w._model_combo, provider_id, w._model_combo.currentText())
            w._refresh_btn.setText("↻ Refresh")

        def _on_error(provider_id: str, message: str) -> None:
            _release_signals()
            w = widget_ref()
            if w is None:
                return
            logger.error("ModelFetchWorker error for %s: %s", provider_id, message)
            w._refresh_btn.setText("✖ Error")
            QTimer.singleShot(3000, lambda: w._refresh_btn.setText("↻ Refresh"))

        signals.finished.connect(_on_finished, Qt.QueuedConnection)
        signals.error.connect(_on_error, Qt.QueuedConnection)
        self._refresh_btn.setText("…")
        QThreadPool.globalInstance().start(worker)

    # ── API key table ─────────────────────────────────────────────────────────

    def _refresh_keys_table(self) -> None:
        """Repopulate the API keys table from Keychain."""
        self._keys_table.setRowCount(0)
        for pid, pdata in PROVIDERS.items():
            key = load_api_key(pid)
            if not key and pdata.get("base_url") is None:
                # Local provider — show URL instead of key
                display = "http://localhost:11434 (local)"
            elif key:
                # Mask: keep last 4 chars
                masked = "•" * max(4, len(key) - 4) + key[-4:] if len(key) > 4 else "••••"
                display = masked
            else:
                continue  # skip providers with no key set

            row = self._keys_table.rowCount()
            self._keys_table.insertRow(row)
            self._keys_table.setItem(row, 0, QTableWidgetItem(pdata["name"]))
            self._keys_table.setItem(row, 1, QTableWidgetItem(display))
            del_btn = QPushButton("Delete")
            del_btn.setFocusPolicy(Qt.TabFocus)
            del_btn.clicked.connect(lambda checked=False, p=pid: self._on_delete_key(p))
            self._keys_table.setCellWidget(row, 2, del_btn)

    def _on_add_key(self) -> None:
        """Open inline dialog to add an API key."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dialog = QDialog(self)
        dialog.setWindowTitle("Add API Key")
        dialog.setMinimumWidth(420)
        dlayout = QFormLayout(dialog)

        provider_combo = QComboBox()
        provider_combo.setFocusPolicy(Qt.StrongFocus)
        for pid, pdata in PROVIDERS.items():
            provider_combo.addItem(pdata["name"], pid)

        key_edit = QLineEdit()
        key_edit.setFocusPolicy(Qt.StrongFocus)
        key_edit.setPlaceholderText("sk-…  or  your API key")
        key_edit.setEchoMode(QLineEdit.Password)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        dlayout.addRow("Provider:", provider_combo)
        dlayout.addRow("API Key:", key_edit)
        dlayout.addRow(buttons)

        if dialog.exec() == QDialog.Accepted:
            pid = provider_combo.currentData()
            key = key_edit.text().strip()
            if pid and key:
                try:
                    save_api_key(pid, key)
                    self._refresh_keys_table()
                    logger.info("API key saved for provider %s", pid)
                except Exception as e:
                    logger.error("save_api_key(%s) failed: %s", pid, e, exc_info=True)

    def _on_delete_key(self, provider_id: str) -> None:
        """Delete an API key from Keychain.

        Args:
            provider_id: Provider ID whose key should be deleted.
        """
        try:
            delete_api_key(provider_id)
            self._refresh_keys_table()
            logger.info("API key deleted for provider %s", provider_id)
        except Exception as e:
            logger.error("delete_api_key(%s) failed: %s", provider_id, e, exc_info=True)

    # ── System Health checks ──────────────────────────────────────────────────

    def _on_tab_changed(self, index: int) -> None:
        """Run health checks when System Health tab becomes active (pull-on-open)."""
        if self._tabs.tabText(index) == "System Health":
            self._run_health_checks()

    def _run_health_checks(self) -> None:
        """Execute all dependency checks and update status labels."""
        self._check_ffmpeg()
        self._check_anki()
        self._check_ollama()
        self._check_mecab()
        self._check_keychain()
        self._check_ankimorphs()

    def _set_status(self, key: str, text: str, color: str) -> None:
        """Update a health row status label.

        Args:
            key: Health row key.
            text: Status text.
            color: CSS color string.
        """
        _, status_label = self._health_rows[key]
        status_label.setText(text)
        status_label.setStyleSheet(f"color: {color};")

    def _check_ffmpeg(self) -> None:
        """Check if ffmpeg is in PATH."""
        path = shutil.which("ffmpeg")
        if path:
            self._set_status("ffmpeg", f"{STATUS_OK}  ({path})", _DARK_GREEN)
        else:
            self._set_status("ffmpeg", f"{STATUS_OFFLINE}  — brew install ffmpeg", _DARK_RED)

    def _check_anki(self) -> None:
        """Check if AnkiConnect is reachable on :8765."""
        try:
            import requests
            resp = requests.post(
                "http://localhost:8765",
                json={"action": "version", "version": 6},
                timeout=2,
            )
            if resp.status_code == 200:
                self._set_status("anki", STATUS_OK, _DARK_GREEN)
            else:
                self._set_status("anki", f"{STATUS_AMBER}  (status {resp.status_code})", _DARK_AMBER)
        except Exception:
            self._set_status("anki", f"{STATUS_OFFLINE}  — start Anki with AnkiConnect", _DARK_AMBER)

    def _check_ollama(self) -> None:
        """Check if Ollama is running and has models."""
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                count = len(models)
                first = models[0]["name"] if models else "none"
                self._set_status("ollama", f"{STATUS_OK}  ({count} model(s); {first})", _DARK_GREEN)
            else:
                self._set_status("ollama", f"{STATUS_AMBER}  (status {resp.status_code})", _DARK_AMBER)
        except Exception:
            self._set_status("ollama", f"{STATUS_OFFLINE}  — run: ollama serve", _DARK_AMBER)

    def _check_mecab(self) -> None:
        """Check if MeCab/fugashi is available."""
        try:
            import fugashi
            tagger = fugashi.Tagger()
            self._set_status("mecab", STATUS_OK, _DARK_GREEN)
        except Exception as e:
            logger.error("MeCab check failed: %s", e, exc_info=True)
            self._set_status(
                "mecab",
                f"{STATUS_AMBER}  — RegexTokenizer active (pip install fugashi unidic-lite)",
                _DARK_AMBER,
            )

    def _check_keychain(self) -> None:
        """Check if macOS Keychain is accessible."""
        try:
            import keyring
            keyring.get_password("ImmersionStudio", "_healthcheck_probe")
            self._set_status("keychain", STATUS_OK, _DARK_GREEN)
        except Exception as e:
            logger.error("Keychain check failed: %s", e, exc_info=True)
            self._set_status("keychain", f"{STATUS_OFFLINE}  — unlock Keychain", _DARK_RED)

    def _check_ankimorphs(self) -> None:
        """Check known_vocab row count and last import date."""
        try:
            from core.db import DatabaseManager
            db = DatabaseManager(path=self._db_path)
            with db.connect() as conn:
                count = conn.execute("SELECT COUNT(*) FROM known_vocab").fetchone()[0]
                last_row = conn.execute(
                    "SELECT MAX(added_at) FROM known_vocab WHERE source='ankimorph'"
                ).fetchone()[0]
            if count > 0:
                last = last_row[:10] if last_row else "unknown date"
                self._set_status(
                    "ankimorphs",
                    f"● Synced  (last: {last}, {count:,} morphs)",
                    _DARK_GREEN,
                )
            else:
                self._set_status(
                    "ankimorphs",
                    f"{STATUS_OFFLINE}  — import via Settings → Data",
                    _DARK_AMBER,
                )
        except Exception as e:
            logger.error("AnkiMorphs check failed: %s", e, exc_info=True)
            self._set_status("ankimorphs", f"{STATUS_AMBER}  — DB error: {e}", _DARK_AMBER)
