"""Immersion Studio — NWD Media File Queue / Tracker panel.

Single responsibility: display a scrollable list of media files with NWD
zone badges (🟢/🟡/🔴), allow the user to add files, and show the top-10
unknown words as a tooltip on hover.

Architecture:
    TrackerWidget          — top-level panel widget
        ↓ contains
    _MediaListWidget       — QListWidget subclass with custom item delegates
        ↓ data from
    core/nwd.py            — calculate_nwd(), get_frequent_unknowns()
    core/pipeline.py       — precompute_subtitle_tokens()
    core/db.py             — DatabaseManager (media_info, media_subtitles)

Worker dispatch:
    NWDWorker (QRunnable) runs on QThreadPool. All signal connections use
    Qt.QueuedConnection.  The widget uses weakref to avoid accessing a
    destroyed widget from a background thread.

Used by: ui/main_window.py (PANEL_TRACKER slot).

STATUS: new
DIVERGES_FROM_AIST: True
Changes: written from scratch per Phase 2 Step 8 spec.
"""

from __future__ import annotations

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: new
# DIVERGES_FROM_AIST: True
# Changes: written from scratch.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import weakref
from pathlib import Path

from PySide6.QtCore import (
    QObject, QRunnable, QSize, Qt, QThreadPool, Signal,
)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from core.db import DatabaseManager

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MEDIA_FILE_FILTER: str = (
    "Media files (*.mkv *.mp4 *.avi *.mov *.m4v *.webm *.mp3 *.flac *.aac *.wav)"
    ";;All files (*)"
)
TRACKER_EMPTY_TEXT:     str = "Add media files to begin NWD scoring."
ADD_BTN_LABEL:          str = "Add Files…"
RESCORE_BTN_LABEL:      str = "Re-score"
STATUS_IDLE:            str = ""
STATUS_SCORING:         str = "Scoring…"
UNKNOWN_TOOLTIP_HEADER: str = "Top unknown words:\n"
ITEM_FONT_SIZE:         int = 13
BADGE_FONT_SIZE:        int = 16


# ══════════════════════════════════════════════════════════════════════════════
# Qt WORKER — NWD SCORING
# ══════════════════════════════════════════════════════════════════════════════

class _NWDSignals(QObject):
    """Signals emitted by NWDWorker."""
    scored   = Signal(str, float, str, list)  # (media_path, score, zone, top_unknowns)
    error    = Signal(str, str)               # (media_path, message)


class NWDWorker(QRunnable):
    """Background worker: tokenize subtitles + compute NWD for one media file.

    # ⚠️ WORKER THREAD — runs on QThreadPool.  Never touch Qt widgets directly.
    # Emit self.signals.scored/error and let Qt dispatch to main thread.

    Args:
        media_path: Path to the media file.
        srt_path:   Path to the corresponding SRT file.  If absent, scoring
                    falls back to any existing tokens in ``media_subtitles``.
        db:         DatabaseManager instance.
    """

    def __init__(
        self,
        media_path: str,
        srt_path: str | None,
        db: DatabaseManager,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.signals    = _NWDSignals()
        self.media_path = media_path
        self.srt_path   = srt_path
        self.db         = db

    # ⚠️ WORKER THREAD
    def run(self) -> None:
        """Execute: tokenize (if SRT given) → calculate_nwd → get_frequent_unknowns."""
        try:
            from core.pipeline import precompute_subtitle_tokens
            from core.nwd import calculate_nwd, get_frequent_unknowns

            if self.srt_path and Path(self.srt_path).exists():
                precompute_subtitle_tokens(
                    Path(self.srt_path),
                    self.media_path,
                    self.db,
                    overwrite=False,
                )

            score    = calculate_nwd(self.media_path, self.db)
            unknowns = get_frequent_unknowns(self.media_path, self.db, limit=10)
            from core.nwd import nwd_zone
            zone = nwd_zone(score)
            self.signals.scored.emit(self.media_path, score, zone, unknowns)

        except Exception as exc:
            logger.error("NWDWorker failed for %s: %s", self.media_path, exc, exc_info=True)
            self.signals.error.emit(self.media_path, str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# TrackerWidget
# ══════════════════════════════════════════════════════════════════════════════

class TrackerWidget(QWidget):
    """NWD media file queue with zone badges and unknown-word tooltips.

    Layout (top → bottom):
        Button row:  [Add Files…]  [Re-score]  [status label]
        List:        scrollable list of media entries with badges + tooltips

    Each list item shows:
        <zone badge> <filename>  (<score%>)
    Hovering an item shows a tooltip listing the top-10 unknown words.

    Args:
        db_path: Path to the Immersion Studio SQLite database.
        parent:  Optional parent widget.
    """

    def __init__(self, db_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.TabFocus)
        self._db_path = db_path
        self._db      = DatabaseManager(path=db_path)
        # maps media_path → {score, zone, top_unknowns}
        self._item_data: dict[str, dict] = {}
        # keeps NWDWorker.signals alive until callbacks fire (prevents SIGBUS)
        self._pending_signals: list = []
        self._build_ui()
        self._load_existing()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # Button row
        btn_row = QHBoxLayout()
        self._add_btn = QPushButton(ADD_BTN_LABEL)
        self._add_btn.setFocusPolicy(Qt.TabFocus)
        self._add_btn.clicked.connect(self._on_add_files)

        self._rescore_btn = QPushButton(RESCORE_BTN_LABEL)
        self._rescore_btn.setFocusPolicy(Qt.TabFocus)
        self._rescore_btn.setEnabled(False)
        self._rescore_btn.clicked.connect(self._on_rescore_all)

        self._status_label = QLabel(STATUS_IDLE)
        self._status_label.setFocusPolicy(Qt.TabFocus)
        self._status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._rescore_btn)
        btn_row.addWidget(self._status_label, stretch=1)
        root.addLayout(btn_row)

        # Media list
        self._list = QListWidget()
        self._list.setFocusPolicy(Qt.TabFocus)
        self._list.setMouseTracking(True)  # needed for hover tooltips
        self._list.setAlternatingRowColors(True)
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.setIconSize(QSize(24, 24))
        self._list.itemEntered.connect(self._on_item_hovered)
        root.addWidget(self._list)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_existing(self) -> None:
        """Populate list from previously scored entries in media_info."""
        try:
            with self._db.connect() as conn:
                rows = conn.execute(
                    "SELECT path, title, nwd_score, nwd_zone FROM media_info ORDER BY path"
                ).fetchall()
        except Exception as exc:
            logger.error("TrackerWidget._load_existing failed: %s", exc, exc_info=True)
            return

        for path, title, score, zone in rows:
            score  = score or 0.0
            zone   = zone  or "🔴"
            self._item_data[path] = {"score": score, "zone": zone, "top_unknowns": []}
            self._add_list_item(path, score, zone)

        if self._list.count():
            self._rescore_btn.setEnabled(True)

    # ── List management ───────────────────────────────────────────────────────

    def _add_list_item(self, media_path: str, score: float, zone: str) -> None:
        """Insert or update a list item for *media_path*."""
        filename = Path(media_path).name
        pct      = f"{score * 100:.1f}%"
        label    = f"{zone}  {filename}  ({pct})"

        # Find existing item to update in place
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.UserRole) == media_path:
                item.setText(label)
                return

        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, media_path)
        item.setToolTip(TRACKER_EMPTY_TEXT)  # placeholder until scored
        item.setSizeHint(QSize(0, 32))
        self._list.addItem(item)

    def _update_item_tooltip(self, media_path: str) -> None:
        """Update the tooltip of the list item for *media_path* with unknowns."""
        data = self._item_data.get(media_path, {})
        unknowns: list[tuple[str, int]] = data.get("top_unknowns", [])
        if not unknowns:
            tooltip = "No unknown words found (or not yet scored)."
        else:
            lines = [f"  {w}  ×{n}" for w, n in unknowns]
            tooltip = UNKNOWN_TOOLTIP_HEADER + "\n".join(lines)
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.UserRole) == media_path:
                item.setToolTip(tooltip)
                break

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_add_files(self) -> None:
        """Open a file picker and add selected media files to the queue."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Media Files", "", MEDIA_FILE_FILTER
        )
        if not paths:
            return

        for media_path in paths:
            if media_path not in self._item_data:
                self._item_data[media_path] = {"score": 0.0, "zone": "🔴", "top_unknowns": []}
                self._add_list_item(media_path, 0.0, "🔴")

        self._rescore_btn.setEnabled(True)
        self._score_paths(paths)

    def _on_rescore_all(self) -> None:
        """Re-score all items in the queue."""
        paths = list(self._item_data.keys())
        if paths:
            self._score_paths(paths)

    def _on_item_hovered(self, item: QListWidgetItem) -> None:
        """Show tooltip for the hovered item."""
        media_path = item.data(Qt.UserRole)
        if not media_path:
            return
        data = self._item_data.get(media_path, {})
        unknowns: list[tuple[str, int]] = data.get("top_unknowns", [])
        if unknowns:
            lines = [f"  {w}  ×{n}" for w, n in unknowns]
            tip = UNKNOWN_TOOLTIP_HEADER + "\n".join(lines)
        else:
            tip = "Not yet scored — click Re-score."
        QToolTip.showText(self._list.mapToGlobal(self._list.visualItemRect(item).topLeft()), tip)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_paths(self, paths: list[str]) -> None:
        """Dispatch NWDWorker for each path in *paths*."""
        self._status_label.setText(STATUS_SCORING)
        self._add_btn.setEnabled(False)
        self._rescore_btn.setEnabled(False)
        self._pending = len(paths)

        weak_self = weakref.ref(self)

        for media_path in paths:
            # Look for a sidecar SRT in the same directory
            srt_candidate = Path(media_path).with_suffix(".srt")
            srt_path = str(srt_candidate) if srt_candidate.exists() else None

            worker = NWDWorker(media_path, srt_path, self._db)

            # Hold strong ref to signals until the callback fires.
            # NWDWorker uses autoDelete=True, so worker is deleted by Qt's
            # thread pool after run() completes. Without this, worker.signals
            # is GC'd and the QueuedConnection delivery crashes (SIGBUS).
            signals = worker.signals
            self._pending_signals.append(signals)

            def _make_release(sig):
                def _release():
                    try:
                        self._pending_signals.remove(sig)
                    except ValueError:
                        pass
                return _release

            _release = _make_release(signals)

            def _on_scored(mp: str, score: float, zone: str, unknowns: list,
                           _rel=_release) -> None:
                _rel()
                w = weak_self()
                if w is not None:
                    w._handle_scored(mp, score, zone, unknowns)

            def _on_error(mp: str, message: str, _rel=_release) -> None:
                _rel()
                w = weak_self()
                if w is not None:
                    w._handle_error(mp, message)

            signals.scored.connect(_on_scored, Qt.QueuedConnection)
            signals.error.connect(_on_error, Qt.QueuedConnection)
            QThreadPool.globalInstance().start(worker)

    def _handle_scored(self, media_path: str, score: float, zone: str, unknowns: list) -> None:
        """Update the list item after successful scoring."""
        self._item_data[media_path] = {"score": score, "zone": zone, "top_unknowns": unknowns}
        self._add_list_item(media_path, score, zone)
        self._update_item_tooltip(media_path)
        self._decrement_pending()

    def _handle_error(self, media_path: str, message: str) -> None:
        """Mark item with error indicator after failed scoring."""
        logger.error("NWD scoring error for %s: %s", media_path, message)
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.UserRole) == media_path:
                filename = Path(media_path).name
                item.setText(f"⚠  {filename}  (error)")
                item.setToolTip(f"Scoring failed: {message}")
                break
        self._decrement_pending()

    def _decrement_pending(self) -> None:
        """Decrement the pending counter and re-enable controls when done."""
        self._pending = max(0, getattr(self, "_pending", 1) - 1)
        if self._pending == 0:
            self._status_label.setText(STATUS_IDLE)
            self._add_btn.setEnabled(True)
            self._rescore_btn.setEnabled(True)
