"""Immersion Studio — HVPT Pitch Accent trainer panel.

Single responsibility: embed Kuuuube's minimal-pairs local web app in a
QWebEngineView.  No network traffic — the HTML/JS/CSS is loaded from the
cloned repo at resources/minimal-pairs/index.html via QUrl.fromLocalFile.

Clone the repo once:
    git clone --depth=1 https://github.com/Kuuuube/minimal-pairs \
              resources/minimal-pairs

Used by: ui/main_window.py (PANEL_PITCH slot).

STATUS: new
DIVERGES_FROM_AIST: True
Changes: written from scratch per Step 12 spec.
"""

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: new
# DIVERGES_FROM_AIST: True
# Changes: written from scratch — AIST has no pitch trainer widget.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Path is resolved relative to this file: ui/widgets/ → ../../resources/minimal-pairs
_MODULE_DIR: Path = Path(__file__).resolve().parent
MINIMAL_PAIRS_INDEX: Path = (
    _MODULE_DIR / ".." / ".." / "resources" / "minimal-pairs" / "index.html"
).resolve()

MISSING_LABEL_TEXT: str = (
    "Minimal-pairs not found.\n\n"
    "Clone the repo once:\n"
    "  git clone --depth=1 https://github.com/Kuuuube/minimal-pairs \\\n"
    "            resources/minimal-pairs"
)


class PitchWidget(QWidget):
    """HVPT pitch accent trainer panel.

    Embeds Kuuuube's minimal-pairs app (``resources/minimal-pairs/index.html``)
    in a QWebEngineView loaded from a local file URL.  If the index.html is
    not present a human-readable setup message is shown instead.

    Args:
        parent: Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.TabFocus)
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the panel: WebEngineView if index.html exists, else label."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if not MINIMAL_PAIRS_INDEX.exists():
            logger.warning(
                "PitchWidget: minimal-pairs index.html not found at %s",
                MINIMAL_PAIRS_INDEX,
            )
            label = QLabel(MISSING_LABEL_TEXT)
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)
            label.setFocusPolicy(Qt.TabFocus)
            layout.addWidget(label)
            return

        # Import here so the app can still launch if QtWebEngine is not installed
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
        except ImportError:
            logger.error(
                "PitchWidget: PySide6-WebEngine not installed — "
                "run: pip install PySide6-WebEngine"
            )
            label = QLabel(
                "PySide6-WebEngine is required for the Pitch trainer.\n\n"
                "Install it with:\n    pip install PySide6-WebEngine"
            )
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)
            label.setFocusPolicy(Qt.TabFocus)
            layout.addWidget(label)
            return

        self._web_view = QWebEngineView(self)
        self._web_view.setFocusPolicy(Qt.TabFocus)

        # Allow audio playback without a prior user gesture (autoplay) and
        # allow local files to reference other local files (e.g. audio data).
        try:
            from PySide6.QtWebEngineCore import QWebEngineSettings
            ws = self._web_view.settings()
            ws.setAttribute(
                QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False
            )
            ws.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessLocalUrls, True
            )
        except Exception as e:
            logger.warning("Could not configure WebEngine audio settings: %s", e)

        url = QUrl.fromLocalFile(str(MINIMAL_PAIRS_INDEX))
        self._web_view.load(url)
        layout.addWidget(self._web_view)
        logger.info("PitchWidget: loaded %s", url.toString())
