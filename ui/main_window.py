"""Immersion Studio main application window.

Single responsibility: PySide6 sidebar shell, panel routing, global keyboard
shortcuts, dark theme, system tray session timer, and graceful shutdown.

Used by: main.py (instantiated directly).
Consumes from: ui/widgets/*.py (each panel), core/config.py (settings).

Signals emitted (none — MainWindow is the top-level consumer).
"""

# ── Constants ─────────────────────────────────────────────────────────────────
import logging
import weakref
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QThreadPool, QSize
from PySide6.QtGui import (
    QColor, QPalette, QKeySequence, QAction, QIcon,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QStackedWidget, QLabel, QSizePolicy,
    QSystemTrayIcon, QMenu,
)

logger = logging.getLogger(__name__)

# ── Sidebar panel indexes ──────────────────────────────────────────────────────
PANEL_DASHBOARD:   int = 0
PANEL_PITCH:       int = 1
PANEL_CHORUSING:   int = 2
PANEL_SETTINGS:    int = 3

# ── Sidebar appearance ─────────────────────────────────────────────────────────
SIDEBAR_WIDTH_PX:    int = 56
SIDEBAR_BTN_SIZE_PX: int = 44
TOPBAR_HEIGHT_PX:    int = 36

# ── Shutdown ──────────────────────────────────────────────────────────────────
THREADPOOL_DRAIN_TIMEOUT_MS: int = 3000

# ── Dark theme palette ────────────────────────────────────────────────────────
_DARK_BG:      str = "#0d1117"
_DARK_PANEL:   str = "#161b22"
_DARK_BORDER:  str = "#30363d"
_DARK_TEXT:    str = "#e6edf3"
_DARK_MUTED:   str = "#7d8590"
_DARK_ACCENT:  str = "#e05c2a"
_DARK_SUCCESS: str = "#3fb950"
_DARK_ERROR:   str = "#f85149"


def _build_dark_palette() -> QPalette:
    """Construct a dark QPalette matching the Immersion Studio theme.

    Returns:
        QPalette with dark background and light text.
    """
    palette = QPalette()
    bg      = QColor(_DARK_BG)
    panel   = QColor(_DARK_PANEL)
    text    = QColor(_DARK_TEXT)
    muted   = QColor(_DARK_MUTED)
    accent  = QColor(_DARK_ACCENT)
    palette.setColor(QPalette.Window,          bg)
    palette.setColor(QPalette.WindowText,      text)
    palette.setColor(QPalette.Base,            panel)
    palette.setColor(QPalette.AlternateBase,   QColor("#1c2128"))
    palette.setColor(QPalette.Text,            text)
    palette.setColor(QPalette.Button,          QColor("#21262d"))
    palette.setColor(QPalette.ButtonText,      text)
    palette.setColor(QPalette.Highlight,       accent)
    palette.setColor(QPalette.HighlightedText, text)
    palette.setColor(QPalette.PlaceholderText, muted)
    palette.setColor(QPalette.ToolTipBase,     panel)
    palette.setColor(QPalette.ToolTipText,     text)
    return palette


class _SidebarButton(QPushButton):
    """Icon-only sidebar navigation button.

    Keyboard accessible via TabFocus. Emits clicked() signal when activated
    with Enter or Space (handled by QPushButton default).
    """

    def __init__(self, icon_text: str, tooltip: str, parent: Optional[QWidget] = None):
        super().__init__(icon_text, parent)
        self.setFocusPolicy(Qt.TabFocus)
        self.setToolTip(tooltip)
        self.setFixedSize(SIDEBAR_BTN_SIZE_PX, SIDEBAR_BTN_SIZE_PX)
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {_DARK_MUTED};
                font-size: 20px;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background: {_DARK_PANEL};
                color: {_DARK_TEXT};
            }}
            QPushButton:checked, QPushButton:pressed {{
                background: {_DARK_BORDER};
                color: {_DARK_ACCENT};
            }}
            QPushButton:focus {{
                outline: 1px solid {_DARK_ACCENT};
            }}
        """)
        self.setCheckable(True)


class MainWindow(QMainWindow):
    """Immersion Studio main application window.

    Layout:
        ┌──────┬─────────────────────────────────────────────────────┐
        │      │  Top bar: [NWD: 82% 🟢] [Session: 1:23] [Sync ↺]   │
        │ 📊   ├─────────────────────────────────────────────────────┤
        │ 🎵   │                                                       │
        │ 🎤   │              Active Panel                              │
        │ ⚙️   │                                                       │
        └──────┴─────────────────────────────────────────────────────┘

    Global keyboard shortcuts:
        Ctrl+1 … Ctrl+4  Switch panels
        Ctrl+,           Open Settings
        Escape           Close dialog / dismiss overlay
        /                Focus search input (delegated to active panel)
        Tab / Shift+Tab  Cycle interactive elements
        Enter / Space    Activate focused button (handled by Qt defaults)
        Arrow keys       Navigate lists (handled by focused widget)

    Signals:
        None emitted — MainWindow is the top-level consumer.
    """

    def __init__(self, db_path: Path, parent: Optional[QWidget] = None):
        """Initialize MainWindow.

        Args:
            db_path: Path to the SQLite database (passed to widgets that need DB).
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self.setFocusPolicy(Qt.TabFocus)
        self._db_path = db_path

        # ── Apply dark theme ──────────────────────────────────────────────────
        QApplication.setStyle("Fusion")
        QApplication.setPalette(_build_dark_palette())

        self.setWindowTitle("Immersion Studio")
        self.resize(1100, 720)

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_ui()

        # ── Global keyboard shortcuts ─────────────────────────────────────────
        self._setup_shortcuts()

        # ── Session timer (top bar + system tray) ────────────────────────────
        self._session_seconds: int = 0
        self._session_running: bool = False
        self._session_timer = QTimer(self)
        self._session_timer.setInterval(1000)
        self._session_timer.timeout.connect(self._on_session_tick)

        # ── System tray ───────────────────────────────────────────────────────
        self._setup_tray()

        logger.info("MainWindow initialized")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct sidebar, top bar, and stacked panel area."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Left sidebar ──────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(SIDEBAR_WIDTH_PX)
        sidebar.setStyleSheet(f"background: {_DARK_PANEL}; border-right: 1px solid {_DARK_BORDER};")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(4, 8, 4, 8)
        sidebar_layout.setSpacing(4)

        self._btn_dashboard = _SidebarButton("📊", "Dashboard (Ctrl+1)")
        self._btn_pitch     = _SidebarButton("🎵", "Pitch Training (Ctrl+2)")
        self._btn_chorusing = _SidebarButton("🎤", "Chorusing (Ctrl+3)")
        self._btn_settings  = _SidebarButton("⚙", "Settings (Ctrl+4 or Ctrl+,)")

        self._sidebar_buttons = [
            self._btn_dashboard, self._btn_pitch,
            self._btn_chorusing, self._btn_settings,
        ]
        for btn in self._sidebar_buttons:
            sidebar_layout.addWidget(btn, alignment=Qt.AlignHCenter)

        sidebar_layout.addStretch()
        root_layout.addWidget(sidebar)

        # ── Right area (top bar + stacked panels) ─────────────────────────────
        right_area = QWidget()
        right_layout = QVBoxLayout(right_area)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Top bar
        top_bar = self._build_top_bar()
        right_layout.addWidget(top_bar)

        # Stacked panels
        self._stack = QStackedWidget()
        self._stack.setFocusPolicy(Qt.TabFocus)
        right_layout.addWidget(self._stack)

        # Add placeholder panels (real panels added lazily on first visit)
        self._panels: dict[int, Optional[QWidget]] = {}
        for i in range(4):
            placeholder = QWidget()
            placeholder.setFocusPolicy(Qt.TabFocus)
            self._stack.addWidget(placeholder)
            self._panels[i] = None  # marks as not yet loaded

        root_layout.addWidget(right_area)

        # Wire sidebar buttons
        self._btn_dashboard.clicked.connect(lambda: self._switch_panel(PANEL_DASHBOARD))
        self._btn_pitch.clicked.connect(lambda: self._switch_panel(PANEL_PITCH))
        self._btn_chorusing.clicked.connect(lambda: self._switch_panel(PANEL_CHORUSING))
        self._btn_settings.clicked.connect(lambda: self._switch_panel(PANEL_SETTINGS))

        # Show dashboard on launch
        self._switch_panel(PANEL_DASHBOARD)

    def _build_top_bar(self) -> QWidget:
        """Build the top bar with NWD display, session timer, and sync button.

        Returns:
            QWidget top bar.
        """
        bar = QWidget()
        bar.setFixedHeight(TOPBAR_HEIGHT_PX)
        bar.setStyleSheet(
            f"background: {_DARK_PANEL}; border-bottom: 1px solid {_DARK_BORDER};"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(16)

        self._nwd_label = QLabel("NWD: —")
        self._nwd_label.setStyleSheet(f"color: {_DARK_MUTED}; font-size: 12px;")

        self._session_label = QLabel("Session: —")
        self._session_label.setStyleSheet(f"color: {_DARK_MUTED}; font-size: 12px;")

        self._session_start_btn = QPushButton("▶ Start")
        self._session_start_btn.setFocusPolicy(Qt.TabFocus)
        self._session_start_btn.setFixedHeight(24)
        self._session_start_btn.setStyleSheet(f"font-size: 11px; color: {_DARK_TEXT};")
        self._session_start_btn.clicked.connect(self._toggle_session)

        self._sync_btn = QPushButton("↺ Sync")
        self._sync_btn.setFocusPolicy(Qt.TabFocus)
        self._sync_btn.setFixedHeight(24)
        self._sync_btn.setStyleSheet(f"font-size: 11px; color: {_DARK_TEXT};")
        self._sync_btn.setToolTip("Sync AnkiMorphs known vocab")
        self._sync_btn.clicked.connect(self._on_sync_clicked)

        layout.addWidget(self._nwd_label)
        layout.addStretch()
        layout.addWidget(self._session_label)
        layout.addWidget(self._session_start_btn)
        layout.addWidget(self._sync_btn)
        return bar

    def _setup_shortcuts(self) -> None:
        """Register global keyboard shortcuts."""
        # Panel switching Ctrl+1 … Ctrl+4
        for idx, key in enumerate(["1", "2", "3", "4"]):
            action = QAction(self)
            action.setShortcut(QKeySequence(f"Ctrl+{key}"))
            panel_idx = idx
            action.triggered.connect(lambda checked=False, i=panel_idx: self._switch_panel(i))
            self.addAction(action)

        # Ctrl+, → Settings
        settings_action = QAction(self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(lambda: self._switch_panel(PANEL_SETTINGS))
        self.addAction(settings_action)

        # Escape → close any overlay / go back to dashboard
        escape_action = QAction(self)
        escape_action.setShortcut(QKeySequence(Qt.Key_Escape))
        escape_action.triggered.connect(self._on_escape)
        self.addAction(escape_action)

        # / → focus search (delegated to active panel)
        search_action = QAction(self)
        search_action.setShortcut(QKeySequence(Qt.Key_Slash))
        search_action.triggered.connect(self._focus_search)
        self.addAction(search_action)

    def _setup_tray(self) -> None:
        """Create macOS system tray icon with session timer menu."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("System tray not available on this platform")
            return
        self._tray = QSystemTrayIcon(self)
        self._tray.setToolTip("Immersion Studio")
        tray_menu = QMenu()
        _show_action = tray_menu.addAction("Show Window")
        _show_action.triggered.connect(self.show)
        _show_action.triggered.connect(self.raise_)
        tray_menu.addSeparator()
        _quit_action = tray_menu.addAction("Quit")
        _quit_action.triggered.connect(QApplication.quit)
        self._tray.setContextMenu(tray_menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    # ── Panel switching ───────────────────────────────────────────────────────

    def _switch_panel(self, panel_idx: int) -> None:
        """Switch the active panel by index.

        Lazy-loads the real widget on first visit.

        Args:
            panel_idx: One of PANEL_DASHBOARD, PANEL_PITCH, PANEL_CHORUSING,
                       PANEL_SETTINGS.
        """
        # Update sidebar button checked states
        for i, btn in enumerate(self._sidebar_buttons):
            btn.setChecked(i == panel_idx)

        # Lazy-load the real panel if not yet created
        if self._panels.get(panel_idx) is None:
            widget = self._create_panel(panel_idx)
            if widget is not None:
                # Replace the placeholder widget in the stack
                old = self._stack.widget(panel_idx)
                self._stack.removeWidget(old)
                self._stack.insertWidget(panel_idx, widget)
                self._panels[panel_idx] = widget

        self._stack.setCurrentIndex(panel_idx)

    def _create_panel(self, panel_idx: int) -> Optional[QWidget]:
        """Instantiate the widget for a given panel.

        Args:
            panel_idx: Panel index constant.

        Returns:
            The created QWidget, or a fallback placeholder on import error.
        """
        try:
            if panel_idx == PANEL_DASHBOARD:
                from ui.widgets.translation_widget import TranslationWidget
                return TranslationWidget()

            elif panel_idx == PANEL_PITCH:
                try:
                    from ui.widgets.pitch_widget import PitchWidget
                    return PitchWidget()
                except Exception as e:
                    logger.error("PitchWidget failed to load: %s", e, exc_info=True)
                    return self._make_error_panel(f"Pitch widget error: {e}")

            elif panel_idx == PANEL_CHORUSING:
                placeholder = QWidget()
                placeholder.setFocusPolicy(Qt.TabFocus)
                lbl = QLabel("🎤 Chorusing — Phase 3")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet(f"color: {_DARK_MUTED}; font-size: 16px;")
                from PySide6.QtWidgets import QVBoxLayout as VBL
                VBL(placeholder).addWidget(lbl)
                return placeholder

            elif panel_idx == PANEL_SETTINGS:
                from ui.widgets.settings_widget import SettingsWidget
                return SettingsWidget(db_path=self._db_path)

        except Exception as e:
            logger.error("_create_panel(%d) failed: %s", panel_idx, e, exc_info=True)
            return self._make_error_panel(f"Panel load error: {e}")

        return None

    def _make_error_panel(self, message: str) -> QWidget:
        """Return a simple error placeholder panel.

        Args:
            message: Error description to display.

        Returns:
            QWidget containing the error label.
        """
        w = QWidget()
        w.setFocusPolicy(Qt.TabFocus)
        lbl = QLabel(message)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {_DARK_ERROR}; font-size: 13px; padding: 24px;")
        from PySide6.QtWidgets import QVBoxLayout as VBL
        VBL(w).addWidget(lbl)
        return w

    # ── Session timer ─────────────────────────────────────────────────────────

    def _toggle_session(self) -> None:
        """Start or stop the session timer."""
        if self._session_running:
            self._session_timer.stop()
            self._session_running = False
            self._session_start_btn.setText("▶ Start")
            logger.info("Session stopped at %ds", self._session_seconds)
        else:
            self._session_timer.start()
            self._session_running = True
            self._session_start_btn.setText("⏹ Stop")
            logger.info("Session started")

    def _on_session_tick(self) -> None:
        """Called every second while session is running."""
        self._session_seconds += 1
        m, s = divmod(self._session_seconds, 60)
        h, m = divmod(m, 60)
        if h:
            text = f"Session: {h}:{m:02d}:{s:02d}"
        else:
            text = f"Session: {m}:{s:02d}"
        self._session_label.setText(text)
        if hasattr(self, "_tray"):
            self._tray.setToolTip(f"Immersion Studio — {text}")

    def _on_sync_clicked(self) -> None:
        """Handle top-bar sync button click — navigate to Settings → Data."""
        self._switch_panel(PANEL_SETTINGS)

    # ── Keyboard event handlers ───────────────────────────────────────────────

    def _on_escape(self) -> None:
        """Escape: close any open dialog or return to dashboard."""
        focused = QApplication.focusWidget()
        if focused and focused is not self:
            focused.clearFocus()

    def _focus_search(self) -> None:
        """/ shortcut: delegate to active panel's search widget if available."""
        panel = self._stack.currentWidget()
        if panel:
            search = panel.findChild(QWidget, "search_input")
            if search:
                search.setFocus()

    # ── Tray ──────────────────────────────────────────────────────────────────

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Show window on tray icon double-click."""
        if reason == QSystemTrayIcon.DoubleClick:
            self.show()
            self.raise_()
            self.activateWindow()

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Drain QThreadPool before closing to avoid worker crashes.

        Waits up to THREADPOOL_DRAIN_TIMEOUT_MS ms for background workers.
        """
        logger.info("closeEvent: draining QThreadPool (timeout=%dms)", THREADPOOL_DRAIN_TIMEOUT_MS)
        QThreadPool.globalInstance().waitForDone(THREADPOOL_DRAIN_TIMEOUT_MS)
        super().closeEvent(event)
