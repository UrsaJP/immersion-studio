"""Immersion Studio — application entry point.

Single responsibility: bootstrap logging, database, backup, and Qt application.
Launches MainWindow after all pre-flight setup is complete.

Dependencies: core/db.py (schema init), core/backup.py (auto backup),
              ui/main_window.py (MainWindow).
"""

# ── Constants ─────────────────────────────────────────────────────────────────
import logging
import logging.handlers
import sys
from pathlib import Path

# ── Step 1: Logging setup MUST happen before QApplication ─────────────────────
_APP_SUPPORT_DIR: Path = (
    Path.home() / "Library" / "Application Support" / "ImmersionStudio"
)
_LOG_DIR: Path = _APP_SUPPORT_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_log_handler = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "immersion_studio.log",
    maxBytes=10 * 1024 * 1024,   # 10 MB per file
    backupCount=3,
)
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(
    logging.Formatter("%(levelname)s %(name)s: %(message)s")
)
logging.getLogger().addHandler(_log_handler)
logging.getLogger().addHandler(_console_handler)
logging.getLogger().setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.info("Immersion Studio starting up")

# ── Step 2: Import Qt AFTER logging is configured ─────────────────────────────
# Deferred so logging captures any import-time errors from core modules.
try:
    from PySide6.QtWidgets import QApplication
except ImportError as e:
    logger.error("PySide6 not installed: %s", e, exc_info=True)
    print("ERROR: PySide6 is not installed. Run: pip install PySide6", file=sys.stderr)
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
_DB_PATH: Path = _APP_SUPPORT_DIR / "immersion_studio.db"
_BACKUP_DIR: Path = _APP_SUPPORT_DIR / "backups"


def _bootstrap_db() -> None:
    """Initialize DatabaseManager and V1 schema before any UI is created.

    This must run before MainWindow so all tables exist when widgets start.
    """
    try:
        from core.db import DatabaseManager, init_schema
        db = DatabaseManager(path=_DB_PATH)
        init_schema(db)
        logger.info("Database schema initialized at %s", _DB_PATH)
    except Exception as e:
        logger.error("Database bootstrap failed: %s", e, exc_info=True)
        # Non-fatal: app can still launch without DB, but will degrade.


def _auto_backup() -> None:
    """Run daily backup if needed. Called before QApplication.exec()."""
    try:
        from core.backup import auto_backup_if_needed
        auto_backup_if_needed(db_path=_DB_PATH, backup_dir=_BACKUP_DIR)
    except Exception as e:
        logger.error("auto_backup_if_needed failed: %s", e, exc_info=True)
        # Non-fatal: continue even if backup fails.


def main() -> int:
    """Application entry point.

    Returns:
        Exit code (0 = success).
    """
    # ── Database schema init ───────────────────────────────────────────────────
    _bootstrap_db()

    # ── Auto-backup (before exec, before any UI) ──────────────────────────────
    _auto_backup()

    # ── QApplication ──────────────────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setApplicationName("ImmersionStudio")
    app.setOrganizationName("ImmersionStudio")
    app.setApplicationDisplayName("Immersion Studio")

    # ── MainWindow ────────────────────────────────────────────────────────────
    try:
        from ui.main_window import MainWindow
    except Exception as e:
        logger.error("Failed to import MainWindow: %s", e, exc_info=True)
        raise

    window = MainWindow(db_path=_DB_PATH)
    window.show()

    logger.info("Entering Qt event loop")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
