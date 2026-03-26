"""Immersion Studio database backup utilities.

Single responsibility: create, prune, and restore SQLite database backups.
Backups are plain SQLite files written atomically via the SQLite Online Backup
API (sqlite3.Connection.backup()).  Only the most recent N backups are kept.

Used by: main.py (auto_backup_if_needed on startup).

STATUS: new
DIVERGES_FROM_AIST: True
Changes: written from scratch — AIST has no backup module.
"""

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: new
# DIVERGES_FROM_AIST: True
# Changes: written from scratch per Step 11 spec.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BACKUP_KEEP: int = 7                          # max backups to retain
BACKUP_INTERVAL_HOURS: int = 24               # minimum hours between auto-backups
BACKUP_FILENAME_FORMAT: str = "%Y%m%d_%H%M%S" # strftime pattern for backup filename
BACKUP_SUFFIX: str = ".bak.db"               # suffix appended to each backup file
BACKUP_SENTINEL_FILENAME: str = ".last_backup"# tracks timestamp of last successful backup
BACKUP_PAGES: int = 0                         # 0 → copy whole DB in one shot (no progress)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def auto_backup_if_needed(db_path: str | Path, backup_dir: str | Path) -> bool:
    """Create a backup of *db_path* only if BACKUP_INTERVAL_HOURS have elapsed.

    Reads a sentinel file in *backup_dir* to determine when the last backup
    ran.  Safe to call on every app launch.

    Args:
        db_path: Absolute path to the live SQLite database.
        backup_dir: Directory where backup files are stored.  Created if absent.

    Returns:
        True if a new backup was created, False if skipped (too recent) or
        if the database file does not yet exist.
    """
    db_path = Path(db_path)
    backup_dir = Path(backup_dir)

    if not db_path.exists():
        logger.debug("auto_backup_if_needed: db not found, skipping (%s)", db_path)
        return False

    backup_dir.mkdir(parents=True, exist_ok=True)
    sentinel = backup_dir / BACKUP_SENTINEL_FILENAME

    if sentinel.exists():
        try:
            last_ts = datetime.fromisoformat(sentinel.read_text().strip())
            if datetime.now() - last_ts < timedelta(hours=BACKUP_INTERVAL_HOURS):
                logger.debug(
                    "auto_backup_if_needed: last backup %s, interval not elapsed, skipping",
                    last_ts.isoformat(),
                )
                return False
        except ValueError:
            logger.warning(
                "auto_backup_if_needed: corrupt sentinel %s, will re-backup", sentinel
            )

    create_backup(db_path, backup_dir)
    sentinel.write_text(datetime.now().isoformat())
    _prune_backups(backup_dir, keep=BACKUP_KEEP)
    return True


def create_backup(db_path: str | Path, backup_dir: str | Path) -> Path:
    """Write a full, consistent backup of *db_path* using the SQLite Online
    Backup API.

    The backup is written to a temp file first, then atomically renamed into
    place so that a crash during write never leaves a corrupt backup file.

    Args:
        db_path: Path to the source database.
        backup_dir: Directory to write the backup file into.

    Returns:
        Path to the newly created backup file.

    Raises:
        RuntimeError: If the backup operation fails.
    """
    db_path = Path(db_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime(BACKUP_FILENAME_FORMAT)
    dest = backup_dir / f"{db_path.stem}_{stamp}{BACKUP_SUFFIX}"
    tmp = dest.with_suffix(".tmp")

    try:
        src_conn = sqlite3.connect(str(db_path))
        dst_conn = sqlite3.connect(str(tmp))
        try:
            src_conn.backup(dst_conn, pages=BACKUP_PAGES)
            logger.info("create_backup: wrote backup to %s", tmp)
        finally:
            dst_conn.close()
            src_conn.close()
    except Exception as exc:
        logger.error("create_backup failed: %s", exc, exc_info=True)
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Backup failed: {exc}") from exc

    # Atomic rename — safe against interrupted writes
    tmp.replace(dest)
    logger.info("create_backup: backup committed → %s", dest)
    return dest


def restore_backup(backup_path: str | Path, db_path: str | Path) -> None:
    """Replace the live database with a previously created backup.

    The current database is preserved as *<db_path>.pre_restore.db* before
    replacement so the user can recover from an accidental restore.

    Args:
        backup_path: Path to the backup file to restore (must end in BACKUP_SUFFIX).
        db_path: Path to the live database file that will be overwritten.

    Raises:
        FileNotFoundError: If *backup_path* does not exist.
        RuntimeError: If the copy operation fails.
    """
    backup_path = Path(backup_path)
    db_path = Path(db_path)

    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    safety_copy = db_path.with_suffix(".pre_restore.db")
    if db_path.exists():
        shutil.copy2(str(db_path), str(safety_copy))
        logger.info("restore_backup: saved current db as %s", safety_copy)

    tmp = db_path.with_suffix(".restore_tmp.db")
    try:
        shutil.copy2(str(backup_path), str(tmp))
        tmp.replace(db_path)
    except Exception as exc:
        logger.error("restore_backup failed: %s", exc, exc_info=True)
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Restore failed: {exc}") from exc

    logger.info("restore_backup: restored %s → %s", backup_path, db_path)


def list_backups(backup_dir: str | Path) -> list[Path]:
    """Return backup files in *backup_dir*, newest first.

    Args:
        backup_dir: Directory containing backup files.

    Returns:
        List of Path objects matching ``*BACKUP_SUFFIX``, sorted newest first.
        Empty list if the directory does not exist or contains no backups.
    """
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        return []
    files = sorted(
        backup_dir.glob(f"*{BACKUP_SUFFIX}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _prune_backups(backup_dir: Path, keep: int = BACKUP_KEEP) -> None:
    """Delete oldest backup files, retaining the most recent *keep* files.

    Args:
        backup_dir: Directory containing backup files.
        keep: Number of backup files to retain.
    """
    files = list_backups(backup_dir)
    to_delete = files[keep:]
    for old in to_delete:
        try:
            old.unlink()
            logger.debug("_prune_backups: deleted %s", old)
        except OSError as exc:
            logger.warning("_prune_backups: could not delete %s: %s", old, exc)
