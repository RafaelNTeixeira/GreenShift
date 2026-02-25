"""
File: backup_manager.py
Description: This module defines the BackupManager class, which is responsible for managing automated backups of the SQLite databases and JSON state files used by the Green Shift component. 
It provides functionality to create timestamped backups, organize them by type (automatic, manual, startup, shutdown), clean up old backups to save space and restore from specific backups when needed. 
The backup operations are designed to be safe for active databases and are executed asynchronously to avoid blocking the main thread.
"""

import logging
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import asyncio

_LOGGER = logging.getLogger(__name__)


class BackupManager:
    """Manages automated backups for SQLite databases and JSON state files."""

    def __init__(self, data_dir: Path):
        """Initialize backup manager.

        Args:
            data_dir: Path to the green_shift_data directory
        """
        self.data_dir = Path(data_dir)
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # Create organized subdirectories for each backup type
        self.auto_dir = self.backup_dir / "auto"
        self.startup_dir = self.backup_dir / "startup"
        self.shutdown_dir = self.backup_dir / "shutdown"
        self.manual_dir = self.backup_dir / "manual"

        self.auto_dir.mkdir(exist_ok=True)
        self.startup_dir.mkdir(exist_ok=True)
        self.shutdown_dir.mkdir(exist_ok=True)
        self.manual_dir.mkdir(exist_ok=True)

        self.db_path = self.data_dir / "sensor_data.db"
        self.research_db_path = self.data_dir / "research_data.db"
        self.state_file = self.data_dir / "state.json"

        _LOGGER.info("Backup manager initialized at: %s", self.backup_dir)

    async def create_backup(self, backup_type: str = "auto") -> bool:
        """Create a timestamped backup of all data files.

        Args:
            backup_type: Type of backup ('auto', 'manual', 'startup', 'shutdown')

        Returns:
            True if backup successful, False otherwise
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Organize backups into type-specific subdirectories
            type_dir_map = {
                "auto": self.auto_dir,
                "startup": self.startup_dir,
                "shutdown": self.shutdown_dir,
                "manual": self.manual_dir
            }

            parent_dir = type_dir_map.get(backup_type, self.backup_dir)
            backup_subdir = parent_dir / timestamp
            backup_subdir.mkdir(exist_ok=True)

            # Backup SQLite databases using the proper backup API
            tasks = []
            if self.db_path.exists():
                tasks.append(self._backup_sqlite(self.db_path, backup_subdir / "sensor_data.db"))

            if self.research_db_path.exists():
                tasks.append(self._backup_sqlite(self.research_db_path, backup_subdir / "research_data.db"))

            # Backup JSON state file
            if self.state_file.exists():
                tasks.append(self._backup_file(self.state_file, backup_subdir / "state.json"))

            # Execute all backups concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check if any failed
            failures = [r for r in results if isinstance(r, Exception)]
            if failures:
                _LOGGER.error("Some backups failed: %s", failures)
                return False

            _LOGGER.info("Backup created successfully: %s", backup_subdir)
            return True

        except Exception as e:
            _LOGGER.error("Backup failed: %s", e)
            return False

    async def _backup_sqlite(self, source_db: Path, dest_db: Path) -> None:
        """Backup SQLite database using SQLite's backup API (safe for active databases).

        Args:
            source_db: Source database path
            dest_db: Destination database path
        """
        def _do_backup():
            # Connect to source and destination
            source = sqlite3.connect(str(source_db))
            dest = sqlite3.connect(str(dest_db))

            # Use SQLite's backup API (handles locking properly)
            with dest:
                source.backup(dest)

            source.close()
            dest.close()
            _LOGGER.debug("SQLite backup complete: %s -> %s", source_db.name, dest_db.name)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_backup)

    async def _backup_file(self, source_file: Path, dest_file: Path) -> None:
        """Backup a regular file.

        Args:
            source_file: Source file path
            dest_file: Destination file path
        """
        def _do_backup():
            shutil.copy2(str(source_file), str(dest_file))
            _LOGGER.debug("File backup complete: %s -> %s", source_file.name, dest_file.name)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_backup)

    async def cleanup_old_backups(self, keep_auto: int = 3, keep_startup: int = 10, keep_shutdown: int = 10) -> None:
        """Remove old backups to save space.

        Note: Manual backups are never auto-deleted.

        Args:
            keep_auto: Number of automatic backups to keep
            keep_startup: Number of startup backups to keep
            keep_shutdown: Number of shutdown backups to keep
        """
        try:
            def _cleanup():
                total_removed = 0

                # Clean auto backups
                auto_backups = sorted([d for d in self.auto_dir.iterdir() if d.is_dir()], reverse=True)
                for backup in auto_backups[keep_auto:]:
                    shutil.rmtree(backup)
                    total_removed += 1
                    _LOGGER.debug("Removed old auto backup: %s", backup.name)

                # Clean startup backups
                startup_backups = sorted([d for d in self.startup_dir.iterdir() if d.is_dir()], reverse=True)
                for backup in startup_backups[keep_startup:]:
                    shutil.rmtree(backup)
                    total_removed += 1
                    _LOGGER.debug("Removed old startup backup: %s", backup.name)

                # Clean shutdown backups
                shutdown_backups = sorted([d for d in self.shutdown_dir.iterdir() if d.is_dir()], reverse=True)
                for backup in shutdown_backups[keep_shutdown:]:
                    shutil.rmtree(backup)
                    total_removed += 1
                    _LOGGER.debug("Removed old shutdown backup: %s", backup.name)

                if total_removed > 0:
                    _LOGGER.info("Cleanup: removed %d old backups (keeping auto:%d, startup:%d, shutdown:%d)",
                                total_removed, keep_auto, keep_startup, keep_shutdown)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _cleanup)

        except Exception as e:
            _LOGGER.error("Backup cleanup failed: %s", e)

    async def restore_from_backup(self, backup_name: str) -> bool:
        """Restore data from a specific backup.

        Args:
            backup_name: Full path to backup (e.g., 'auto/20260218_100000' or just '20260218_100000')

        Returns:
            True if restore successful, False otherwise
        """
        try:
            # Support both full path and just timestamp
            if '/' in backup_name or '\\' in backup_name:
                backup_path = self.backup_dir / backup_name
            else:
                # Try to find in all type directories
                for type_dir in [self.auto_dir, self.startup_dir, self.shutdown_dir, self.manual_dir]:
                    potential_path = type_dir / backup_name
                    if potential_path.exists():
                        backup_path = potential_path
                        break
                else:
                    _LOGGER.error("Backup not found: %s", backup_name)
                    return False

            if not backup_path.exists():
                _LOGGER.error("Backup not found: %s", backup_name)
                return False

            def _do_restore():
                # Restore databases
                backup_sensor_db = backup_path / "sensor_data.db"
                backup_research_db = backup_path / "research_data.db"
                backup_state = backup_path / "state.json"

                if backup_sensor_db.exists():
                    shutil.copy2(str(backup_sensor_db), str(self.db_path))
                    _LOGGER.info("Restored sensor database from backup")

                if backup_research_db.exists():
                    shutil.copy2(str(backup_research_db), str(self.research_db_path))
                    _LOGGER.info("Restored research database from backup")

                if backup_state.exists():
                    shutil.copy2(str(backup_state), str(self.state_file))
                    _LOGGER.info("Restored state file from backup")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _do_restore)

            _LOGGER.info("Restore complete from backup: %s", backup_name)
            return True

        except Exception as e:
            _LOGGER.error("Restore failed: %s", e)
            return False

    def list_backups(self) -> list:
        """List all available backups organized by type.

        Returns:
            List of backup paths in format 'type/timestamp', sorted by date (newest first)
        """
        try:
            backups = []

            # Collect from all type directories
            for type_name, type_dir in [("auto", self.auto_dir), ("startup", self.startup_dir),
                                        ("shutdown", self.shutdown_dir), ("manual", self.manual_dir)]:
                for backup in type_dir.iterdir():
                    if backup.is_dir():
                        backups.append(f"{type_name}/{backup.name}")

            # Sort by timestamp (newest first)
            backups.sort(reverse=True)
            return backups
        except Exception as e:
            _LOGGER.error("Failed to list backups: %s", e)
            return []
