"""
Backup management for Green Shift data.
Provides automatic backups and recovery mechanisms.
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
        
        self.db_path = self.data_dir / "sensor_data.db"
        self.research_db_path = self.data_dir / "research_data.db"
        self.state_file = self.data_dir / "state.json"
        
        _LOGGER.info("Backup manager initialized at: %s", self.backup_dir)
    
    async def create_backup(self, backup_type: str = "auto") -> bool:
        """Create a timestamped backup of all data files.
        
        Args:
            backup_type: Type of backup ('auto', 'manual', 'daily', 'weekly')
            
        Returns:
            True if backup successful, False otherwise
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = self.backup_dir / f"{backup_type}_{timestamp}"
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
    
    async def cleanup_old_backups(self, keep_auto: int = 48, keep_daily: int = 7, keep_weekly: int = 4) -> None:
        """Remove old backups to save space.
        
        Args:
            keep_auto: Number of automatic backups to keep (default: 48 = 2 days if hourly)
            keep_daily: Number of daily backups to keep
            keep_weekly: Number of weekly backups to keep
        """
        try:
            def _cleanup():
                # Get all backup directories
                auto_backups = sorted([d for d in self.backup_dir.glob("auto_*")], reverse=True)
                daily_backups = sorted([d for d in self.backup_dir.glob("daily_*")], reverse=True)
                weekly_backups = sorted([d for d in self.backup_dir.glob("weekly_*")], reverse=True)
                
                # Remove old auto backups
                for backup in auto_backups[keep_auto:]:
                    shutil.rmtree(backup)
                    _LOGGER.debug("Removed old auto backup: %s", backup.name)
                
                # Remove old daily backups
                for backup in daily_backups[keep_daily:]:
                    shutil.rmtree(backup)
                    _LOGGER.debug("Removed old daily backup: %s", backup.name)
                
                # Remove old weekly backups
                for backup in weekly_backups[keep_weekly:]:
                    shutil.rmtree(backup)
                    _LOGGER.debug("Removed old weekly backup: %s", backup.name)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _cleanup)
            
            _LOGGER.info("Backup cleanup complete")
            
        except Exception as e:
            _LOGGER.error("Backup cleanup failed: %s", e)
    
    async def restore_from_backup(self, backup_name: str) -> bool:
        """Restore data from a specific backup.
        
        Args:
            backup_name: Name of the backup directory to restore from
            
        Returns:
            True if restore successful, False otherwise
        """
        try:
            backup_path = self.backup_dir / backup_name
            
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
        """List all available backups.
        
        Returns:
            List of backup directory names, sorted by date (newest first)
        """
        try:
            backups = sorted(
                [d.name for d in self.backup_dir.iterdir() if d.is_dir()],
                reverse=True
            )
            return backups
        except Exception as e:
            _LOGGER.error("Failed to list backups: %s", e)
            return []
