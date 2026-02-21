"""
Tests for backup_manager.py

Covers:
- Directory structure creation on init
- create_backup: correct subdirectory, files copied
- cleanup_old_backups: removes excess by type, never touches manual
- list_backups: sorted newest-first, correct format
- restore_from_backup: files copied back; missing backup returns False
"""
import pytest
import asyncio
import sqlite3
import shutil
from pathlib import Path
from unittest.mock import patch

import sys, types, pathlib

# Provide a minimal logging stub so backup_manager doesn't need the full stack
if "homeassistant" not in sys.modules:
    sys.modules["homeassistant"] = types.ModuleType("homeassistant")

spec = __import__("importlib").util.spec_from_file_location(
    "backup_manager",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "backup_manager.py"
)
backup_mod = __import__("importlib").util.module_from_spec(spec)
spec.loader.exec_module(backup_mod)
BackupManager = backup_mod.BackupManager


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_backup_dir(bm: BackupManager, btype: str, name: str) -> Path:
    """Create a fake backup subdirectory."""
    d = {"auto": bm.auto_dir, "startup": bm.startup_dir,
         "shutdown": bm.shutdown_dir, "manual": bm.manual_dir}[btype] / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _create_dummy_db(path: Path):
    """Create a minimal valid SQLite file."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "green_shift_data"


@pytest.fixture
def bm(data_dir):
    data_dir.mkdir()
    return BackupManager(data_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestBackupManagerInit:

    def test_backup_dir_created(self, bm):
        assert bm.backup_dir.exists()

    def test_type_subdirs_created(self, bm):
        assert bm.auto_dir.exists()
        assert bm.startup_dir.exists()
        assert bm.shutdown_dir.exists()
        assert bm.manual_dir.exists()


# ─────────────────────────────────────────────────────────────────────────────
# create_backup
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateBackup:

    @pytest.mark.asyncio
    async def test_returns_true_when_no_files_exist(self, bm):
        result = await bm.create_backup("auto")
        assert result is True

    @pytest.mark.asyncio
    async def test_backup_subdir_created_under_type_dir(self, bm):
        await bm.create_backup("auto")
        subdirs = list(bm.auto_dir.iterdir())
        assert len(subdirs) == 1

    @pytest.mark.asyncio
    async def test_state_file_copied(self, bm, data_dir):
        state = data_dir / "state.json"
        state.write_text('{"phase": "baseline"}')
        await bm.create_backup("manual")
        backup_dirs = list(bm.manual_dir.iterdir())
        assert (backup_dirs[0] / "state.json").exists()

    @pytest.mark.asyncio
    async def test_sqlite_db_backed_up(self, bm, data_dir):
        _create_dummy_db(data_dir / "sensor_data.db")
        await bm.create_backup("startup")
        backup_dirs = list(bm.startup_dir.iterdir())
        assert (backup_dirs[0] / "sensor_data.db").exists()

    @pytest.mark.asyncio
    async def test_both_dbs_backed_up(self, bm, data_dir):
        _create_dummy_db(data_dir / "sensor_data.db")
        _create_dummy_db(data_dir / "research_data.db")
        await bm.create_backup("shutdown")
        backup_dirs = list(bm.shutdown_dir.iterdir())
        assert (backup_dirs[0] / "sensor_data.db").exists()
        assert (backup_dirs[0] / "research_data.db").exists()

    @pytest.mark.asyncio
    async def test_unknown_type_uses_root_backup_dir(self, bm):
        result = await bm.create_backup("unknown_type")
        assert result is True


# ─────────────────────────────────────────────────────────────────────────────
# cleanup_old_backups
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanupOldBackups:

    @pytest.mark.asyncio
    async def test_removes_excess_auto_backups(self, bm):
        # Create 5 auto backups, keep only 3
        for i in range(5):
            _make_backup_dir(bm, "auto", f"2026021{i}_100000")
        await bm.cleanup_old_backups(keep_auto=3, keep_startup=10, keep_shutdown=10)
        remaining = list(bm.auto_dir.iterdir())
        assert len(remaining) == 3

    @pytest.mark.asyncio
    async def test_keeps_newest_auto_backups(self, bm):
        names = [f"20260218_1000{i:02d}" for i in range(5)]
        for n in names:
            _make_backup_dir(bm, "auto", n)
        await bm.cleanup_old_backups(keep_auto=2, keep_startup=10, keep_shutdown=10)
        kept = sorted([d.name for d in bm.auto_dir.iterdir()], reverse=True)
        assert kept == sorted(names, reverse=True)[:2]

    @pytest.mark.asyncio
    async def test_manual_backups_never_deleted(self, bm):
        for i in range(10):
            _make_backup_dir(bm, "manual", f"2026021{i}_120000")
        await bm.cleanup_old_backups(keep_auto=1, keep_startup=1, keep_shutdown=1)
        # Manual dir untouched
        assert len(list(bm.manual_dir.iterdir())) == 10

    @pytest.mark.asyncio
    async def test_cleanup_startup_and_shutdown(self, bm):
        for i in range(5):
            _make_backup_dir(bm, "startup", f"2026021{i}_080000")
            _make_backup_dir(bm, "shutdown", f"2026021{i}_200000")
        await bm.cleanup_old_backups(keep_auto=10, keep_startup=2, keep_shutdown=2)
        assert len(list(bm.startup_dir.iterdir())) == 2
        assert len(list(bm.shutdown_dir.iterdir())) == 2


# ─────────────────────────────────────────────────────────────────────────────
# list_backups
# ─────────────────────────────────────────────────────────────────────────────

class TestListBackups:

    def test_empty_returns_empty_list(self, bm):
        assert bm.list_backups() == []

    def test_format_is_type_slash_timestamp(self, bm):
        _make_backup_dir(bm, "auto", "20260218_100000")
        backups = bm.list_backups()
        assert "auto/20260218_100000" in backups

    def test_all_types_included(self, bm):
        _make_backup_dir(bm, "auto", "20260218_100000")
        _make_backup_dir(bm, "startup", "20260218_090000")
        _make_backup_dir(bm, "shutdown", "20260218_200000")
        _make_backup_dir(bm, "manual", "20260218_150000")
        backups = bm.list_backups()
        types_found = {b.split("/")[0] for b in backups}
        assert types_found == {"auto", "startup", "shutdown", "manual"}

    def test_sorted_newest_first(self, bm):
        for ts in ["20260215_100000", "20260218_100000", "20260217_100000"]:
            _make_backup_dir(bm, "auto", ts)
        backups = bm.list_backups()
        auto_backups = [b for b in backups if b.startswith("auto/")]
        assert auto_backups == sorted(auto_backups, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# restore_from_backup
# ─────────────────────────────────────────────────────────────────────────────

class TestRestoreFromBackup:

    @pytest.mark.asyncio
    async def test_returns_false_for_missing_backup(self, bm):
        result = await bm.restore_from_backup("nonexistent/20260101_000000")
        assert result is False

    @pytest.mark.asyncio
    async def test_restores_state_file(self, bm, data_dir):
        # Create a backup with a state file
        backup_path = bm.auto_dir / "20260218_100000"
        backup_path.mkdir()
        (backup_path / "state.json").write_text('{"phase": "active"}')

        result = await bm.restore_from_backup("auto/20260218_100000")
        assert result is True
        assert (data_dir / "state.json").read_text() == '{"phase": "active"}'

    @pytest.mark.asyncio
    async def test_restore_by_timestamp_only(self, bm, data_dir):
        """Should find backup across type directories by timestamp alone."""
        backup_path = bm.startup_dir / "20260218_090000"
        backup_path.mkdir()
        (backup_path / "state.json").write_text('{"phase": "baseline"}')

        result = await bm.restore_from_backup("20260218_090000")
        assert result is True
