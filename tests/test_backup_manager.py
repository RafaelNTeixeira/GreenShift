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
         "shutdown": bm.shutdown_dir, "manual": bm.manual_dir,
         "pre_restore": bm.pre_restore_dir}[btype] / name
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
    async def test_removes_excess_pre_restore_backups(self, bm):
        for i in range(5):
            _make_backup_dir(bm, "pre_restore", f"2026021{i}_090000")
        await bm.cleanup_old_backups(keep_auto=10, keep_startup=10, keep_shutdown=10, keep_pre_restore=2)
        assert len(list(bm.pre_restore_dir.iterdir())) == 2

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

    def test_sorted_by_timestamp_across_types(self, bm):
        """Cross-type ordering must use only the timestamp, not the type prefix.

        Without the fix, alphabetical ordering puts 'startup/' before 'manual/' before 'auto/',
        so the oldest backup (startup/10:00) would appear first instead of the newest (auto/12:00).
        """
        # startup > manual > auto alphabetically, but timestamps must dominate
        _make_backup_dir(bm, "auto",    "20260218_120000")  # newest
        _make_backup_dir(bm, "manual",  "20260218_110000")  # middle
        _make_backup_dir(bm, "startup", "20260218_100000")  # oldest
        backups = bm.list_backups()
        timestamps = [b.split("/")[1] for b in backups]
        assert timestamps == sorted(timestamps, reverse=True)
        assert backups[0] == "auto/20260218_120000"    # newest first
        assert backups[-1] == "startup/20260218_100000"  # oldest last

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

    @pytest.mark.asyncio
    async def test_creates_pre_restore_backup(self, bm, data_dir):
        """A pre_restore safety backup must be created before any file is overwritten."""
        backup_path = bm.auto_dir / "20260218_100000"
        backup_path.mkdir()
        (backup_path / "state.json").write_text('{"phase": "active"}')

        assert len(list(bm.pre_restore_dir.iterdir())) == 0
        await bm.restore_from_backup("auto/20260218_100000")
        assert len(list(bm.pre_restore_dir.iterdir())) == 1

    @pytest.mark.asyncio
    async def test_returns_false_for_corrupt_sqlite(self, bm, data_dir):
        """Restore must fail if a backup SQLite file fails integrity_check."""
        backup_path = bm.auto_dir / "20260218_100000"
        backup_path.mkdir()
        # Write bytes that are not a valid SQLite database
        (backup_path / "sensor_data.db").write_bytes(b"this is not a valid sqlite database file")

        result = await bm.restore_from_backup("auto/20260218_100000")
        assert result is False
        # Live DB must not have been overwritten
        assert not (data_dir / "sensor_data.db").exists()

    @pytest.mark.asyncio
    async def test_returns_false_when_integrity_check_not_ok(self, bm):
        """Restore must fail when PRAGMA integrity_check does not return 'ok'."""
        backup_path = bm.auto_dir / "20260218_100000"
        backup_path.mkdir()
        _create_dummy_db(backup_path / "sensor_data.db")

        class _DummyConn:
            def execute(self, _query):
                class _Res:
                    @staticmethod
                    def fetchone():
                        return ("corrupt",)

                return _Res()

            def close(self):
                return None

        class _DummyLoop:
            async def run_in_executor(self, _executor, func):
                func()

        with patch.object(backup_mod.sqlite3, "connect", return_value=_DummyConn()):
            with patch.object(backup_mod.asyncio, "get_event_loop", return_value=_DummyLoop()):
                result = await bm.restore_from_backup("auto/20260218_100000")

        assert result is False

    @pytest.mark.asyncio
    async def test_restore_unlinks_stale_wal_files_inline_executor(self, bm, data_dir):
        """When WAL/SHM files exist, restore removes them for both DBs."""
        backup_path = bm.auto_dir / "20260218_100001"
        backup_path.mkdir()
        _create_dummy_db(backup_path / "sensor_data.db")
        _create_dummy_db(backup_path / "research_data.db")

        # Plant stale files that must be removed by restore
        sensor_wal = Path(str(bm.db_path) + "-wal")
        sensor_shm = Path(str(bm.db_path) + "-shm")
        research_wal = Path(str(bm.research_db_path) + "-wal")
        research_shm = Path(str(bm.research_db_path) + "-shm")
        for f in (sensor_wal, sensor_shm, research_wal, research_shm):
            f.write_text("stale")

        class _InlineLoop:
            async def run_in_executor(self, _executor, func):
                func()

        with patch.object(backup_mod.asyncio, "get_event_loop", return_value=_InlineLoop()):
            result = await bm.restore_from_backup("auto/20260218_100001")

        assert result is True
        assert not sensor_wal.exists()
        assert not sensor_shm.exists()
        assert not research_wal.exists()
        assert not research_shm.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Error / edge-case paths
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorPaths:

    @pytest.mark.asyncio
    async def test_create_backup_returns_false_on_file_backup_failure(self, bm, data_dir):
        """If _backup_file raises, the gathered results contain an Exception -> False."""
        (data_dir / "state.json").write_text("{}")

        original = bm._backup_file

        async def failing_backup_file(src, dst):
            raise OSError("disk full")

        bm._backup_file = failing_backup_file

        result = await bm.create_backup("auto")
        assert result is False

        bm._backup_file = original  # restore

    @pytest.mark.asyncio
    async def test_create_backup_returns_false_on_top_level_exception(self, bm):
        """If mkdir itself raises, the outer except block must return False."""
        original_mkdir = bm.auto_dir.mkdir

        def exploding_mkdir(*a, **kw):
            raise PermissionError("no write access")

        with patch.object(type(bm.auto_dir), "__truediv__",
                          side_effect=PermissionError("boom")):
            result = await bm.create_backup("auto")

        assert result is False

    @pytest.mark.asyncio
    async def test_restore_returns_false_for_full_path_not_found(self, bm):
        """Full path (with /) that doesn't exist on disk must return False."""
        result = await bm.restore_from_backup("auto/20991231_999999")
        assert result is False

    @pytest.mark.asyncio
    async def test_restore_returns_false_when_timestamp_not_in_any_dir(self, bm):
        """Timestamp-only form not found in any type dir must return False."""
        result = await bm.restore_from_backup("20991231_999999")
        assert result is False

    @pytest.mark.asyncio
    async def test_restore_with_backslash_path_not_found(self, bm):
        """restore_from_backup treats backslash paths the same as forward-slash (full path)."""
        result = await bm.restore_from_backup("auto\\20991231_000000")
        assert result is False

    @pytest.mark.asyncio
    async def test_restore_restores_both_databases(self, bm, data_dir):
        """Both sensor_data.db and research_data.db are restored from backup."""
        backup_path = bm.manual_dir / "20260218_150000"
        backup_path.mkdir()
        _create_dummy_db(backup_path / "sensor_data.db")
        _create_dummy_db(backup_path / "research_data.db")
        (backup_path / "state.json").write_text('{"phase":"active"}')

        result = await bm.restore_from_backup("manual/20260218_150000")
        assert result is True
        assert (data_dir / "sensor_data.db").exists()
        assert (data_dir / "research_data.db").exists()
        assert (data_dir / "state.json").read_text() == '{"phase":"active"}'

    @pytest.mark.asyncio
    async def test_cleanup_exception_is_handled(self, bm):
        """cleanup_old_backups must not raise even if an internal error occurs."""
        from unittest.mock import patch as _patch
        with _patch("shutil.rmtree", side_effect=OSError("locked")):
            # Create excess backups so rmtree is called
            for i in range(3):
                _make_backup_dir(bm, "auto", f"2026021{i}_100000")
            # Should not raise
            await bm.cleanup_old_backups(keep_auto=1, keep_startup=10, keep_shutdown=10)

    def test_list_backups_returns_empty_on_exception(self, bm):
        """list_backups must return [] rather than raising when iterdir fails."""
        from unittest.mock import patch as _patch

        def bad_iterdir(self_dir):
            raise OSError("permission denied")

        with _patch.object(type(bm.auto_dir), "iterdir", bad_iterdir):
            result = bm.list_backups()

        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Data isolation: two backups with different data, restore one -> exact match
# ─────────────────────────────────────────────────────────────────────────────

class TestRestoreDataIsolation:
    """
    Core correctness tests: after restoring a specific backup the live files must
    contain exactly the data that was in that backup — no contamination from
    the other backup or from the current (live) state.
    """

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _create_db_with_marker(self, path: Path, marker: str) -> None:
        """Create a minimal SQLite DB containing a single identifiable row."""
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE IF NOT EXISTS backup_marker (value TEXT)")
        conn.execute("DELETE FROM backup_marker")
        conn.execute("INSERT INTO backup_marker VALUES (?)", (marker,))
        conn.commit()
        conn.close()

    def _read_marker(self, path: Path) -> str:
        """Return the marker value stored in a restored DB, or None if absent."""
        conn = sqlite3.connect(str(path))
        row = conn.execute("SELECT value FROM backup_marker").fetchone()
        conn.close()
        return row[0] if row else None

    # ── Core round-trip: restore A does not contain B's data ─────────────────

    @pytest.mark.asyncio
    async def test_restore_backup_a_excludes_backup_b_data(self, bm, data_dir):
        """Restoring backup A after B has been written must yield A's data."""
        # ── Backup A ─────────────────────────────────────────────────────────
        self._create_db_with_marker(data_dir / "sensor_data.db", "sensor_A")
        self._create_db_with_marker(data_dir / "research_data.db", "research_A")
        (data_dir / "state.json").write_text('{"phase": "baseline", "snapshot": "A"}')
        assert await bm.create_backup("manual") is True
        backup_a = "manual/" + list(bm.manual_dir.iterdir())[0].name

        # ── Overwrite live files with B's content ────────────────────────────
        (data_dir / "sensor_data.db").unlink()
        (data_dir / "research_data.db").unlink()
        self._create_db_with_marker(data_dir / "sensor_data.db", "sensor_B")
        self._create_db_with_marker(data_dir / "research_data.db", "research_B")
        (data_dir / "state.json").write_text('{"phase": "active", "snapshot": "B"}')

        # ── Restore A ────────────────────────────────────────────────────────
        assert await bm.restore_from_backup(backup_a) is True

        # ── Verify live data == A (not B) ─────────────────────────────────────
        assert self._read_marker(data_dir / "sensor_data.db") == "sensor_A"
        assert self._read_marker(data_dir / "research_data.db") == "research_A"
        state_text = (data_dir / "state.json").read_text()
        assert '"snapshot": "A"' in state_text
        assert '"snapshot": "B"' not in state_text

    @pytest.mark.asyncio
    async def test_restore_backup_b_excludes_backup_a_data(self, bm, data_dir):
        """Restoring backup B must yield B's data even though A was created first."""
        # ── Backup A (manual dir) ─────────────────────────────────────────────
        self._create_db_with_marker(data_dir / "sensor_data.db", "sensor_A")
        (data_dir / "state.json").write_text('{"snapshot": "A"}')
        await bm.create_backup("manual")

        # ── Backup B (auto dir) ───────────────────────────────────────────────
        (data_dir / "sensor_data.db").unlink()
        self._create_db_with_marker(data_dir / "sensor_data.db", "sensor_B")
        (data_dir / "state.json").write_text('{"snapshot": "B"}')
        assert await bm.create_backup("auto") is True
        backup_b = "auto/" + list(bm.auto_dir.iterdir())[0].name

        # ── Overwrite live with "C" (simulate time passing) ───────────────────
        (data_dir / "sensor_data.db").unlink()
        self._create_db_with_marker(data_dir / "sensor_data.db", "sensor_C")
        (data_dir / "state.json").write_text('{"snapshot": "C"}')

        # ── Restore B ────────────────────────────────────────────────────────
        assert await bm.restore_from_backup(backup_b) is True

        assert self._read_marker(data_dir / "sensor_data.db") == "sensor_B"
        state_text = (data_dir / "state.json").read_text()
        assert '"snapshot": "B"' in state_text
        assert '"snapshot": "A"' not in state_text
        assert '"snapshot": "C"' not in state_text

    # ── Two-backup round-trip: swap between A and B, then back ───────────────

    @pytest.mark.asyncio
    async def test_alternating_restores_always_match_their_backup(self, bm, data_dir):
        """
        Create A and B in different type dirs, then restore A -> B -> A.
        Each restore must yield exactly the data from that backup.
        """
        # ── Backup A (manual) ────────────────────────────────────────────────
        self._create_db_with_marker(data_dir / "sensor_data.db", "epoch_1")
        self._create_db_with_marker(data_dir / "research_data.db", "r_epoch_1")
        (data_dir / "state.json").write_text('{"epoch": 1}')
        await bm.create_backup("manual")
        backup_a = "manual/" + list(bm.manual_dir.iterdir())[0].name

        # ── Backup B (auto) ───────────────────────────────────────────────────
        (data_dir / "sensor_data.db").unlink()
        (data_dir / "research_data.db").unlink()
        self._create_db_with_marker(data_dir / "sensor_data.db", "epoch_2")
        self._create_db_with_marker(data_dir / "research_data.db", "r_epoch_2")
        (data_dir / "state.json").write_text('{"epoch": 2}')
        await bm.create_backup("auto")
        backup_b = "auto/" + list(bm.auto_dir.iterdir())[0].name

        # ── Restore A ────────────────────────────────────────────────────────
        assert await bm.restore_from_backup(backup_a) is True
        assert self._read_marker(data_dir / "sensor_data.db") == "epoch_1"
        assert self._read_marker(data_dir / "research_data.db") == "r_epoch_1"
        assert '"epoch": 1' in (data_dir / "state.json").read_text()

        # ── Restore B ────────────────────────────────────────────────────────
        assert await bm.restore_from_backup(backup_b) is True
        assert self._read_marker(data_dir / "sensor_data.db") == "epoch_2"
        assert self._read_marker(data_dir / "research_data.db") == "r_epoch_2"
        assert '"epoch": 2' in (data_dir / "state.json").read_text()

        # ── Restore A again ───────────────────────────────────────────────────
        assert await bm.restore_from_backup(backup_a) is True
        assert self._read_marker(data_dir / "sensor_data.db") == "epoch_1"
        assert self._read_marker(data_dir / "research_data.db") == "r_epoch_1"
        assert '"epoch": 1' in (data_dir / "state.json").read_text()

    # ── Cross-DB isolation ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_restore_does_not_mix_sensor_and_research_db_data(self, bm, data_dir):
        """sensor_data.db and research_data.db must each restore their own content."""
        self._create_db_with_marker(data_dir / "sensor_data.db", "sensor_ALPHA")
        self._create_db_with_marker(data_dir / "research_data.db", "research_ALPHA")
        await bm.create_backup("manual")
        backup_name = "manual/" + list(bm.manual_dir.iterdir())[0].name

        # Overwrite live with different content
        (data_dir / "sensor_data.db").unlink()
        (data_dir / "research_data.db").unlink()
        self._create_db_with_marker(data_dir / "sensor_data.db", "sensor_BETA")
        self._create_db_with_marker(data_dir / "research_data.db", "research_BETA")

        assert await bm.restore_from_backup(backup_name) is True

        assert self._read_marker(data_dir / "sensor_data.db") == "sensor_ALPHA"
        assert self._read_marker(data_dir / "research_data.db") == "research_ALPHA"
        # Make sure they weren't swapped
        assert self._read_marker(data_dir / "sensor_data.db") != "research_ALPHA"
        assert self._read_marker(data_dir / "research_data.db") != "sensor_ALPHA"

    # ── WAL / SHM cleanup ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_stale_sensor_wal_and_shm_deleted_after_restore(self, bm, data_dir):
        """Stale -wal and -shm files for sensor_data.db are removed after restore."""
        self._create_db_with_marker(data_dir / "sensor_data.db", "marker")
        await bm.create_backup("auto")
        backup_name = "auto/" + list(bm.auto_dir.iterdir())[0].name

        # Plant stale WAL/SHM companion files
        wal = Path(str(bm.db_path) + "-wal")
        shm = Path(str(bm.db_path) + "-shm")
        wal.write_bytes(b"stale wal bytes")
        shm.write_bytes(b"stale shm bytes")

        assert await bm.restore_from_backup(backup_name) is True

        assert not wal.exists(), "-wal file was not removed after restore"
        assert not shm.exists(), "-shm file was not removed after restore"

    @pytest.mark.asyncio
    async def test_stale_research_wal_and_shm_deleted_after_restore(self, bm, data_dir):
        """Stale -wal and -shm files for research_data.db are removed after restore."""
        self._create_db_with_marker(data_dir / "research_data.db", "research_marker")
        await bm.create_backup("auto")
        backup_name = "auto/" + list(bm.auto_dir.iterdir())[0].name

        wal = Path(str(bm.research_db_path) + "-wal")
        shm = Path(str(bm.research_db_path) + "-shm")
        wal.write_bytes(b"stale wal bytes")
        shm.write_bytes(b"stale shm bytes")

        assert await bm.restore_from_backup(backup_name) is True

        assert not wal.exists(), "research -wal file was not removed after restore"
        assert not shm.exists(), "research -shm file was not removed after restore"

    @pytest.mark.asyncio
    async def test_restore_succeeds_when_no_wal_files_present(self, bm, data_dir):
        """Restore must succeed cleanly when no WAL/SHM files exist."""
        self._create_db_with_marker(data_dir / "sensor_data.db", "clean_marker")
        await bm.create_backup("auto")
        backup_name = "auto/" + list(bm.auto_dir.iterdir())[0].name

        # Ensure no WAL files exist
        for suffix in ("-wal", "-shm"):
            f = Path(str(bm.db_path) + suffix)
            assert not f.exists()

        assert await bm.restore_from_backup(backup_name) is True
        assert self._read_marker(data_dir / "sensor_data.db") == "clean_marker"

    # ── Pre-restore safety backup captures the live state ────────────────────

    @pytest.mark.asyncio
    async def test_pre_restore_backup_captures_overwritten_state(self, bm, data_dir):
        """The pre_restore backup created before overwriting must hold the pre-restore data."""
        # Backup A
        self._create_db_with_marker(data_dir / "sensor_data.db", "to_restore")
        (data_dir / "state.json").write_text('{"snapshot": "target"}')
        await bm.create_backup("manual")
        backup_a = "manual/" + list(bm.manual_dir.iterdir())[0].name

        # Change live data
        (data_dir / "sensor_data.db").unlink()
        self._create_db_with_marker(data_dir / "sensor_data.db", "current_live")
        (data_dir / "state.json").write_text('{"snapshot": "live"}')

        # Restore A — this should first snapshot the live state into pre_restore
        assert await bm.restore_from_backup(backup_a) is True

        # Pre-restore dir must have exactly one entry (the safety snapshot)
        pre_restore_entries = list(bm.pre_restore_dir.iterdir())
        assert len(pre_restore_entries) == 1
        pre_restore_state = pre_restore_entries[0] / "state.json"
        assert pre_restore_state.exists()
        assert '"snapshot": "live"' in pre_restore_state.read_text()

        # And the live state is now what was in the backup
        assert '"snapshot": "target"' in (data_dir / "state.json").read_text()
