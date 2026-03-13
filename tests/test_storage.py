"""
Tests for storage.py

Covers:
- Database initialization (sensor_data.db, research_data.db)
- store_sensor_snapshot: data insertion, working hours flag, UTC epoch storage
- store_area_snapshot: area-based data insertion
- get_history: temporal queries with filters
- get_area_history: area-specific temporal queries
- get_all_areas: unique area listing
- get_area_stats: aggregation (avg, min, max, stddev)
- get_recent_values: most recent N values
- save_daily_tasks / get_today_tasks: task persistence
- Data cleanup: old temporal data removal
- RL episode cleanup: retention policy enforcement
- compute_daily_aggregates: avg_power_working_hours / avg_power_off_hours columns
- get_active_phase_savings: prefers working-hours representative power"""
import pytest
import sys
import types
import pathlib
import importlib.util
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, AsyncMock, patch

# ── Minimal HA stubs ────────────────────────────────────────────────────────
for mod_name in [
    "homeassistant", 
    "homeassistant.core", 
    "homeassistant.helpers", 
    "homeassistant.helpers.area_registry",
    "homeassistant.helpers.entity_registry",
    "homeassistant.helpers.device_registry"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Import real const module
const_spec = importlib.util.spec_from_file_location(
    "custom_components.green_shift.const",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "const.py"
)
const_mod = importlib.util.module_from_spec(const_spec)
const_mod.__package__ = "custom_components.green_shift"
const_spec.loader.exec_module(const_mod)
sys.modules["custom_components.green_shift.const"] = const_mod

helpers_spec = importlib.util.spec_from_file_location(
    "custom_components.green_shift.helpers",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "helpers.py"
)
helpers_mod = importlib.util.module_from_spec(helpers_spec)
helpers_mod.__package__ = "custom_components.green_shift"
helpers_spec.loader.exec_module(helpers_mod)
sys.modules["custom_components.green_shift.helpers"] = helpers_mod

# Load storage module
storage_spec = importlib.util.spec_from_file_location(
    "storage",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "storage.py"
)
storage_mod = importlib.util.module_from_spec(storage_spec)
storage_mod.__package__ = "custom_components.green_shift"
storage_spec.loader.exec_module(storage_mod)
StorageManager = storage_mod.StorageManager


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_hass(tmp_path):
    hass = MagicMock()
    hass.config = MagicMock()
    # Make config.path a callable that returns a string
    hass.config.path = lambda x: str(tmp_path / x)
    # Mock async executor job - execute function directly in async wrapper
    async def async_executor_job(func, *args):
        return func(*args)
    hass.async_add_executor_job = async_executor_job
    return hass


@pytest.fixture
def storage(mock_hass, event_loop):
    """Fixture that creates and sets up storage synchronously."""
    sm = StorageManager(mock_hass)
    event_loop.run_until_complete(sm.setup())
    return sm


# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────

class TestStorageInit:

    @pytest.mark.asyncio
    async def test_creates_data_directory(self, storage):
        assert storage.config_dir.exists()

    @pytest.mark.asyncio
    async def test_creates_sensor_database(self, storage):
        assert storage.db_path.exists()

    @pytest.mark.asyncio
    async def test_creates_research_database(self, storage):
        assert storage.research_db_path.exists()

    @pytest.mark.asyncio
    async def test_creates_all_sensor_tables(self, storage):
        """Verify all expected tables exist in sensor_data.db."""
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "sensor_history", "area_sensor_history", "daily_tasks"
        }
        assert expected.issubset(tables)

    @pytest.mark.asyncio
    async def test_creates_research_tables(self, storage):
        """Verify research database has expected tables."""
        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        # Should include RL episodes, notifications, tasks, etc.
        assert "research_rl_episodes" in tables
        assert "research_blocked_notifications" in tables


# ─────────────────────────────────────────────────────────────────────────────
# store_sensor_snapshot
# ─────────────────────────────────────────────────────────────────────────────

class TestStoreSensorSnapshot:

    @pytest.mark.asyncio
    async def test_inserts_power_data(self, storage):
        now = datetime.now()
        await storage.store_sensor_snapshot(timestamp=now, power=500.0)

        history = await storage.get_history("power", hours=1)
        assert len(history) == 1
        assert history[0][1] == 500.0

    @pytest.mark.asyncio
    async def test_inserts_all_metrics(self, storage):
        now = datetime.now()
        await storage.store_sensor_snapshot(
            timestamp=now,
            power=500.0,
            energy=2.5,
            temperature=21.5,
            humidity=50.0,
            illuminance=300.0,
            occupancy=True
        )

        assert (await storage.get_history("power", hours=1))[0][1] == 500.0
        assert (await storage.get_history("energy", hours=1))[0][1] == 2.5
        assert (await storage.get_history("temperature", hours=1))[0][1] == 21.5
        assert (await storage.get_history("humidity", hours=1))[0][1] == 50.0
        assert (await storage.get_history("illuminance", hours=1))[0][1] == 300.0
        assert (await storage.get_history("occupancy", hours=1))[0][1] == 1.0

    @pytest.mark.asyncio
    async def test_working_hours_flag_stored(self, storage):
        now = datetime.now()
        await storage.store_sensor_snapshot(
            timestamp=now, power=500.0, within_working_hours=True
        )

        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT within_working_hours FROM sensor_history WHERE power = 500.0")
        result = cursor.fetchone()
        conn.close()
        flag = result[0] if result else None
        assert flag == 1

    @pytest.mark.asyncio
    async def test_null_values_allowed(self, storage):
        now = datetime.now()
        await storage.store_sensor_snapshot(timestamp=now, power=None, temperature=22.0)

        # Should not raise, and temperature should be stored
        temp_history = await storage.get_history("temperature", hours=1)
        assert len(temp_history) == 1
        assert temp_history[0][1] == 22.0

    @pytest.mark.asyncio
    async def test_naive_datetime_stored_as_local_epoch(self, storage):
        """A naive datetime must be treated as local time (not UTC) when converting
        to a Unix epoch.  

        We verify that the stored epoch equals naive_ref.timestamp() (which Python
        interprets as local time), NOT datetime(..., tzinfo=UTC).timestamp()."""
        naive_ref = datetime(2024, 6, 1, 12, 0, 0)   # noon, no tzinfo -> local time
        expected_ts = naive_ref.timestamp()            # Python treats naive as local

        await storage.store_sensor_snapshot(timestamp=naive_ref, power=42.0)

        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp FROM sensor_history WHERE power = 42.0")
        row = cursor.fetchone()
        conn.close()

        assert row is not None, "Snapshot was not stored"
        stored_ts = row[0]
        assert stored_ts == pytest.approx(expected_ts, abs=1), (
            f"Expected local-time epoch {expected_ts} but got {stored_ts}; "
            "naive timestamps must be treated as local time, not UTC."
        )

    @pytest.mark.asyncio
    async def test_aware_utc_datetime_stored_correctly(self, storage):
        """A timezone-aware UTC datetime must be stored as the correct UTC epoch."""
        utc_ref = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        expected_ts = utc_ref.timestamp()

        await storage.store_sensor_snapshot(timestamp=utc_ref, power=99.0)

        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp FROM sensor_history WHERE power = 99.0")
        row = cursor.fetchone()
        conn.close()

        assert row is not None, "Snapshot was not stored"
        stored_ts = row[0]
        assert stored_ts == pytest.approx(expected_ts, abs=1), (
            f"Expected UTC epoch {expected_ts} but got {stored_ts}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# store_area_snapshot
# ─────────────────────────────────────────────────────────────────────────────

class TestStoreAreaSnapshot:

    @pytest.mark.asyncio
    async def test_inserts_area_data(self, storage):
        now = datetime.now()
        await storage.store_area_snapshot(
            timestamp=now,
            area_name="Living Room",
            power=300.0,
            temperature=22.0
        )

        history = await storage.get_area_history("Living Room", "power", hours=1)
        assert len(history) == 1
        assert history[0][1] == 300.0

    @pytest.mark.asyncio
    async def test_multiple_areas_stored_separately(self, storage):
        now = datetime.now()
        await storage.store_area_snapshot(now, "Kitchen", power=200.0)
        await storage.store_area_snapshot(now, "Bedroom", power=100.0)

        kitchen = await storage.get_area_history("Kitchen", "power", hours=1)
        bedroom = await storage.get_area_history("Bedroom", "power", hours=1)

        assert kitchen[0][1] == 200.0
        assert bedroom[0][1] == 100.0

    @pytest.mark.asyncio
    async def test_working_hours_flag_for_areas(self, storage):
        now = datetime.now()
        await storage.store_area_snapshot(
            now, "Office", power=500.0, within_working_hours=False
        )

        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT within_working_hours FROM area_sensor_history WHERE area_name = 'Office'"
        )
        result = cursor.fetchone()
        conn.close()
        flag = result[0] if result else None
        assert flag == 0


# ─────────────────────────────────────────────────────────────────────────────
# get_history
# ─────────────────────────────────────────────────────────────────────────────

class TestGetHistory:

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_data(self, storage):
        result = await storage.get_history("power", hours=1)
        assert result == []

    @pytest.mark.asyncio
    async def test_hours_filter(self, storage):
        now = datetime.now()
        await storage.store_sensor_snapshot(now - timedelta(hours=2), power=100.0)
        await storage.store_sensor_snapshot(now - timedelta(minutes=30), power=200.0)

        result = await storage.get_history("power", hours=1)
        assert len(result) == 1
        assert result[0][1] == 200.0

    @pytest.mark.asyncio
    async def test_days_filter(self, storage):
        now = datetime.now()
        await storage.store_sensor_snapshot(now - timedelta(days=2), power=100.0)
        await storage.store_sensor_snapshot(now - timedelta(hours=12), power=200.0)

        result = await storage.get_history("power", days=1)
        assert len(result) == 1
        assert result[0][1] == 200.0

    @pytest.mark.asyncio
    async def test_working_hours_only_filter(self, storage):
        now = datetime.now()
        await storage.store_sensor_snapshot(now, power=100.0, within_working_hours=True)
        await storage.store_sensor_snapshot(now, power=200.0, within_working_hours=False)

        result = await storage.get_history("power", hours=1, working_hours_only=True)
        assert len(result) == 1
        assert result[0][1] == 100.0

    @pytest.mark.asyncio
    async def test_sorted_chronologically(self, storage):
        now = datetime.now()
        await storage.store_sensor_snapshot(now - timedelta(minutes=20), power=100.0)
        await storage.store_sensor_snapshot(now - timedelta(minutes=10), power=200.0)
        await storage.store_sensor_snapshot(now, power=300.0)

        result = await storage.get_history("power", hours=1)
        assert len(result) == 3
        assert result[0][1] == 100.0
        assert result[1][1] == 200.0
        assert result[2][1] == 300.0


# ─────────────────────────────────────────────────────────────────────────────
# get_area_history
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAreaHistory:

    @pytest.mark.asyncio
    async def test_filters_by_area(self, storage):
        now = datetime.now()
        await storage.store_area_snapshot(now, "Kitchen", power=200.0)
        await storage.store_area_snapshot(now, "Bedroom", power=100.0)

        result = await storage.get_area_history("Kitchen", "power", hours=1)
        assert len(result) == 1
        assert result[0][1] == 200.0

    @pytest.mark.asyncio
    async def test_time_filtering(self, storage):
        now = datetime.now()
        await storage.store_area_snapshot(now - timedelta(hours=2), "Kitchen", power=100.0)
        await storage.store_area_snapshot(now, "Kitchen", power=200.0)

        result = await storage.get_area_history("Kitchen", "power", hours=1)
        assert len(result) == 1
        assert result[0][1] == 200.0

    @pytest.mark.asyncio
    async def test_working_hours_filter(self, storage):
        now = datetime.now()
        await storage.store_area_snapshot(
            now, "Office", power=100.0, within_working_hours=True
        )
        await storage.store_area_snapshot(
            now, "Office", power=200.0, within_working_hours=False
        )

        result = await storage.get_area_history(
            "Office", "power", hours=1, working_hours_only=True
        )
        assert len(result) == 1
        assert result[0][1] == 100.0

    @pytest.mark.asyncio
    async def test_get_history_non_working_hours_filter_false(self, storage):
        now = datetime.now()
        await storage.store_sensor_snapshot(now, power=100.0, within_working_hours=True)
        await storage.store_sensor_snapshot(now, power=200.0, within_working_hours=False)

        result = await storage.get_history("power", hours=1, working_hours_only=False)
        assert len(result) == 1
        assert result[0][1] == 200.0

    @pytest.mark.asyncio
    async def test_get_area_history_non_working_hours_filter_false(self, storage):
        now = datetime.now()
        await storage.store_area_snapshot(now, "Office", power=100.0, within_working_hours=True)
        await storage.store_area_snapshot(now, "Office", power=200.0, within_working_hours=False)

        result = await storage.get_area_history("Office", "power", hours=1, working_hours_only=False)
        assert len(result) == 1
        assert result[0][1] == 200.0


class TestMetricWhitelist:
    """get_history and get_area_history must reject unknown metric names."""

    @pytest.mark.asyncio
    async def test_get_history_raises_for_invalid_metric(self, storage):
        """An unknown metric name must raise ValueError before any DB query."""
        with pytest.raises(ValueError, match="Invalid metric"):
            await storage.get_history("nonexistent_column", hours=1)

    @pytest.mark.asyncio
    async def test_get_history_raises_for_sql_injection_attempt(self, storage):
        """A crafted metric string that could inject SQL must be rejected."""
        with pytest.raises(ValueError, match="Invalid metric"):
            await storage.get_history("power; DROP TABLE sensor_history;--", hours=1)

    @pytest.mark.asyncio
    async def test_get_area_history_raises_for_invalid_metric(self, storage):
        """get_area_history must also validate the metric whitelist."""
        with pytest.raises(ValueError, match="Invalid metric"):
            await storage.get_area_history("Living Room", "bad_column", hours=1)

    @pytest.mark.asyncio
    async def test_get_history_accepts_all_valid_metrics(self, storage):
        """Each known metric must be accepted without raising."""
        valid_metrics = ["power", "energy", "temperature", "humidity", "illuminance", "occupancy"]
        for metric in valid_metrics:
            # Should not raise; empty result is fine.
            result = await storage.get_history(metric, hours=1)
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_area_history_accepts_all_valid_metrics(self, storage):
        """Each known metric must be accepted without raising for area queries."""
        valid_metrics = ["power", "energy", "temperature", "humidity", "illuminance", "occupancy"]
        for metric in valid_metrics:
            result = await storage.get_area_history("Kitchen", metric, hours=1)
            assert isinstance(result, list)


class TestLocalMidnightBoundaries:
    """compute_area_daily_aggregates and compute_daily_aggregates must use local midnight as day boundaries."""

    @pytest.mark.asyncio
    async def test_reading_at_local_midnight_is_included(self, storage):
        """A reading at exactly local midnight must appear in that day's aggregate."""
        date_str = "2026-03-01"
        # Construct the naive local midnight and convert to epoch (local time)
        local_midnight_ts = datetime(2026, 3, 1, 0, 0, 0).timestamp()

        conn = sqlite3.connect(storage.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO area_sensor_history (timestamp, area_name, power) VALUES (?, ?, ?)",
            (local_midnight_ts, "Office", 500.0),
        )
        conn.commit()
        conn.close()

        await storage.compute_area_daily_aggregates(date=date_str, phase="active")

        res_conn = sqlite3.connect(storage.research_db_path)
        res_cursor = res_conn.cursor()
        res_cursor.execute(
            "SELECT avg_power_w FROM research_area_daily_stats WHERE date = ? AND area_name = ?",
            (date_str, "Office"),
        )
        row = res_cursor.fetchone()
        res_conn.close()

        assert row is not None, (
            "Area aggregate row missing; local-midnight reading was not included. "
            "compute_area_daily_aggregates must use local midnight as the day start."
        )
        assert row[0] == pytest.approx(500.0, abs=1.0)

    @pytest.mark.asyncio
    async def test_reading_one_second_before_local_midnight_excluded(self, storage):
        """A reading 1 second before local midnight must NOT appear in that day's aggregate."""
        date_str = "2026-03-02"
        local_midnight_ts = datetime(2026, 3, 2, 0, 0, 0).timestamp()
        just_before = local_midnight_ts - 1  # belongs to 2026-03-01

        conn = sqlite3.connect(storage.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO area_sensor_history (timestamp, area_name, power) VALUES (?, ?, ?)",
            (just_before, "Office", 999.0),
        )
        conn.commit()
        conn.close()

        await storage.compute_area_daily_aggregates(date=date_str, phase="active")

        res_conn = sqlite3.connect(storage.research_db_path)
        res_cursor = res_conn.cursor()
        res_cursor.execute(
            "SELECT avg_power_w FROM research_area_daily_stats WHERE date = ? AND area_name = ?",
            (date_str, "Office"),
        )
        row = res_cursor.fetchone()
        res_conn.close()

        assert row is None, (
            "A reading from before local midnight was incorrectly included in the next day. "
            "compute_area_daily_aggregates must use LOCAL midnight as the day boundary."
        )

    @pytest.mark.asyncio
    async def test_reading_at_local_midnight_included_in_daily_aggregates(self, storage):
        """compute_daily_aggregates also uses local midnight boundaries."""
        date_str = "2026-03-01"
        local_midnight_ts = datetime(2026, 3, 1, 0, 0, 0).timestamp()

        conn = sqlite3.connect(storage.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO sensor_history (timestamp, power, within_working_hours) VALUES (?, ?, ?)",
            (local_midnight_ts, 800.0, 1),
        )
        conn.commit()
        conn.close()

        await storage.compute_daily_aggregates(date=date_str, phase="active")

        res_conn = sqlite3.connect(storage.research_db_path)
        res_cursor = res_conn.cursor()
        res_cursor.execute(
            "SELECT avg_power_w FROM research_daily_aggregates WHERE date = ?",
            (date_str,),
        )
        row = res_cursor.fetchone()
        res_conn.close()

        assert row is not None, "Daily aggregate row missing for local-midnight reading."
        assert row[0] == pytest.approx(800.0, abs=1.0)

# ─────────────────────────────────────────────────────────────────────────────
# get_all_areas
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAllAreas:

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_data(self, storage):
        result = await storage.get_all_areas()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_unique_areas(self, storage):
        now = datetime.now()
        await storage.store_area_snapshot(now, "Kitchen", power=100.0)
        await storage.store_area_snapshot(now, "Kitchen", power=150.0)
        await storage.store_area_snapshot(now, "Bedroom", power=200.0)

        result = await storage.get_all_areas()
        assert set(result) == {"Kitchen", "Bedroom"}

    @pytest.mark.asyncio
    async def test_sorted_alphabetically(self, storage):
        now = datetime.now()
        for area in ["Bedroom", "Kitchen", "Living Room"]:
            await storage.store_area_snapshot(now, area, power=100.0)

        result = await storage.get_all_areas()
        assert result == sorted(result)


# ─────────────────────────────────────────────────────────────────────────────
# get_area_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAreaStats:

    @pytest.mark.asyncio
    async def test_calculates_avg_min_max(self, storage):
        now = datetime.now()
        for val in [100.0, 200.0, 300.0]:
            await storage.store_area_snapshot(
                now - timedelta(minutes=val), "Kitchen", power=val
            )

        stats = await storage.get_area_stats("Kitchen", "power", hours=24)
        assert stats["mean"] == pytest.approx(200.0)
        assert stats["min"] == pytest.approx(100.0)
        assert stats["max"] == pytest.approx(300.0)

    @pytest.mark.asyncio
    async def test_returns_none_for_no_data(self, storage):
        stats = await storage.get_area_stats("NonExistent", "power", hours=1)
        assert stats["mean"] == 0
        assert stats["min"] == 0
        assert stats["max"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# get_recent_values
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRecentValues:

    @pytest.mark.asyncio
    async def test_limits_to_count(self, storage):
        now = datetime.now()
        for i in range(10):
            await storage.store_sensor_snapshot(
                now - timedelta(minutes=i), power=float(i * 100)
            )

        result = await storage.get_recent_values("power", count=5)
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_returns_most_recent_first(self, storage):
        now = datetime.now()
        await storage.store_sensor_snapshot(now - timedelta(minutes=2), power=100.0)
        await storage.store_sensor_snapshot(now - timedelta(minutes=1), power=200.0)
        await storage.store_sensor_snapshot(now, power=300.0)

        result = await storage.get_recent_values("power", count=3)
        # Returns in chronological order (oldest first)
        assert result[0] == 100.0
        assert result[1] == 200.0
        assert result[2] == 300.0


# ─────────────────────────────────────────────────────────────────────────────
# Daily tasks
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyTasks:

    @pytest.mark.asyncio
    async def test_save_and_retrieve_tasks(self, storage):
        today = datetime.now().strftime("%Y-%m-%d")
        tasks = [
            {
                "task_id": "t1",
                "task_type": "power_reduction",
                "date": today,
                "title": "Reduce power",
                "description": "Test task 1"
            },
            {
                "task_id": "t2",
                "task_type": "temperature",
                "date": today,
                "title": "Adjust temperature",
                "description": "Test task 2"
            },
        ]
        await storage.save_daily_tasks(tasks)

        retrieved = await storage.get_today_tasks()
        assert len(retrieved) == 2
        assert retrieved[0]["task_id"] == "t1"

    @pytest.mark.asyncio
    async def test_peak_hour_persisted_and_retrieved(self, storage):
        """peak_hour saved with a peak_avoidance task must be returned unchanged."""
        today = datetime.now().strftime("%Y-%m-%d")
        tasks = [
            {
                "task_id": "peak_persist",
                "task_type": "peak_avoidance",
                "date": today,
                "title": "Avoid peak",
                "description": "Peak avoidance test",
                "peak_hour": 14,
            }
        ]
        await storage.save_daily_tasks(tasks)

        retrieved = await storage.get_today_tasks()
        assert len(retrieved) == 1
        assert retrieved[0]["peak_hour"] == 14

    @pytest.mark.asyncio
    async def test_non_peak_task_has_none_peak_hour(self, storage):
        """Non-peak tasks should round-trip with peak_hour=None."""
        today = datetime.now().strftime("%Y-%m-%d")
        tasks = [
            {
                "task_id": "no_peak",
                "task_type": "power_reduction",
                "date": today,
                "title": "Power task",
                "description": "No peak_hour field",
                # No peak_hour key at all
            }
        ]
        await storage.save_daily_tasks(tasks)

        retrieved = await storage.get_today_tasks()
        assert len(retrieved) == 1
        assert retrieved[0]["peak_hour"] is None

    @pytest.mark.asyncio
    async def test_created_at_returned_for_task(self, storage):
        """get_today_tasks must include created_at for each task (verification time anchor)."""
        today = datetime.now().strftime("%Y-%m-%d")
        tasks = [
            {
                "task_id": "anchor_test",
                "task_type": "power_reduction",
                "date": today,
                "title": "Power task",
                "description": "Anchor test",
            }
        ]
        await storage.save_daily_tasks(tasks)

        retrieved = await storage.get_today_tasks()
        assert len(retrieved) == 1
        assert "created_at" in retrieved[0]
        assert retrieved[0]["created_at"] is not None

    @pytest.mark.asyncio
    async def test_created_at_is_local_time(self, storage):
        """created_at stored by save_daily_tasks must be a local-time ISO string,
        not the UTC CURRENT_TIMESTAMP that SQLite would insert by default.
        The value must fall within a 2-second window around datetime.now()."""
        today = datetime.now().strftime("%Y-%m-%d")
        before = datetime.now()
        tasks = [
            {
                "task_id": "ts_local",
                "task_type": "power_reduction",
                "date": today,
                "title": "T",
                "description": "D",
            }
        ]
        await storage.save_daily_tasks(tasks)
        after = datetime.now()

        retrieved = await storage.get_today_tasks()
        assert len(retrieved) == 1

        created_at = datetime.fromisoformat(retrieved[0]["created_at"])
        assert before - timedelta(seconds=2) <= created_at <= after + timedelta(seconds=2), (
            f"created_at={created_at} is not in local-time window [{before}, {after}]"
        )

    @pytest.mark.asyncio
    async def test_get_today_tasks_filters_by_date(self, storage):
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        yesterday_tasks = [
            {
                "task_id": "old",
                "task_type": "power",
                "date": yesterday,
                "title": "Old task",
                "description": "Yesterday's task"
            }
        ]
        today_tasks = [
            {
                "task_id": "new",
                "task_type": "power",
                "date": today,
                "title": "New task",
                "description": "Today's task"
            }
        ]

        await storage.save_daily_tasks(yesterday_tasks)
        await storage.save_daily_tasks(today_tasks)

        retrieved = await storage.get_today_tasks()
        # Should only get today's tasks
        assert len(retrieved) == 1
        assert retrieved[0]["task_id"] == "new"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_tasks(self, storage):
        result = await storage.get_today_tasks()
        assert result == []

# ─────────────────────────────────────────────────────────────────────────────
# Data cleanup
# ─────────────────────────────────────────────────────────────────────────────

class TestDataCleanup:

    @pytest.mark.asyncio
    async def test_removes_old_sensor_data(self, storage):
        now = datetime.now()
        # Insert data older than 14 days
        await storage.store_sensor_snapshot(now - timedelta(days=15), power=100.0)
        # Insert recent data
        await storage.store_sensor_snapshot(now, power=200.0)

        await storage._cleanup_old_data()

        all_data = await storage.get_history("power", days=30)
        # Old data should be removed, only recent remains
        assert len(all_data) == 1
        assert all_data[0][1] == 200.0

    @pytest.mark.asyncio
    async def test_removes_old_area_data(self, storage):
        now = datetime.now()
        await storage.store_area_snapshot(now - timedelta(days=15), "Kitchen", power=100.0)
        await storage.store_area_snapshot(now, "Kitchen", power=200.0)

        await storage._cleanup_old_data()

        all_data = await storage.get_area_history("Kitchen", "power", days=30)
        assert len(all_data) == 1
        assert all_data[0][1] == 200.0


class TestRLEpisodeCleanup:

    @pytest.mark.asyncio
    async def test_removes_old_episodes(self, storage):
        """Test that RL episodes older than retention period are removed."""
        now = datetime.now()
        old_date = now - timedelta(days=121)  # Older than 120 days

        def insert_episode(date):
            conn = sqlite3.connect(storage.research_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO research_rl_episodes (timestamp, state_vector, action, reward)
                VALUES (?, ?, ?, ?)
            """, (date.timestamp(), "[]", 1, 0.5))
            conn.commit()
            conn.close()

        # Insert old and recent episodes
        insert_episode(old_date)
        insert_episode(now)

        await storage._cleanup_old_research_data()

        def count_episodes():
            conn = sqlite3.connect(storage.research_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM research_rl_episodes")
            count = cursor.fetchone()[0]
            conn.close()
            return count

        count = count_episodes()
        assert count == 1  # Only recent episode remains

    @pytest.mark.asyncio
    async def test_removes_old_nudge_log_rows(self, storage):
        """research_nudge_log rows older than 120 days must be purged."""
        now = datetime.now()
        old_ts = (now - timedelta(days=121)).timestamp()
        new_ts = now.timestamp()

        def insert(ts):
            conn = sqlite3.connect(storage.research_db_path)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO research_nudge_log (timestamp) VALUES (?)",
                (ts,),
            )
            conn.commit()
            conn.close()

        insert(old_ts)
        insert(new_ts)
        await storage._cleanup_old_research_data()

        conn = sqlite3.connect(storage.research_db_path)
        count = conn.execute("SELECT COUNT(*) FROM research_nudge_log").fetchone()[0]
        conn.close()
        assert count == 1, "Old nudge_log row must be deleted; recent one must remain."

    @pytest.mark.asyncio
    async def test_removes_old_blocked_notifications_rows(self, storage):
        """research_blocked_notifications rows older than 120 days must be purged."""
        now = datetime.now()
        old_ts = (now - timedelta(days=121)).timestamp()
        new_ts = now.timestamp()

        def insert(ts):
            conn = sqlite3.connect(storage.research_db_path)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO research_blocked_notifications (timestamp, block_reason) VALUES (?, ?)",
                (ts, "cooldown"),
            )
            conn.commit()
            conn.close()

        insert(old_ts)
        insert(new_ts)
        await storage._cleanup_old_research_data()

        conn = sqlite3.connect(storage.research_db_path)
        count = conn.execute("SELECT COUNT(*) FROM research_blocked_notifications").fetchone()[0]
        conn.close()
        assert count == 1, "Old blocked_notifications row must be deleted; recent one must remain."

    @pytest.mark.asyncio
    async def test_removes_old_task_interactions_rows(self, storage):
        """research_task_interactions rows older than 120 days (by generation_timestamp) must be purged."""
        now = datetime.now()
        old_ts = (now - timedelta(days=121)).timestamp()
        new_ts = now.timestamp()

        def insert(ts, task_id):
            conn = sqlite3.connect(storage.research_db_path)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO research_task_interactions (task_id, date, generation_timestamp) VALUES (?, ?, ?)",
                (task_id, now.strftime("%Y-%m-%d"), ts),
            )
            conn.commit()
            conn.close()

        insert(old_ts, "task-old-1")
        insert(new_ts, "task-new-1")
        await storage._cleanup_old_research_data()

        conn = sqlite3.connect(storage.research_db_path)
        count = conn.execute("SELECT COUNT(*) FROM research_task_interactions").fetchone()[0]
        conn.close()
        assert count == 1, "Old task_interactions row must be deleted; recent one must remain."

    @pytest.mark.asyncio
    async def test_removes_old_area_daily_stats_rows(self, storage):
        """research_area_daily_stats rows older than 120 days must be purged."""
        now = datetime.now()
        old_date = (now - timedelta(days=121)).strftime("%Y-%m-%d")
        new_date = now.strftime("%Y-%m-%d")

        def insert(d):
            conn = sqlite3.connect(storage.research_db_path)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO research_area_daily_stats (date, area_name) VALUES (?, ?)",
                (d, "kitchen"),
            )
            conn.commit()
            conn.close()

        insert(old_date)
        insert(new_date)
        await storage._cleanup_old_research_data()

        conn = sqlite3.connect(storage.research_db_path)
        count = conn.execute("SELECT COUNT(*) FROM research_area_daily_stats").fetchone()[0]
        conn.close()
        assert count == 1, "Old area_daily_stats row must be deleted; recent one must remain."

    @pytest.mark.asyncio
    async def test_phase_metadata_never_purged(self, storage):
        """research_phase_metadata must NEVER be touched by the cleanup job."""
        now = datetime.now()
        old_ts = (now - timedelta(days=200)).timestamp()
        new_ts = now.timestamp()

        def insert(ts):
            conn = sqlite3.connect(storage.research_db_path)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO research_phase_metadata (phase, start_timestamp) VALUES (?, ?)",
                ("baseline", ts),
            )
            conn.commit()
            conn.close()

        insert(old_ts)
        insert(new_ts)
        await storage._cleanup_old_research_data()

        conn = sqlite3.connect(storage.research_db_path)
        count = conn.execute("SELECT COUNT(*) FROM research_phase_metadata").fetchone()[0]
        conn.close()
        assert count == 2, "research_phase_metadata rows must never be deleted by cleanup."


# ─────────────────────────────────────────────────────────────────────────────
# get_active_phase_savings
# ─────────────────────────────────────────────────────────────────────────────

class TestActivePhraseSavings:
    """Tests for get_active_phase_savings (uses research_daily_aggregates)."""

    def _insert_daily_aggregate(self, storage, date: str, phase: str, avg_power_w: float):
        """Helper: insert a minimal research_daily_aggregates row."""
        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO research_daily_aggregates
                (date, phase, avg_power_w)
            VALUES (?, ?, ?)
        """, (date, phase, avg_power_w))
        conn.commit()
        conn.close()

    @pytest.mark.asyncio
    async def test_no_data_returns_zero(self, storage):
        result = await storage.get_active_phase_savings("2026-01-01", baseline_w=1000.0)
        assert result["total_savings_kwh"] == 0.0
        assert result["days_with_data"] == 0
        assert result["overall_avg_power_w"] == 0.0

    @pytest.mark.asyncio
    async def test_single_day_savings_calculated(self, storage):
        """One active day consuming 200 W below 1000 W baseline = 4.8 kWh saved."""
        self._insert_daily_aggregate(storage, "2026-03-01", "active", 800.0)
        result = await storage.get_active_phase_savings("2026-03-01", baseline_w=1000.0)
        # saving_w = 200 W; over 24h = 4.8 kWh
        assert result["days_with_data"] == 1
        assert result["total_savings_kwh"] == pytest.approx(4.8, abs=0.01)
        assert result["overall_avg_power_w"] == pytest.approx(800.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_multiple_days_accumulated(self, storage):
        """3 days at 500 W below 1000 W baseline => 3 * 12 = 36 kWh."""
        for day in ("2026-03-01", "2026-03-02", "2026-03-03"):
            self._insert_daily_aggregate(storage, day, "active", 500.0)
        result = await storage.get_active_phase_savings("2026-03-01", baseline_w=1000.0)
        assert result["days_with_data"] == 3
        assert result["total_savings_kwh"] == pytest.approx(36.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_baseline_rows_excluded(self, storage):
        """baseline-phase rows must not contribute to savings calculations."""
        self._insert_daily_aggregate(storage, "2026-02-01", "baseline", 200.0)
        self._insert_daily_aggregate(storage, "2026-03-01", "active", 800.0)
        result = await storage.get_active_phase_savings("2026-02-01", baseline_w=1000.0)
        # Only the active row contributes
        assert result["days_with_data"] == 1
        assert result["total_savings_kwh"] == pytest.approx(4.8, abs=0.01)

    @pytest.mark.asyncio
    async def test_rows_before_active_since_excluded(self, storage):
        """Active rows older than active_since_date must be ignored."""
        self._insert_daily_aggregate(storage, "2026-02-28", "active", 700.0)
        self._insert_daily_aggregate(storage, "2026-03-01", "active", 800.0)
        # Only rows from 2026-03-01 onward should be included
        result = await storage.get_active_phase_savings("2026-03-01", baseline_w=1000.0)
        assert result["days_with_data"] == 1

    @pytest.mark.asyncio
    async def test_negative_saving_still_summed(self, storage):
        """Days where consumption exceeds baseline produce negative savings."""
        self._insert_daily_aggregate(storage, "2026-03-01", "active", 1200.0)
        result = await storage.get_active_phase_savings("2026-03-01", baseline_w=1000.0)
        # saving_w = -200 W; over 24h = -4.8 kWh
        assert result["total_savings_kwh"] == pytest.approx(-4.8, abs=0.01)

    @pytest.mark.asyncio
    async def test_null_avg_power_rows_skipped(self, storage):
        """Rows with NULL avg_power_w must be skipped without raising."""
        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO research_daily_aggregates (date, phase, avg_power_w) VALUES (?, ?, ?)",
            ("2026-03-01", "active", None)
        )
        conn.commit()
        conn.close()
        result = await storage.get_active_phase_savings("2026-03-01", baseline_w=1000.0)
        assert result["days_with_data"] == 0
        assert result["total_savings_kwh"] == 0.0

    def _insert_daily_aggregate_with_wh(
        self, storage, date: str, phase: str,
        avg_power_w: float,
        avg_power_working_hours: Optional[float] = None,
        avg_power_off_hours: Optional[float] = None,
    ):
        """Helper: insert a row with all three power columns."""
        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO research_daily_aggregates
                (date, phase, avg_power_w, avg_power_working_hours, avg_power_off_hours)
            VALUES (?, ?, ?, ?, ?)
        """, (date, phase, avg_power_w, avg_power_working_hours, avg_power_off_hours))
        conn.commit()
        conn.close()

    @pytest.mark.asyncio
    async def test_savings_uses_working_hours_power_when_available(self, storage):
        """get_active_phase_savings must use avg_power_working_hours as the
        representative power (not the 24-hour avg_power_w) when the column is
        non-NULL, avoiding false savings caused by including off-hours low load."""

        storage.config_data = {"environment_mode": "office"}

        # 24-h average is artificially low (office empty at night).
        # Working-hours average reflects real occupied load.
        self._insert_daily_aggregate_with_wh(
            storage, "2026-03-01", "active",
            avg_power_w=400.0,              # 24-h avg (distorted by night)
            avg_power_working_hours=900.0,  # truth: near-baseline during working hours
            avg_power_off_hours=50.0,
        )
        # baseline_w=1000 W; working-hours avg=900 W -> saving = 100 W × 10 h = 1.0 kWh
        result = await storage.get_active_phase_savings("2026-03-01", baseline_w=1000.0)
        assert result["days_with_data"] == 1
        # The representative power must be 900 W.
        assert result["overall_avg_power_w"] == pytest.approx(900.0, abs=0.1)
        # Calculate based on working hours duration (10h)
        assert result["total_savings_kwh"] == pytest.approx(1.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_savings_falls_back_to_avg_power_w_when_wh_null(self, storage):
        """When avg_power_working_hours IS NULL the function must fall back to
        avg_power_w so older rows (pre-migration) are still usable."""
        self._insert_daily_aggregate_with_wh(
            storage, "2026-03-01", "active",
            avg_power_w=800.0,
            avg_power_working_hours=None,  # not yet populated
            avg_power_off_hours=None,
        )
        result = await storage.get_active_phase_savings("2026-03-01", baseline_w=1000.0)
        assert result["days_with_data"] == 1
        assert result["overall_avg_power_w"] == pytest.approx(800.0, abs=0.1)
        assert result["total_savings_kwh"] == pytest.approx(4.8, abs=0.05)

    @pytest.mark.asyncio
    async def test_compute_daily_aggregates_populates_wh_columns(self, storage):
        """compute_daily_aggregates must fill avg_power_working_hours and
        avg_power_off_hours from sensor_history.within_working_hours."""
        from datetime import timezone as _tz

        # Insert sensor rows for 2026-03-01 UTC.
        date_str = "2026-03-01"
        utc_midnight = datetime(2026, 3, 1, 0, 0, 0, tzinfo=_tz.utc).timestamp()

        def insert_sensor_row(offset_secs: float, power: float, wh_flag: int):
            conn = sqlite3.connect(storage.db_path)
            c = conn.cursor()
            c.execute(
                "INSERT INTO sensor_history (timestamp, power, within_working_hours) VALUES (?, ?, ?)",
                (utc_midnight + offset_secs, power, wh_flag),
            )
            conn.commit()
            conn.close()

        # 3 working-hours readings: 1000 W, 1200 W, 800 W -> avg = 1000 W
        insert_sensor_row(3600, 1000.0, 1)
        insert_sensor_row(7200, 1200.0, 1)
        insert_sensor_row(10800, 800.0, 1)
        # 2 off-hours readings: 100 W, 150 W -> avg = 125 W
        insert_sensor_row(50400, 100.0, 0)
        insert_sensor_row(54000, 150.0, 0)

        await storage.compute_daily_aggregates(date=date_str)

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT avg_power_working_hours, avg_power_off_hours "
            "FROM research_daily_aggregates WHERE date = ?",
            (date_str,),
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None, "Daily aggregate row not created"
        avg_wh, avg_off = row
        assert avg_wh == pytest.approx(1000.0, abs=1.0), (
            f"Expected avg_power_working_hours≈1000 W, got {avg_wh}"
        )
        assert avg_off == pytest.approx(125.0, abs=1.0), (
            f"Expected avg_power_off_hours≈125 W, got {avg_off}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# get_history / get_area_history with no time filter
# ─────────────────────────────────────────────────────────────────────────────

class TestGetHistoryNoTimeFilter:
    """get_history must return all rows when neither hours nor days is given."""

    @pytest.mark.asyncio
    async def test_returns_all_rows_when_no_filter(self, storage):
        now = datetime.now()
        for offset in [60, 30, 0]:  # minutes ago
            await storage.store_sensor_snapshot(now - timedelta(minutes=offset), power=float(offset * 10))

        result = await storage.get_history("power")
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_area_returns_all_rows_when_no_filter(self, storage):
        now = datetime.now()
        for offset in [60, 30, 0]:
            await storage.store_area_snapshot(now - timedelta(minutes=offset), "Office", power=float(offset))

        result = await storage.get_area_history("Office", "power")
        assert len(result) == 3


# ─────────────────────────────────────────────────────────────────────────────
# get_recent_values: invalid metric
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRecentValuesInvalidMetric:

    @pytest.mark.asyncio
    async def test_raises_for_invalid_metric(self, storage):
        with pytest.raises(ValueError, match="Invalid metric"):
            await storage.get_recent_values("not_a_column", count=10)


# ─────────────────────────────────────────────────────────────────────────────
# get_tasks_for_date
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTasksForDate:

    @pytest.mark.asyncio
    async def test_get_tasks_by_date_string(self, storage):
        target_date = "2026-01-15"
        tasks = [{"task_id": "t_date1", "date": target_date, "task_type": "power_reduction",
                  "title": "Task A", "description": "Desc A"}]
        await storage.save_daily_tasks(tasks)

        result = await storage.get_tasks_for_date(target_date)
        assert len(result) == 1
        assert result[0]["task_id"] == "t_date1"

    @pytest.mark.asyncio
    async def test_get_tasks_by_date_object(self, storage):
        from datetime import date as date_cls
        target = date_cls(2026, 1, 20)
        tasks = [{"task_id": "t_date2", "date": "2026-01-20", "task_type": "temperature",
                  "title": "Task B", "description": "Desc B"}]
        await storage.save_daily_tasks(tasks)

        result = await storage.get_tasks_for_date(target)
        assert len(result) == 1
        assert result[0]["task_id"] == "t_date2"

    @pytest.mark.asyncio
    async def test_returns_empty_for_date_with_no_tasks(self, storage):
        result = await storage.get_tasks_for_date("2020-01-01")
        assert result == []

    @pytest.mark.asyncio
    async def test_does_not_return_other_dates(self, storage):
        await storage.save_daily_tasks([
            {"task_id": "t_jan", "date": "2026-01-10", "task_type": "power_reduction",
             "title": "Jan", "description": "Jan task"},
            {"task_id": "t_feb", "date": "2026-02-10", "task_type": "power_reduction",
             "title": "Feb", "description": "Feb task"},
        ])
        result = await storage.get_tasks_for_date("2026-01-10")
        assert len(result) == 1
        assert result[0]["task_id"] == "t_jan"


class TestStorageErrorAndStateBranches:

    @pytest.mark.asyncio
    async def test_store_sensor_snapshot_handles_insert_exception(self, storage):
        with patch.object(storage_mod.sqlite3, "connect", side_effect=RuntimeError("db down")):
            await storage.store_sensor_snapshot(timestamp=datetime.now(), power=123.0)

    @pytest.mark.asyncio
    async def test_store_sensor_snapshot_rolls_back_when_insert_fails(self, storage):
        class BadCursor:
            def execute(self, *args, **kwargs):
                raise RuntimeError("insert fail")

        class BadConn:
            def __init__(self):
                self.rolled_back = False
                self.closed = False

            def cursor(self):
                return BadCursor()

            def commit(self):
                return None

            def rollback(self):
                self.rolled_back = True

            def close(self):
                self.closed = True

        conn = BadConn()
        with patch.object(storage_mod.sqlite3, "connect", return_value=conn):
            await storage.store_sensor_snapshot(timestamp=datetime.now(), power=1.0)
        assert conn.rolled_back is True
        assert conn.closed is True

    @pytest.mark.asyncio
    async def test_store_area_snapshot_handles_insert_exception(self, storage):
        with patch.object(storage_mod.sqlite3, "connect", side_effect=RuntimeError("db down")):
            await storage.store_area_snapshot(timestamp=datetime.now(), area_name="Office", power=123.0)

    @pytest.mark.asyncio
    async def test_store_area_snapshot_rolls_back_when_insert_fails(self, storage):
        class BadCursor:
            def execute(self, *args, **kwargs):
                raise RuntimeError("insert fail")

        class BadConn:
            def __init__(self):
                self.rolled_back = False
                self.closed = False

            def cursor(self):
                return BadCursor()

            def commit(self):
                return None

            def rollback(self):
                self.rolled_back = True

            def close(self):
                self.closed = True

        conn = BadConn()
        with patch.object(storage_mod.sqlite3, "connect", return_value=conn):
            await storage.store_area_snapshot(timestamp=datetime.now(), area_name="Office", power=1.0)
        assert conn.rolled_back is True
        assert conn.closed is True

    @pytest.mark.asyncio
    async def test_save_daily_tasks_returns_false_on_individual_task_error(self, storage):
        bad_task = {
            "task_id": "bad_task",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "task_type": "power_reduction",
            "title": "Bad",
            # Missing description triggers KeyError in save loop
        }
        result = await storage.save_daily_tasks([bad_task])
        assert result is False

    @pytest.mark.asyncio
    async def test_save_state_cleans_temp_file_when_write_fails(self, storage):
        temp_path = storage.state_file.with_suffix('.tmp')
        temp_path.write_text("partial")

        with patch.object(storage_mod.json, "dump", side_effect=RuntimeError("write fail")):
            with pytest.raises(RuntimeError):
                await storage.save_state({"k": "v"})

        assert not temp_path.exists()

    @pytest.mark.asyncio
    async def test_load_state_returns_empty_on_invalid_json(self, storage):
        storage.state_file.write_text("{bad json")
        result = await storage.load_state()
        assert result == {}

    @pytest.mark.asyncio
    async def test_has_phase_metadata_true_and_false(self, storage):
        assert await storage.has_phase_metadata() is False

        conn = sqlite3.connect(storage.research_db_path)
        conn.execute(
            "INSERT INTO research_phase_metadata (phase, start_timestamp) VALUES (?, ?)",
            ("baseline", datetime.now().timestamp()),
        )
        conn.commit()
        conn.close()

        assert await storage.has_phase_metadata() is True

    @pytest.mark.asyncio
    async def test_compute_daily_aggregates_default_date_branch(self, storage):
        await storage.compute_daily_aggregates(date=None, phase="active")

    @pytest.mark.asyncio
    async def test_compute_daily_aggregates_weather_sensor_parse_and_exception_paths(self, storage):
        storage.config_data = {
            "outdoor_temp_sensor": "sensor.outdoor_bad",
            "weather_entity": "sensor.raises",
        }

        now_ts = datetime.now().timestamp()
        conn = sqlite3.connect(storage.db_path)
        conn.execute(
            "INSERT INTO sensor_history (timestamp, power, within_working_hours) VALUES (?, ?, ?)",
            (now_ts, 500.0, 1),
        )
        conn.commit()
        conn.close()

        bad_temp_state = MagicMock()
        bad_temp_state.state = "not-a-number"
        bad_temp_state.attributes = {}

        def get_state(entity_id):
            if entity_id == "sensor.raises":
                raise RuntimeError("boom")
            if entity_id == "sensor.outdoor_bad":
                return bad_temp_state
            return None

        storage.hass.states.get = MagicMock(side_effect=get_state)
        await storage.compute_daily_aggregates(date=None, phase="active")

    @pytest.mark.asyncio
    async def test_compute_daily_aggregates_nudge_stats_none_fallback(self, storage):
        date_str = "2026-03-20"
        ts = datetime(2026, 3, 20, 12, 0, 0).timestamp()

        conn = sqlite3.connect(storage.db_path)
        conn.execute(
            "INSERT INTO sensor_history (timestamp, power, within_working_hours) VALUES (?, ?, ?)",
            (ts, 600.0, 1),
        )
        conn.commit()
        conn.close()

        storage.hass.states.get = MagicMock(return_value=None)

        real_connect = storage_mod.sqlite3.connect

        class ConnProxy:
            def __init__(self, conn):
                self._conn = conn

            def cursor(self):
                real_cursor = self._conn.cursor()

                class CursorProxy:
                    def __init__(self, cur):
                        self._cur = cur
                        self._last_sql = ""

                    def execute(self, sql, params=()):
                        self._last_sql = sql
                        return self._cur.execute(sql, params)

                    def fetchone(self):
                        if "FROM research_nudge_log" in self._last_sql:
                            return (None, None, None, None)
                        return self._cur.fetchone()

                    def fetchall(self):
                        return self._cur.fetchall()

                return CursorProxy(real_cursor)

            def commit(self):
                return self._conn.commit()

            def close(self):
                return self._conn.close()

            def rollback(self):
                return self._conn.rollback()

        def connect_proxy(path):
            return ConnProxy(real_connect(path))

        with patch.object(storage_mod.sqlite3, "connect", side_effect=connect_proxy):
            await storage.compute_daily_aggregates(date=date_str, phase="active")

        rconn = sqlite3.connect(storage.research_db_path)
        row = rconn.execute(
            "SELECT nudges_sent, nudges_accepted, nudges_dismissed, nudges_ignored FROM research_daily_aggregates WHERE date = ?",
            (date_str,),
        ).fetchone()
        rconn.close()
        assert row is not None

    @pytest.mark.asyncio
    async def test_compute_area_daily_aggregates_skips_area_when_total_readings_zero(self, storage):
        date_str = "2026-03-22"
        ts = datetime(2026, 3, 22, 9, 0, 0).timestamp()

        conn = sqlite3.connect(storage.db_path)
        conn.execute(
            "INSERT INTO area_sensor_history (timestamp, area_name, power, occupancy) VALUES (?, ?, ?, ?)",
            (ts, "Office", 200.0, 1),
        )
        conn.commit()
        conn.close()

        real_connect = storage_mod.sqlite3.connect

        class ConnProxy:
            def __init__(self, conn):
                self._conn = conn

            def cursor(self):
                real_cursor = self._conn.cursor()

                class CursorProxy:
                    def __init__(self, cur):
                        self._cur = cur
                        self._last_sql = ""

                    def execute(self, sql, params=()):
                        self._last_sql = sql
                        return self._cur.execute(sql, params)

                    def fetchone(self):
                        if "AVG(power) as avg_power" in self._last_sql:
                            return (0, 0, 0, 0, 0, 0, 0, 0)
                        return self._cur.fetchone()

                    def fetchall(self):
                        return self._cur.fetchall()

                return CursorProxy(real_cursor)

            def commit(self):
                return self._conn.commit()

            def close(self):
                return self._conn.close()

            def rollback(self):
                return self._conn.rollback()

        def connect_proxy(path):
            return ConnProxy(real_connect(path))

        with patch.object(storage_mod.sqlite3, "connect", side_effect=connect_proxy):
            await storage.compute_area_daily_aggregates(date=date_str, phase="active")

    @pytest.mark.asyncio
    async def test_compute_area_daily_aggregates_no_areas_early_return(self, storage):
        # No area rows inserted for today, should hit early return branch.
        await storage.compute_area_daily_aggregates(date=None, phase="active")

    @pytest.mark.asyncio
    async def test_compute_area_daily_aggregates_phase_fallback_to_baseline(self, storage):
        date_str = "2026-03-15"
        ts = datetime(2026, 3, 15, 10, 0, 0).timestamp()

        conn = sqlite3.connect(storage.db_path)
        conn.execute(
            "INSERT INTO area_sensor_history (timestamp, area_name, power, occupancy) VALUES (?, ?, ?, ?)",
            (ts, "Office", 300.0, 1),
        )
        conn.commit()
        conn.close()

        await storage.compute_area_daily_aggregates(date=date_str, phase=None)

        rconn = sqlite3.connect(storage.research_db_path)
        row = rconn.execute(
            "SELECT phase FROM research_area_daily_stats WHERE date = ? AND area_name = ?",
            (date_str, "Office"),
        ).fetchone()
        rconn.close()

        assert row is not None
        assert row[0] == "baseline"

    @pytest.mark.asyncio
    async def test_close_closes_existing_connection(self, storage):
        class DummyConn:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        dummy = DummyConn()
        storage._conn = dummy

        await storage.close()
        assert dummy.closed is True


# ─────────────────────────────────────────────────────────────────────────────
# get_total_completed_tasks_count (30-day rolling window)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTotalCompletedTasksCount:

    async def _insert_verified_task(self, storage, date_str: str, verified: int = 1):
        conn = sqlite3.connect(storage.db_path)
        conn.execute(
            "INSERT INTO daily_tasks (task_id, date, task_type, title, description, verified) "
            "VALUES (?, ?, 'power', 'T', 'D', ?)",
            (f"vtask_{date_str}_{verified}", date_str, verified)
        )
        conn.commit()
        conn.close()

    @pytest.mark.asyncio
    async def test_counts_verified_tasks_in_window(self, storage):
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        await self._insert_verified_task(storage, today)
        await self._insert_verified_task(storage, yesterday)

        count = await storage.get_total_completed_tasks_count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_excludes_tasks_outside_30_day_window(self, storage):
        old_date = (datetime.now() - timedelta(days=31)).strftime("%Y-%m-%d")
        await self._insert_verified_task(storage, old_date)

        count = await storage.get_total_completed_tasks_count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_excludes_unverified_tasks(self, storage):
        today = datetime.now().strftime("%Y-%m-%d")
        await self._insert_verified_task(storage, today, verified=0)

        count = await storage.get_total_completed_tasks_count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_tasks(self, storage):
        count = await storage.get_total_completed_tasks_count()
        assert count == 0


# ─────────────────────────────────────────────────────────────────────────────
# get_total_completed_tasks_count_alltime (research DB)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTotalCompletedTasksCountAlltime:

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_research_tasks(self, storage):
        count = await storage.get_total_completed_tasks_count_alltime()
        assert count == 0

    @pytest.mark.asyncio
    async def test_counts_completed_research_tasks(self, storage):
        conn = sqlite3.connect(storage.research_db_path)
        conn.execute(
            "INSERT INTO research_task_interactions "
            "(task_id, date, completed) VALUES (?, ?, 1)",
            ("rt1", "2026-01-01")
        )
        conn.execute(
            "INSERT INTO research_task_interactions "
            "(task_id, date, completed) VALUES (?, ?, 1)",
            ("rt2", "2026-01-02")
        )
        conn.execute(
            "INSERT INTO research_task_interactions "
            "(task_id, date, completed) VALUES (?, ?, 0)",
            ("rt3", "2026-01-03")
        )
        conn.commit()
        conn.close()

        count = await storage.get_total_completed_tasks_count_alltime()
        assert count == 2


# ─────────────────────────────────────────────────────────────────────────────
# mark_task_verified
# ─────────────────────────────────────────────────────────────────────────────

class TestMarkTaskVerified:

    @pytest.mark.asyncio
    async def test_marks_task_as_verified(self, storage):
        today = datetime.now().strftime("%Y-%m-%d")
        await storage.save_daily_tasks([{
            "task_id": "verif_1", "date": today, "task_type": "power_reduction",
            "title": "Task", "description": "Desc"
        }])

        success = await storage.mark_task_verified("verif_1", verified=True)
        assert success is True

        tasks = await storage.get_today_tasks()
        assert tasks[0]["verified"] is True
        assert tasks[0]["completed"] is True

    @pytest.mark.asyncio
    async def test_unverifies_task(self, storage):
        today = datetime.now().strftime("%Y-%m-%d")
        await storage.save_daily_tasks([{
            "task_id": "verif_2", "date": today, "task_type": "power_reduction",
            "title": "Task", "description": "Desc"
        }])

        await storage.mark_task_verified("verif_2", verified=True)
        success = await storage.mark_task_verified("verif_2", verified=False)
        assert success is True

        tasks = await storage.get_today_tasks()
        assert tasks[0]["verified"] is False

    @pytest.mark.asyncio
    async def test_returns_false_for_non_existent_task(self, storage):
        success = await storage.mark_task_verified("does_not_exist", verified=True)
        assert success is False


# ─────────────────────────────────────────────────────────────────────────────
# save_task_feedback
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveTaskFeedback:

    @pytest.mark.asyncio
    async def test_saves_feedback_to_task(self, storage):
        today = datetime.now().strftime("%Y-%m-%d")
        await storage.save_daily_tasks([{
            "task_id": "fb_1", "date": today, "task_type": "power_reduction",
            "title": "Task", "description": "Desc"
        }])

        success = await storage.save_task_feedback("fb_1", "too_easy")
        assert success is True

        tasks = await storage.get_today_tasks()
        assert tasks[0]["user_feedback"] == "too_easy"

    @pytest.mark.asyncio
    async def test_saves_feedback_to_difficulty_history(self, storage):
        today = datetime.now().strftime("%Y-%m-%d")
        await storage.save_daily_tasks([{
            "task_id": "fb_2", "date": today, "task_type": "temperature",
            "title": "Task", "description": "Desc", "difficulty_level": 3
        }])

        await storage.save_task_feedback("fb_2", "just_right")

        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT task_type, feedback FROM task_difficulty_history WHERE task_type = 'temperature'")
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][1] == "just_right"

    @pytest.mark.asyncio
    async def test_returns_false_for_non_existent_task(self, storage):
        success = await storage.save_task_feedback("no_such_task", "too_hard")
        assert success is False


# ─────────────────────────────────────────────────────────────────────────────
# delete_today_tasks
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteTodayTasks:

    @pytest.mark.asyncio
    async def test_deletes_todays_tasks(self, storage):
        today = datetime.now().strftime("%Y-%m-%d")
        await storage.save_daily_tasks([{
            "task_id": "del_1", "date": today, "task_type": "power_reduction",
            "title": "Task", "description": "Desc"
        }])

        result = await storage.delete_today_tasks()
        assert result is True

        tasks = await storage.get_today_tasks()
        assert tasks == []

    @pytest.mark.asyncio
    async def test_does_not_delete_other_days(self, storage):
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        await storage.save_daily_tasks([{
            "task_id": "old_del", "date": yesterday, "task_type": "power_reduction",
            "title": "Old", "description": "Old"
        }])

        await storage.delete_today_tasks()

        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM daily_tasks WHERE date = ?", (yesterday,))
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1


# ─────────────────────────────────────────────────────────────────────────────
# get_task_difficulty_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTaskDifficultyStats:

    async def _seed_history(self, storage, task_type: str, feedbacks: list):
        today = datetime.now().strftime("%Y-%m-%d")
        conn = sqlite3.connect(storage.db_path)
        for i, fb in enumerate(feedbacks):
            conn.execute(
                "INSERT INTO task_difficulty_history (task_type, difficulty_level, feedback, date) "
                "VALUES (?, 3, ?, ?)",
                (task_type, fb, today)
            )
        conn.commit()
        conn.close()

    @pytest.mark.asyncio
    async def test_returns_default_stats_when_no_history(self, storage):
        stats = await storage.get_task_difficulty_stats("unknown_type")
        assert stats["too_easy_count"] == 0
        assert stats["too_hard_count"] == 0
        assert stats["suggested_adjustment"] == 0

    @pytest.mark.asyncio
    async def test_counts_feedback_correctly(self, storage):
        await self._seed_history(storage, "power_reduction",
                                 ["too_easy", "too_easy", "too_easy", "just_right"])

        stats = await storage.get_task_difficulty_stats("power_reduction")
        assert stats["too_easy_count"] == 3
        assert stats["just_right_count"] == 1
        assert stats["suggested_adjustment"] == 1  # Increase difficulty

    @pytest.mark.asyncio
    async def test_suggests_decrease_when_mostly_too_hard(self, storage):
        await self._seed_history(storage, "temperature",
                                 ["too_hard", "too_hard", "too_hard", "just_right"])

        stats = await storage.get_task_difficulty_stats("temperature")
        assert stats["suggested_adjustment"] == -1  # Decrease difficulty

    @pytest.mark.asyncio
    async def test_no_adjustment_when_balanced(self, storage):
        await self._seed_history(storage, "daylight",
                                 ["too_easy", "too_hard", "just_right"])

        stats = await storage.get_task_difficulty_stats("daylight")
        assert stats["suggested_adjustment"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# save_state / load_state
# ─────────────────────────────────────────────────────────────────────────────

class TestStatePersistence:

    @pytest.mark.asyncio
    async def test_save_and_load_state_roundtrip(self, storage):
        state = {"phase": "active", "baseline": 500.0, "episode": 42}
        await storage.save_state(state)

        loaded = await storage.load_state()
        assert loaded["phase"] == "active"
        assert loaded["baseline"] == 500.0
        assert loaded["episode"] == 42

    @pytest.mark.asyncio
    async def test_load_state_returns_empty_when_no_file(self, storage):
        if storage.state_file.exists():
            storage.state_file.unlink()

        loaded = await storage.load_state()
        assert loaded == {}

    @pytest.mark.asyncio
    async def test_save_state_overwrites_existing(self, storage):
        await storage.save_state({"phase": "baseline"})
        await storage.save_state({"phase": "active", "new_key": True})

        loaded = await storage.load_state()
        assert loaded["phase"] == "active"
        assert loaded["new_key"] is True
        assert "baseline" not in loaded or loaded.get("phase") == "active"

    @pytest.mark.asyncio
    async def test_save_state_serialises_datetime(self, storage):
        dt = datetime(2026, 3, 1, 12, 0, 0)
        await storage.save_state({"last_update": dt})

        loaded = await storage.load_state()
        # Datetime becomes ISO string
        assert "2026-03-01" in loaded["last_update"]


# ─────────────────────────────────────────────────────────────────────────────
# Research data: log_* and record_phase_change
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchData:

    @pytest.mark.asyncio
    async def test_record_phase_change_inserts_row(self, storage):
        await storage.record_phase_change("active", baseline_consumption=800.0)

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT phase, baseline_consumption_W FROM research_phase_metadata")
        rows = cursor.fetchall()
        conn.close()

        phases = [r[0] for r in rows]
        assert "active" in phases
        baseline_row = next(r for r in rows if r[0] == "active")
        assert baseline_row[1] == pytest.approx(800.0)

    @pytest.mark.asyncio
    async def test_record_phase_change_closes_previous_phase(self, storage):
        await storage.record_phase_change("baseline")
        await storage.record_phase_change("active")

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT end_timestamp FROM research_phase_metadata WHERE phase = 'baseline'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] is not None  # end_timestamp set

    @pytest.mark.asyncio
    async def test_log_rl_decision_inserts_row(self, storage):
        episode = {
            "episode": 1, "phase": "active", "state_vector": [0.1, 0.2],
            "action": 1, "action_name": "specific", "reward": 0.5,
            "q_values": {0: 0.0, 1: 0.5}, "epsilon": 0.1,
            "power": 600.0, "anomaly_index": 0.3,
        }
        await storage.log_rl_decision(episode)

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT action, reward FROM research_rl_episodes")
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == 1
        assert rows[0][1] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_log_nudge_sent_inserts_row(self, storage):
        nudge = {
            "notification_id": "nid_001", "phase": "active",
            "action_type": "specific", "title": "Save energy",
            "message": "Turn off lights", "current_power": 700.0,
        }
        await storage.log_nudge_sent(nudge)

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT notification_id, action_type FROM research_nudge_log WHERE notification_id = 'nid_001'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[1] == "specific"

    @pytest.mark.asyncio
    async def test_log_nudge_response_updates_row(self, storage):
        nudge = {"notification_id": "nid_resp", "phase": "active",
                 "action_type": "anomaly", "title": "T", "message": "M"}
        await storage.log_nudge_sent(nudge)
        await storage.log_nudge_response("nid_resp", accepted=True)

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT responded, accepted FROM research_nudge_log WHERE notification_id = 'nid_resp'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row[0] == 1  # responded
        assert row[1] == 1  # accepted

    @pytest.mark.asyncio
    async def test_log_nudge_response_rejected(self, storage):
        nudge = {"notification_id": "nid_rej", "phase": "active",
                 "action_type": "anomaly", "title": "T", "message": "M"}
        await storage.log_nudge_sent(nudge)
        await storage.log_nudge_response("nid_rej", accepted=False)

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT accepted FROM research_nudge_log WHERE notification_id = 'nid_rej'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row[0] == 0

    @pytest.mark.asyncio
    async def test_log_blocked_notification_inserts_row(self, storage):
        block = {
            "phase": "active", "block_reason": "cooldown",
            "opportunity_score": 0.4, "current_power": 300.0,
            "fatigue_index": 0.8, "notification_count_today": 5,
        }
        await storage.log_blocked_notification(block)

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT block_reason, notification_count_today FROM research_blocked_notifications"
        )
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "cooldown"
        assert rows[0][1] == 5

    @pytest.mark.asyncio
    async def test_log_task_generation_inserts_row(self, storage):
        task = {
            "task_id": "gen_t1", "date": "2026-03-01", "phase": "active",
            "task_type": "power_reduction", "difficulty_level": 2,
            "target_value": 0.5, "baseline_value": 1.0,
        }
        await storage.log_task_generation(task)

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT task_type, difficulty_level FROM research_task_interactions WHERE task_id = 'gen_t1'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "power_reduction"
        assert row[1] == 2

    @pytest.mark.asyncio
    async def test_log_task_generation_replace_on_duplicate(self, storage):
        """Re-generating logs must overwrite the previous row (INSERT OR REPLACE)."""
        task = {"task_id": "gen_dup", "date": "2026-03-01", "phase": "active",
                "task_type": "temperature", "difficulty_level": 1}
        await storage.log_task_generation(task)
        await storage.log_task_generation({**task, "difficulty_level": 5})

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT difficulty_level FROM research_task_interactions WHERE task_id = 'gen_dup'"
        )
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1  # only one row (replaced)
        assert rows[0][0] == 5

    @pytest.mark.asyncio
    async def test_log_task_completion_updates_row(self, storage):
        task = {"task_id": "comp_t1", "date": "2026-03-01", "phase": "active",
                "task_type": "power_reduction", "difficulty_level": 2}
        await storage.log_task_generation(task)
        await storage.log_task_completion("comp_t1", completion_value=0.8)

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT completed, completion_value FROM research_task_interactions WHERE task_id = 'comp_t1'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row[0] == 1
        assert row[1] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_log_task_feedback_updates_row(self, storage):
        task = {"task_id": "fb_t1", "date": "2026-03-01", "phase": "active",
                "task_type": "power_reduction", "difficulty_level": 2}
        await storage.log_task_generation(task)
        await storage.log_task_feedback("fb_t1", "too_easy")

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_feedback FROM research_task_interactions WHERE task_id = 'fb_t1'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "too_easy"

    @pytest.mark.asyncio
    async def test_log_weekly_challenge_inserts_row(self, storage):
        challenge = {
            "week_start_date": "2026-02-23", "week_end_date": "2026-03-01",
            "phase": "active", "target_percentage": 10.0,
            "baseline_W": 1000.0, "actual_W": 900.0,
            "savings_W": 100.0, "savings_percentage": 10.0,
            "achieved": True,
        }
        await storage.log_weekly_challenge(challenge)

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT achieved, savings_W FROM research_weekly_challenges WHERE week_start_date = '2026-02-23'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == 1
        assert row[1] == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_log_rl_decision_with_accepted_flag(self, storage):
        """accepted=True must be stored as 1, accepted=False as 0."""
        await storage.log_rl_decision({
            "episode": 5, "action": 2, "reward": 1.0, "accepted": True, "gamma_used": 0.95
        })
        await storage.log_rl_decision({
            "episode": 6, "action": 0, "reward": 0.0, "accepted": False, "gamma_used": 0.0
        })

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT episode_number, accepted, gamma_used FROM research_rl_episodes ORDER BY episode_number")
        rows = cursor.fetchall()
        conn.close()

        assert rows[0][1] == 1   # accepted=True -> 1
        assert rows[0][2] == pytest.approx(0.95)
        assert rows[1][1] == 0   # accepted=False -> 0
        assert rows[1][2] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_log_nudge_response_calculates_response_time(self, storage):
        """response_time_seconds should be set (positive) after logging nudge response."""
        nudge = {"notification_id": "nid_time", "phase": "active",
                 "action_type": "specific", "title": "T", "message": "M"}
        await storage.log_nudge_sent(nudge)
        await storage.log_nudge_response("nid_time", accepted=True)

        conn = sqlite3.connect(storage.research_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT response_time_seconds FROM research_nudge_log WHERE notification_id = 'nid_time'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] is not None
        assert row[0] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_daily_aggregates: weather entity processing
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeAggregatesWeather:
    """compute_daily_aggregates must extract outdoor temp / condition from hass.states."""

    def _make_weather_state(self, hass, entity_id: str, condition: str, temp: float):
        state_obj = MagicMock()
        state_obj.state = condition
        state_obj.attributes = {"temperature": temp}
        hass.states.get = MagicMock(side_effect=lambda eid: state_obj if eid == entity_id else None)

    @pytest.mark.asyncio
    async def test_weather_entity_populates_outdoor_temp(self, mock_hass, tmp_path):
        """outdoor_temp_celsius must be populated from a weather.* entity attribute."""
        sm = StorageManager(mock_hass)
        await sm.setup()

        self._make_weather_state(mock_hass, "weather.home", "sunny", 18.5)
        sm.config_data = {"weather_entity": "weather.home"}

        date_str = "2026-03-01"
        ts = datetime(2026, 3, 1, 12, 0, 0).timestamp()
        conn = sqlite3.connect(sm.db_path)
        conn.execute(
            "INSERT INTO sensor_history (timestamp, power, within_working_hours) VALUES (?, ?, 1)",
            (ts, 500.0)
        )
        conn.commit()
        conn.close()

        await sm.compute_daily_aggregates(date=date_str, phase="active")

        res_conn = sqlite3.connect(sm.research_db_path)
        cursor = res_conn.cursor()
        cursor.execute(
            "SELECT outdoor_temp_celsius, weather_condition FROM research_daily_aggregates WHERE date = ?",
            (date_str,)
        )
        row = cursor.fetchone()
        res_conn.close()

        assert row is not None
        assert row[0] == pytest.approx(18.5)
        assert row[1] == "sunny"

    @pytest.mark.asyncio
    async def test_no_weather_entity_still_computes(self, mock_hass, tmp_path):
        """compute_daily_aggregates must not crash when no weather entity is available."""
        sm = StorageManager(mock_hass)
        await sm.setup()

        mock_hass.states.get = MagicMock(return_value=None)

        date_str = "2026-03-02"
        ts = datetime(2026, 3, 2, 12, 0, 0).timestamp()
        conn = sqlite3.connect(sm.db_path)
        conn.execute(
            "INSERT INTO sensor_history (timestamp, power, within_working_hours) VALUES (?, ?, 1)",
            (ts, 400.0)
        )
        conn.commit()
        conn.close()

        # Should not raise
        await sm.compute_daily_aggregates(date=date_str, phase="active")

        res_conn = sqlite3.connect(sm.research_db_path)
        cursor = res_conn.cursor()
        cursor.execute(
            "SELECT avg_power_w FROM research_daily_aggregates WHERE date = ?",
            (date_str,)
        )
        row = cursor.fetchone()
        res_conn.close()

        assert row is not None


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent access safety (lock regression tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrentAccess:
    """StorageManager must use asyncio.Lock to serialize access to critical sections of code 
    that interact with shared resources (state file, main DB, research DB). 
    This prevents race conditions and data corruption when multiple async calls happen concurrently.
    """

    @pytest.mark.asyncio
    async def test_storage_has_three_named_locks(self, storage):
        """StorageManager must expose exactly three per-resource asyncio.Lock objects."""
        assert isinstance(storage._state_lock, asyncio.Lock), "_state_lock must be asyncio.Lock"
        assert isinstance(storage._db_lock, asyncio.Lock), "_db_lock must be asyncio.Lock"
        assert isinstance(storage._research_db_lock, asyncio.Lock), "_research_db_lock must be asyncio.Lock"

    @pytest.mark.asyncio
    async def test_legacy_bare_lock_is_gone(self, storage):
        """The old self._lock must no longer exist; only the three named locks remain."""
        assert not hasattr(storage, "_lock"), (
            "Found legacy self._lock - it must be replaced by _state_lock, "
            "_db_lock, and _research_db_lock to avoid the false sense of security."
        )

    @pytest.mark.asyncio
    async def test_concurrent_sensor_writes_all_persist(self, storage):
        """
        Fire 50 concurrent store_sensor_snapshot calls via asyncio.gather.
        Every one must land in the database - no rows dropped due to lock contention.
        """
        now = datetime.now()
        tasks = [
            storage.store_sensor_snapshot(
                timestamp=now - timedelta(seconds=i),
                power=float(i * 10),
            )
            for i in range(50)
        ]
        await asyncio.gather(*tasks)

        history = await storage.get_history("power")
        assert len(history) == 50, (
            f"Expected 50 readings but only {len(history)} survived. "
            "Concurrent inserts are dropping rows - lock all async_add_executor_job calls."
        )

    @pytest.mark.asyncio
    async def test_concurrent_area_writes_all_persist(self, storage):
        """30 concurrent store_area_snapshot calls must all commit without error."""
        now = datetime.now()
        tasks = [
            storage.store_area_snapshot(
                timestamp=now - timedelta(seconds=i),
                area_name="ConcurrentRoom",
                power=float(i),
            )
            for i in range(30)
        ]
        await asyncio.gather(*tasks)

        history = await storage.get_area_history("ConcurrentRoom", "power")
        assert len(history) == 30, (
            f"Expected 30 area readings but got {len(history)}."
        )

    @pytest.mark.asyncio
    async def test_concurrent_research_writes_all_persist(self, storage):
        """20 concurrent log_rl_decision calls must all insert without raising."""
        tasks = [
            storage.log_rl_decision({
                "episode": i,
                "phase": "active",
                "state_vector": [0.1],
                "action": 1,
                "reward": 0.5,
            })
            for i in range(20)
        ]
        await asyncio.gather(*tasks)

        conn = sqlite3.connect(storage.research_db_path)
        count = conn.execute("SELECT COUNT(*) FROM research_rl_episodes").fetchone()[0]
        conn.close()
        assert count == 20, (
            f"Expected 20 RL episodes but got {count}. "
            "Concurrent research writes are colliding - _research_db_lock must guard all research DB calls."
        )

    @pytest.mark.asyncio
    async def test_interleaved_snapshot_and_aggregation_no_data_loss(self, storage):
        """
        A daily aggregation running while snapshot inserts are in flight must not
        cause any insert to be silently dropped.
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        # Mix: 20 snapshot inserts + 5 aggregation runs all at once.
        insert_tasks = [
            storage.store_sensor_snapshot(
                timestamp=now - timedelta(seconds=i),
                power=float(i * 5),
                within_working_hours=True,
            )
            for i in range(20)
        ]
        agg_tasks = [
            storage.compute_daily_aggregates(date=date_str, phase="active")
            for _ in range(5)
        ]

        await asyncio.gather(*insert_tasks, *agg_tasks)

        history = await storage.get_history("power")
        assert len(history) == 20, (
            f"Expected 20 snapshots after concurrent aggregation but got {len(history)}. "
            "Heavy aggregation must not discard concurrent fast inserts."
        )

    @pytest.mark.asyncio
    async def test_reset_all_data_acquires_state_lock(self, storage):
        """
        reset_all_data deletes state.json inside the executor job.  If _state_lock
        is not held, a concurrent save_state/load_state can race with the unlink().
        Verify that _state_lock is included in the locking hierarchy (outermost).
        """
        # Write some state so the file exists before reset.
        await storage.save_state({"phase": "active", "test": True})
        assert storage.state_file.exists()

        # Simulate a concurrent save racing with reset_all_data.
        # Both coroutines are launched together; _state_lock must serialise them.
        await asyncio.gather(
            storage.reset_all_data(),
            storage.save_state({"phase": "post_reset"}),
        )

        # After both complete the state file must be in a consistent state:
        # either the post-reset save won (file with valid JSON) or reset won
        # (file absent).  What must never happen is a corrupt / partially-written
        # file that crashes load_state.
        loaded = await storage.load_state()
        assert isinstance(loaded, dict), (
            "load_state must return a dict even after a concurrent reset; "
            "likely _state_lock was not held during reset_all_data."
        )
