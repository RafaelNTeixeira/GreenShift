"""
Tests for storage.py

Covers:
- Database initialization (sensor_data.db, research_data.db)
- store_sensor_snapshot: data insertion, working hours flag
- store_area_snapshot: area-based data insertion
- get_history: temporal queries with filters
- get_area_history: area-specific temporal queries
- get_all_areas: unique area listing
- get_area_stats: aggregation (avg, min, max, stddev)
- get_recent_values: most recent N values
- save_daily_tasks / get_today_tasks: task persistence
- Data cleanup: old temporal data removal
- RL episode cleanup: retention policy enforcement
"""
import pytest
import sys
import types
import pathlib
import importlib.util
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# ── Minimal HA stubs ────────────────────────────────────────────────────────
for mod_name in ["homeassistant", "homeassistant.core"]:
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

        await storage._cleanup_old_rl_episodes()

        def count_episodes():
            conn = sqlite3.connect(storage.research_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM research_rl_episodes")
            count = cursor.fetchone()[0]
            conn.close()
            return count

        count = count_episodes()
        assert count == 1  # Only recent episode remains


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
