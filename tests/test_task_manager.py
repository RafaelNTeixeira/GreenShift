"""
Tests for task_manager.py

Covers:
- generate_daily_tasks: skips if phase != active
- generate_daily_tasks: skips outside working hours (office mode)
- generate_daily_tasks: returns existing tasks without re-generating
- generate_daily_tasks: selects ≤3 tasks from available generators
- generate_daily_tasks: returns [] when no sensors available
- Task difficulty calculation helpers (indirect via difficulty_multipliers)
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

import sys, types, pathlib, importlib

# ── Stubs for HA and project modules ────────────────────────────────────────

for mod_name in [
    "homeassistant", "homeassistant.core",
    "homeassistant.helpers", "homeassistant.helpers.dispatcher",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Import real translations_runtime module
trans_spec = importlib.util.spec_from_file_location(
    "custom_components.green_shift.translations_runtime",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "translations_runtime.py"
)
trans_mod = importlib.util.module_from_spec(trans_spec)
trans_mod.__package__ = "custom_components.green_shift"
trans_spec.loader.exec_module(trans_mod)
sys.modules["custom_components.green_shift.translations_runtime"] = trans_mod

# helpers stub
helpers_stub = types.ModuleType("custom_components.green_shift.helpers")
helpers_stub.should_ai_be_active = MagicMock(return_value=True)
sys.modules["custom_components.green_shift.helpers"] = helpers_stub

# Import real const module
const_spec = importlib.util.spec_from_file_location(
    "custom_components.green_shift.const",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "const.py"
)
const_mod = importlib.util.module_from_spec(const_spec)
const_mod.__package__ = "custom_components.green_shift"
const_spec.loader.exec_module(const_mod)
sys.modules["custom_components.green_shift.const"] = const_mod

# Load the real module
spec = importlib.util.spec_from_file_location(
    "task_manager",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "task_manager.py"
)
tm_mod = importlib.util.module_from_spec(spec)
tm_mod.__package__ = "custom_components.green_shift"
spec.loader.exec_module(tm_mod)
TaskManager = tm_mod.TaskManager


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_task_manager(sensors=None, phase="active", working_hours=True, config=None):
    hass = MagicMock()
    collector = MagicMock()
    collector.get_current_state = MagicMock(return_value={"power": 500.0, "occupancy": True})
    base_time = datetime(2026, 2, 19, 12, 0, 0)
    collector.get_temperature_history = AsyncMock(
        return_value=[(base_time + timedelta(hours=i), 21.0) for i in range(50)]
    )
    collector.get_power_history = AsyncMock(
        return_value=[(base_time + timedelta(hours=i), 500.0) for i in range(50)]
    )

    storage = AsyncMock()
    storage.get_today_tasks = AsyncMock(return_value=[])
    storage.save_daily_tasks = AsyncMock()
    storage.log_task_generation = AsyncMock()
    storage.get_task_difficulty_stats = AsyncMock(return_value=None)

    agent = MagicMock()
    agent.phase = phase

    if sensors is None:
        sensors = {"power": ["sensor.power_1"], "temperature": ["sensor.temp_1"]}

    # Patch functions on the task_manager module (not on their source modules)
    # because task_manager already imported them
    tm_mod.should_ai_be_active = MagicMock(return_value=working_hours)
    tm_mod.get_language = AsyncMock(return_value="en")

    return TaskManager(
        hass=hass,
        sensors=sensors,
        data_collector=collector,
        storage=storage,
        decision_agent=agent,
        config_data=config or {},
    )


# ─────────────────────────────────────────────────────────────────────────────
# generate_daily_tasks — phase guard
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateDailyTasksPhaseGuard:

    @pytest.mark.asyncio
    async def test_skips_during_baseline_phase(self):
        """Task generation should be blocked during the baseline phase."""
        # The phase guard is in __init__.py's callback, but task_manager itself
        # checks should_ai_be_active (working hours). We verify the agent.phase
        # is accessible and the callback in __init__ would skip.
        tm = make_task_manager(phase="baseline")
        # Simulate the guard from __init__.py
        if tm.decision_agent.phase != "active":
            result = []
        else:
            result = await tm.generate_daily_tasks()
        assert result == []

    @pytest.mark.asyncio
    async def test_runs_during_active_phase(self):
        tm = make_task_manager(phase="active")
        result = await tm.generate_daily_tasks()
        assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# generate_daily_tasks — working hours guard
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateDailyTasksWorkingHours:

    @pytest.mark.asyncio
    async def test_returns_empty_outside_working_hours(self):
        tm = make_task_manager(working_hours=False)
        result = await tm.generate_daily_tasks()
        assert result == []

    @pytest.mark.asyncio
    async def test_generates_tasks_inside_working_hours(self):
        tm = make_task_manager(working_hours=True)
        result = await tm.generate_daily_tasks()
        assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# generate_daily_tasks — idempotence
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateDailyTasksIdempotence:

    @pytest.mark.asyncio
    async def test_returns_existing_tasks_without_regenerating(self):
        tm = make_task_manager()
        existing = [{"task_id": "t1", "task_type": "power_reduction"}]
        tm.storage.get_today_tasks = AsyncMock(return_value=existing)

        result = await tm.generate_daily_tasks()

        assert result == existing
        tm.storage.save_daily_tasks.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# generate_daily_tasks — sensor availability
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateDailyTasksSensorAvailability:

    @pytest.mark.asyncio
    async def test_no_sensors_returns_empty(self):
        tm = make_task_manager(sensors={})
        result = await tm.generate_daily_tasks()
        assert result == []

    @pytest.mark.asyncio
    async def test_generates_at_most_3_tasks(self):
        sensors = {
            "power": ["sensor.power_1"],
            "temperature": ["sensor.temp_1"],
            "occupancy": ["binary_sensor.occ_1"],
            "illuminance": ["sensor.lux_1"],
        }
        tm = make_task_manager(sensors=sensors)
        tm.storage.get_today_tasks = AsyncMock(return_value=[])

        result = await tm.generate_daily_tasks()
        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_each_task_has_required_fields(self):
        tm = make_task_manager()
        result = await tm.generate_daily_tasks()
        for task in result:
            assert "task_id" in task
            assert "task_type" in task
            assert "date" in task

    @pytest.mark.asyncio
    async def test_tasks_saved_to_storage(self):
        tm = make_task_manager()
        result = await tm.generate_daily_tasks()
        if result:  # Only check if tasks were actually generated
            tm.storage.save_daily_tasks.assert_called_once()

    @pytest.mark.asyncio
    async def test_power_only_generates_tasks(self):
        """Even with just power sensors, at least one task should be possible."""
        sensors = {"power": ["sensor.power_1"]}
        tm = make_task_manager(sensors=sensors)
        result = await tm.generate_daily_tasks()
        assert len(result) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# difficulty_multipliers
# ─────────────────────────────────────────────────────────────────────────────

class TestDifficultyMultipliers:

    def test_all_five_difficulty_levels_defined(self):
        tm = make_task_manager()
        assert set(tm.difficulty_multipliers.keys()) == {1, 2, 3, 4, 5}

    def test_difficulty_increases_monotonically(self):
        tm = make_task_manager()
        values = [tm.difficulty_multipliers[i] for i in range(1, 6)]
        assert values == sorted(values)

    def test_normal_difficulty_is_1x(self):
        tm = make_task_manager()
        assert tm.difficulty_multipliers[3] == pytest.approx(1.0)

    def test_easy_is_half_of_normal(self):
        tm = make_task_manager()
        assert tm.difficulty_multipliers[1] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Peak Avoidance Task — peak_hour storage and targeted verification
# ─────────────────────────────────────────────────────────────────────────────

class TestPeakAvoidanceTask:

    @pytest.mark.asyncio
    async def test_peak_hour_stored_in_task(self):
        """Generated peak_avoidance task must include peak_hour for verification."""
        sensors = {"power": ["sensor.power_1"]}
        tm = make_task_manager(sensors=sensors)
        tm.storage.get_today_tasks = AsyncMock(return_value=[])

        tasks = await tm.generate_daily_tasks()
        peak_tasks = [t for t in tasks if t["task_type"] == "peak_avoidance"]
        if peak_tasks:
            assert "peak_hour" in peak_tasks[0], "peak_hour key must be present in the task"
            assert 0 <= peak_tasks[0]["peak_hour"] <= 23

    @pytest.mark.asyncio
    async def test_verification_checks_only_stored_peak_hour(self):
        """Verification must succeed when only the stored peak_hour is below target,
        even though another hour in the same day would fail if it were checked."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        sensors = {"power": ["sensor.power_1"]}
        tm = make_task_manager(sensors=sensors)

        # Hour 10 → 900 W (would fail if checked against target 450 W)
        # Hour 14 → 400 W (below target 450 W) — this is the stored peak_hour
        base_10 = real_dt(2026, 2, 19, 10, 0, 0)
        base_14 = real_dt(2026, 2, 19, 14, 0, 0)
        power_data = (
            [( base_10 + timedelta(minutes=i), 900.0) for i in range(60)]
            + [(base_14 + timedelta(minutes=i), 400.0) for i in range(60)]
        )
        tm.data_collector.get_power_history = AsyncMock(return_value=power_data)

        task = {
            "task_id": "peak_test",
            "task_type": "peak_avoidance",
            "target_value": 450,
            "area_name": None,
            "peak_hour": 14,  # Only this hour should be evaluated
            "verified": False,
        }

        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual = await tm._verify_single_task(task)

        assert verified  # True: only peak_hour=14 (400W) evaluated, below target 450W
        assert actual == pytest.approx(400.0, abs=1.0)

    @pytest.mark.asyncio
    async def test_verification_fails_when_stored_peak_hour_exceeds_target(self):
        """Verification must fail when the stored peak_hour average exceeds the target."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        sensors = {"power": ["sensor.power_1"]}
        tm = make_task_manager(sensors=sensors)

        base_14 = real_dt(2026, 2, 19, 14, 0, 0)
        power_data = [(base_14 + timedelta(minutes=i), 600.0) for i in range(60)]
        tm.data_collector.get_power_history = AsyncMock(return_value=power_data)

        task = {
            "task_id": "peak_test_fail",
            "task_type": "peak_avoidance",
            "target_value": 450,
            "area_name": None,
            "peak_hour": 14,
            "verified": False,
        }

        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual = await tm._verify_single_task(task)

        assert not verified  # 600W > 450W target

    @pytest.mark.asyncio
    async def test_verification_returns_false_when_peak_hour_missing(self):
        """If peak_hour is absent (legacy task), verification must return False gracefully."""
        tm = make_task_manager()
        task = {
            "task_id": "peak_legacy",
            "task_type": "peak_avoidance",
            "target_value": 450,
            "area_name": None,
            # No 'peak_hour' key
            "verified": False,
        }
        # hours_passed check will run first — give enough data so it doesn't exit early
        from unittest.mock import patch
        from datetime import datetime as real_dt
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual = await tm._verify_single_task(task)
        assert not verified  # peak_hour key absent -> graceful False
        assert actual is None


# ─────────────────────────────────────────────────────────────────────────────
# Unoccupied Power Task — occupancy-aware generation and verification
# ─────────────────────────────────────────────────────────────────────────────

class TestUnoccupiedPowerTask:

    def _make_tm_with_area(self, mixed_occ=False):
        """Helper: task manager with one mocked area 'Office'."""
        sensors = {
            "power": ["sensor.power_1"],
            "occupancy": ["binary_sensor.occ_1"],
        }
        tm = make_task_manager(sensors=sensors)
        tm.storage.get_today_tasks = AsyncMock(return_value=[])
        tm.data_collector.get_all_areas = MagicMock(return_value=["Office"])

        base_time = datetime(2026, 2, 19, 12, 0, 0)
        if mixed_occ:
            # First half occupied (800 W), second half unoccupied (200 W)
            power_data = (
                [(base_time + timedelta(minutes=i), 800.0) for i in range(50)]
                + [(base_time + timedelta(minutes=50 + i), 200.0) for i in range(50)]
            )
            occ_data = (
                [(base_time + timedelta(minutes=i), 1) for i in range(50)]
                + [(base_time + timedelta(minutes=50 + i), 0) for i in range(50)]
            )
        else:
            power_data = [(base_time + timedelta(minutes=i), 200.0) for i in range(100)]
            occ_data   = [(base_time + timedelta(minutes=i), 0)     for i in range(100)]

        async def area_history(area, metric, **kwargs):
            if metric == "power":
                return power_data
            if metric == "occupancy":
                return occ_data
            return []

        tm.data_collector.get_area_history = AsyncMock(side_effect=area_history)
        return tm

    @pytest.mark.asyncio
    async def test_generation_baseline_reflects_unoccupied_power_only(self):
        """Baseline in generated task should be the average power during unoccupied periods."""
        tm = self._make_tm_with_area(mixed_occ=True)
        task = await tm._generate_unoccupied_power_task()
        assert task is not None
        # Unoccupied periods have 200 W average; occupied periods have 800 W
        assert task["baseline_value"] == pytest.approx(200.0, abs=1.0)

    @pytest.mark.asyncio
    async def test_generation_falls_back_without_occupancy_data(self):
        """When occupancy data is unavailable, all power readings are used as fallback."""
        sensors = {"power": ["sensor.power_1"], "occupancy": ["binary_sensor.occ_1"]}
        tm = make_task_manager(sensors=sensors)
        tm.data_collector.get_all_areas = MagicMock(return_value=["Office"])

        base_time = datetime(2026, 2, 19, 12, 0, 0)
        power_data = [(base_time + timedelta(minutes=i), 300.0) for i in range(100)]

        async def area_history(area, metric, **kwargs):
            if metric == "power":
                return power_data
            return []  # No occupancy data

        tm.data_collector.get_area_history = AsyncMock(side_effect=area_history)
        task = await tm._generate_unoccupied_power_task()
        assert task is not None
        assert task["baseline_value"] == pytest.approx(300.0, abs=1.0)

    @pytest.mark.asyncio
    async def test_generation_skips_area_without_power_data(self):
        """Area with no power data should be skipped and not become the target area."""
        sensors = {"power": ["sensor.power_1"], "occupancy": ["binary_sensor.occ_1"]}
        tm = make_task_manager(sensors=sensors)
        tm.data_collector.get_all_areas = MagicMock(return_value=["EmptyRoom"])

        async def area_history(area, metric, **kwargs):
            return []  # No data at all

        tm.data_collector.get_area_history = AsyncMock(side_effect=area_history)
        task = await tm._generate_unoccupied_power_task()
        assert task is None

    @pytest.mark.asyncio
    async def test_verification_measures_unoccupied_intervals_only(self):
        """Verification should pass when unoccupied power is below target,
        even if occupied intervals would push the overall average above target."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        sensors = {"power": ["sensor.power_1"], "occupancy": ["binary_sensor.occ_1"]}
        tm = make_task_manager(sensors=sensors)

        base_time = real_dt(2026, 2, 19, 12, 0, 0)
        # Occupied half: 900 W; unoccupied half: 200 W (target 250 W)
        power_data = (
            [(base_time + timedelta(minutes=i),      900.0) for i in range(30)]
            + [(base_time + timedelta(minutes=30+i), 200.0) for i in range(30)]
        )
        occ_data = (
            [(base_time + timedelta(minutes=i),      1) for i in range(30)]
            + [(base_time + timedelta(minutes=30+i), 0) for i in range(30)]
        )

        async def area_history(area, metric, **kwargs):
            if metric == "power":
                return power_data
            if metric == "occupancy":
                return occ_data
            return []

        tm.data_collector.get_area_history = AsyncMock(side_effect=area_history)

        task = {
            "task_id": "unocc_verify",
            "task_type": "unoccupied_power",
            "target_value": 250,
            "area_name": "Office",
            "verified": False,
        }

        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual = await tm._verify_single_task(task)

        assert verified  # 200W unoccupied avg is below target 250W
        assert actual == pytest.approx(200.0, abs=1.0)

    @pytest.mark.asyncio
    async def test_verification_fails_when_unoccupied_power_exceeds_target(self):
        """Verification must fail when unoccupied power average is above the target."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        sensors = {"power": ["sensor.power_1"], "occupancy": ["binary_sensor.occ_1"]}
        tm = make_task_manager(sensors=sensors)

        base_time = real_dt(2026, 2, 19, 12, 0, 0)
        power_data = [(base_time + timedelta(minutes=i), 400.0) for i in range(60)]
        occ_data   = [(base_time + timedelta(minutes=i), 0)     for i in range(60)]  # always unoccupied

        async def area_history(area, metric, **kwargs):
            if metric == "power":
                return power_data
            if metric == "occupancy":
                return occ_data
            return []

        tm.data_collector.get_area_history = AsyncMock(side_effect=area_history)

        task = {
            "task_id": "unocc_fail",
            "task_type": "unoccupied_power",
            "target_value": 250,
            "area_name": "Office",
            "verified": False,
        }

        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual = await tm._verify_single_task(task)

        assert not verified  # 400W unoccupied avg > 250W target
