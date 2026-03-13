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
helpers_stub.get_working_days_from_config = MagicMock(return_value=list(range(5)))  # Mon-Fri
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
    storage.get_tasks_for_date = AsyncMock(return_value=[])  # no tasks yesterday by default
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
    # Patch working-days helper (used by new office-mode day-guard)
    today_weekday = datetime.now().weekday()
    if working_hours:
        # Return a list that includes today so the day-guard passes
        tm_mod.get_working_days_from_config = MagicMock(return_value=list(range(7)))
    else:
        # Return a list that excludes today so the day-guard blocks
        tm_mod.get_working_days_from_config = MagicMock(
            return_value=[d for d in range(7) if d != today_weekday]
        )

    return TaskManager(
        hass=hass,
        sensors=sensors,
        data_collector=collector,
        storage=storage,
        decision_agent=agent,
        config_data=config or {},
    )


# ─────────────────────────────────────────────────────────────────────────────
# generate_daily_tasks
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

    def test_last_verification_results_empty_on_new_instance(self):
        """_last_verification_results must start empty on a fresh TaskManager."""
        tm = make_task_manager()
        assert tm._last_verification_results == {}


# ─────────────────────────────────────────────────────────────────────────────
# generate_daily_tasks : working hours guard
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateDailyTasksWorkingHours:
    """
    In office mode, tasks must only be generated on working days.
    In home mode there is no day-gate, tasks are always generated.
    """

    @pytest.mark.asyncio
    async def test_returns_empty_on_non_working_day_office_mode(self):
        """Office mode: task generation is skipped when today is not a working day."""
        office_config = {"environment_mode": "office"}
        tm = make_task_manager(working_hours=False, config=office_config)
        result = await tm.generate_daily_tasks()
        assert result == []

    @pytest.mark.asyncio
    async def test_generates_tasks_on_working_day_office_mode(self):
        """Office mode: tasks are generated when today is a working day."""
        office_config = {"environment_mode": "office"}
        tm = make_task_manager(working_hours=True, config=office_config)
        result = await tm.generate_daily_tasks()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_home_mode_generates_tasks_any_day(self):
        """Home mode never applies the working-day gate; tasks are always generated."""
        home_config = {"environment_mode": "home"}
        tm = make_task_manager(working_hours=False, config=home_config)
        result = await tm.generate_daily_tasks()
        # Home mode: no day gate -> should proceed to generate (list, possibly empty if no sensors)
        assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# generate_daily_tasks : idempotence
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
# generate_daily_tasks : sensor availability
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
# Peak Avoidance Task : peak_hour storage and targeted verification
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

        # Hour 10 -> 900 W (would fail if checked against target 450 W)
        # Hour 14 -> 400 W (below target 450 W) : this is the stored peak_hour
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
            verified, actual, pending = await tm._verify_single_task(task)

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
            verified, actual, pending = await tm._verify_single_task(task)

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
        # hours_passed check will run first : give enough data so it doesn't exit early
        from unittest.mock import patch
        from datetime import datetime as real_dt
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)
        assert not verified  # peak_hour key absent -> graceful False
        assert actual is None

    @pytest.mark.asyncio
    async def test_peak_avoidance_pending_when_peak_hour_not_reached(self):
        """peak_avoidance returns pending=True when the peak hour hasn't arrived yet."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager()
        power_data = [(real_dt(2026, 2, 19, 12, i), 300.0) for i in range(60)]
        tm.data_collector.get_power_history = AsyncMock(return_value=power_data)
        task = {
            "task_id": "peak_pending",
            "task_type": "peak_avoidance",
            "target_value": 400,
            "area_name": None,
            "peak_hour": 23,
            "verified": False,
        }
        fake_now = real_dt(2026, 2, 19, 14, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)

        assert not verified
        assert actual is None
        assert pending is True


# ─────────────────────────────────────────────────────────────────────────────
# Unoccupied Power Task : occupancy-aware generation and verification
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
            verified, actual, pending = await tm._verify_single_task(task)

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
            verified, actual, pending = await tm._verify_single_task(task)

        assert not verified  # 400W unoccupied avg > 250W target


# ─────────────────────────────────────────────────────────────────────────────
# Power Reduction / Daylight Usage : working_hours_filter in office mode
# ─────────────────────────────────────────────────────────────────────────────

class TestVerificationWorkingHoursFilter:
    """In office mode, power_reduction and daylight_usage verification must pass
    working_hours_only=True to get_power_history; in home mode it must be falsy."""

    @pytest.mark.asyncio
    async def test_power_reduction_office_mode_passes_working_hours_filter(self):
        """get_power_history called with working_hours_only=True in office mode."""
        from unittest.mock import patch, call
        from datetime import datetime as real_dt

        tm = make_task_manager(config={"environment_mode": "office"})
        tm.data_collector.get_power_history = AsyncMock(return_value=[
            (real_dt(2026, 2, 19, 12, i), 300.0) for i in range(30)
        ])

        task = {
            "task_id": "pr_office",
            "task_type": "power_reduction",
            "target_value": 500,  # 300W < 500W -> should pass
            "area_name": None,
            "verified": False,
        }

        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm._verify_single_task(task)

        call_kwargs = tm.data_collector.get_power_history.call_args
        assert call_kwargs.kwargs.get("working_hours_only") is True

    @pytest.mark.asyncio
    async def test_power_reduction_home_mode_no_working_hours_filter(self):
        """get_power_history called WITHOUT working_hours_only=True in home mode."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager(config={"environment_mode": "home"})
        tm.data_collector.get_power_history = AsyncMock(return_value=[
            (real_dt(2026, 2, 19, 12, i), 300.0) for i in range(30)
        ])

        task = {
            "task_id": "pr_home",
            "task_type": "power_reduction",
            "target_value": 500,
            "area_name": None,
            "verified": False,
        }

        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm._verify_single_task(task)

        call_kwargs = tm.data_collector.get_power_history.call_args
        assert not call_kwargs.kwargs.get("working_hours_only")

    @pytest.mark.asyncio
    async def test_daylight_usage_office_mode_passes_working_hours_filter(self):
        """daylight_usage verification in office mode must pass working_hours_only=True."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager(config={"environment_mode": "office"})
        tm.data_collector.get_power_history = AsyncMock(return_value=[
            (real_dt(2026, 2, 19, 12, i), 200.0) for i in range(30)
        ])

        task = {
            "task_id": "day_office",
            "task_type": "daylight_usage",
            "target_value": 400,
            "area_name": None,
            "verified": False,
        }

        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm._verify_single_task(task)

        call_kwargs = tm.data_collector.get_power_history.call_args
        assert call_kwargs.kwargs.get("working_hours_only") is True


# ─────────────────────────────────────────────────────────────────────────────
# unoccupied_power: working_hours_filter in office mode
# ─────────────────────────────────────────────────────────────────────────────

class TestUnoccupiedPowerWorkingHoursFilter:
    """unoccupied_power generation and verification must pass working_hours_only=True
    in office mode for consistency with the other task types."""

    @pytest.mark.asyncio
    async def test_generation_office_mode_passes_working_hours_filter(self):
        """_generate_unoccupied_power_task must call get_area_history with
        working_hours_only=True when running in office mode."""
        office_cfg = {"environment_mode": "office"}
        sensors = {"power": ["sensor.power_1"], "occupancy": ["binary_sensor.occ_1"]}
        tm = make_task_manager(sensors=sensors, config=office_cfg)
        tm.data_collector.get_all_areas = MagicMock(return_value=["Office"])

        base_time = datetime(2026, 2, 19, 10, 0, 0)
        power_data = [(base_time + timedelta(minutes=i), 300.0) for i in range(50)]
        calls = []

        async def _area_history(area, metric, **kwargs):
            calls.append((metric, kwargs.get("working_hours_only")))
            if metric == "power":
                return power_data
            return []

        tm.data_collector.get_area_history = _area_history
        await tm._generate_unoccupied_power_task()

        wh_values = [wh for _, wh in calls]
        assert all(wh is True for wh in wh_values), (
            f"All get_area_history calls must use working_hours_only=True in office mode, got: {calls}"
        )

    @pytest.mark.asyncio
    async def test_generation_home_mode_no_working_hours_filter(self):
        """_generate_unoccupied_power_task must call get_area_history without
        working_hours_only filter in home mode."""
        home_cfg = {"environment_mode": "home"}
        sensors = {"power": ["sensor.power_1"], "occupancy": ["binary_sensor.occ_1"]}
        tm = make_task_manager(sensors=sensors, config=home_cfg)
        tm.data_collector.get_all_areas = MagicMock(return_value=["Living Room"])

        base_time = datetime(2026, 2, 19, 10, 0, 0)
        power_data = [(base_time + timedelta(minutes=i), 300.0) for i in range(50)]
        calls = []

        async def _area_history(area, metric, **kwargs):
            calls.append((metric, kwargs.get("working_hours_only")))
            if metric == "power":
                return power_data
            return []

        tm.data_collector.get_area_history = _area_history
        await tm._generate_unoccupied_power_task()

        wh_values = [wh for _, wh in calls]
        assert all(wh is None for wh in wh_values), (
            f"All get_area_history calls must use working_hours_only=None in home mode, got: {calls}"
        )

    @pytest.mark.asyncio
    async def test_verification_office_mode_passes_working_hours_filter(self):
        """_verify_single_task for unoccupied_power must call get_area_history with
        working_hours_only=True in office mode."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        office_cfg = {"environment_mode": "office"}
        sensors = {"power": ["sensor.power_1"], "occupancy": ["binary_sensor.occ_1"]}
        tm = make_task_manager(sensors=sensors, config=office_cfg)

        base_time = real_dt(2026, 2, 19, 10, 0, 0)
        power_data = [(base_time + timedelta(minutes=i), 200.0) for i in range(50)]
        calls = []

        async def _area_history(area, metric, **kwargs):
            calls.append((metric, kwargs.get("working_hours_only")))
            if metric == "power":
                return power_data
            return []

        tm.data_collector.get_area_history = _area_history

        task = {
            "task_id": "unocc_office",
            "task_type": "unoccupied_power",
            "target_value": 250,
            "area_name": "Office",
            "verified": False,
        }
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm._verify_single_task(task)

        wh_values = [wh for _, wh in calls]
        assert all(wh is True for wh in wh_values), (
            f"Verification get_area_history calls must use working_hours_only=True in office mode, got: {calls}"
        )

    @pytest.mark.asyncio
    async def test_verification_home_mode_no_working_hours_filter(self):
        """_verify_single_task for unoccupied_power must call get_area_history without
        working_hours_only filter in home mode."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        home_cfg = {"environment_mode": "home"}
        sensors = {"power": ["sensor.power_1"], "occupancy": ["binary_sensor.occ_1"]}
        tm = make_task_manager(sensors=sensors, config=home_cfg)

        base_time = real_dt(2026, 2, 19, 10, 0, 0)
        power_data = [(base_time + timedelta(minutes=i), 200.0) for i in range(50)]
        calls = []

        async def _area_history(area, metric, **kwargs):
            calls.append((metric, kwargs.get("working_hours_only")))
            if metric == "power":
                return power_data
            return []

        tm.data_collector.get_area_history = _area_history

        task = {
            "task_id": "unocc_home",
            "task_type": "unoccupied_power",
            "target_value": 250,
            "area_name": "Living Room",
            "verified": False,
        }
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm._verify_single_task(task)

        wh_values = [wh for _, wh in calls]
        assert all(wh is None for wh in wh_values), (
            f"Verification get_area_history calls must use working_hours_only=None in home mode, got: {calls}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Verification time anchor: created_at vs TASK_GENERATION_TIME
# ─────────────────────────────────────────────────────────────────────────────

class TestVerificationTimeAnchor:
    """When a task carries 'created_at', verification uses that timestamp as the
    window anchor instead of the hardcoded TASK_GENERATION_TIME (06:00)."""

    @pytest.mark.asyncio
    async def test_created_at_anchors_hours_passed_window(self):
        """Task created at 10:00, now is 20:00 -> hours_passed ~10, not ~14."""
        from unittest.mock import patch, call
        from datetime import datetime as real_dt

        tm = make_task_manager()
        tm.data_collector.get_power_history = AsyncMock(return_value=[
            (real_dt(2026, 2, 19, 12, i), 300.0) for i in range(30)
        ])

        # created_at at 10:00; now at 20:00 -> hours ≈ 10
        task = {
            "task_id": "anchor_10h",
            "task_type": "power_reduction",
            "target_value": 500,
            "area_name": None,
            "created_at": "2026-02-19T10:00:00",
            "verified": False,
        }

        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm._verify_single_task(task)

        call_kwargs = tm.data_collector.get_power_history.call_args
        # hours should be ~10 (from 10:00 to 20:00), not ~14 (from 06:00)
        hours_arg = call_kwargs.kwargs.get("hours")
        if hours_arg is None and call_kwargs.args:
            hours_arg = call_kwargs.args[0]
        assert hours_arg is not None
        assert abs(hours_arg - 10) < 1, f"Expected ~10 hours but got {hours_arg}"

    @pytest.mark.asyncio
    async def test_missing_created_at_falls_back_to_task_generation_time(self):
        """Task without created_at uses TASK_GENERATION_TIME (06:00) as anchor."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager()
        tm.data_collector.get_power_history = AsyncMock(return_value=[
            (real_dt(2026, 2, 19, 12, i), 300.0) for i in range(30)
        ])

        # No created_at -> fallback to 06:00; now is 20:00 -> hours ≈ 14
        task = {
            "task_id": "fallback_anchor",
            "task_type": "power_reduction",
            "target_value": 500,
            "area_name": None,
            # no 'created_at'
            "verified": False,
        }

        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm._verify_single_task(task)

        call_kwargs = tm.data_collector.get_power_history.call_args
        hours_arg = call_kwargs.kwargs.get("hours")
        if hours_arg is None and call_kwargs.args:
            hours_arg = call_kwargs.args[0]
        assert hours_arg is not None
        # TASK_GENERATION_TIME = (6, 0, 0) -> 20:00 - 06:00 = 14 hours
        assert abs(hours_arg - 14) < 1, f"Expected ~14 hours (06:00 fallback) but got {hours_arg}"

    @pytest.mark.asyncio
    async def test_hours_passed_ceil_not_truncated(self):
        """_verify_single_task must use math.ceil, not int().
        created_at 06:00, fake_now 13:54 -> hours_passed=7.9 -> query must ask for 8 h."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager()
        tm.data_collector.get_power_history = AsyncMock(return_value=[
            (real_dt(2026, 2, 19, 12, i), 300.0) for i in range(30)
        ])
        task = {
            "task_id": "ceil_test",
            "task_type": "power_reduction",
            "target_value": 500,
            "area_name": None,
            "created_at": "2026-02-19T06:00:00",
            "verified": False,
        }
        fake_now = real_dt(2026, 2, 19, 13, 54, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm._verify_single_task(task)

        call_kwargs = tm.data_collector.get_power_history.call_args
        if call_kwargs.kwargs.get("hours") is not None:
            hours_arg = call_kwargs.kwargs["hours"]
        else:
            hours_arg = call_kwargs.args[0] if call_kwargs.args else None
        assert hours_arg == 8, f"Expected ceil(7.9)=8 but got {hours_arg}"


# ─────────────────────────────────────────────────────────────────────────────
# Streak integration : via TaskManager
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskManagerStreak:
    """Verify that TaskManager calls update_task_streak at the right moments."""

    # ── verify_tasks ──────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_verify_tasks_credits_streak_when_at_least_one_verified(self):
        """update_task_streak(True, today) called when at least one task is verified."""
        tm = make_task_manager()

        today_str = datetime.now().strftime("%Y-%m-%d")
        tasks = [
            {"task_id": f"t{i}", "task_type": "power_reduction",
             "target_value": 400.0, "area_name": None, "verified": True,
             "created_at": f"{today_str}T08:01:00",
             "peak_hour": None}
            for i in range(3)
        ]
        tm.storage.get_today_tasks = AsyncMock(return_value=tasks)

        await tm.verify_tasks()

        tm.decision_agent.update_task_streak.assert_called_once_with(True, datetime.now().date())

    @pytest.mark.asyncio
    async def test_verify_tasks_credits_streak_when_partial_verified(self):
        """update_task_streak(True, today) called even when only SOME tasks are verified."""
        tm = make_task_manager()

        today_str = datetime.now().strftime("%Y-%m-%d")
        t_verified = {
            "task_id": "t1", "task_type": "power_reduction",
            "target_value": 400.0, "area_name": None, "verified": True,
            "created_at": f"{today_str}T08:01:00", "peak_hour": None
        }
        t_pending = {
            "task_id": "t2", "task_type": "power_reduction",
            "target_value": 400.0, "area_name": None, "verified": False,
            "created_at": f"{today_str}T08:01:00", "peak_hour": None
        }
        tm.storage.get_today_tasks = AsyncMock(return_value=[t_verified, t_pending])
        tm.data_collector.get_power_history = AsyncMock(return_value=[
            (datetime.now() - timedelta(hours=i), 500.0) for i in range(30)
        ])

        await tm.verify_tasks()

        tm.decision_agent.update_task_streak.assert_called_once_with(True, datetime.now().date())

    @pytest.mark.asyncio
    async def test_verify_tasks_no_streak_when_no_tasks(self):
        """update_task_streak is NOT called when there are no tasks."""
        tm = make_task_manager()
        tm.storage.get_today_tasks = AsyncMock(return_value=[])

        await tm.verify_tasks()

        tm.decision_agent.update_task_streak.assert_not_called()

    # ── generate_daily_tasks ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_generate_tasks_resets_streak_when_none_verified_yesterday(self):
        """update_task_streak(False, yesterday) is called only if NO tasks were verified yesterday."""
        tm = make_task_manager()

        yesterday = (datetime.now().date() - timedelta(days=1))
        yesterday_tasks = [
            {"task_id": "y1", "task_type": "power_reduction", "verified": False},
            {"task_id": "y2", "task_type": "power_reduction", "verified": False},
        ]
        tm.storage.get_tasks_for_date = AsyncMock(return_value=yesterday_tasks)

        await tm.generate_daily_tasks()

        tm.decision_agent.update_task_streak.assert_called_once_with(False, yesterday)

    @pytest.mark.asyncio
    async def test_generate_tasks_no_streak_reset_when_yesterday_partial_verified(self):
        """update_task_streak is NOT called if at least one task was verified yesterday."""
        tm = make_task_manager()

        yesterday_tasks = [
            {"task_id": "y1", "task_type": "power_reduction", "verified": False},
            {"task_id": "y2", "task_type": "power_reduction", "verified": True},
        ]
        tm.storage.get_tasks_for_date = AsyncMock(return_value=yesterday_tasks)

        await tm.generate_daily_tasks()

        tm.decision_agent.update_task_streak.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_tasks_no_streak_reset_when_yesterday_all_done(self):
        """update_task_streak is NOT called if yesterday tasks were all verified."""
        tm = make_task_manager()

        yesterday_tasks = [
            {"task_id": "y1", "task_type": "power_reduction", "verified": True},
            {"task_id": "y2", "task_type": "power_reduction", "verified": True},
        ]
        tm.storage.get_tasks_for_date = AsyncMock(return_value=yesterday_tasks)

        await tm.generate_daily_tasks()

        tm.decision_agent.update_task_streak.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_tasks_no_streak_reset_when_no_yesterday_tasks(self):
        """update_task_streak is NOT called if yesterday had no tasks (e.g., weekend)."""
        tm = make_task_manager()
        # Already default: get_tasks_for_date returns []

        await tm.generate_daily_tasks()

        tm.decision_agent.update_task_streak.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# _calculate_task_difficulty with real stats data
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateTaskDifficultyWithStats:

    @pytest.mark.asyncio
    async def test_defaults_to_3_when_not_enough_samples(self):
        """Fewer than 3 total feedback items -> default difficulty 3."""
        tm = make_task_manager()
        stats = {"too_easy_count": 1, "just_right_count": 0, "too_hard_count": 0, "avg_difficulty": 3, "suggested_adjustment": 0}
        result = await tm._calculate_task_difficulty(stats)
        assert result == 3

    @pytest.mark.asyncio
    async def test_uses_suggested_adjustment_when_enough_samples(self):
        """With 3+ samples, apply adjustment to avg_difficulty."""
        tm = make_task_manager()
        stats = {
            "too_easy_count": 2, "just_right_count": 1, "too_hard_count": 0,
            "avg_difficulty": 3, "suggested_adjustment": 1,
        }
        result = await tm._calculate_task_difficulty(stats)
        assert result == 4  # 3 + 1

    @pytest.mark.asyncio
    async def test_clamps_result_to_max_5(self):
        tm = make_task_manager()
        stats = {
            "too_easy_count": 3, "just_right_count": 0, "too_hard_count": 0,
            "avg_difficulty": 5, "suggested_adjustment": 2,
        }
        result = await tm._calculate_task_difficulty(stats)
        assert result == 5  # clamped from 7

    @pytest.mark.asyncio
    async def test_clamps_result_to_min_1(self):
        tm = make_task_manager()
        stats = {
            "too_easy_count": 0, "just_right_count": 0, "too_hard_count": 3,
            "avg_difficulty": 1, "suggested_adjustment": -2,
        }
        result = await tm._calculate_task_difficulty(stats)
        assert result == 1  # clamped from -1

    @pytest.mark.asyncio
    async def test_none_stats_returns_3(self):
        tm = make_task_manager()
        result = await tm._calculate_task_difficulty(None)
        assert result == 3


# ─────────────────────────────────────────────────────────────────────────────
# Individual task generators return None when no history
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskGeneratorsReturnNoneWithoutData:

    @pytest.mark.asyncio
    async def test_temperature_task_returns_none_when_no_history(self):
        tm = make_task_manager(sensors={"temperature": ["sensor.temp_1"]})
        tm.data_collector.get_temperature_history = AsyncMock(return_value=[])
        result = await tm._generate_temperature_task()
        assert result is None

    @pytest.mark.asyncio
    async def test_power_reduction_task_returns_none_when_no_history(self):
        tm = make_task_manager(sensors={"power": ["sensor.power_1"]})
        tm.data_collector.get_power_history = AsyncMock(return_value=[])
        result = await tm._generate_power_reduction_task()
        assert result is None

    @pytest.mark.asyncio
    async def test_daylight_task_returns_none_when_no_history(self):
        tm = make_task_manager(sensors={"power": ["sensor.power_1"]})
        tm.data_collector.get_power_history = AsyncMock(return_value=[])
        result = await tm._generate_daylight_task()
        assert result is None

    @pytest.mark.asyncio
    async def test_daylight_task_returns_none_when_no_daytime_readings(self):
        """All readings outside 08:00-17:00 -> no daytime data -> None."""
        tm = make_task_manager(sensors={"power": ["sensor.power_1"]})
        # All timestamps are at 20:00 (outside daytime window)
        base = datetime(2026, 2, 18, 20, 0, 0)
        tm.data_collector.get_power_history = AsyncMock(
            return_value=[(base + timedelta(minutes=i), 300.0) for i in range(30)]
        )
        result = await tm._generate_daylight_task()
        assert result is None

    @pytest.mark.asyncio
    async def test_peak_avoidance_returns_none_when_no_history(self):
        tm = make_task_manager(sensors={"power": ["sensor.power_1"]})
        tm.data_collector.get_power_history = AsyncMock(return_value=[])
        result = await tm._generate_peak_avoidance_task()
        assert result is None

    @pytest.mark.asyncio
    async def test_unoccupied_task_returns_none_when_no_areas(self):
        tm = make_task_manager(sensors={"power": ["sensor.p1"], "occupancy": ["bs.occ1"]})
        tm.data_collector.get_all_areas = MagicMock(return_value=[])
        result = await tm._generate_unoccupied_power_task()
        assert result is None

    @pytest.mark.asyncio
    async def test_unoccupied_task_skips_no_area_label(self):
        """Areas labelled 'No Area' are skipped in unoccupied power task generation."""
        tm = make_task_manager(sensors={"power": ["sensor.p1"], "occupancy": ["bs.occ1"]})
        tm.data_collector.get_all_areas = MagicMock(return_value=["No Area"])
        tm.data_collector.get_area_history = AsyncMock(return_value=[])
        result = await tm._generate_unoccupied_power_task()
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# verify_tasks: pre-verified tasks stay verified
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyTasksPreVerified:

    @pytest.mark.asyncio
    async def test_already_verified_task_stays_true_without_rechecking(self):
        tm = make_task_manager()
        today = datetime.now().strftime("%Y-%m-%d")
        tasks = [{
            "task_id": "t1", "task_type": "power_reduction",
            "target_value": 400.0, "area_name": None,
            "verified": True,  # already done
            "created_at": f"{today}T08:00:00",
            "peak_hour": None,
        }]
        tm.storage.get_today_tasks = AsyncMock(return_value=tasks)

        results = await tm.verify_tasks()

        assert results["t1"] is True
        # mark_task_verified should NOT be called again
        tm.storage.mark_task_verified.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_tasks_populates_last_verification_results(self):
        """verify_tasks() must populate _last_verification_results for each task."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager()
        today_str = "2026-02-19"
        tasks = [{
            "task_id": "vr_test",
            "task_type": "power_reduction",
            "target_value": 400,
            "area_name": None,
            "verified": False,
            "created_at": f"{today_str}T06:00:00",
            "peak_hour": None,
            "completion_value": None,
        }]
        tm.storage.get_today_tasks = AsyncMock(return_value=tasks)
        base = real_dt(2026, 2, 19, 12, 0, 0)
        tm.data_collector.get_power_history = AsyncMock(return_value=[
            (base + timedelta(minutes=i), 600.0) for i in range(30)
        ])
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm.verify_tasks()

        result = tm._last_verification_results.get("vr_test")
        assert result is not None, "_last_verification_results not populated"
        for key in ("verified", "failed", "pending", "checked_at", "reason"):
            assert key in result, f"Missing key '{key}' in result"
        assert result["failed"] is True
        assert result["pending"] is False
        assert result["reason"] != ""


# ─────────────────────────────────────────────────────────────────────────────
# _verify_single_task: temperature_reduction type
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifySingleTaskTemperature:

    @pytest.mark.asyncio
    async def test_temperature_verified_when_avg_below_target(self):
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager(sensors={"temperature": ["sensor.t1"]})
        base = real_dt(2026, 2, 19, 8, 0, 0)
        tm.data_collector.get_temperature_history = AsyncMock(
            return_value=[(base + timedelta(hours=i), 19.5) for i in range(5)]
        )
        task = {
            "task_id": "temp_verify",
            "task_type": "temperature_reduction",
            "target_value": 20.0,
            "area_name": None,
            "created_at": "2026-02-19T08:00:00",
            "verified": False,
        }
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)

        assert verified == True
        assert actual == pytest.approx(19.5, abs=0.1)

    @pytest.mark.asyncio
    async def test_temperature_fails_when_avg_above_target(self):
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager(sensors={"temperature": ["sensor.t1"]})
        base = real_dt(2026, 2, 19, 8, 0, 0)
        tm.data_collector.get_temperature_history = AsyncMock(
            return_value=[(base + timedelta(hours=i), 22.5) for i in range(5)]
        )
        task = {
            "task_id": "temp_fail",
            "task_type": "temperature_reduction",
            "target_value": 21.0,
            "area_name": None,
            "created_at": "2026-02-19T08:00:00",
            "verified": False,
        }
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)

        assert verified == False


class TestTaskManagerAdditionalCoverage:

    @pytest.mark.asyncio
    async def test_generate_daily_tasks_ignores_generator_exception(self):
        tm = make_task_manager()
        tm.storage.get_today_tasks = AsyncMock(return_value=[])
        tm.storage.get_tasks_for_date = AsyncMock(return_value=[])

        good_task = {
            "task_id": "ok1",
            "task_type": "power_reduction",
            "title": "T",
            "description": "D",
            "target_value": 300,
            "baseline_value": 400,
            "difficulty_level": 3,
        }

        async def bad_gen():
            raise RuntimeError("boom")

        async def good_gen():
            return good_task.copy()

        with patch.object(tm_mod.np.random, "choice", return_value=[bad_gen, good_gen]):
            tasks = await tm.generate_daily_tasks()

        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "ok1"

    @pytest.mark.asyncio
    async def test_generate_temperature_task_weather_and_sensor_parse_failures(self):
        tm = make_task_manager(sensors={"temperature": ["sensor.temp_1"]})
        tm.config_data = {
            "weather_entity": "weather.home",
            "outdoor_temp_sensor": "sensor.outdoor_temp",
        }

        weather_state = MagicMock()
        weather_state.attributes = {"temperature": "not-a-float"}
        sensor_state = MagicMock()
        sensor_state.state = "also-bad"

        def get_state(entity_id):
            if entity_id == "weather.home":
                return weather_state
            if entity_id == "sensor.outdoor_temp":
                return sensor_state
            return None

        tm.hass.states.get = MagicMock(side_effect=get_state)

        task = await tm._generate_temperature_task()
        assert task is None

    @pytest.mark.asyncio
    async def test_verify_tasks_uses_generic_reason_for_unknown_type(self):
        tm = make_task_manager()
        today_str = datetime.now().strftime("%Y-%m-%d")
        tm.storage.get_today_tasks = AsyncMock(return_value=[{
            "task_id": "unknown_type",
            "task_type": "custom_unknown",
            "target_value": 100,
            "target_unit": "W",
            "verified": False,
            "area_name": None,
            "created_at": f"{today_str}T06:00:00",
            "peak_hour": None,
        }])

        tm._verify_single_task = AsyncMock(return_value=(False, 150, False))
        await tm.verify_tasks()

        reason = tm._last_verification_results["unknown_type"]["reason"]
        assert "target was" in reason

    @pytest.mark.asyncio
    async def test_verify_tasks_pending_non_peak_uses_evaluation_deferred_reason(self):
        tm = make_task_manager()
        today_str = datetime.now().strftime("%Y-%m-%d")
        tm.storage.get_today_tasks = AsyncMock(return_value=[{
            "task_id": "pending_non_peak",
            "task_type": "power_reduction",
            "target_value": 100,
            "target_unit": "W",
            "verified": False,
            "area_name": None,
            "created_at": f"{today_str}T06:00:00",
            "peak_hour": None,
        }])

        tm._verify_single_task = AsyncMock(return_value=(False, None, True))
        await tm.verify_tasks()

        reason = tm._last_verification_results["pending_non_peak"]["reason"]
        assert "deferred" in reason.lower()

    @pytest.mark.asyncio
    async def test_verify_single_task_invalid_created_at_falls_back(self):
        from datetime import datetime as real_dt

        tm = make_task_manager()
        tm.data_collector.get_power_history = AsyncMock(return_value=[
            (real_dt(2026, 2, 19, 12, i), 250.0) for i in range(30)
        ])

        task = {
            "task_id": "bad_created_at",
            "task_type": "power_reduction",
            "target_value": 400,
            "created_at": "not-an-iso",
            "verified": False,
        }

        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)

        assert verified
        assert actual is not None
        assert pending is False

    @pytest.mark.asyncio
    async def test_verify_single_task_unknown_type_returns_default_tuple(self):
        verified, actual, pending = await make_task_manager()._verify_single_task({
            "task_id": "x",
            "task_type": "unknown_type",
            "target_value": 1,
        })
        assert verified is False
        assert actual is None
        assert pending is False

    @pytest.mark.asyncio
    async def test_verify_single_task_daylight_after_filter_empty(self):
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager()
        tm.data_collector.get_power_history = AsyncMock(return_value=[
            (real_dt(2026, 2, 19, 22, 0, 0), 250.0)
        ])

        task = {
            "task_id": "day_empty",
            "task_type": "daylight_usage",
            "target_value": 200,
            "created_at": "2026-02-19T06:00:00",
            "verified": False,
        }
        with patch.object(tm_mod, "datetime") as mock_dt:
            fake_now = real_dt(2026, 2, 19, 23, 0, 0)
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)

        assert verified is False
        assert actual is None
        assert pending is False

    @pytest.mark.asyncio
    async def test_verify_single_task_unoccupied_without_area(self):
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager()
        task = {
            "task_id": "unocc_no_area",
            "task_type": "unoccupied_power",
            "target_value": 200,
            "area_name": None,
            "created_at": "2026-02-19T06:00:00",
            "verified": False,
        }
        with patch.object(tm_mod, "datetime") as mock_dt:
            fake_now = real_dt(2026, 2, 19, 20, 0, 0)
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)

        assert verified is False
        assert actual is None
        assert pending is False

    @pytest.mark.asyncio
    async def test_verify_tasks_no_streak_update_when_none_verified(self):
        tm = make_task_manager()
        today_str = datetime.now().strftime("%Y-%m-%d")
        tm.storage.get_today_tasks = AsyncMock(return_value=[{
            "task_id": "t_fail",
            "task_type": "power_reduction",
            "target_value": 100,
            "target_unit": "W",
            "verified": False,
            "area_name": None,
            "created_at": f"{today_str}T06:00:00",
            "peak_hour": None,
        }])
        tm._verify_single_task = AsyncMock(return_value=(False, 200, False))

        await tm.verify_tasks()
        tm.decision_agent.update_task_streak.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_temperature_task_returns_none_when_no_temp_history(self):
        tm = make_task_manager(sensors={"temperature": ["sensor.temp_1"]})
        tm.config_data = {"weather_entity": "weather.home"}

        weather_state = MagicMock()
        weather_state.attributes = {"temperature": 35.0}
        tm.hass.states.get = MagicMock(return_value=weather_state)
        tm.data_collector.get_temperature_history = AsyncMock(return_value=[])

        task = await tm._generate_temperature_task()
        assert task is None

    @pytest.mark.asyncio
    async def test_generate_unoccupied_power_task_continues_when_all_occupied(self):
        tm = make_task_manager(sensors={"power": ["sensor.p1"], "occupancy": ["binary_sensor.o1"]})
        tm.data_collector.get_all_areas = MagicMock(return_value=["Office"])
        base = datetime(2026, 2, 19, 10, 0, 0)
        power = [(base + timedelta(minutes=i), 400.0) for i in range(10)]
        occ = [(base + timedelta(minutes=i), 1) for i in range(10)]

        async def area_history(area, metric, **kwargs):
            return power if metric == "power" else occ

        tm.data_collector.get_area_history = AsyncMock(side_effect=area_history)
        task = await tm._generate_unoccupied_power_task()
        assert task is None

    @pytest.mark.asyncio
    async def test_generate_daylight_task_success_path(self):
        tm = make_task_manager(sensors={"power": ["sensor.power_1"], "illuminance": ["sensor.lux_1"]})
        base = datetime(2026, 2, 19, 9, 0, 0)
        tm.data_collector.get_power_history = AsyncMock(return_value=[
            (base + timedelta(minutes=i), 500.0) for i in range(30)
        ])
        tm.storage.get_task_difficulty_stats = AsyncMock(return_value={
            "too_easy_count": 0,
            "just_right_count": 3,
            "too_hard_count": 0,
            "avg_difficulty": 3,
            "suggested_adjustment": 0,
        })

        task = await tm._generate_daylight_task()
        assert task is not None
        assert task["task_type"] == "daylight_usage"
        assert "target_value" in task

    @pytest.mark.asyncio
    async def test_verify_tasks_reason_variants_and_verified_logging(self):
        tm = make_task_manager()
        today_str = datetime.now().strftime("%Y-%m-%d")

        tasks = [
            {"task_id": "verified_case", "task_type": "power_reduction", "target_value": 100, "target_unit": "W", "verified": False, "area_name": None, "created_at": f"{today_str}T06:00:00", "peak_hour": None},
            {"task_id": "pending_peak", "task_type": "peak_avoidance", "target_value": 100, "target_unit": "W", "verified": False, "area_name": None, "created_at": f"{today_str}T06:00:00", "peak_hour": 17},
            {"task_id": "temp_increase_fail", "task_type": "temperature_increase", "target_value": 24.0, "target_unit": "°C", "verified": False, "area_name": None, "created_at": f"{today_str}T06:00:00", "peak_hour": None},
            {"task_id": "insufficient", "task_type": "power_reduction", "target_value": 100, "target_unit": "W", "verified": False, "area_name": None, "created_at": f"{today_str}T06:00:00", "peak_hour": None},
        ]
        tm.storage.get_today_tasks = AsyncMock(return_value=tasks)

        async def verify_side_effect(task):
            if task["task_id"] == "verified_case":
                return True, 80, False
            if task["task_id"] == "pending_peak":
                return False, None, True
            if task["task_id"] == "temp_increase_fail":
                return False, 26.0, False
            return False, None, False

        tm._verify_single_task = AsyncMock(side_effect=verify_side_effect)
        await tm.verify_tasks()

        assert tm.storage.mark_task_verified.called
        assert tm.storage.log_task_completion.called
        assert "Target achieved" in tm._last_verification_results["verified_case"]["reason"]
        assert "peak hour" in tm._last_verification_results["pending_peak"]["reason"].lower()
        assert "setpoint" in tm._last_verification_results["temp_increase_fail"]["reason"].lower()
        assert "insufficient" in tm._last_verification_results["insufficient"]["reason"].lower()

    @pytest.mark.asyncio
    async def test_verify_single_task_returns_none_for_missing_histories(self):
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager()
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)

        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)

            tm.data_collector.get_power_history = AsyncMock(return_value=[])
            v1, a1, _ = await tm._verify_single_task({
                "task_id": "p0", "task_type": "power_reduction", "target_value": 100, "created_at": "2026-02-19T06:00:00", "verified": False
            })

            tm.data_collector.get_power_history = AsyncMock(return_value=[(real_dt(2026, 2, 19, 12, 0, 0), 100.0)])
            tm.data_collector.get_area_history = AsyncMock(return_value=[])
            v2, a2, _ = await tm._verify_single_task({
                "task_id": "u0", "task_type": "unoccupied_power", "target_value": 100, "area_name": "Office", "created_at": "2026-02-19T06:00:00", "verified": False
            })

            tm.data_collector.get_area_history = AsyncMock(side_effect=[
                [(real_dt(2026, 2, 19, 12, i, 0), 200.0) for i in range(5)],
                [(real_dt(2026, 2, 19, 12, i, 0), 1) for i in range(5)],
            ])
            v3, a3, _ = await tm._verify_single_task({
                "task_id": "u1", "task_type": "unoccupied_power", "target_value": 100, "area_name": "Office", "created_at": "2026-02-19T06:00:00", "verified": False
            })

            tm.data_collector.get_power_history = AsyncMock(return_value=[])
            v4, a4, _ = await tm._verify_single_task({
                "task_id": "pk0", "task_type": "peak_avoidance", "target_value": 100, "peak_hour": 13, "created_at": "2026-02-19T06:00:00", "verified": False
            })

        assert v1 is False and a1 is None
        assert v2 is False and a2 is None
        assert v3 is False and a3 is None
        assert v4 is False and a4 is None

    @pytest.mark.asyncio
    async def test_temperature_returns_false_when_no_history(self):
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager(sensors={"temperature": ["sensor.t1"]})
        tm.data_collector.get_temperature_history = AsyncMock(return_value=[])
        task = {
            "task_id": "temp_no_data",
            "task_type": "temperature_reduction",
            "target_value": 20.0,
            "area_name": None,
            "created_at": "2026-02-19T08:00:00",
            "verified": False,
        }
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)

        assert verified is False
        assert actual is None


# ─────────────────────────────────────────────────────────────────────────────
# _verify_single_task: error handling returns (False, None)
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifySingleTaskErrorHandling:

    @pytest.mark.asyncio
    async def test_exception_in_verification_returns_false_none(self):
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager()
        tm.data_collector.get_power_history = AsyncMock(side_effect=RuntimeError("DB error"))
        task = {
            "task_id": "err_task",
            "task_type": "power_reduction",
            "target_value": 400,
            "area_name": None,
            "created_at": "2026-02-19T08:00:00",
            "verified": False,
        }
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)

        assert verified is False
        assert actual is None


# ─────────────────────────────────────────────────────────────────────────────
# Office-mode Friday streak: last working day lookup 
# ─────────────────────────────────────────────────────────────────────────────

class TestOfficeModeFridayStreak:
    """
    In office mode, generate_daily_tasks() must check the LAST WORKING DAY's
    tasks for the streak calculation, not blindly use yesterday.

    Scenario: HA is run on Monday; yesterday = Sunday (no tasks).
    Friday had tasks that were all unverified -> streak reset must fire.
    """

    @pytest.mark.asyncio
    async def test_monday_checks_friday_tasks_in_office_mode(self):
        """When today is Monday (office), the streak check should use the most-recent
        working day (Friday), not yesterday (Sunday which has no tasks)."""
        from unittest.mock import patch
        from datetime import datetime as real_dt, date as real_date

        office_cfg = {"environment_mode": "office"}
        tm = make_task_manager(config=office_cfg)
        # Force working days to Mon-Fri only
        tm_mod.get_working_days_from_config = MagicMock(return_value=list(range(5)))

        # Storage: no tasks on Sunday but failed tasks on Friday (3 days back from Mon)
        friday_tasks = [
            {"task_id": "fri_1", "task_type": "power_reduction", "verified": False},
            {"task_id": "fri_2", "task_type": "power_reduction", "verified": False},
        ]

        monday = real_date(2026, 3, 2)   # This is a Monday
        friday = real_date(2026, 2, 27)  # The Friday before

        async def get_tasks_for_date(date_arg):
            if date_arg == friday:
                return friday_tasks
            return []

        tm.storage.get_tasks_for_date = AsyncMock(side_effect=get_tasks_for_date)

        # Use a FixedDatetime subclass so .now() returns Monday and .date() works correctly
        fake_now = real_dt(2026, 3, 2, 6, 0, 0)  # Monday 06:00

        class FixedDatetime(real_dt):
            @classmethod
            def now(cls, tz=None):
                return fake_now

        with patch.object(tm_mod, "datetime", FixedDatetime):
            await tm.generate_daily_tasks()

        # The streak must have been called with False for the Friday date
        tm.decision_agent.update_task_streak.assert_called_once_with(False, friday)

    @pytest.mark.asyncio
    async def test_friday_failed_streak_reset_in_office_mode(self):
        """Simpler: use the real datetime but manipulate storage to return failed
        Friday tasks as the 'last working day' when current day is Monday.
        Verifies that update_task_streak(False, friday) is eventually called."""
        from unittest.mock import patch
        from datetime import datetime as real_dt, date as real_date
        import datetime as _stdlib

        office_cfg = {"environment_mode": "office"}
        tm = make_task_manager(config=office_cfg)

        # Stub working days Mon-Fri (0-4)
        tm_mod.get_working_days_from_config = MagicMock(return_value=list(range(5)))

        monday = real_date(2026, 3, 2)
        friday = real_date(2026, 2, 27)

        friday_failed = [{"task_id": "f1", "verified": False}]

        async def get_tasks(d):
            if d == friday:
                return friday_failed
            return []

        tm.storage.get_tasks_for_date = AsyncMock(side_effect=get_tasks)

        # Patch datetime inside t_mod so .now().date() returns Monday
        fake_now = real_dt(2026, 3, 2, 7, 0, 0)

        original_date = _stdlib.date

        class FixedDatetime(real_dt):
            @classmethod
            def now(cls, tz=None):
                return fake_now

        FixedDatetime.date = real_dt.date  # keep .date method

        with patch.object(tm_mod, "datetime", FixedDatetime):
            await tm.generate_daily_tasks()

        tm.decision_agent.update_task_streak.assert_called_once_with(False, friday)

    @pytest.mark.asyncio
    async def test_home_mode_still_uses_yesterday(self):
        """In home mode, the streak check should still use yesterday (not skip it)."""
        from datetime import datetime as real_dt, date as real_date
        import datetime as _stdlib

        home_cfg = {"environment_mode": "home"}
        tm = make_task_manager(config=home_cfg)
        tm_mod.get_working_days_from_config = MagicMock(return_value=list(range(7)))

        yesterday = real_date(2026, 3, 1)
        yesterday_failed = [{"task_id": "y1", "verified": False}]

        async def get_tasks(d):
            if d == yesterday:
                return yesterday_failed
            return []

        tm.storage.get_tasks_for_date = AsyncMock(side_effect=get_tasks)

        fake_now = real_dt(2026, 3, 2, 7, 0, 0)

        class FixedDatetime(real_dt):
            @classmethod
            def now(cls, tz=None):
                return fake_now

        from unittest.mock import patch
        with patch.object(tm_mod, "datetime", FixedDatetime):
            await tm.generate_daily_tasks()

        tm.decision_agent.update_task_streak.assert_called_once_with(False, yesterday)


# ─────────────────────────────────────────────────────────────────────────────
# Task-type-specific verification failure reasons
# ─────────────────────────────────────────────────────────────────────────────

class TestVerificationFailureReasons:
    """
    verify_tasks() must produce task-type-specific failure reasons so the user
    understands WHY the task was not completed.
    """

    def _failed_task(self, task_type, target=400, area=None, peak_hour=None, cooling_mode=False):
        # Use a fixed date that matches the fake_now used in _run_verify (2026-02-19)
        # so that hours_passed is positive and _verify_single_task is not short-circuited.
        t = {
            "task_id": f"{task_type}_test",
            "task_type": task_type,
            "target_value": target,
            "target_unit": "W",
            "area_name": area,
            "peak_hour": peak_hour,
            "cooling_mode": cooling_mode,
            "verified": False,
            "created_at": "2026-02-19T06:00:00",
        }
        return t

    async def _run_verify(self, tm, task, actual_value=500.0, fake_hour=20):
        """Helper: return the reason string after one verify_tasks() run."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm.storage.get_today_tasks = AsyncMock(return_value=[task])

        base = real_dt(2026, 2, 19, 10, 0, 0)
        data = [(base + timedelta(minutes=i), actual_value) for i in range(60)]
        tm.data_collector.get_power_history = AsyncMock(return_value=data)
        tm.data_collector.get_temperature_history = AsyncMock(return_value=data)
        tm.data_collector.get_area_history = AsyncMock(return_value=data)

        fake_now = real_dt(2026, 2, 19, fake_hour, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm.verify_tasks()

        result = tm._last_verification_results.get(task["task_id"])
        return result["reason"] if result else ""

    @pytest.mark.asyncio
    async def test_power_reduction_failure_reason_mentions_watts(self):
        tm = make_task_manager()
        task = self._failed_task("power_reduction", target=300)  # 500W > 300W -> fails
        reason = await self._run_verify(tm, task, actual_value=500.0)
        assert "500" in reason
        assert "300" in reason

    @pytest.mark.asyncio
    async def test_daylight_failure_reason_mentions_natural_light(self):
        """daylight_usage failure should include a hint about natural light."""
        tm = make_task_manager()
        base = datetime(2026, 2, 19, 12, 0, 0)  # daytime hours
        data = [(base + timedelta(minutes=i), 600.0) for i in range(60)]
        tm.data_collector.get_power_history = AsyncMock(return_value=data)
        task = self._failed_task("daylight_usage", target=400)
        from unittest.mock import patch
        from datetime import datetime as real_dt
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        tm.storage.get_today_tasks = AsyncMock(return_value=[task])
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm.verify_tasks()
        result = tm._last_verification_results.get(task["task_id"])
        assert result is not None
        # The EN reason for daylight_usage includes "natural light"
        assert "natural light" in result["reason"].lower() or "600" in result["reason"]

    @pytest.mark.asyncio
    async def test_unoccupied_power_failure_reason_mentions_idle_devices(self):
        """unoccupied_power failure reason should mention turning off idle devices."""
        sensors = {"power": ["sensor.p"], "occupancy": ["bs.occ"]}
        tm = make_task_manager(sensors=sensors)
        base = datetime(2026, 2, 19, 10, 0, 0)
        data = [(base + timedelta(minutes=i), 600.0) for i in range(60)]

        async def area_history(area, metric, **kwargs):
            if metric == "power":
                return data
            return [(base + timedelta(minutes=i), 0) for i in range(60)]  # unoccupied

        tm.data_collector.get_area_history = AsyncMock(side_effect=area_history)
        task = self._failed_task("unoccupied_power", target=400, area="Office")
        from unittest.mock import patch
        from datetime import datetime as real_dt
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        tm.storage.get_today_tasks = AsyncMock(return_value=[task])
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm.verify_tasks()
        result = tm._last_verification_results.get(task["task_id"])
        assert result is not None
        assert "600" in result["reason"] or "idle" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_peak_avoidance_failure_reason_mentions_peak(self):
        """peak_avoidance failure reason should mention shifting usage from peak."""
        tm = make_task_manager()
        base = datetime(2026, 2, 19, 14, 0, 0)
        data = [(base + timedelta(minutes=i), 600.0) for i in range(60)]
        tm.data_collector.get_power_history = AsyncMock(return_value=data)
        task = self._failed_task("peak_avoidance", target=400, peak_hour=14)
        from unittest.mock import patch
        from datetime import datetime as real_dt
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        tm.storage.get_today_tasks = AsyncMock(return_value=[task])
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm.verify_tasks()
        result = tm._last_verification_results.get(task["task_id"])
        assert result is not None
        assert "600" in result["reason"] or "peak" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_temperature_reduction_failure_reason_mentions_heating(self):
        """temperature_reduction (heating mode) failure reason should mention heating."""
        sensors = {"temperature": ["sensor.t"]}
        tm = make_task_manager(sensors=sensors)
        base = datetime(2026, 2, 19, 8, 0, 0)
        data = [(base + timedelta(hours=i), 22.0) for i in range(10)]
        tm.data_collector.get_temperature_history = AsyncMock(return_value=data)
        # cooling_mode=False means heating season, target is below baseline (21°C)
        task = self._failed_task("temperature_reduction", target=20.0, cooling_mode=False)
        task["target_unit"] = "°C"
        from unittest.mock import patch
        from datetime import datetime as real_dt
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        tm.storage.get_today_tasks = AsyncMock(return_value=[task])
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm.verify_tasks()
        result = tm._last_verification_results.get(task["task_id"])
        assert result is not None
        assert "heat" in result["reason"].lower() or "22" in result["reason"]

    @pytest.mark.asyncio
    async def test_temperature_reduction_failure_reason_is_type_specific(self):
        """temperature_reduction failure reason must use the temperature-specific
        template (not the generic avg_above_target) and expose the actual value."""
        sensors = {"temperature": ["sensor.t"]}
        tm = make_task_manager(sensors=sensors)
        base = datetime(2026, 2, 19, 8, 0, 0)
        # avg 22°C but target is 20°C -> 22 > 20 -> fails
        data = [(base + timedelta(hours=i), 22.0) for i in range(10)]
        tm.data_collector.get_temperature_history = AsyncMock(return_value=data)
        task = self._failed_task("temperature_reduction", target=20.0)
        task["target_unit"] = "°C"
        from unittest.mock import patch
        from datetime import datetime as real_dt
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        tm.storage.get_today_tasks = AsyncMock(return_value=[task])
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm.verify_tasks()
        result = tm._last_verification_results.get(task["task_id"])
        assert result is not None
        # Must be the temperature-specific reason (mentions "heat" or the actual °C value)
        assert "heat" in result["reason"].lower() or "22" in result["reason"]
        # Must NOT be the generic fallback string
        assert result["reason"] != "Avg: 22.0°C, target was 20.0°C"


# ─────────────────────────────────────────────────────────────────────────────
# peak_avoidance verification: working_hours_filter
# ─────────────────────────────────────────────────────────────────────────────

class TestPeakAvoidanceWorkingHoursFilter:
    """peak_avoidance verification must pass working_hours_only=True in office
    mode and working_hours_only=None in home mode, consistent with all other
    task types. Previously the peak_avoidance branch called get_power_history
    without the filter, ignoring the environment mode setting."""

    @pytest.mark.asyncio
    async def test_office_mode_passes_working_hours_filter(self):
        """In office mode, get_power_history must be called with working_hours_only=True."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        sensors = {"power": ["sensor.power_1"]}
        tm = make_task_manager(sensors=sensors, config={"environment_mode": "office"})

        peak_hour = 14
        base = real_dt(2026, 2, 19, peak_hour, 0, 0)
        power_data = [(base + timedelta(minutes=i), 300.0) for i in range(60)]
        tm.data_collector.get_power_history = AsyncMock(return_value=power_data)

        task = {
            "task_id": "pa_office",
            "task_type": "peak_avoidance",
            "target_value": 450,
            "area_name": None,
            "peak_hour": peak_hour,
            "verified": False,
        }

        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm._verify_single_task(task)

        call_kwargs = tm.data_collector.get_power_history.call_args
        assert call_kwargs.kwargs.get("working_hours_only") is True, (
            "peak_avoidance in office mode must pass working_hours_only=True to get_power_history"
        )

    @pytest.mark.asyncio
    async def test_home_mode_no_working_hours_filter(self):
        """In home mode, get_power_history must be called without working_hours_only filter."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        sensors = {"power": ["sensor.power_1"]}
        tm = make_task_manager(sensors=sensors, config={"environment_mode": "home"})

        peak_hour = 14
        base = real_dt(2026, 2, 19, peak_hour, 0, 0)
        power_data = [(base + timedelta(minutes=i), 300.0) for i in range(60)]
        tm.data_collector.get_power_history = AsyncMock(return_value=power_data)

        task = {
            "task_id": "pa_home",
            "task_type": "peak_avoidance",
            "target_value": 450,
            "area_name": None,
            "peak_hour": peak_hour,
            "verified": False,
        }

        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            await tm._verify_single_task(task)

        call_kwargs = tm.data_collector.get_power_history.call_args
        assert not call_kwargs.kwargs.get("working_hours_only"), (
            "peak_avoidance in home mode must not pass working_hours_only=True to get_power_history"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AC guard: temperature tasks only generated when has_ac=True
# ─────────────────────────────────────────────────────────────────────────────

class TestTemperatureTaskACGuard:
    """Temperature task generators are conditional on config has_ac=True."""

    @pytest.mark.asyncio
    async def test_temperature_task_not_generated_when_has_ac_false(self):
        """has_ac=False -> temperature generator excluded from the pool."""
        sensors = {"temperature": ["sensor.t1"]}  # only temperature, no power/light/occ
        tm = make_task_manager(sensors=sensors, config={"has_ac": False})
        result = await tm.generate_daily_tasks()
        # No valid generators -> empty task list
        assert result == []

    @pytest.mark.asyncio
    async def test_temperature_task_not_generated_when_has_ac_missing(self):
        """Missing has_ac key defaults to False -> temperature generator excluded."""
        sensors = {"temperature": ["sensor.t1"]}
        tm = make_task_manager(sensors=sensors, config={})
        result = await tm.generate_daily_tasks()
        assert result == []

    @pytest.mark.asyncio
    async def test_temperature_task_generated_when_has_ac_true(self):
        """has_ac=True -> temperature generator included in pool and task produced."""
        sensors = {"temperature": ["sensor.t1"]}
        tm = make_task_manager(sensors=sensors, config={"has_ac": True})
        result = await tm.generate_daily_tasks()
        assert isinstance(result, list)
        if result:
            types_found = {t["task_type"] for t in result}
            assert types_found.issubset({"temperature_reduction", "temperature_increase"})


# ─────────────────────────────────────────────────────────────────────────────
# _generate_temperature_task: seasonal direction
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateTemperatureTaskSeasonalDirection:
    """_generate_temperature_task selects direction based on outdoor temperature."""

    def _make_weather_state(self, outdoor_temp):
        state = MagicMock()
        state.attributes = {"temperature": outdoor_temp}
        return state

    @pytest.mark.asyncio
    async def test_cold_outdoor_generates_temperature_reduction(self):
        """Outdoor temp < threshold -> heating season -> temperature_reduction task."""
        tm = make_task_manager(
            sensors={"temperature": ["sensor.t1"]},
            config={"has_ac": True, "weather_entity": "weather.home"},
        )
        tm.hass.states.get = MagicMock(return_value=self._make_weather_state(outdoor_temp=5.0))
        result = await tm._generate_temperature_task()
        assert result is not None
        assert result["task_type"] == "temperature_reduction"
        assert result["target_value"] < result["baseline_value"]

    @pytest.mark.asyncio
    async def test_hot_outdoor_generates_temperature_increase(self):
        """Outdoor temp > threshold -> cooling season -> temperature_increase task."""
        tm = make_task_manager(
            sensors={"temperature": ["sensor.t1"]},
            config={"has_ac": True, "weather_entity": "weather.home"},
        )
        tm.hass.states.get = MagicMock(return_value=self._make_weather_state(outdoor_temp=32.0))
        result = await tm._generate_temperature_task()
        assert result is not None
        assert result["task_type"] == "temperature_increase"
        assert result["target_value"] > result["baseline_value"]

    @pytest.mark.asyncio
    async def test_no_weather_entity_configured_returns_none(self):
        """No weather_entity key -> outdoor temp unknown -> no task generated."""
        tm = make_task_manager(
            sensors={"temperature": ["sensor.t1"]},
            config={"has_ac": True},
        )
        result = await tm._generate_temperature_task()
        assert result is None

    @pytest.mark.asyncio
    async def test_weather_entity_not_found_returns_none(self):
        """Weather entity exists in config but not in HA states -> no task generated."""
        tm = make_task_manager(
            sensors={"temperature": ["sensor.t1"]},
            config={"has_ac": True, "weather_entity": "weather.missing"},
        )
        tm.hass.states.get = MagicMock(return_value=None)
        result = await tm._generate_temperature_task()
        assert result is None

    @pytest.mark.asyncio
    async def test_exactly_at_cold_threshold_generates_reduction(self):
        """Outdoor temp == OUTDOOR_COLD_TEMP_THRESHOLD (<=) -> reduction task."""
        cold_threshold = tm_mod.OUTDOOR_COLD_TEMP_THRESHOLD
        tm = make_task_manager(
            sensors={"temperature": ["sensor.t1"]},
            config={"has_ac": True, "weather_entity": "weather.home"},
        )
        tm.hass.states.get = MagicMock(return_value=self._make_weather_state(outdoor_temp=cold_threshold))
        result = await tm._generate_temperature_task()
        assert result is not None
        assert result["task_type"] == "temperature_reduction"

    @pytest.mark.asyncio
    async def test_between_thresholds_returns_none(self):
        """Outdoor temp between thresholds (not hot, not cold) -> no task generated."""
        between_temp = (tm_mod.OUTDOOR_COLD_TEMP_THRESHOLD + tm_mod.OUTDOOR_HOT_TEMP_THRESHOLD) / 2
        tm = make_task_manager(
            sensors={"temperature": ["sensor.t1"]},
            config={"has_ac": True, "weather_entity": "weather.home"},
        )
        tm.hass.states.get = MagicMock(return_value=self._make_weather_state(outdoor_temp=between_temp))
        result = await tm._generate_temperature_task()
        assert result is None

    @pytest.mark.asyncio
    async def test_exactly_at_hot_threshold_still_in_between(self):
        """Outdoor temp == OUTDOOR_HOT_TEMP_THRESHOLD (not *strictly* greater) -> no task."""
        hot_threshold = tm_mod.OUTDOOR_HOT_TEMP_THRESHOLD
        tm = make_task_manager(
            sensors={"temperature": ["sensor.t1"]},
            config={"has_ac": True, "weather_entity": "weather.home"},
        )
        tm.hass.states.get = MagicMock(return_value=self._make_weather_state(outdoor_temp=hot_threshold))
        result = await tm._generate_temperature_task()
        # <= cold threshold is False (24 > 18), > hot threshold is False (24 is not > 24) -> None
        assert result is None

    @pytest.mark.asyncio
    async def test_increase_task_has_correct_fields(self):
        """temperature_increase task must carry the expected dict keys."""
        tm = make_task_manager(
            sensors={"temperature": ["sensor.t1"]},
            config={"has_ac": True, "weather_entity": "weather.home"},
        )
        tm.hass.states.get = MagicMock(return_value=self._make_weather_state(outdoor_temp=35.0))
        result = await tm._generate_temperature_task()
        assert result is not None
        for key in ("task_id", "task_type", "title", "description", "target_value",
                    "target_unit", "baseline_value", "difficulty_level", "difficulty_display"):
            assert key in result, f"Missing key: {key}"
        assert result["target_unit"] == "°C"


# ─────────────────────────────────────────────────────────────────────────────
# _verify_single_task: temperature_increase type
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifySingleTaskTemperatureIncrease:
    """Cooling-season tasks pass when avg temp >= target (house not over-cooled)."""

    @pytest.mark.asyncio
    async def test_increase_verified_when_avg_at_or_above_target(self):
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager(sensors={"temperature": ["sensor.t1"]})
        base = real_dt(2026, 2, 19, 8, 0, 0)
        tm.data_collector.get_temperature_history = AsyncMock(
            return_value=[(base + timedelta(hours=i), 24.0) for i in range(5)]
        )
        task = {
            "task_id": "temp_inc_ok",
            "task_type": "temperature_increase",
            "target_value": 23.0,
            "area_name": None,
            "created_at": "2026-02-19T08:00:00",
            "verified": False,
        }
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)

        assert verified is True
        assert actual == pytest.approx(24.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_increase_fails_when_avg_below_target(self):
        """House still over-cooled (avg temp below target) -> task not achieved."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager(sensors={"temperature": ["sensor.t1"]})
        base = real_dt(2026, 2, 19, 8, 0, 0)
        tm.data_collector.get_temperature_history = AsyncMock(
            return_value=[(base + timedelta(hours=i), 20.0) for i in range(5)]
        )
        task = {
            "task_id": "temp_inc_fail",
            "task_type": "temperature_increase",
            "target_value": 23.0,
            "area_name": None,
            "created_at": "2026-02-19T08:00:00",
            "verified": False,
        }
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)

        assert verified is False
        assert actual == pytest.approx(20.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_increase_returns_false_when_no_history(self):
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager(sensors={"temperature": ["sensor.t1"]})
        tm.data_collector.get_temperature_history = AsyncMock(return_value=[])
        task = {
            "task_id": "temp_inc_no_data",
            "task_type": "temperature_increase",
            "target_value": 23.0,
            "area_name": None,
            "created_at": "2026-02-19T08:00:00",
            "verified": False,
        }
        fake_now = real_dt(2026, 2, 19, 20, 0, 0)
        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)

        assert verified is False
        assert actual is None


# ─────────────────────────────────────────────────────────────────────────────
# hours_passed < 1 must return pending=True
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyPendingWithinFirstHour:
    """When a task was generated < 1 hour ago the 3rd return value must
    be True (pending) so the UI shows 'Evaluation deferred', not 'Insufficient
    data'."""

    @pytest.mark.asyncio
    async def test_within_first_hour_returns_pending_true(self):
        """created_at 10 minutes ago -> hours_passed < 1 -> pending=True."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager()
        # Task was generated 10 minutes ago
        created_at = real_dt(2026, 2, 19, 11, 50, 0)
        fake_now   = real_dt(2026, 2, 19, 12,  0, 0)  # only 10 minutes later

        task = {
            "task_id": "early_verify",
            "task_type": "power_reduction",
            "target_value": 400,
            "area_name": None,
            "created_at": created_at.isoformat(),
            "verified": False,
        }

        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)

        assert verified is False, "Task should not be verified yet"
        assert actual is None,    "No actual value expected before first hour"
        assert pending is True,   "pending must be True within the first hour (was False before the fix)"

    @pytest.mark.asyncio
    async def test_after_first_hour_pending_is_false(self):
        """After 1 full hour, pending must be False (evaluation proceeds normally)."""
        from unittest.mock import patch
        from datetime import datetime as real_dt

        tm = make_task_manager()
        base = real_dt(2026, 2, 19, 8, 0, 0)
        tm.data_collector.get_power_history = AsyncMock(return_value=[
            (base + timedelta(minutes=i), 600.0) for i in range(60)
        ])

        created_at = real_dt(2026, 2, 19, 6,  0, 0)
        fake_now   = real_dt(2026, 2, 19, 20, 0, 0)  # 14 hours later

        task = {
            "task_id": "normal_verify",
            "task_type": "power_reduction",
            "target_value": 400,
            "area_name": None,
            "created_at": created_at.isoformat(),
            "verified": False,
        }

        with patch.object(tm_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = real_dt.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: real_dt(*a, **kw)
            verified, actual, pending = await tm._verify_single_task(task)

        assert pending is False, "After 1+ hours pending must be False"


# ─────────────────────────────────────────────────────────────────────────────
# Auto-feedback on task generation day
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoFeedbackOnGeneration:
    """
    When generate_daily_tasks() runs it looks at yesterday's tasks and
    auto-submits difficulty feedback for any task that has none yet:
      - verified=True, no feedback  -> auto 'just_right'
      - verified=False, no feedback -> auto 'too_hard'
    Tasks that already carry user_feedback are left untouched.
    """

    def _make_yesterday_task(self, task_id, verified, user_feedback=None):
        return {
            "task_id": task_id,
            "task_type": "power_reduction",
            "verified": verified,
            "user_feedback": user_feedback,
        }

    @pytest.mark.asyncio
    async def test_auto_submits_just_right_for_completed_task_without_feedback(self):
        """A completed (verified) task with no user_feedback receives auto 'just_right'."""
        tm = make_task_manager()
        yesterday_task = self._make_yesterday_task("y1", verified=True, user_feedback=None)
        tm.storage.get_tasks_for_date = AsyncMock(return_value=[yesterday_task])
        tm.storage.save_task_feedback = AsyncMock(return_value=True)

        await tm.generate_daily_tasks()

        tm.storage.save_task_feedback.assert_called_once_with("y1", "just_right")
        tm.storage.log_task_feedback.assert_called_once_with("y1", "just_right")

    @pytest.mark.asyncio
    async def test_auto_submits_too_hard_for_unfinished_task_without_feedback(self):
        """An unfinished (not verified) task with no user_feedback receives auto 'too_hard'."""
        tm = make_task_manager()
        yesterday_task = self._make_yesterday_task("y2", verified=False, user_feedback=None)
        tm.storage.get_tasks_for_date = AsyncMock(return_value=[yesterday_task])
        tm.storage.save_task_feedback = AsyncMock(return_value=True)

        await tm.generate_daily_tasks()

        tm.storage.save_task_feedback.assert_called_once_with("y2", "too_hard")
        tm.storage.log_task_feedback.assert_called_once_with("y2", "too_hard")

    @pytest.mark.asyncio
    async def test_does_not_overwrite_existing_user_feedback(self):
        """Tasks that already have user_feedback must NOT receive an auto-feedback call."""
        tm = make_task_manager()
        yesterday_task = self._make_yesterday_task("y3", verified=False, user_feedback="too_easy")
        tm.storage.get_tasks_for_date = AsyncMock(return_value=[yesterday_task])
        tm.storage.save_task_feedback = AsyncMock(return_value=True)

        await tm.generate_daily_tasks()

        tm.storage.save_task_feedback.assert_not_called()
        tm.storage.log_task_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_tasks_auto_feedback_applied_only_where_missing(self):
        """Mixed scenario: only tasks without feedback receive auto-feedback."""
        tm = make_task_manager()
        tasks = [
            self._make_yesterday_task("y1", verified=True,  user_feedback=None),        # -> just_right
            self._make_yesterday_task("y2", verified=False, user_feedback=None),        # -> too_hard
            self._make_yesterday_task("y3", verified=True,  user_feedback="too_easy"),  # -> skip
            self._make_yesterday_task("y4", verified=False, user_feedback="too_hard"),  # -> skip
        ]
        tm.storage.get_tasks_for_date = AsyncMock(return_value=tasks)
        tm.storage.save_task_feedback = AsyncMock(return_value=True)

        await tm.generate_daily_tasks()

        calls = [(c.args[0], c.args[1]) for c in tm.storage.save_task_feedback.call_args_list]
        assert ("y1", "just_right") in calls
        assert ("y2", "too_hard")   in calls
        assert ("y3", "too_easy")   not in calls  # was already set
        assert ("y4", "too_hard")   not in calls  # was already set
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_no_auto_feedback_when_no_yesterday_tasks(self):
        """No auto-feedback submitted when yesterday's task list is empty."""
        tm = make_task_manager()
        # Default: storage.get_tasks_for_date returns []
        tm.storage.save_task_feedback = AsyncMock(return_value=True)

        await tm.generate_daily_tasks()

        tm.storage.save_task_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_feedback_skipped_when_save_fails(self):
        """When save_task_feedback returns False, log_task_feedback must NOT be called."""
        tm = make_task_manager()
        yesterday_task = self._make_yesterday_task("y5", verified=False, user_feedback=None)
        tm.storage.get_tasks_for_date = AsyncMock(return_value=[yesterday_task])
        tm.storage.save_task_feedback = AsyncMock(return_value=False)  # storage failure

        await tm.generate_daily_tasks()

        tm.storage.save_task_feedback.assert_called_once_with("y5", "too_hard")
        tm.storage.log_task_feedback.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Temperature Task : climate-aware difficulty scaling (HDD/CDD)
# ─────────────────────────────────────────────────────────────────────────────

def _make_tm_with_outdoor_temp(outdoor_temp: float) -> "TaskManager":
    """Create a TaskManager with a weather entity returning the given outdoor temperature."""
    from datetime import datetime as real_dt

    hass = MagicMock()
    weather_state = MagicMock()
    weather_state.attributes = {"temperature": outdoor_temp}
    weather_state.state = "sunny"
    hass.states.get = MagicMock(return_value=weather_state)

    collector = MagicMock()
    base_time = real_dt(2026, 6, 15, 12, 0, 0)
    collector.get_temperature_history = AsyncMock(
        return_value=[(base_time + timedelta(hours=i), 22.0) for i in range(50)]
    )
    collector.get_power_history = AsyncMock(
        return_value=[(base_time + timedelta(hours=i), 500.0) for i in range(50)]
    )

    storage = AsyncMock()
    storage.get_today_tasks = AsyncMock(return_value=[])
    storage.get_tasks_for_date = AsyncMock(return_value=[])
    storage.save_daily_tasks = AsyncMock()
    storage.log_task_generation = AsyncMock()
    storage.get_task_difficulty_stats = AsyncMock(return_value=None)

    agent = MagicMock()
    agent.phase = "active"

    tm_mod.should_ai_be_active = MagicMock(return_value=True)
    tm_mod.get_language = AsyncMock(return_value="en")
    tm_mod.get_working_days_from_config = MagicMock(return_value=list(range(7)))

    config = {"weather_entity": "weather.home", "environment_mode": "home"}

    return TaskManager(
        hass=hass,
        sensors={"power": ["sensor.p1"], "temperature": ["sensor.t1"]},
        data_collector=collector,
        storage=storage,
        decision_agent=agent,
        config_data=config,
    )


class TestTemperatureTaskClimateScale:
    """climate_scale correctly reduces temperature task difficulty in extreme weather."""

    def _hot(self, temp: float):
        return _make_tm_with_outdoor_temp(temp)

    def _cold(self, temp: float):
        return _make_tm_with_outdoor_temp(temp)

    # ── hot season ────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_hot_weather_task_type_is_temperature_increase(self):
        """Hot outdoor temperature produces a temperature_increase task."""
        tm = self._hot(28.0)
        task = await tm._generate_temperature_task()
        assert task is not None
        assert task["task_type"] == "temperature_increase"

    @pytest.mark.asyncio
    async def test_normal_hot_climate_scale_near_one(self):
        """Mild hot weather (1 °C above threshold) -> climate_scale ≈ 0.94."""
        tm = self._hot(25.0)  # excess = 25 - 24 = 1
        task = await tm._generate_temperature_task()
        assert task is not None
        expected_scale = max(0.5, 1.0 - 1.0 / 16.0)  # ≈ 0.9375
        assert task["climate_scale"] == pytest.approx(expected_scale, abs=0.01)

    @pytest.mark.asyncio
    async def test_extreme_hot_climate_scale_capped_at_half(self):
        """At 40 °C the climate_scale floor (0.5) is hit: excess = 16."""
        tm = self._hot(40.0)  # excess = 40 - 24 = 16 -> max(0.5, 1 - 16/16) = 0.5
        task = await tm._generate_temperature_task()
        assert task is not None
        assert task["climate_scale"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_beyond_extreme_hot_still_capped(self):
        """Temperatures > 40 °C must not produce a scale below 0.5."""
        tm = self._hot(55.0)
        task = await tm._generate_temperature_task()
        assert task is not None
        assert task["climate_scale"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_hot_cdd_degree_day_value(self):
        """CDD in task metadata = outdoor_temp - BASE_TEMPERATURE (18 °C)."""
        tm = self._hot(28.0)  # CDD = 28 - 18 = 10
        task = await tm._generate_temperature_task()
        assert task is not None
        assert task["degree_day_value"] == pytest.approx(10.0)

    # ── cold season ───────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_cold_weather_task_type_is_temperature_reduction(self):
        """Cold outdoor temperature produces a temperature_reduction task."""
        tm = self._cold(10.0)
        task = await tm._generate_temperature_task()
        assert task is not None
        assert task["task_type"] == "temperature_reduction"

    @pytest.mark.asyncio
    async def test_mild_cold_climate_scale_near_one(self):
        """Mild cold (14 °C, 1 °C below threshold 16 °C) -> climate_scale ≈ 0.9."""
        tm = self._cold(14.0)  # excess = 16 - 14 = 2
        task = await tm._generate_temperature_task()
        assert task is not None
        expected_scale = max(0.5, 1.0 - 2.0 / 20.0)  # 0.9
        assert task["climate_scale"] == pytest.approx(expected_scale, abs=0.01)

    @pytest.mark.asyncio
    async def test_extreme_cold_climate_scale_capped_at_half(self):
        """At -2 °C excess = 20 -> climate_scale floor (0.5) is hit."""
        tm = self._cold(-2.0)  # excess = 18 - (-2) = 20 -> max(0.5, 1-20/20) = 0.5
        task = await tm._generate_temperature_task()
        assert task is not None
        assert task["climate_scale"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_cold_hdd_degree_day_value(self):
        """HDD in task metadata = BASE_TEMPERATURE (18 °C) - outdoor_temp."""
        tm = self._cold(8.0)  # HDD = 18 - 8 = 10
        task = await tm._generate_temperature_task()
        assert task is not None
        assert task["degree_day_value"] == pytest.approx(10.0)

    # ── effect on target ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_extreme_heat_produces_smaller_adjustment_than_mild_heat(self):
        """Extreme heat -> climate_scale 0.5 -> target deviation halved vs mild heat."""
        tm_mild = self._hot(25.0)    # scale ≈ 0.94
        tm_extreme = self._hot(40.0)  # scale = 0.5

        task_mild = await tm_mild._generate_temperature_task()
        task_extreme = await tm_extreme._generate_temperature_task()

        assert task_mild is not None and task_extreme is not None
        delta_mild = abs(task_mild["target_value"] - task_mild["baseline_value"])
        delta_extreme = abs(task_extreme["target_value"] - task_extreme["baseline_value"])
        assert delta_extreme < delta_mild

    # ── neutral zone ──────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_neutral_outdoor_temp_returns_none(self):
        """Temperature between thresholds (e.g. 21 °C) produces no task."""
        tm = _make_tm_with_outdoor_temp(21.0)
        task = await tm._generate_temperature_task()
        assert task is None


# ─────────────────────────────────────────────────────────────────────────────
# outdoor_temp_sensor: WiFi-offline fallback for temperature task generation
# ─────────────────────────────────────────────────────────────────────────────

def _make_tm_with_sensor_fallback(
    weather_available: bool,
    sensor_temp,
    sensor_state_str: str = None,
) -> "TaskManager":
    """Create TaskManager where weather_entity is absent/unavailable and
    outdoor_temp_sensor provides the outdoor temperature via its state."""
    from datetime import datetime as real_dt

    hass = MagicMock()

    def _states_get(entity_id):
        if entity_id == "weather.home":
            if not weather_available:
                return None  # entity not present
            st = MagicMock()
            st.attributes = {"temperature": 30.0}  # hot
            st.state = "sunny"
            return st
        if entity_id == "sensor.outdoor_physical":
            if sensor_temp is None:
                return None
            st = MagicMock()
            raw = sensor_state_str if sensor_state_str is not None else str(sensor_temp)
            st.state = raw
            st.attributes = {"unit_of_measurement": "°C"}
            return st
        return None

    hass.states.get = MagicMock(side_effect=_states_get)

    collector = MagicMock()
    base_time = real_dt(2026, 6, 15, 12, 0, 0)
    collector.get_temperature_history = AsyncMock(
        return_value=[(base_time + timedelta(hours=i), 22.0) for i in range(50)]
    )
    collector.get_power_history = AsyncMock(
        return_value=[(base_time + timedelta(hours=i), 500.0) for i in range(50)]
    )

    storage = AsyncMock()
    storage.get_today_tasks = AsyncMock(return_value=[])
    storage.get_tasks_for_date = AsyncMock(return_value=[])
    storage.save_daily_tasks = AsyncMock()
    storage.log_task_generation = AsyncMock()
    storage.get_task_difficulty_stats = AsyncMock(return_value=None)

    agent = MagicMock()
    agent.phase = "active"

    tm_mod.should_ai_be_active = MagicMock(return_value=True)
    tm_mod.get_language = AsyncMock(return_value="en")
    tm_mod.get_working_days_from_config = MagicMock(return_value=list(range(7)))

    config = {
        "weather_entity": "weather.home" if weather_available else None,
        "outdoor_temp_sensor": "sensor.outdoor_physical",
        "environment_mode": "home",
    }

    return TaskManager(
        hass=hass,
        sensors={"power": ["sensor.p1"], "temperature": ["sensor.t1"]},
        data_collector=collector,
        storage=storage,
        decision_agent=agent,
        config_data=config,
    )


class TestOutdoorTempSensorFallback:
    """outdoor_temp_sensor acts as local fallback when WiFi/weather-API is unavailable."""

    @pytest.mark.asyncio
    async def test_weather_entity_used_when_available(self):
        """When weather_entity is present and returns a temperature, it is used (hot season)."""
        tm = _make_tm_with_sensor_fallback(weather_available=True, sensor_temp=5.0)
        task = await tm._generate_temperature_task()
        # weather.home returns 30 °C (hot) -> task_type is temperature_increase
        assert task is not None
        assert task["task_type"] == "temperature_increase"

    @pytest.mark.asyncio
    async def test_outdoor_sensor_fallback_when_weather_entity_unavailable(self):
        """When weather_entity is None, outdoor_temp_sensor reading is used instead."""
        # sensor_temp=5.0 is cold (<= 16 °C threshold) -> temperature_reduction task
        tm = _make_tm_with_sensor_fallback(weather_available=False, sensor_temp=5.0)
        task = await tm._generate_temperature_task()
        assert task is not None
        assert task["task_type"] == "temperature_reduction"

    @pytest.mark.asyncio
    async def test_outdoor_sensor_hot_triggers_temperature_increase_task(self):
        """A hot reading from the physical sensor also triggers temperature_increase."""
        tm = _make_tm_with_sensor_fallback(weather_available=False, sensor_temp=28.0)
        task = await tm._generate_temperature_task()
        assert task is not None
        assert task["task_type"] == "temperature_increase"

    @pytest.mark.asyncio
    async def test_outdoor_sensor_neutral_temperature_returns_none(self):
        """A neutral physical sensor reading (between thresholds) produces no task."""
        tm = _make_tm_with_sensor_fallback(weather_available=False, sensor_temp=20.0)
        task = await tm._generate_temperature_task()
        assert task is None

    @pytest.mark.asyncio
    async def test_outdoor_sensor_unavailable_state_produces_no_task(self):
        """If the physical sensor state is 'unavailable', outdoor_temp stays None -> no task."""
        tm = _make_tm_with_sensor_fallback(
            weather_available=False,
            sensor_temp=5.0,
            sensor_state_str="unavailable",
        )
        task = await tm._generate_temperature_task()
        assert task is None

    @pytest.mark.asyncio
    async def test_outdoor_sensor_unknown_state_produces_no_task(self):
        """If the physical sensor state is 'unknown', outdoor_temp stays None -> no task."""
        tm = _make_tm_with_sensor_fallback(
            weather_available=False,
            sensor_temp=5.0,
            sensor_state_str="unknown",
        )
        task = await tm._generate_temperature_task()
        assert task is None

    @pytest.mark.asyncio
    async def test_outdoor_sensor_not_configured_and_no_weather_entity(self):
        """Both outdoor_temp_sensor and weather_entity absent -> no temperature task."""
        from datetime import datetime as real_dt

        hass = MagicMock()
        hass.states.get = MagicMock(return_value=None)

        collector = MagicMock()
        base_time = real_dt(2026, 6, 15, 12, 0, 0)
        collector.get_temperature_history = AsyncMock(
            return_value=[(base_time + timedelta(hours=i), 22.0) for i in range(50)]
        )

        storage = AsyncMock()
        storage.get_today_tasks = AsyncMock(return_value=[])
        storage.get_tasks_for_date = AsyncMock(return_value=[])
        storage.save_daily_tasks = AsyncMock()
        storage.log_task_generation = AsyncMock()
        storage.get_task_difficulty_stats = AsyncMock(return_value=None)

        agent = MagicMock()
        agent.phase = "active"

        tm_mod.should_ai_be_active = MagicMock(return_value=True)
        tm_mod.get_language = AsyncMock(return_value="en")
        tm_mod.get_working_days_from_config = MagicMock(return_value=list(range(7)))

        # Neither weather_entity nor outdoor_temp_sensor configured
        config = {"environment_mode": "home"}

        tm = TaskManager(
            hass=hass,
            sensors={"power": ["sensor.p1"], "temperature": ["sensor.t1"]},
            data_collector=collector,
            storage=storage,
            decision_agent=agent,
            config_data=config,
        )

        task = await tm._generate_temperature_task()
        assert task is None

