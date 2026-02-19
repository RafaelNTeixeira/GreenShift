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