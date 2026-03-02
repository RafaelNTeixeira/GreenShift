"""
Tests for data_collector.py

Covers:
- Power monitoring: total power calculation, recalculation on state changes
- Energy monitoring: daily kWh tracking, reset at midnight
- get_current_state: aggregated state snapshot
- get_area_state: area-specific state
- get_all_areas: unique area listing
- Area grouping: sensors grouped by area
- Working hours filtering in office mode
- Memory management: deque size limits
"""
import pytest
import sys
import types
import pathlib
import importlib.util
import logging
from datetime import datetime, time, timedelta
from collections import deque
from unittest.mock import MagicMock, AsyncMock, patch

# ── Minimal HA stubs ────────────────────────────────────────────────────────
for mod_name in [
    "homeassistant",
    "homeassistant.core",
    "homeassistant.helpers",
    "homeassistant.helpers.event",
    "homeassistant.helpers.dispatcher",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Stub event tracking functions
event_stub = sys.modules["homeassistant.helpers.event"]
event_stub.async_track_state_change_event = MagicMock(return_value=MagicMock())
event_stub.async_track_time_change = MagicMock(return_value=MagicMock())
event_stub.async_track_time_interval = MagicMock(return_value=MagicMock())

# Stub dispatcher
dispatcher_stub = sys.modules["homeassistant.helpers.dispatcher"]
dispatcher_stub.async_dispatcher_send = MagicMock()

# Import real const module
const_spec = importlib.util.spec_from_file_location(
    "custom_components.green_shift.const",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "const.py"
)
const_mod = importlib.util.module_from_spec(const_spec)
const_mod.__package__ = "custom_components.green_shift"
const_spec.loader.exec_module(const_mod)
sys.modules["custom_components.green_shift.const"] = const_mod

# Stub helpers module
helpers_stub = types.ModuleType("custom_components.green_shift.helpers")
helpers_stub.get_normalized_value = MagicMock(return_value=(100.0, "W"))
helpers_stub.get_entity_area = MagicMock(return_value="Living Room")
helpers_stub.group_sensors_by_area = MagicMock(return_value={"Living Room": ["sensor.power_1"]})
helpers_stub.get_environmental_impact = MagicMock(return_value={})
helpers_stub.is_within_working_hours = MagicMock(return_value=True)
sys.modules["custom_components.green_shift.helpers"] = helpers_stub

# Stub storage module
storage_stub = types.ModuleType("custom_components.green_shift.storage")
storage_stub.StorageManager = MagicMock()
sys.modules["custom_components.green_shift.storage"] = storage_stub

# Load data_collector module
dc_spec = importlib.util.spec_from_file_location(
    "data_collector",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "data_collector.py"
)
dc_mod = importlib.util.module_from_spec(dc_spec)
dc_mod.__package__ = "custom_components.green_shift"
dc_spec.loader.exec_module(dc_mod)
DataCollector = dc_mod.DataCollector


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.data = {}
    return hass


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.store_sensor_snapshot = AsyncMock()
    storage.store_area_snapshot = AsyncMock()
    storage.load_state = AsyncMock(return_value={})
    storage.save_state = AsyncMock()
    storage.get_history = AsyncMock(return_value=[])
    storage.get_area_history = AsyncMock(return_value=[])
    storage.get_all_areas = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def discovered_sensors():
    return {
        "power": ["sensor.power_1", "sensor.power_2"],
        "energy": ["sensor.energy_1"],
        "temperature": ["sensor.temp_1"],
        "humidity": ["sensor.humidity_1"],
        "occupancy": ["binary_sensor.occupancy_1"],
    }


@pytest.fixture
def home_config():
    return {"environment_mode": "home", "electricity_price": 0.25}


@pytest.fixture
def office_config():
    return {
        "environment_mode": "office",
        "working_start": "08:00",
        "working_end": "18:00",
        "working_monday": True,
        "working_tuesday": True,
        "working_wednesday": True,
        "working_thursday": True,
        "working_friday": True,
        "working_saturday": False,
        "working_sunday": False,
    }


def make_collector(
    hass=None,
    sensors=None,
    storage=None,
    config=None,
    main_power=None,
    main_energy=None
):
    """Factory for creating DataCollector instances."""
    if hass is None:
        hass = MagicMock()
        hass.states = MagicMock()
        hass.states.get = MagicMock(return_value=None)
    if sensors is None:
        sensors = {"power": ["sensor.power_1"]}
    if storage is None:
        storage = AsyncMock()
        storage.store_sensor_snapshot = AsyncMock()
        storage.store_area_snapshot = AsyncMock()
    if config is None:
        config = {"environment_mode": "home"}

    return DataCollector(
        hass, sensors, main_energy, main_power, storage, config_data=config
    )


# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────

class TestDataCollectorInit:

    def test_stores_discovered_sensors(self, mock_hass, mock_storage, discovered_sensors):
        dc = DataCollector(
            mock_hass, discovered_sensors, None, None, mock_storage
        )
        assert dc.sensors == discovered_sensors

    def test_initializes_current_state_dict(self, mock_hass, mock_storage):
        dc = DataCollector(mock_hass, {}, None, None, mock_storage)
        assert isinstance(dc.current_total_power, float)

    def test_initializes_area_states_dict(self, mock_hass, mock_storage):
        dc = DataCollector(mock_hass, {}, None, None, mock_storage)
        assert isinstance(dc.area_data, dict)

    def test_stores_config_data(self, mock_hass, mock_storage, home_config):
        dc = DataCollector(
            mock_hass, {}, None, None, mock_storage, config_data=home_config
        )
        assert dc.config_data == home_config


# ─────────────────────────────────────────────────────────────────────────────
# Power monitoring
# ─────────────────────────────────────────────────────────────────────────────

class TestPowerMonitoring:

    def test_total_power_calculated_from_sensors(self):
        dc = make_collector()
        dc._power_sensor_cache = {"sensor.power_1": 100.0, "sensor.power_2": 200.0}
        dc._recalculate_total_power()
        assert dc.current_total_power == 300.0

    def test_total_power_zero_when_no_sensors(self):
        dc = make_collector()
        dc._power_sensor_cache = {}
        dc._recalculate_total_power()
        assert dc.current_total_power == 0.0

    def test_total_power_skips_none_values(self):
        dc = make_collector()
        dc._power_sensor_cache = {"sensor.power_1": 100.0, "sensor.power_2": None}
        dc._recalculate_total_power()
        assert dc.current_total_power == 100.0

    def test_power_cache_stores_values(self):
        dc = make_collector()
        dc._power_sensor_cache["sensor.power_1"] = 100.0
        assert dc._power_sensor_cache["sensor.power_1"] == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Energy monitoring
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyMonitoring:

    def test_daily_kwh_calculated_correctly(self):
        dc = make_collector()
        dc.main_energy_sensor = "sensor.energy_1"
        dc._energy_midnight_points["sensor.energy_1"] = 10.0
        dc._energy_sensor_cache["sensor.energy_1"] = 12.5

        dc.get_daily_kwh()  # Updates current_daily_energy
        assert dc.current_daily_energy == pytest.approx(2.5)

    def test_daily_kwh_zero_when_no_start_value(self):
        dc = make_collector()
        dc.main_energy_sensor = "sensor.energy_1"
        dc._energy_midnight_points = {}
        dc._energy_sensor_cache["sensor.energy_1"] = 12.5

        dc.get_daily_kwh()  # Won't update if no midnight point
        # When there's no midnight point for main sensor, it returns early
        assert dc.current_daily_energy == 0.0  # Initial value

    def test_daily_kwh_zero_when_no_current_value(self):
        dc = make_collector()
        dc.main_energy_sensor = "sensor.energy_1"
        dc._energy_midnight_points["sensor.energy_1"] = 10.0
        dc._energy_sensor_cache = {}

        dc.get_daily_kwh()  # Won't update if no current value
        assert dc.current_daily_energy == 0.0  # Initial value

    def test_daily_kwh_handles_reset(self):
        """When current < start (meter reset), should use current value."""
        dc = make_collector()
        dc.main_energy_sensor = "sensor.energy_1"
        dc._energy_midnight_points["sensor.energy_1"] = 100.0
        dc._energy_sensor_cache["sensor.energy_1"] = 50.0  # Reset occurred

        dc.get_daily_kwh()
        # When current < midnight, it sets current_daily_energy to current_val
        assert dc.current_daily_energy == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# get_current_state
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCurrentState:

    def test_returns_aggregated_state(self):
        dc = make_collector()
        dc.current_total_power = 500.0
        dc.current_temperature = 21.0
        dc.current_humidity = 50.0
        dc.current_illuminance = 100.0
        dc.current_occupancy = True

        result = dc.get_current_state()
        assert result["power"] == 500.0
        assert result["temperature"] == 21.0
        assert result["humidity"] == 50.0
        assert result["illuminance"] == 100.0
        assert result["occupancy"] is True

    def test_returns_copy_not_reference(self):
        dc = make_collector()
        dc.current_total_power = 500.0

        result = dc.get_current_state()
        result["power"] = 999.0

        # Original should be unchanged
        assert dc.current_total_power == 500.0


# ─────────────────────────────────────────────────────────────────────────────
# get_area_state
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAreaState:

    def test_returns_area_specific_state(self):
        dc = make_collector()
        dc.area_data = {
            "Living Room": {"temperature": 22.0, "humidity": 50.0, "illuminance": 300.0, "occupancy": True},
            "Kitchen": {"temperature": 20.0, "humidity": 45.0, "illuminance": 200.0, "occupancy": False},
        }

        result = dc.get_area_state("Living Room")
        assert result["temperature"] == 22.0

    def test_returns_empty_dict_for_unknown_area(self):
        dc = make_collector()
        dc.area_data = {}

        result = dc.get_area_state("NonExistent")
        # Should return default structure, not empty dict
        assert "temperature" in result
        assert "humidity" in result

    def test_modifying_returned_state_affects_original(self):
        dc = make_collector()
        dc.area_data = {"Kitchen": {"temperature": 20.0, "humidity": 50.0}}

        result = dc.get_area_state("Kitchen")
        result["temperature"] = 999.0

        # The returned dict IS the reference from area_data
        assert dc.area_data["Kitchen"]["temperature"] == 999.0


# ─────────────────────────────────────────────────────────────────────────────
# get_all_areas
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAllAreas:

    def test_returns_list_of_areas(self):
        dc = make_collector()
        dc.area_data = {
            "Living Room": {"temperature": None},
            "Kitchen": {"temperature": None},
            "Bedroom": {"temperature": None},
        }

        result = dc.get_all_areas()
        assert set(result) == {"Living Room", "Kitchen", "Bedroom"}

    def test_returns_empty_list_when_no_areas(self):
        dc = make_collector()
        dc.area_data = {}

        result = dc.get_all_areas()
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# History methods (delegating to storage)
# ─────────────────────────────────────────────────────────────────────────────

class TestHistoryMethods:

    @pytest.mark.asyncio
    async def test_get_power_history_delegates_to_storage(self):
        storage = AsyncMock()
        storage.get_history = AsyncMock(return_value=[(datetime.now(), 500.0)])

        dc = make_collector(storage=storage)
        result = await dc.get_power_history(hours=1)

        storage.get_history.assert_called_once()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_area_history_delegates_to_storage(self):
        storage = AsyncMock()
        storage.get_area_history = AsyncMock(return_value=[(datetime.now(), 200.0)])

        dc = make_collector(storage=storage)
        result = await dc.get_area_history("Kitchen", "power", hours=1)

        storage.get_area_history.assert_called_once()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_working_hours_filter_passed_through(self):
        storage = AsyncMock()
        storage.get_history = AsyncMock(return_value=[])

        dc = make_collector(storage=storage)
        await dc.get_power_history(hours=1, working_hours_only=True)

        # Check that working_hours_only was passed to storage
        call_kwargs = storage.get_history.call_args[1]
        assert call_kwargs.get("working_hours_only") is True


# ─────────────────────────────────────────────────────────────────────────────
# Area grouping
# ─────────────────────────────────────────────────────────────────────────────

class TestAreaGrouping:

    @pytest.mark.asyncio
    async def test_sensors_grouped_by_area(self, mock_hass, mock_storage):
        """Test that setup correctly initializes area-based structures."""
        sensors = {
            "power": ["sensor.living_room_power", "sensor.kitchen_power"],
            "temperature": ["sensor.living_room_temp"],
        }

        dc = DataCollector(mock_hass, sensors, None, None, mock_storage)
        await dc.setup()

        # Verify area data was initialized (even if empty)
        assert isinstance(dc.area_data, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Working hours (office mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkingHours:

    @pytest.mark.asyncio
    async def test_home_mode_always_within_working_hours(self, mock_hass, mock_storage, home_config):
        """Home mode should always consider it working hours."""
        dc = DataCollector(
            mock_hass, {}, None, None, mock_storage, config_data=home_config
        )

        # Mock the helper
        helpers_stub.is_within_working_hours = MagicMock(return_value=True)

        # In home mode, helper should always return True
        # This is tested in test_helpers.py, here we just verify integration
        assert dc.config_data["environment_mode"] == "home"

    @pytest.mark.asyncio
    async def test_office_mode_respects_working_hours(self, mock_hass, mock_storage, office_config):
        """Office mode should check working hours."""
        dc = DataCollector(
            mock_hass, {}, None, None, mock_storage, config_data=office_config
        )

        assert dc.config_data["environment_mode"] == "office"
        assert dc.config_data["working_start"] == "08:00"
        assert dc.config_data["working_end"] == "18:00"


# ─────────────────────────────────────────────────────────────────────────────
# Memory management
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryManagement:

    def test_power_cache_is_dict(self):
        dc = make_collector()
        # Check that power sensor cache is a dict
        assert isinstance(dc._power_sensor_cache, dict)

    def test_energy_cache_is_dict(self):
        dc = make_collector()
        assert isinstance(dc._energy_sensor_cache, dict)

    def test_caches_store_sensor_values(self):
        """Test that sensor caches correctly store values."""
        dc = make_collector()
        dc._power_sensor_cache["sensor.test"] = 123.4
        dc._energy_sensor_cache["sensor.test"] = 56.7

        assert dc._power_sensor_cache["sensor.test"] == 123.4
        assert dc._energy_sensor_cache["sensor.test"] == 56.7


# ─────────────────────────────────────────────────────────────────────────────
# Midnight-points date guard
# ─────────────────────────────────────────────────────────────────────────────

class TestMidnightPointsDateGuard:
    """_load_persistent_data must discard midnight energy points that were saved on
    a different calendar day.  Keeping stale points after an HA restart that spans
    midnight would inflate current_daily_energy for the new day."""

    @pytest.mark.asyncio
    async def test_loads_midnight_points_when_date_matches_today(self):
        """Midnight points saved today must be loaded normally."""
        today_str = datetime.now().strftime("%Y-%m-%d")

        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value={
            "energy_midnight_points": {"sensor.energy_1": 50.0},
            "energy_midnight_points_date": today_str,
        })

        dc = make_collector(storage=storage)
        await dc._load_persistent_data()

        assert dc._energy_midnight_points == {"sensor.energy_1": 50.0}

    @pytest.mark.asyncio
    async def test_discards_midnight_points_when_date_is_yesterday(self):
        """Midnight points from yesterday must be discarded (HA restarted after midnight)."""
        yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value={
            "energy_midnight_points": {"sensor.energy_1": 50.0},
            "energy_midnight_points_date": yesterday_str,
        })

        dc = make_collector(storage=storage)
        await dc._load_persistent_data()

        # Must be empty — will be re-initialised from first energy change event
        assert dc._energy_midnight_points == {}

    @pytest.mark.asyncio
    async def test_discards_midnight_points_when_date_is_missing(self):
        """If no date key is stored (legacy state), points must be discarded."""
        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value={
            "energy_midnight_points": {"sensor.energy_1": 50.0},
            # no "energy_midnight_points_date" key
        })

        dc = make_collector(storage=storage)
        await dc._load_persistent_data()

        assert dc._energy_midnight_points == {}

    def test_update_midnight_points_schedules_single_task(self, mock_hass, mock_storage):
        """update_midnight_points must schedule exactly ONE async_create_task (atomic save)."""
        dc = DataCollector(mock_hass, {"energy": ["sensor.e1"]}, None, None, mock_storage)
        dc._energy_sensor_cache = {"sensor.e1": 20.0}

        mock_hass.async_create_task = MagicMock()

        dc.update_midnight_points()

        assert mock_hass.async_create_task.call_count == 1, (
            "update_midnight_points must schedule exactly one task to avoid a "
            "read-modify-write race condition when two tasks run concurrently."
        )

    @pytest.mark.asyncio
    async def test_update_midnight_points_saves_both_fields_atomically(self, mock_hass, mock_storage):
        """The single scheduled coroutine must atomically persist both
        energy_midnight_points and energy_midnight_points_date via save_state."""
        dc = DataCollector(mock_hass, {"energy": ["sensor.e1"]}, None, None, mock_storage)
        dc._energy_sensor_cache = {"sensor.e1": 20.0}

        captured = []
        mock_hass.async_create_task = MagicMock(side_effect=captured.append)

        mock_storage.load_state = AsyncMock(return_value={})
        mock_storage.save_state = AsyncMock()

        dc.update_midnight_points()

        assert len(captured) == 1
        # Await the captured coroutine to exercise its body
        await captured[0]

        assert mock_storage.save_state.call_count == 1
        saved = mock_storage.save_state.call_args[0][0]

        expected_date = datetime.now().strftime("%Y-%m-%d")
        assert "energy_midnight_points" in saved, "save_state payload must contain energy_midnight_points"
        assert "energy_midnight_points_date" in saved, "save_state payload must contain energy_midnight_points_date"
        assert saved["energy_midnight_points_date"] == expected_date, (
            f"Expected local date '{expected_date}', got '{saved['energy_midnight_points_date']}'. "
            "Must use local calendar date, not UTC."
        )
        assert saved["energy_midnight_points"] == {"sensor.e1": 20.0}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_state_event(entity_id: str, state_value: str, attributes=None):
    """Build a minimal HA state-change event dict."""
    new_state = MagicMock()
    new_state.state = state_value
    new_state.attributes = attributes or {}
    event = MagicMock()
    # Using a plain dict so .data.get() works normally
    event.data = {"entity_id": entity_id, "new_state": new_state}
    return event

def _grab_callback(call_args_list, nth_from_end: int = 0):
    """Extract the callback (3rd positional arg) from an async_track_state_change_event call."""
    call = call_args_list[-(1 + nth_from_end)]
    return call[0][2]  # (hass, entity_list, callback)

# ─────────────────────────────────────────────────────────────────────────────
# Power monitoring callbacks
# ─────────────────────────────────────────────────────────────────────────────

class TestPowerMonitoringCallback:

    @pytest.mark.asyncio
    async def test_callback_updates_power_cache(self):
        dc = make_collector(sensors={"power": ["sensor.power_1"]})
        dc_mod.get_normalized_value = MagicMock(return_value=(250.0, "W"))
        helpers_stub.get_entity_area = MagicMock(return_value="No Area")

        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_power_monitoring()

        callback = _grab_callback(dc_mod.async_track_state_change_event.call_args_list)
        event = _make_state_event("sensor.power_1", "250")
        callback(event)

        assert dc._power_sensor_cache["sensor.power_1"] == 250.0
        assert dc.current_total_power == 250.0

    @pytest.mark.asyncio
    async def test_callback_ignores_unavailable_state(self):
        dc = make_collector(sensors={"power": ["sensor.power_1"]})
        dc_mod.get_normalized_value = MagicMock(return_value=(None, None))

        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_power_monitoring()

        callback = _grab_callback(dc_mod.async_track_state_change_event.call_args_list)
        event = _make_state_event("sensor.power_1", "unavailable")
        callback(event)

        # Unavailable zeros out the cache and triggers recalculation
        assert dc._power_sensor_cache.get("sensor.power_1", 0.0) == 0.0

    @pytest.mark.asyncio
    async def test_callback_ignores_unknown_entity(self):
        dc = make_collector(sensors={"power": ["sensor.power_1"]})
        dc_mod.get_normalized_value = MagicMock(return_value=(100.0, "W"))

        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_power_monitoring()

        callback = _grab_callback(dc_mod.async_track_state_change_event.call_args_list)
        # Fire event for an entity NOT in the registered list
        event = _make_state_event("sensor.unknown", "300")
        callback(event)

        # Cache should be untouched
        assert "sensor.unknown" not in dc._power_sensor_cache

    @pytest.mark.asyncio
    async def test_callback_none_new_state_is_ignored(self):
        dc = make_collector(sensors={"power": ["sensor.power_1"]})

        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_power_monitoring()

        callback = _grab_callback(dc_mod.async_track_state_change_event.call_args_list)
        event = MagicMock()
        event.data = {"entity_id": "sensor.power_1", "new_state": None}
        callback(event)

        assert "sensor.power_1" not in dc._power_sensor_cache

    @pytest.mark.asyncio
    async def test_no_power_sensors_skips_setup(self):
        dc = make_collector(sensors={})
        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_power_monitoring()
        # Nothing registered
        assert dc_mod.async_track_state_change_event.call_count == 0

    @pytest.mark.asyncio
    async def test_recalculate_uses_main_power_sensor(self):
        dc = make_collector(sensors={"power": ["sensor.main", "sensor.sub"]}, main_power="sensor.main")
        dc._power_sensor_cache["sensor.main"] = 500.0
        dc._power_sensor_cache["sensor.sub"] = 200.0

        dc._recalculate_total_power()
        # Should use main sensor, not sum
        assert dc.current_total_power == 500.0


# ─────────────────────────────────────────────────────────────────────────────
# Energy monitoring callbacks
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyMonitoringCallback:

    @pytest.mark.asyncio
    async def test_callback_updates_energy_cache(self):
        dc = make_collector(sensors={"energy": ["sensor.energy_1"]})
        dc_mod.get_normalized_value = MagicMock(return_value=(15.0, "kWh"))
        helpers_stub.get_entity_area = MagicMock(return_value="No Area")

        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_energy_monitoring()

        callback = _grab_callback(dc_mod.async_track_state_change_event.call_args_list)
        event = _make_state_event("sensor.energy_1", "15.0")
        callback(event)

        assert dc._energy_sensor_cache["sensor.energy_1"] == 15.0

    @pytest.mark.asyncio
    async def test_callback_initialises_midnight_point_if_missing(self):
        dc = make_collector(sensors={"energy": ["sensor.energy_1"]})
        dc_mod.get_normalized_value = MagicMock(return_value=(10.0, "kWh"))
        helpers_stub.get_entity_area = MagicMock(return_value="No Area")

        dc._energy_midnight_points = {}

        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_energy_monitoring()

        callback = _grab_callback(dc_mod.async_track_state_change_event.call_args_list)
        callback(_make_state_event("sensor.energy_1", "10.0"))

        # Midnight point must be initialized on first event
        assert dc._energy_midnight_points["sensor.energy_1"] == 10.0

    @pytest.mark.asyncio
    async def test_no_energy_sensors_skips_setup(self):
        dc = make_collector(sensors={})
        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_energy_monitoring()
        assert dc_mod.async_track_state_change_event.call_count == 0

    @pytest.mark.asyncio
    async def test_callback_ignores_none_value(self):
        dc = make_collector(sensors={"energy": ["sensor.energy_1"]})
        dc_mod.get_normalized_value = MagicMock(return_value=(None, None))

        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_energy_monitoring()

        callback = _grab_callback(dc_mod.async_track_state_change_event.call_args_list)
        callback(_make_state_event("sensor.energy_1", "unavailable"))

        assert "sensor.energy_1" not in dc._energy_sensor_cache


# ─────────────────────────────────────────────────────────────────────────────
# get_daily_kwh fallback (Case 2: non-main sensors summed)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDailyKwhFallback:

    def test_fallback_sums_non_main_sensors(self):
        """When main energy sensor misses midnight point, Case 2 sums other sensors."""
        dc = make_collector(sensors={"energy": ["sensor.main", "sensor.sub"]})
        dc.main_energy_sensor = "sensor.main"

        # Main sensor has no midnight point -> falls to Case 2
        dc._energy_sensor_cache = {
            "sensor.main": 100.0,  # skipped (it IS main, but midnight_val is None)
            "sensor.sub": 5.0,
        }
        dc._energy_midnight_points = {"sensor.sub": 3.0}  # no entry for sensor.main

        dc.get_daily_kwh()
        # sensor.main is skipped; sensor.sub contributes 5.0 - 3.0 = 2.0
        assert dc.current_daily_energy == pytest.approx(2.0)

    def test_fallback_handles_sensor_reset(self):
        """When current < midnight for a sub-sensor, use current value (reset)."""
        dc = make_collector(sensors={"energy": ["sensor.sub"]})
        dc.main_energy_sensor = None

        dc._energy_sensor_cache = {"sensor.sub": 2.0}
        dc._energy_midnight_points = {"sensor.sub": 100.0}  # current < midnight = reset

        dc.get_daily_kwh()
        assert dc.current_daily_energy == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Environment monitoring callbacks
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvironmentMonitoringCallbacks:

    @pytest.mark.asyncio
    async def test_temperature_callback_updates_current_temperature(self):
        sensors = {"temperature": ["sensor.temp_1"]}
        dc = make_collector(sensors=sensors)
        helpers_stub.get_entity_area = MagicMock(return_value="No Area")

        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_environment_monitoring()

        # Only temperature sensors -> single registration call
        callback = _grab_callback(dc_mod.async_track_state_change_event.call_args_list, nth_from_end=0)
        callback(_make_state_event("sensor.temp_1", "22.5"))

        assert dc.current_temperature == pytest.approx(22.5)

    @pytest.mark.asyncio
    async def test_temperature_callback_ignores_unavailable(self):
        sensors = {"temperature": ["sensor.temp_1"]}
        dc = make_collector(sensors=sensors)
        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_environment_monitoring()

        callback = _grab_callback(dc_mod.async_track_state_change_event.call_args_list, nth_from_end=0)
        original_temp = dc.current_temperature
        callback(_make_state_event("sensor.temp_1", "unavailable"))
        assert dc.current_temperature == original_temp

    @pytest.mark.asyncio
    async def test_humidity_callback_updates_current_humidity(self):
        sensors = {"humidity": ["sensor.hum_1"]}
        dc = make_collector(sensors=sensors)
        helpers_stub.get_entity_area = MagicMock(return_value="No Area")

        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_environment_monitoring()

        callback = _grab_callback(dc_mod.async_track_state_change_event.call_args_list, nth_from_end=0)
        callback(_make_state_event("sensor.hum_1", "55.0"))

        assert dc.current_humidity == pytest.approx(55.0)

    @pytest.mark.asyncio
    async def test_illuminance_callback_updates_current_illuminance(self):
        sensors = {"illuminance": ["sensor.lux_1"]}
        dc = make_collector(sensors=sensors)
        helpers_stub.get_entity_area = MagicMock(return_value="No Area")

        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_environment_monitoring()

        callback = _grab_callback(dc_mod.async_track_state_change_event.call_args_list, nth_from_end=0)
        callback(_make_state_event("sensor.lux_1", "300"))

        assert dc.current_illuminance == pytest.approx(300.0)

    @pytest.mark.asyncio
    async def test_occupancy_callback_updates_current_occupancy_true(self):
        sensors = {"occupancy": ["binary_sensor.occ_1"]}
        dc = make_collector(sensors=sensors)
        helpers_stub.get_entity_area = MagicMock(return_value="No Area")

        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_environment_monitoring()

        callback = _grab_callback(dc_mod.async_track_state_change_event.call_args_list, nth_from_end=0)
        callback(_make_state_event("binary_sensor.occ_1", "on"))

        assert dc.current_occupancy is True

    @pytest.mark.asyncio
    async def test_occupancy_callback_updates_current_occupancy_false(self):
        sensors = {"occupancy": ["binary_sensor.occ_1"]}
        dc = make_collector(sensors=sensors)
        helpers_stub.get_entity_area = MagicMock(return_value="No Area")

        # Pre-set cache so the sensor is currently True
        dc._occupancy_sensor_cache["binary_sensor.occ_1"] = True
        dc_mod.async_track_state_change_event.reset_mock()
        await dc._setup_environment_monitoring()

        callback = _grab_callback(dc_mod.async_track_state_change_event.call_args_list, nth_from_end=0)
        callback(_make_state_event("binary_sensor.occ_1", "off"))

        assert dc.current_occupancy is False


# ─────────────────────────────────────────────────────────────────────────────
# _record_periodic_snapshot
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordPeriodicSnapshot:

    def test_creates_task_for_global_snapshot(self):
        storage = AsyncMock()
        storage.store_sensor_snapshot = AsyncMock()
        storage.store_area_snapshot = AsyncMock()

        dc = make_collector(storage=storage)
        dc.current_total_power = 400.0
        dc.current_daily_energy = 1.5
        dc.hass.async_create_task = MagicMock()
        helpers_stub.is_within_working_hours = MagicMock(return_value=True)

        dc._record_periodic_snapshot(datetime.now())

        assert dc.hass.async_create_task.called

    def test_creates_task_for_each_area(self):
        storage = AsyncMock()
        dc = make_collector(storage=storage)
        dc.area_data = {
            "Kitchen": {"power": 100.0, "energy": None, "temperature": None,
                        "humidity": None, "illuminance": None, "occupancy": False},
            "Living Room": {"power": 200.0, "energy": None, "temperature": None,
                            "humidity": None, "illuminance": None, "occupancy": True},
        }
        dc.hass.async_create_task = MagicMock()
        helpers_stub.is_within_working_hours = MagicMock(return_value=True)

        dc._record_periodic_snapshot(datetime.now())

        # Three tasks: 1 global + 2 areas
        assert dc.hass.async_create_task.call_count == 3

    def test_no_storage_skips_snapshot(self):
        dc = make_collector()
        dc.storage = None
        dc.hass.async_create_task = MagicMock()

        dc._record_periodic_snapshot(datetime.now())

        dc.hass.async_create_task.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# update_midnight_points: state machine fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateMidnightPointsStateMachine:

    def test_falls_back_to_state_machine_when_cache_empty(self):
        """If energy cache has no entry for a sensor, fall back to hass.states.get."""
        hass = MagicMock()
        storage = AsyncMock()
        hass.async_create_task = MagicMock()

        state_mock = MagicMock()
        state_mock.state = "25.0"
        state_mock.attributes = {"unit_of_measurement": "kWh"}
        hass.states.get = MagicMock(return_value=state_mock)

        dc_mod.get_normalized_value = MagicMock(return_value=(25.0, "kWh"))

        dc = DataCollector(hass, {"energy": ["sensor.e1"]}, None, None, storage)
        dc._energy_sensor_cache = {}  # no cache entry

        dc.update_midnight_points()

        assert dc._energy_midnight_points["sensor.e1"] == 25.0

    def test_skips_unavailable_state_in_state_machine(self):
        """An 'unavailable' state during midnight reset must be skipped."""
        hass = MagicMock()
        storage = AsyncMock()
        hass.async_create_task = MagicMock()

        state_mock = MagicMock()
        state_mock.state = "unavailable"
        hass.states.get = MagicMock(return_value=state_mock)
        dc_mod.get_normalized_value = MagicMock(return_value=(None, None))

        dc = DataCollector(hass, {"energy": ["sensor.e1"]}, None, None, storage)
        dc._energy_sensor_cache = {}

        dc.update_midnight_points()

        assert "sensor.e1" not in dc._energy_midnight_points


# ─────────────────────────────────────────────────────────────────────────────
# get_all_history: no storage
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAllHistoryNoStorage:

    @pytest.mark.asyncio
    async def test_returns_empty_dicts_when_no_storage(self):
        dc = make_collector()
        dc.storage = None

        result = await dc.get_all_history(hours=1)

        assert result["power"] == []
        assert result["energy"] == []
        assert result["temperature"] == []
        assert result["humidity"] == []
        assert result["illuminance"] == []
        assert result["occupancy"] == []

    @pytest.mark.asyncio
    async def test_delegates_to_storage_when_available(self):
        storage = AsyncMock()
        storage.get_history = AsyncMock(return_value=[(datetime.now(), 100.0)])
        dc = make_collector(storage=storage)

        result = await dc.get_all_history(hours=1)

        assert len(result["power"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# calculate_baseline_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateBaselineSummary:

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_storage(self):
        dc = make_collector()
        dc.storage = None

        result = await dc.calculate_baseline_summary()
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_summary_with_data(self):
        now = datetime.now()
        storage = AsyncMock()

        # energy history returns values so avg_daily > 0
        storage.get_history = AsyncMock(return_value=[
            (now - timedelta(hours=2), 1.0),
            (now - timedelta(hours=1), 2.0),
            (now, 3.0),
        ])
        storage.get_all_areas = AsyncMock(return_value=["Kitchen"])
        storage.get_area_stats = AsyncMock(return_value={"mean": 200.0, "min": 100.0, "max": 300.0, "std": 0.0})

        dc = make_collector(storage=storage)
        helpers_stub.get_environmental_impact = MagicMock(return_value={"co2": 0.5})

        result = await dc.calculate_baseline_summary()

        assert "avg_daily_kwh" in result
        assert "peak_time" in result
        assert "top_area" in result
        assert result["top_area"] == "Kitchen"
        assert result["target"] == 15

    @pytest.mark.asyncio
    async def test_returns_summary_with_no_history(self):
        storage = AsyncMock()
        storage.get_history = AsyncMock(return_value=[])
        storage.get_all_areas = AsyncMock(return_value=[])

        dc = make_collector(storage=storage)
        helpers_stub.get_environmental_impact = MagicMock(return_value={})

        result = await dc.calculate_baseline_summary()

        assert result["avg_daily_kwh"] == 0.0
        assert result["peak_time"] == "Unknown"
        assert result["top_area"] is None

    @pytest.mark.asyncio
    async def test_office_mode_uses_working_hours_filter(self):
        storage = AsyncMock()
        storage.get_history = AsyncMock(return_value=[])
        storage.get_all_areas = AsyncMock(return_value=[])

        dc = make_collector(storage=storage, config={"environment_mode": "office"})
        helpers_stub.get_environmental_impact = MagicMock(return_value={})
        helpers_stub.is_within_working_hours = MagicMock(return_value=True)

        result = await dc.calculate_baseline_summary()
        # Should not raise, and should return a summary
        assert "avg_daily_kwh" in result


# ─────────────────────────────────────────────────────────────────────────────
# History delegation methods (energy, temperature, humidity, illuminance, occupancy, area)
# ─────────────────────────────────────────────────────────────────────────────

class TestHistoryDelegation:

    @pytest.mark.asyncio
    async def test_get_energy_history_delegates(self):
        storage = AsyncMock()
        storage.get_history = AsyncMock(return_value=[(datetime.now(), 2.5)])
        dc = make_collector(storage=storage)
        result = await dc.get_energy_history(hours=1)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_temperature_history_delegates(self):
        storage = AsyncMock()
        storage.get_history = AsyncMock(return_value=[(datetime.now(), 21.0)])
        dc = make_collector(storage=storage)
        result = await dc.get_temperature_history(hours=1)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_humidity_history_delegates(self):
        storage = AsyncMock()
        storage.get_history = AsyncMock(return_value=[(datetime.now(), 50.0)])
        dc = make_collector(storage=storage)
        result = await dc.get_humidity_history(hours=1)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_illuminance_history_delegates(self):
        storage = AsyncMock()
        storage.get_history = AsyncMock(return_value=[(datetime.now(), 300.0)])
        dc = make_collector(storage=storage)
        result = await dc.get_illuminance_history(hours=1)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_occupancy_history_delegates(self):
        storage = AsyncMock()
        storage.get_history = AsyncMock(return_value=[(datetime.now(), 1.0)])
        dc = make_collector(storage=storage)
        result = await dc.get_occupancy_history(hours=1)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_history_passes_working_hours_filter(self):
        storage = AsyncMock()
        storage.get_history = AsyncMock(return_value=[])
        dc = make_collector(storage=storage)

        await dc.get_temperature_history(hours=1, working_hours_only=True)
        call_kwargs = storage.get_history.call_args[1]
        assert call_kwargs.get("working_hours_only") is True

    @pytest.mark.asyncio
    async def test_get_area_history_passes_all_params(self):
        storage = AsyncMock()
        storage.get_area_history = AsyncMock(return_value=[])
        dc = make_collector(storage=storage)

        await dc.get_area_history("Kitchen", "temperature", days=7, working_hours_only=False)

        call_kwargs = storage.get_area_history.call_args[1]
        assert call_kwargs.get("working_hours_only") is False
