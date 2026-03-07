"""
Tests for __init__.py

Covers (unit tests — no HomeAssistant event loop required):
- async_discover_sensors: categorizes entities by device class, unit and keyword;
  skips own-platform entities, entities without devices and HA-manufactured entities
- sync_helper_entities: calls correct helper services with the data from config entry
- trigger_phase_transition_notification: sends the persistent notification with correct
  notification_id and payload
"""

import sys
import types
import pathlib
import importlib.util
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# ── Minimal HA stubs ─────────────────────────────────────────────────────────

for mod_name in [
    "homeassistant",
    "homeassistant.config_entries",
    "homeassistant.core",
    "homeassistant.helpers",
    "homeassistant.helpers.event",
    "homeassistant.helpers.entity_registry",
    "homeassistant.helpers.device_registry",
    "homeassistant.helpers.dispatcher",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Stub event helpers (so periodic listeners don't fire)
event_stub = sys.modules["homeassistant.helpers.event"]
event_stub.async_track_time_interval = MagicMock(return_value=MagicMock())
event_stub.async_track_state_change_event = MagicMock(return_value=MagicMock())
event_stub.async_track_time_change = MagicMock(return_value=MagicMock())

dispatcher_stub = sys.modules["homeassistant.helpers.dispatcher"]
dispatcher_stub.async_dispatcher_send = MagicMock()

# Keep references to stubs for monkeypatching inside tests
er_stub = sys.modules["homeassistant.helpers.entity_registry"]
dr_stub = sys.modules["homeassistant.helpers.device_registry"]

# ── Real const module ─────────────────────────────────────────────────────────

const_spec = importlib.util.spec_from_file_location(
    "custom_components.green_shift.const",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "const.py",
)
const_mod = importlib.util.module_from_spec(const_spec)
const_mod.__package__ = "custom_components.green_shift"
const_spec.loader.exec_module(const_mod)
sys.modules["custom_components.green_shift.const"] = const_mod

# ── Stub remaining green_shift sub-modules ────────────────────────────────────

for mod_name in [
    "custom_components.green_shift.data_collector",
    "custom_components.green_shift.decision_agent",
    "custom_components.green_shift.storage",
    "custom_components.green_shift.task_manager",
    "custom_components.green_shift.backup_manager",
    "custom_components.green_shift.helpers",
    "custom_components.green_shift.translations_runtime",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

sys.modules["custom_components.green_shift.data_collector"].DataCollector = MagicMock()
sys.modules["custom_components.green_shift.decision_agent"].DecisionAgent = MagicMock()
sys.modules["custom_components.green_shift.storage"].StorageManager = MagicMock()
sys.modules["custom_components.green_shift.task_manager"].TaskManager = MagicMock()
sys.modules["custom_components.green_shift.backup_manager"].BackupManager = MagicMock()
sys.modules["custom_components.green_shift.helpers"].get_normalized_value = MagicMock()

translations_stub = sys.modules["custom_components.green_shift.translations_runtime"]
translations_stub.get_language = AsyncMock(return_value="en")
translations_stub.get_phase_transition_template = MagicMock(return_value={
    "title": "Green Shift: Active Phase",
    "message": (
        "avg: {avg_daily_kwh:.1f} kWh, peak: {peak_time}, "
        "{top_area_section}target: {target}%, "
        "co2: {co2_kg} kg, trees: {trees}, flights: {flights}"
    ),
})

# numpy is required by __init__.py
import numpy as np  # noqa: E402 - already available in test env

# ── Load __init__.py ──────────────────────────────────────────────────────────

init_spec = importlib.util.spec_from_file_location(
    "gs_init",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "__init__.py",
)
init_mod = importlib.util.module_from_spec(init_spec)
init_mod.__package__ = "custom_components.green_shift"
init_spec.loader.exec_module(init_mod)

async_discover_sensors = init_mod.async_discover_sensors
sync_helper_entities = init_mod.sync_helper_entities
trigger_phase_transition_notification = init_mod.trigger_phase_transition_notification
async_setup_services = init_mod.async_setup_services


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_entity(entity_id, device_class=None, unit=None, original_name="",
                 platform="third_party", device_id="dev1"):
    entity = MagicMock()
    entity.entity_id = entity_id
    entity.device_class = device_class
    entity.unit_of_measurement = unit
    entity.original_name = original_name
    entity.platform = platform
    entity.device_id = device_id
    return entity


def _build_hass(entities, manufacturer="TestManufacturer"):
    """Build a minimal hass mock with entity/device registries configured."""
    hass = MagicMock()

    entity_reg = MagicMock()
    entity_reg.entities = {e.entity_id: e for e in entities}
    er_stub.async_get = MagicMock(return_value=entity_reg)

    device = MagicMock()
    device.manufacturer = manufacturer
    device_reg = MagicMock()
    device_reg.devices = {"dev1": device}
    dr_stub.async_get = MagicMock(return_value=device_reg)

    hass.states.get = MagicMock(return_value=None)
    return hass


# ─────────────────────────────────────────────────────────────────────────────
# async_discover_sensors
# ─────────────────────────────────────────────────────────────────────────────

class TestAsyncDiscoverSensors:

    @pytest.mark.asyncio
    async def test_result_contains_all_categories(self):
        hass = _build_hass([])
        result = await async_discover_sensors(hass)
        expected = {"power", "energy", "temperature", "humidity", "illuminance", "occupancy"}
        assert set(result.keys()) == expected

    @pytest.mark.asyncio
    async def test_power_sensor_classified_by_unit_W(self):
        entities = [_make_entity("sensor.power_w", unit="W")]
        hass = _build_hass(entities)
        result = await async_discover_sensors(hass)
        assert "sensor.power_w" in result["power"]

    @pytest.mark.asyncio
    async def test_power_sensor_classified_by_unit_kW(self):
        entities = [_make_entity("sensor.power_kw", unit="kW")]
        hass = _build_hass(entities)
        result = await async_discover_sensors(hass)
        assert "sensor.power_kw" in result["power"]

    @pytest.mark.asyncio
    async def test_energy_sensor_classified_by_unit_kWh(self):
        entities = [_make_entity("sensor.energy_kwh", unit="kWh")]
        hass = _build_hass(entities)
        result = await async_discover_sensors(hass)
        assert "sensor.energy_kwh" in result["energy"]

    @pytest.mark.asyncio
    async def test_energy_sensor_classified_by_unit_Wh(self):
        entities = [_make_entity("sensor.energy_wh", unit="Wh")]
        hass = _build_hass(entities)
        result = await async_discover_sensors(hass)
        assert "sensor.energy_wh" in result["energy"]

    @pytest.mark.asyncio
    async def test_temperature_sensor_classified_by_keyword_in_entity_id(self):
        entities = [_make_entity("sensor.living_room_temperature")]
        hass = _build_hass(entities)
        result = await async_discover_sensors(hass)
        assert "sensor.living_room_temperature" in result["temperature"]

    @pytest.mark.asyncio
    async def test_temperature_sensor_classified_by_keyword_temp(self):
        entities = [_make_entity("sensor.outdoor_temp")]
        hass = _build_hass(entities)
        result = await async_discover_sensors(hass)
        assert "sensor.outdoor_temp" in result["temperature"]

    @pytest.mark.asyncio
    async def test_humidity_sensor_classified_by_keyword(self):
        entities = [_make_entity("sensor.bathroom_humidity")]
        hass = _build_hass(entities)
        result = await async_discover_sensors(hass)
        assert "sensor.bathroom_humidity" in result["humidity"]

    @pytest.mark.asyncio
    async def test_occupancy_sensor_classified_by_keyword(self):
        entities = [_make_entity("binary_sensor.living_room_occupancy")]
        hass = _build_hass(entities)
        result = await async_discover_sensors(hass)
        assert "binary_sensor.living_room_occupancy" in result["occupancy"]

    @pytest.mark.asyncio
    async def test_motion_sensor_classified_as_occupancy_by_keyword(self):
        entities = [_make_entity("binary_sensor.hallway_motion")]
        hass = _build_hass(entities)
        result = await async_discover_sensors(hass)
        assert "binary_sensor.hallway_motion" in result["occupancy"]

    @pytest.mark.asyncio
    async def test_green_shift_platform_entities_skipped(self):
        entities = [_make_entity("sensor.gs_metric", platform="green_shift")]
        hass = _build_hass(entities)
        result = await async_discover_sensors(hass)
        for cat in result.values():
            assert "sensor.gs_metric" not in cat

    @pytest.mark.asyncio
    async def test_entity_without_device_id_skipped(self):
        entity = _make_entity("sensor.orphan")
        entity.device_id = None
        entity_reg = MagicMock()
        entity_reg.entities = {"sensor.orphan": entity}
        er_stub.async_get = MagicMock(return_value=entity_reg)

        hass = MagicMock()
        hass.states.get = MagicMock(return_value=None)
        result = await async_discover_sensors(hass)
        for cat in result.values():
            assert "sensor.orphan" not in cat

    @pytest.mark.asyncio
    async def test_home_assistant_manufacturer_skipped(self):
        entities = [_make_entity("sensor.ha_virtual")]
        hass = _build_hass(entities, manufacturer="Home Assistant")
        result = await async_discover_sensors(hass)
        for cat in result.values():
            assert "sensor.ha_virtual" not in cat

    @pytest.mark.asyncio
    async def test_unrecognized_sensor_not_categorized(self):
        entities = [_make_entity(
            "sensor.random_device",
            device_class=None,
            unit=None,
            original_name="Some Weird Sensor",
        )]
        hass = _build_hass(entities)
        result = await async_discover_sensors(hass)
        for cat in result.values():
            assert "sensor.random_device" not in cat

    @pytest.mark.asyncio
    async def test_sensor_classified_by_original_name_keyword(self):
        entities = [_make_entity(
            "sensor.env_123",
            original_name="Room Temperature Sensor",
        )]
        hass = _build_hass(entities)
        result = await async_discover_sensors(hass)
        assert "sensor.env_123" in result["temperature"]

    @pytest.mark.asyncio
    async def test_multiple_entities_classified_correctly(self):
        entities = [
            _make_entity("sensor.power_device", unit="W"),
            _make_entity("sensor.energy_meter", unit="kWh"),
            # "thermometer" contains "meter" (an energy keyword), so use a neutral
            # entity_id and rely on original_name keyword classification instead.
            _make_entity("sensor.env_456", original_name="Room Temperature"),
        ]
        hass = _build_hass(entities)
        result = await async_discover_sensors(hass)
        assert "sensor.power_device" in result["power"]
        assert "sensor.energy_meter" in result["energy"]
        assert "sensor.env_456" in result["temperature"]


# ─────────────────────────────────────────────────────────────────────────────
# sync_helper_entities
# ─────────────────────────────────────────────────────────────────────────────

class TestSyncHelperEntities:

    @pytest.mark.asyncio
    async def test_syncs_currency_to_input_select(self):
        hass = MagicMock()
        hass.services.async_call = AsyncMock()

        entry = MagicMock()
        entry.data = {"currency": "GBP", "electricity_price": 0.30, "energy_saving_target": 20}

        await sync_helper_entities(hass, entry)

        calls = hass.services.async_call.call_args_list
        currency_call = next(
            (c for c in calls
             if c.args[0] == "input_select" and c.args[1] == "select_option"),
            None,
        )
        assert currency_call is not None
        assert currency_call.args[2]["option"] == "GBP"

    @pytest.mark.asyncio
    async def test_syncs_electricity_price_to_input_number(self):
        hass = MagicMock()
        hass.services.async_call = AsyncMock()

        entry = MagicMock()
        entry.data = {"currency": "EUR", "electricity_price": 0.18, "energy_saving_target": 15}

        await sync_helper_entities(hass, entry)

        calls = hass.services.async_call.call_args_list
        price_call = next(
            (c for c in calls
             if c.args[0] == "input_number" and c.args[1] == "set_value"
             and "electricity_price" in str(c.args[2].get("entity_id", ""))),
            None,
        )
        assert price_call is not None
        assert price_call.args[2]["value"] == 0.18

    @pytest.mark.asyncio
    async def test_syncs_energy_saving_target_to_input_number(self):
        hass = MagicMock()
        hass.services.async_call = AsyncMock()

        entry = MagicMock()
        entry.data = {"currency": "EUR", "electricity_price": 0.25, "energy_saving_target": 20}

        await sync_helper_entities(hass, entry)

        calls = hass.services.async_call.call_args_list
        target_call = next(
            (c for c in calls
             if c.args[0] == "input_number" and c.args[1] == "set_value"
             and "energy_saving_target" in str(c.args[2].get("entity_id", ""))),
            None,
        )
        assert target_call is not None
        assert target_call.args[2]["value"] == 20

    @pytest.mark.asyncio
    async def test_defaults_to_eur_when_currency_missing(self):
        hass = MagicMock()
        hass.services.async_call = AsyncMock()

        entry = MagicMock()
        entry.data = {}  # No currency

        await sync_helper_entities(hass, entry)

        calls = hass.services.async_call.call_args_list
        currency_call = next(
            (c for c in calls
             if c.args[0] == "input_select" and c.args[1] == "select_option"),
            None,
        )
        assert currency_call is not None
        assert currency_call.args[2]["option"] == "EUR"

    @pytest.mark.asyncio
    async def test_service_exception_does_not_propagate(self):
        """Service failures must be caught and logged, not raised."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock(side_effect=Exception("HA unavailable"))

        entry = MagicMock()
        entry.data = {"currency": "EUR", "electricity_price": 0.25}

        # Should not raise
        await sync_helper_entities(hass, entry)

    @pytest.mark.asyncio
    async def test_exactly_three_service_calls_made(self):
        """Three helper entities must be synced: currency, price, target."""
        hass = MagicMock()
        hass.services.async_call = AsyncMock()

        entry = MagicMock()
        entry.data = {"currency": "USD", "electricity_price": 0.15, "energy_saving_target": 10}

        await sync_helper_entities(hass, entry)

        assert hass.services.async_call.call_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# trigger_phase_transition_notification
# ─────────────────────────────────────────────────────────────────────────────

class TestTriggerPhaseTransitionNotification:

    def _make_collector(self, top_area=None):
        collector = MagicMock()
        collector.calculate_baseline_summary = AsyncMock(return_value={
            "avg_daily_kwh": 5.2,
            "peak_time": "18:00 - 19:00",
            "top_area": top_area,
            "target": 15,
            "impact": {"co2_kg": 3.0, "trees": 0.14, "flights": 0.02},
        })
        return collector

    @pytest.mark.asyncio
    async def test_sends_persistent_notification_service(self):
        hass = MagicMock()
        hass.services.async_call = AsyncMock()

        await trigger_phase_transition_notification(
            hass, MagicMock(), self._make_collector()
        )

        hass.services.async_call.assert_called_once()
        args = hass.services.async_call.call_args.args
        assert args[0] == "persistent_notification"
        assert args[1] == "create"

    @pytest.mark.asyncio
    async def test_notification_id_is_gs_phase_transition(self):
        hass = MagicMock()
        hass.services.async_call = AsyncMock()

        await trigger_phase_transition_notification(
            hass, MagicMock(), self._make_collector()
        )

        payload = hass.services.async_call.call_args.args[2]
        assert payload["notification_id"] == "gs_phase_transition"

    @pytest.mark.asyncio
    async def test_notification_payload_contains_title(self):
        hass = MagicMock()
        hass.services.async_call = AsyncMock()

        await trigger_phase_transition_notification(
            hass, MagicMock(), self._make_collector()
        )

        payload = hass.services.async_call.call_args.args[2]
        assert "title" in payload
        assert payload["title"]  # Non-empty

    @pytest.mark.asyncio
    async def test_uses_summary_data_in_message(self):
        hass = MagicMock()
        hass.services.async_call = AsyncMock()

        await trigger_phase_transition_notification(
            hass, MagicMock(), self._make_collector()
        )

        payload = hass.services.async_call.call_args.args[2]
        message = payload["message"]
        assert "5.2" in message  # avg_daily_kwh
        assert "18:00 - 19:00" in message  # peak_time

    @pytest.mark.asyncio
    async def test_no_top_area_section_when_missing(self):
        hass = MagicMock()
        hass.services.async_call = AsyncMock()

        await trigger_phase_transition_notification(
            hass, MagicMock(), self._make_collector(top_area=None)
        )

        payload = hass.services.async_call.call_args.args[2]
        # The top_area_section should be empty string when no area is provided
        assert payload["message"]  # Still sends a message

    @pytest.mark.asyncio
    async def test_top_area_included_in_message_when_present(self):
        hass = MagicMock()
        hass.services.async_call = AsyncMock()

        await trigger_phase_transition_notification(
            hass, MagicMock(), self._make_collector(top_area="Living Room")
        )

        payload = hass.services.async_call.call_args.args[2]
        assert "Living Room" in payload["message"]


# ─────────────────────────────────────────────────────────────────────────────
# restore_backup service: in-memory reload after restore
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN = "green_shift"


def _build_services_hass(backup_success: bool, include_agent: bool = True):
    """Build a minimal hass mock suitable for calling async_setup_services."""
    hass = MagicMock()

    # Capture registered service handlers so tests can call them directly.
    registered_handlers = {}

    def _register(domain, name, handler):
        registered_handlers[name] = handler

    hass.services.async_register = _register
    hass.services.async_call = AsyncMock()

    backup_manager_mock = MagicMock()
    backup_manager_mock.restore_from_backup = AsyncMock(return_value=backup_success)
    backup_manager_mock.create_backup = AsyncMock(return_value=True)
    backup_manager_mock.list_backups = MagicMock(return_value=[])

    agent_mock = MagicMock()
    agent_mock._load_persistent_state = AsyncMock()

    task_manager_mock = MagicMock()
    task_manager_mock.generate_daily_tasks = AsyncMock(return_value=[])
    task_manager_mock.verify_tasks = AsyncMock(return_value={})

    storage_mock = MagicMock()
    storage_mock.load_state = AsyncMock(return_value={})
    storage_mock.save_state = AsyncMock()
    storage_mock.get_today_tasks = AsyncMock(return_value=[])
    storage_mock.get_task_by_index = AsyncMock(return_value=None)
    storage_mock.record_task_feedback = AsyncMock()
    storage_mock.record_rl_episode = AsyncMock()
    storage_mock._cleanup_old_research_data = AsyncMock()

    data = {
        DOMAIN: {
            "backup_manager": backup_manager_mock,
            "task_manager": task_manager_mock,
            "storage": storage_mock,
            "notification_select": MagicMock(),
        }
    }
    if include_agent:
        data[DOMAIN]["agent"] = agent_mock

    hass.data = data
    return hass, registered_handlers, backup_manager_mock, agent_mock


class TestRestoreBackupService:
    """Verify that restore_backup reloads in-memory agent state after a successful restore."""

    @pytest.mark.asyncio
    async def test_agent_state_reloaded_after_successful_restore(self):
        """_load_persistent_state must be awaited when restore succeeds."""
        hass, handlers, _, agent_mock = _build_services_hass(backup_success=True)
        await async_setup_services(hass)

        call = MagicMock()
        call.data = {"backup_name": "auto/20260218_100000"}
        await handlers["restore_backup"](call)

        agent_mock._load_persistent_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agent_state_not_reloaded_after_failed_restore(self):
        """_load_persistent_state must NOT be called when restore fails."""
        hass, handlers, _, agent_mock = _build_services_hass(backup_success=False)
        await async_setup_services(hass)

        call = MagicMock()
        call.data = {"backup_name": "auto/20260218_100000"}
        await handlers["restore_backup"](call)

        agent_mock._load_persistent_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_restore_succeeds_without_agent_in_hass_data(self):
        """If agent is not present the restore must still complete without raising."""
        hass, handlers, backup_manager_mock, _ = _build_services_hass(
            backup_success=True, include_agent=False
        )
        await async_setup_services(hass)

        call = MagicMock()
        call.data = {"backup_name": "auto/20260218_100000"}
        # Must not raise
        await handlers["restore_backup"](call)
        backup_manager_mock.restore_from_backup.assert_awaited_once_with("auto/20260218_100000")

    @pytest.mark.asyncio
    async def test_restore_returns_early_when_no_backup_name(self):
        """Missing backup_name must return early without calling restore."""
        hass, handlers, backup_manager_mock, agent_mock = _build_services_hass(backup_success=True)
        await async_setup_services(hass)

        call = MagicMock()
        call.data = {}  # no backup_name
        await handlers["restore_backup"](call)

        backup_manager_mock.restore_from_backup.assert_not_called()
        agent_mock._load_persistent_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_restore_returns_early_when_no_backup_manager(self):
        """Missing backup_manager must return early without calling restore."""
        hass, handlers, _, agent_mock = _build_services_hass(backup_success=True)
        del hass.data[DOMAIN]["backup_manager"]
        await async_setup_services(hass)

        call = MagicMock()
        call.data = {"backup_name": "auto/20260218_100000"}
        await handlers["restore_backup"](call)

        agent_mock._load_persistent_state.assert_not_called()
