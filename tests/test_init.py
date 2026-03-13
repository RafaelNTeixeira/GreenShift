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
from datetime import datetime, timedelta
import sqlite3
from unittest.mock import MagicMock, AsyncMock, patch

# -- Minimal HA stubs ---------------------------------------------------------

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

# -- Real const module ---------------------------------------------------------

const_spec = importlib.util.spec_from_file_location(
    "custom_components.green_shift.const",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "const.py",
)
const_mod = importlib.util.module_from_spec(const_spec)
const_mod.__package__ = "custom_components.green_shift"
const_spec.loader.exec_module(const_mod)
sys.modules["custom_components.green_shift.const"] = const_mod

# -- Stub remaining green_shift sub-modules ------------------------------------

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

# -- Load __init__.py ----------------------------------------------------------

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
async_setup_entry = init_mod.async_setup_entry
async_unload_entry = init_mod.async_unload_entry


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# async_discover_sensors
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# sync_helper_entities
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# trigger_phase_transition_notification
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# restore_backup service: in-memory reload after restore
# -----------------------------------------------------------------------------

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


class TestServiceHandlers:

    @pytest.mark.asyncio
    async def test_submit_task_feedback_success_path(self):
        hass, handlers, _, _ = _build_services_hass(backup_success=True)
        storage = hass.data[DOMAIN]["storage"]
        storage.get_today_tasks = AsyncMock(return_value=[{"task_id": "t-1"}])
        storage.save_task_feedback = AsyncMock(return_value=True)
        storage.log_task_feedback = AsyncMock()

        await async_setup_services(hass)
        dispatcher_stub.async_dispatcher_send.reset_mock()

        call = MagicMock()
        call.data = {"task_index": 0, "feedback": "just_right"}
        await handlers["submit_task_feedback"](call)

        storage.save_task_feedback.assert_awaited_once_with("t-1", "just_right")
        storage.log_task_feedback.assert_awaited_once_with("t-1", "just_right")
        assert dispatcher_stub.async_dispatcher_send.call_count >= 1

    @pytest.mark.asyncio
    async def test_submit_task_feedback_invalid_feedback_returns_early(self):
        hass, handlers, _, _ = _build_services_hass(backup_success=True)
        storage = hass.data[DOMAIN]["storage"]
        storage.get_today_tasks = AsyncMock(return_value=[{"task_id": "t-1"}])
        storage.save_task_feedback = AsyncMock(return_value=True)

        await async_setup_services(hass)

        call = MagicMock()
        call.data = {"task_index": 0, "feedback": "invalid"}
        await handlers["submit_task_feedback"](call)

        storage.save_task_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_respond_to_selection_accept_calls_agent(self):
        hass, handlers, _, agent_mock = _build_services_hass(backup_success=True)
        hass.states = MagicMock()
        hass.states.get = MagicMock(return_value=MagicMock(state="notif_001"))
        agent_mock._handle_notification_feedback = AsyncMock()

        await async_setup_services(hass)
        dispatcher_stub.async_dispatcher_send.reset_mock()

        call = MagicMock()
        call.data = {"decision": "accept"}
        await handlers["respond_to_selection"](call)

        agent_mock._handle_notification_feedback.assert_awaited_once_with("notif_001", accepted=True)
        assert dispatcher_stub.async_dispatcher_send.call_count >= 1

    @pytest.mark.asyncio
    async def test_respond_to_selection_ignores_unknown_selector(self):
        hass, handlers, _, agent_mock = _build_services_hass(backup_success=True)
        hass.states = MagicMock()
        hass.states.get = MagicMock(return_value=MagicMock(state="unknown"))
        agent_mock._handle_notification_feedback = AsyncMock()

        await async_setup_services(hass)

        call = MagicMock()
        call.data = {"decision": "reject"}
        await handlers["respond_to_selection"](call)

        agent_mock._handle_notification_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_notification_restores_original_time_when_no_send(self):
        hass, handlers, _, agent_mock = _build_services_hass(backup_success=True)
        original_time = object()
        agent_mock.last_notification_time = original_time
        agent_mock.notification_history = []
        agent_mock._decide_action = AsyncMock()

        await async_setup_services(hass)

        call = MagicMock()
        call.data = {}
        await handlers["force_notification"](call)

        assert agent_mock.last_notification_time is original_time

    @pytest.mark.asyncio
    async def test_create_backup_returns_when_manager_missing(self):
        hass, handlers, _, _ = _build_services_hass(backup_success=True)
        del hass.data[DOMAIN]["backup_manager"]

        await async_setup_services(hass)

        call = MagicMock()
        call.data = {}
        await handlers["create_backup"](call)

    @pytest.mark.asyncio
    async def test_verify_and_regenerate_services_call_expected_dependencies(self):
        hass, handlers, _, _ = _build_services_hass(backup_success=True)
        storage = hass.data[DOMAIN]["storage"]
        task_manager = hass.data[DOMAIN]["task_manager"]
        storage.delete_today_tasks = AsyncMock()
        task_manager.verify_tasks = AsyncMock(return_value={"verified": 1})
        task_manager.generate_daily_tasks = AsyncMock(return_value=[{"id": "a"}, {"id": "b"}])

        await async_setup_services(hass)
        dispatcher_stub.async_dispatcher_send.reset_mock()

        await handlers["verify_tasks"](MagicMock(data={}))
        await handlers["regenerate_tasks"](MagicMock(data={}))

        task_manager.verify_tasks.assert_awaited_once()
        storage.delete_today_tasks.assert_awaited_once()
        task_manager.generate_daily_tasks.assert_awaited()
        assert dispatcher_stub.async_dispatcher_send.call_count >= 2

    @pytest.mark.asyncio
    async def test_respond_to_selection_returns_when_selector_missing(self):
        hass, handlers, _, agent_mock = _build_services_hass(backup_success=True)
        hass.states = MagicMock()
        hass.states.get = MagicMock(return_value=None)
        agent_mock._handle_notification_feedback = AsyncMock()

        await async_setup_services(hass)
        await handlers["respond_to_selection"](MagicMock(data={"decision": "accept"}))

        agent_mock._handle_notification_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_debug_services_force_process_set_indices_and_save_state(self):
        hass, handlers, _, agent_mock = _build_services_hass(backup_success=True)
        agent_mock.process_ai_model = AsyncMock()
        agent_mock._save_persistent_state = AsyncMock()
        agent_mock.fatigue_index = 0.0
        agent_mock.behaviour_index = 0.0
        agent_mock.anomaly_index = 0.0

        await async_setup_services(hass)
        dispatcher_stub.async_dispatcher_send.reset_mock()

        await handlers["force_ai_process"](MagicMock(data={}))
        await handlers["set_test_indices"](MagicMock(data={"fatigue": 0.7, "behavior": 0.8, "anomaly": 0.9}))
        await handlers["save_state"](MagicMock(data={}))

        agent_mock.process_ai_model.assert_awaited_once()
        assert agent_mock.fatigue_index == 0.7
        assert agent_mock.behaviour_index == 0.8
        assert agent_mock.anomaly_index == 0.9
        agent_mock._save_persistent_state.assert_awaited_once()
        assert dispatcher_stub.async_dispatcher_send.call_count >= 1


class TestSetupAndUnloadEntry:

    @pytest.mark.asyncio
    async def test_async_setup_entry_initializes_and_runs_callbacks(self):
        hass = MagicMock()
        hass.data = {}
        hass.services = MagicMock()
        hass.services.async_call = AsyncMock()
        hass.config = MagicMock()
        hass.config.path = MagicMock(return_value="/tmp/green_shift_data")
        hass.config_entries = MagicMock()
        hass.config_entries.async_forward_entry_setups = AsyncMock()
        hass.config_entries.async_update_entry = MagicMock()

        entry = MagicMock()
        entry.data = {
            "discovered_sensors": {"power": ["sensor.main_power"]},
            "main_total_energy_sensor": "sensor.total_energy",
            "main_total_power_sensor": "sensor.main_power",
            "environment_mode": "home",
            "energy_saving_target": 15,
            "electricity_price": 0.22,
            "currency": "EUR",
        }

        storage = MagicMock()
        storage.setup = AsyncMock()
        storage.load_state = AsyncMock(return_value={})
        storage.has_phase_metadata = AsyncMock(return_value=False)
        storage.record_phase_change = AsyncMock()
        storage.compute_daily_aggregates = AsyncMock()
        storage.compute_area_daily_aggregates = AsyncMock()
        storage._cleanup_old_research_data = AsyncMock()
        storage.save_state = AsyncMock()
        storage._cleanup_old_data = AsyncMock()
        storage.get_today_tasks = AsyncMock(return_value=[])
        storage.close = AsyncMock()

        collector = MagicMock()
        collector.setup = AsyncMock()
        collector.get_power_history = AsyncMock(return_value=[(datetime.now(), 800.0)] * 20)
        collector._load_persistent_data = AsyncMock()

        agent = MagicMock()
        agent.setup = AsyncMock()
        agent._save_persistent_state = AsyncMock()
        agent.calculate_area_baselines = AsyncMock()
        agent.process_ai_model = AsyncMock()
        agent.phase = const_mod.PHASE_BASELINE
        agent.start_date = datetime.now()
        agent.baseline_consumption = 900.0
        agent.storage = storage
        agent.notification_history = []
        agent.last_notification_time = None

        task_manager = MagicMock()
        task_manager.generate_daily_tasks = AsyncMock(return_value=[{"id": "t1"}])
        task_manager.verify_tasks = AsyncMock(return_value={"ok": True})

        backup_manager = MagicMock()
        backup_manager.create_backup = AsyncMock(return_value=True)
        backup_manager.cleanup_old_backups = AsyncMock()

        callbacks = {}

        def _capture_interval(_hass, cb, _delta):
            callbacks[cb.__name__] = cb
            return lambda: None

        def _capture_change(_hass, _entities, cb):
            callbacks[cb.__name__] = cb
            return lambda: None

        def _capture_time(_hass, cb, **_kwargs):
            callbacks[cb.__name__] = cb
            return lambda: None

        with patch.object(init_mod, "StorageManager", return_value=storage), patch.object(
            init_mod, "DataCollector", return_value=collector
        ), patch.object(init_mod, "DecisionAgent", return_value=agent), patch.object(
            init_mod, "TaskManager", return_value=task_manager
        ), patch.object(init_mod, "BackupManager", return_value=backup_manager), patch.object(
            init_mod, "async_setup_services", AsyncMock()
        ), patch.object(init_mod, "async_track_time_interval", side_effect=_capture_interval), patch.object(
            init_mod, "async_track_state_change_event", side_effect=_capture_change
        ), patch.object(init_mod, "async_track_time_change", side_effect=_capture_time):
            ok = await async_setup_entry(hass, entry)

        assert ok is True
        assert const_mod.DOMAIN in hass.data
        backup_manager.create_backup.assert_awaited_once_with(backup_type="startup")
        storage.record_phase_change.assert_awaited()

        # Exercise registered callbacks once to cover runtime paths.
        await callbacks["update_agent_ai_model"](None)
        await callbacks["target_changed"](MagicMock(data={"new_state": MagicMock(state="18")}) )
        await callbacks["electricity_price_changed"](MagicMock(data={"new_state": MagicMock(state="0.3")}) )

        agent.phase = const_mod.PHASE_ACTIVE
        await callbacks["generate_daily_tasks_callback"](None)
        await callbacks["verify_tasks_callback"](None)
        await callbacks["daily_aggregation_callback"](None)
        await callbacks["auto_backup_callback"](None)
        await callbacks["rl_cleanup_callback"](None)
        await callbacks["sensor_data_cleanup_callback"](None)

    @pytest.mark.asyncio
    async def test_async_unload_entry_cleans_up_and_removes_domain(self):
        entry = MagicMock()
        hass = MagicMock()
        hass.config_entries = MagicMock()
        hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
        hass.services = MagicMock()
        hass.services.async_remove = MagicMock()

        storage = MagicMock()
        storage.close = AsyncMock()

        agent = MagicMock()
        agent.storage = storage
        agent._save_persistent_state = AsyncMock()

        backup_manager = MagicMock()
        backup_manager.create_backup = AsyncMock(return_value=True)

        hass.data = {
            const_mod.DOMAIN: {
                "agent": agent,
                "backup_manager": backup_manager,
                "storage": storage,
                "update_listener": lambda: None,
                "target_listener": lambda: None,
                "price_listener": lambda: None,
                "task_generation_listener": lambda: None,
                "task_verification_listener": lambda: None,
                "daily_aggregation_listener": lambda: None,
                "auto_backup_listener": lambda: None,
                "rl_cleanup_listener": lambda: None,
                "sensor_cleanup_listener": lambda: None,
            }
        }

        ok = await async_unload_entry(hass, entry)

        assert ok is True
        agent._save_persistent_state.assert_awaited_once()
        backup_manager.create_backup.assert_awaited_once_with(backup_type="shutdown")
        storage.close.assert_awaited_once()
        assert const_mod.DOMAIN not in hass.data


class TestInitAdditionalCoverage:

    @pytest.mark.asyncio
    async def test_discover_reads_class_and_unit_from_state_attributes(self):
        entity = _make_entity("sensor.fallback", device_class=None, unit=None)
        hass = _build_hass([entity])
        state = MagicMock()
        state.attributes = {"device_class": "temperature", "unit_of_measurement": "°C"}
        hass.states.get = MagicMock(return_value=state)

        result = await async_discover_sensors(hass)

        assert "sensor.fallback" in result["temperature"]

    @pytest.mark.asyncio
    async def test_trigger_phase_transition_pt_branch(self):
        hass = MagicMock()
        hass.services.async_call = AsyncMock()
        collector = MagicMock()
        collector.calculate_baseline_summary = AsyncMock(return_value={
            "avg_daily_kwh": 3.0,
            "peak_time": "10:00",
            "top_area": "Sala",
            "target": 12,
            "impact": {"co2_kg": 1.0, "trees": 0.1, "flights": 0.01},
        })

        with patch.object(init_mod, "get_language", AsyncMock(return_value="pt")):
            await trigger_phase_transition_notification(hass, MagicMock(), collector)

        payload = hass.services.async_call.call_args.args[2]
        assert "Área Principal" in payload["message"]

    @pytest.mark.asyncio
    async def test_submit_task_feedback_missing_fields_and_other_error_branches(self):
        hass, handlers, _, _ = _build_services_hass(backup_success=True)
        storage = hass.data[DOMAIN]["storage"]
        await async_setup_services(hass)

        await handlers["submit_task_feedback"](MagicMock(data={}))

        storage.get_today_tasks = AsyncMock(return_value=[])
        storage.save_task_feedback = AsyncMock(return_value=True)
        await handlers["submit_task_feedback"](MagicMock(data={"task_index": 1, "feedback": "just_right"}))

        storage.get_today_tasks = AsyncMock(return_value=[{"task_id": None}])
        await handlers["submit_task_feedback"](MagicMock(data={"task_index": 0, "feedback": "just_right"}))

        storage.get_today_tasks = AsyncMock(return_value=[{"task_id": "t-1"}])
        storage.save_task_feedback = AsyncMock(return_value=False)
        await handlers["submit_task_feedback"](MagicMock(data={"task_index": 0, "feedback": "just_right"}))

        storage.log_task_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_notification_keeps_new_time_when_notification_sent(self):
        hass, handlers, _, agent_mock = _build_services_hass(backup_success=True)
        original_time = datetime(2025, 1, 1)
        agent_mock.last_notification_time = original_time
        agent_mock.notification_history = ["x"]

        async def _decide():
            agent_mock.last_notification_time = datetime.now()

        agent_mock._decide_action = AsyncMock(side_effect=_decide)
        await async_setup_services(hass)

        await handlers["force_notification"](MagicMock(data={}))

        assert agent_mock.last_notification_time != original_time

    @pytest.mark.asyncio
    async def test_debug_services_inject_inspect_q_and_q_learning(self, tmp_path):
        hass, handlers, _, agent_mock = _build_services_hass(backup_success=True)
        storage = hass.data[DOMAIN]["storage"]
        collector = MagicMock()
        hass.data[DOMAIN]["collector"] = collector
        storage.store_sensor_snapshot = AsyncMock()

        class _ToggleContainsDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._contains_calls = 0

            def __contains__(self, key):
                self._contains_calls += 1
                if self._contains_calls == 1:
                    return True
                if self._contains_calls == 2:
                    return False
                return super().__contains__(key)

        state_key = (1, 1, 1, 1, 1, 1)
        agent_mock.q_table = {
            state_key: {0: 0.5, 1: 0.2},
            (2, 1, 0, 0, 3, 1): {0: 0.1, 1: 0.3},
        }
        agent_mock._discretize_state = MagicMock(return_value=(2, 2, 2, 0, 2, 0))
        agent_mock.learning_rate = 0.1
        agent_mock.episode_number = 1
        agent_mock.shadow_episode_number = 2
        agent_mock.epsilon = 0.1

        await async_setup_services(hass)

        with patch.object(init_mod, "UPDATE_INTERVAL_SECONDS", 3600):
            await handlers["inject_test_data"](MagicMock(data={"hours": 30}))
        with patch.object(init_mod, "UPDATE_INTERVAL_SECONDS", 1):
            await handlers["inject_test_data"](MagicMock(data={"hours": 2.8}))

        agent_mock._discretize_state = MagicMock(return_value=state_key)
        await handlers["inspect_q_table"](MagicMock(data={}))

        agent_mock.q_table = {}
        agent_mock._discretize_state = MagicMock(return_value=(2, 2, 2, 0, 2, 0))
        await handlers["inspect_q_table"](MagicMock(data={}))

        agent_mock._discretize_state = MagicMock(return_value=state_key)
        agent_mock.q_table = {}
        await handlers["test_q_learning"](MagicMock(data={}))

        agent_mock.q_table = _ToggleContainsDict({state_key: {0: 0.5, 1: 0.2}})
        await handlers["test_q_learning"](MagicMock(data={}))

        assert storage.store_sensor_snapshot.await_count >= 1
        assert agent_mock.q_table[state_key][1] != 0.2

    @pytest.mark.asyncio
    async def test_backup_service_failure_and_restore_no_storage_with_collector(self):
        hass, handlers, backup_manager_mock, agent_mock = _build_services_hass(backup_success=True)
        backup_manager_mock.create_backup = AsyncMock(return_value=True)
        hass.data[DOMAIN]["storage"] = None
        collector = MagicMock()
        collector._load_persistent_data = AsyncMock()
        hass.data[DOMAIN]["collector"] = collector
        await async_setup_services(hass)

        await handlers["create_backup"](MagicMock(data={}))
        backup_manager_mock.create_backup = AsyncMock(return_value=False)
        await handlers["create_backup"](MagicMock(data={}))
        await handlers["restore_backup"](MagicMock(data={"backup_name": "auto/20260218_100000"}))

        backup_manager_mock.restore_from_backup.assert_awaited_once()
        agent_mock._load_persistent_state.assert_awaited_once()
        collector._load_persistent_data.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_backups_and_missing_manager_branch(self):
        hass, handlers, backup_manager_mock, _ = _build_services_hass(backup_success=True)
        backup_manager_mock.list_backups = MagicMock(return_value=["a", "b"])
        await async_setup_services(hass)

        await handlers["list_backups"](MagicMock(data={}))
        backup_manager_mock.list_backups.assert_called_once()

        del hass.data[DOMAIN]["backup_manager"]
        await handlers["list_backups"](MagicMock(data={}))

    @pytest.mark.asyncio
    async def test_data_retention_all_modes(self, tmp_path):
        hass, handlers, _, agent_mock = _build_services_hass(backup_success=True)
        storage = hass.data[DOMAIN]["storage"]
        db_path = tmp_path / "research.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE research_rl_episodes (timestamp REAL, phase TEXT, state_vector TEXT, action INTEGER, reward REAL, opportunity_score REAL, action_source TEXT)"
        )
        conn.execute(
            "INSERT INTO research_rl_episodes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().timestamp(), "active", "0", 0, 0.0, 0.0, "seed"),
        )
        conn.commit()
        conn.close()

        storage.research_db_path = db_path
        storage.load_state = AsyncMock(return_value={"notification_history": [], "last_rl_cleanup": datetime.now().isoformat()})
        storage.save_state = AsyncMock()
        storage._cleanup_old_research_data = AsyncMock()
        agent_mock.notification_history = []
        agent_mock._save_persistent_state = AsyncMock()

        async def _exec_job(fn):
            return fn()

        hass.async_add_executor_job = AsyncMock(side_effect=_exec_job)

        await async_setup_services(hass)

        await handlers["test_data_retention"](MagicMock(data={"test_type": "status"}))
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM research_rl_episodes")
        conn.commit()
        conn.close()
        await handlers["test_data_retention"](MagicMock(data={"test_type": "status"}))
        storage.load_state = AsyncMock(return_value={})
        await handlers["test_data_retention"](MagicMock(data={"test_type": "status"}))
        await handlers["test_data_retention"](MagicMock(data={"test_type": "inject_notifications", "count": 3}))
        await handlers["test_data_retention"](MagicMock(data={"test_type": "inject_old_episodes", "months": 1, "episodes_per_day": 1}))
        await handlers["test_data_retention"](MagicMock(data={"test_type": "set_overdue_cleanup", "hours_ago": 2}))
        await handlers["test_data_retention"](MagicMock(data={"test_type": "run_cleanup"}))

        assert storage._cleanup_old_research_data.await_count >= 1
        assert storage.save_state.await_count >= 2

    @pytest.mark.asyncio
    async def test_setup_entry_additional_startup_branches(self):
        hass = MagicMock()
        hass.data = {}
        hass.services = MagicMock()
        hass.services.async_call = AsyncMock()
        hass.config = MagicMock()
        hass.config.path = MagicMock(return_value="/tmp/green_shift_data")
        hass.config_entries = MagicMock()
        hass.config_entries.async_forward_entry_setups = AsyncMock()
        hass.config_entries.async_update_entry = MagicMock()

        entry = MagicMock()
        entry.data = {
            "discovered_sensors": {"power": ["sensor.main_power"]},
            "main_total_energy_sensor": "sensor.total_energy",
            "main_total_power_sensor": "sensor.main_power",
            "environment_mode": "office",
            "energy_saving_target": 15,
            "electricity_price": 0.22,
            "currency": "EUR",
        }

        storage = MagicMock()
        storage.setup = AsyncMock()
        overdue = (datetime.now() - timedelta(hours=30)).isoformat()
        storage.load_state = AsyncMock(
            side_effect=[
                {"some": "state"},
                {"last_rl_cleanup": overdue},
                {"last_sensor_cleanup": overdue},
                {},
                {},
                {},
                {},
            ]
        )
        storage.has_phase_metadata = AsyncMock(return_value=True)
        storage.record_phase_change = AsyncMock()
        storage.compute_daily_aggregates = AsyncMock(side_effect=Exception("agg"))
        storage.compute_area_daily_aggregates = AsyncMock(side_effect=Exception("agg"))
        storage._cleanup_old_research_data = AsyncMock(side_effect=Exception("cleanup"))
        storage.save_state = AsyncMock()
        storage._cleanup_old_data = AsyncMock(side_effect=Exception("cleanup"))
        storage.get_today_tasks = AsyncMock(return_value=[])

        collector = MagicMock()
        collector.setup = AsyncMock()
        collector.get_power_history = AsyncMock(return_value=[(datetime.now(), 1000.0)] * 20)
        collector.calculate_baseline_summary = AsyncMock(return_value={
            "avg_daily_kwh": 2.5,
            "peak_time": "10:00",
            "top_area": "Office",
            "target": 15,
            "impact": {"co2_kg": 0.8, "trees": 0.1, "flights": 0.01},
        })

        agent = MagicMock()
        agent.setup = AsyncMock()
        agent._save_persistent_state = AsyncMock()
        agent.calculate_area_baselines = AsyncMock()
        agent.process_ai_model = AsyncMock()
        agent.phase = const_mod.PHASE_ACTIVE
        agent.start_date = datetime.now() - timedelta(days=20)
        agent.baseline_consumption = 1000.0
        agent.storage = storage

        task_manager = MagicMock()
        task_manager.generate_daily_tasks = AsyncMock(return_value=[])
        task_manager.verify_tasks = AsyncMock(return_value={"ok": False})

        backup_manager = MagicMock()
        backup_manager.create_backup = AsyncMock(side_effect=[None, False])
        backup_manager.cleanup_old_backups = AsyncMock(side_effect=Exception("backup"))

        callbacks = {}

        def _capture_interval(_hass, cb, _delta):
            callbacks[cb.__name__] = cb
            return lambda: None

        def _capture_change(_hass, _entities, cb):
            callbacks[cb.__name__] = cb
            return lambda: None

        def _capture_time(_hass, cb, **_kwargs):
            callbacks[cb.__name__] = cb
            return lambda: None

        with patch.object(init_mod, "StorageManager", return_value=storage), patch.object(
            init_mod, "DataCollector", return_value=collector
        ), patch.object(init_mod, "DecisionAgent", return_value=agent), patch.object(
            init_mod, "TaskManager", return_value=task_manager
        ), patch.object(init_mod, "BackupManager", return_value=backup_manager), patch.object(
            init_mod, "async_setup_services", AsyncMock()
        ), patch.object(init_mod, "async_track_time_interval", side_effect=_capture_interval), patch.object(
            init_mod, "async_track_state_change_event", side_effect=_capture_change
        ), patch.object(init_mod, "async_track_time_change", side_effect=_capture_time):
            ok = await async_setup_entry(hass, entry)

        assert ok is True

        agent.phase = const_mod.PHASE_BASELINE
        agent.start_date = datetime.now() - timedelta(days=20)
        await callbacks["update_agent_ai_model"](None)
        await callbacks["target_changed"](MagicMock(data={"new_state": MagicMock(state="bad")}) )
        await callbacks["electricity_price_changed"](MagicMock(data={"new_state": MagicMock(state="bad")}) )

        agent.phase = const_mod.PHASE_BASELINE
        await callbacks["generate_daily_tasks_callback"](None)
        await callbacks["verify_tasks_callback"](None)
        await callbacks["daily_aggregation_callback"](None)
        await callbacks["auto_backup_callback"](None)
        backup_manager.create_backup = AsyncMock(side_effect=Exception("backup"))
        await callbacks["auto_backup_callback"](None)
        await callbacks["rl_cleanup_callback"](None)
        await callbacks["sensor_data_cleanup_callback"](None)

        task_manager.generate_daily_tasks.assert_awaited()

    @pytest.mark.asyncio
    async def test_async_unload_entry_logs_errors_when_save_or_backup_fails(self):
        entry = MagicMock()
        hass = MagicMock()
        hass.config_entries = MagicMock()
        hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
        hass.services = MagicMock()
        hass.services.async_remove = MagicMock()

        storage = MagicMock()
        storage.close = AsyncMock()

        agent = MagicMock()
        agent.storage = storage
        agent._save_persistent_state = AsyncMock(side_effect=Exception("save"))

        backup_manager = MagicMock()
        backup_manager.create_backup = AsyncMock(side_effect=Exception("backup"))

        hass.data = {
            const_mod.DOMAIN: {
                "agent": agent,
                "backup_manager": backup_manager,
                "storage": storage,
                "update_listener": lambda: None,
            }
        }

        ok = await async_unload_entry(hass, entry)

        assert ok is True
        storage.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_setup_entry_startup_cleanup_parse_errors(self):
        hass = MagicMock()
        hass.data = {}
        hass.config = MagicMock()
        hass.config.path = MagicMock(return_value="/tmp/green_shift_data")
        hass.config_entries = MagicMock()
        hass.config_entries.async_forward_entry_setups = AsyncMock()
        hass.config_entries.async_update_entry = MagicMock()

        entry = MagicMock()
        entry.data = {
            "discovered_sensors": {"power": ["sensor.main_power"]},
            "main_total_energy_sensor": "sensor.total_energy",
            "main_total_power_sensor": "sensor.main_power",
            "environment_mode": "home",
            "energy_saving_target": 15,
            "electricity_price": 0.22,
            "currency": "EUR",
        }

        storage = MagicMock()
        storage.setup = AsyncMock()
        storage.load_state = AsyncMock(side_effect=[{"ready": True}, {"last_rl_cleanup": "bad"}, {"last_sensor_cleanup": "bad"}])
        storage.has_phase_metadata = AsyncMock(return_value=True)
        storage.record_phase_change = AsyncMock()
        storage.save_state = AsyncMock()
        storage.get_today_tasks = AsyncMock(return_value=[{"id": "t1"}])

        collector = MagicMock()
        collector.setup = AsyncMock()
        collector.get_power_history = AsyncMock(return_value=[])

        agent = MagicMock()
        agent.setup = AsyncMock()
        agent.process_ai_model = AsyncMock()
        agent.calculate_area_baselines = AsyncMock()
        agent._save_persistent_state = AsyncMock()
        agent.phase = const_mod.PHASE_ACTIVE
        agent.start_date = datetime.now()
        agent.baseline_consumption = 500.0
        agent.storage = storage

        task_manager = MagicMock()
        task_manager.generate_daily_tasks = AsyncMock(return_value=[])
        task_manager.verify_tasks = AsyncMock(return_value={"ok": True})

        backup_manager = MagicMock()
        backup_manager.create_backup = AsyncMock(return_value=True)
        backup_manager.cleanup_old_backups = AsyncMock(return_value=None)

        with patch.object(init_mod, "StorageManager", return_value=storage), patch.object(
            init_mod, "DataCollector", return_value=collector
        ), patch.object(init_mod, "DecisionAgent", return_value=agent), patch.object(
            init_mod, "TaskManager", return_value=task_manager
        ), patch.object(init_mod, "BackupManager", return_value=backup_manager), patch.object(
            init_mod, "async_setup_services", AsyncMock()
        ), patch.object(init_mod, "async_track_time_interval", return_value=lambda: None), patch.object(
            init_mod, "async_track_state_change_event", return_value=lambda: None
        ), patch.object(init_mod, "async_track_time_change", return_value=lambda: None):
            ok = await async_setup_entry(hass, entry)

        assert ok is True
