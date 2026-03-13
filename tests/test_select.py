"""
Tests for select.py

Covers:
- GreenShiftAreaViewSelect._update_options: area filtering, sorting, "No Area" exclusion
- GreenShiftAreaViewSelect.current_option: property returns correct value
- GreenShiftAreaViewSelect.async_select_option: updates current option
- GreenShiftNotificationSelect._update_options: pending / no-pending scenarios
- GreenShiftNotificationSelect.current_option: property returns correct value
- GreenShiftNotificationSelect.async_select_option: only valid options accepted
"""

import sys
import types
import pathlib
import importlib.util
import pytest
from unittest.mock import MagicMock, AsyncMock

# -- Minimal HA stubs ---------------------------------------------------------

for mod_name in [
    "homeassistant",
    "homeassistant.components",
    "homeassistant.components.select",
    "homeassistant.config_entries",
    "homeassistant.core",
    "homeassistant.helpers",
    "homeassistant.helpers.entity_platform",
    "homeassistant.helpers.dispatcher",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Minimal SelectEntity base class
class _SelectEntityBase:
    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_translation_key = None
    _attr_unique_id = None
    _attr_icon = None
    _attr_current_option = None
    _attr_options = []
    hass = None

    async def async_added_to_hass(self):
        pass

    def async_write_ha_state(self):
        pass

    def async_on_remove(self, unsub):
        pass


sys.modules["homeassistant.components.select"].SelectEntity = _SelectEntityBase
sys.modules["homeassistant.core"].callback = lambda f: f
sys.modules["homeassistant.helpers.dispatcher"].async_dispatcher_connect = MagicMock(
    return_value=MagicMock()
)

# -- Real const module ---------------------------------------------------------

const_spec = importlib.util.spec_from_file_location(
    "custom_components.green_shift.const",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "const.py",
)
const_mod = importlib.util.module_from_spec(const_spec)
const_mod.__package__ = "custom_components.green_shift"
const_spec.loader.exec_module(const_mod)
sys.modules["custom_components.green_shift.const"] = const_mod

# -- Load select module --------------------------------------------------------

select_spec = importlib.util.spec_from_file_location(
    "gs_select",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "select.py",
)
select_mod = importlib.util.module_from_spec(select_spec)
select_mod.__package__ = "custom_components.green_shift"
select_spec.loader.exec_module(select_mod)

GreenShiftAreaViewSelect = select_mod.GreenShiftAreaViewSelect
GreenShiftNotificationSelect = select_mod.GreenShiftNotificationSelect


class TestSelectSetupEntry:

    @pytest.mark.asyncio
    async def test_async_setup_entry_adds_both_select_entities(self):
        hass = MagicMock()
        collector = make_collector(["Kitchen"])
        agent = make_agent([pending_notif("n1")])
        hass.data = {const_mod.DOMAIN: {"collector": collector, "agent": agent}}
        async_add_entities = MagicMock()

        await select_mod.async_setup_entry(hass, MagicMock(), async_add_entities)

        async_add_entities.assert_called_once()
        entities = async_add_entities.call_args.args[0]
        assert len(entities) == 2
        assert isinstance(entities[0], GreenShiftAreaViewSelect)
        assert isinstance(entities[1], GreenShiftNotificationSelect)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_collector(areas=None):
    collector = MagicMock()
    collector.get_all_areas = MagicMock(return_value=areas or [])
    return collector


def make_agent(notifications=None):
    agent = MagicMock()
    agent.notification_history = notifications or []
    return agent


def pending_notif(nid):
    return {"notification_id": nid, "responded": False}


def responded_notif(nid):
    return {"notification_id": nid, "responded": True}


# -----------------------------------------------------------------------------
# GreenShiftAreaViewSelect
# -----------------------------------------------------------------------------

class TestAreaViewSelectInit:

    def test_initial_current_option_is_all_areas(self):
        sel = GreenShiftAreaViewSelect(make_collector())
        assert sel._attr_current_option == "All Areas"

    def test_initial_options_list_contains_only_all_areas(self):
        sel = GreenShiftAreaViewSelect(make_collector())
        assert sel._attr_options == ["All Areas"]


class TestAreaViewSelectUpdateOptions:

    def test_empty_collector_keeps_only_all_areas(self):
        sel = GreenShiftAreaViewSelect(make_collector([]))
        sel._update_options()
        assert sel._attr_options == ["All Areas"]

    def test_no_area_is_filtered_out(self):
        sel = GreenShiftAreaViewSelect(make_collector(["Living Room", "No Area", "Kitchen"]))
        sel._update_options()
        assert "No Area" not in sel._attr_options

    def test_all_areas_always_prepended(self):
        sel = GreenShiftAreaViewSelect(make_collector(["Kitchen"]))
        sel._update_options()
        assert sel._attr_options[0] == "All Areas"

    def test_areas_sorted_alphabetically(self):
        sel = GreenShiftAreaViewSelect(make_collector(["Zebra Room", "Alpha Room", "Middle Room"]))
        sel._update_options()
        assert sel._attr_options == ["All Areas", "Alpha Room", "Middle Room", "Zebra Room"]

    def test_current_option_reset_to_all_areas_when_invalid(self):
        sel = GreenShiftAreaViewSelect(make_collector(["Old Room"]))
        sel._update_options()
        sel._attr_current_option = "Old Room"
        sel._collector.get_all_areas.return_value = ["New Room"]
        sel._update_options()
        assert sel._attr_current_option == "All Areas"

    def test_current_option_kept_when_still_in_list(self):
        sel = GreenShiftAreaViewSelect(make_collector(["Kitchen", "Bedroom"]))
        sel._update_options()
        sel._attr_current_option = "Kitchen"
        sel._collector.get_all_areas.return_value = ["Kitchen", "Bedroom", "Living Room"]
        sel._update_options()
        assert sel._attr_current_option == "Kitchen"

    def test_multiple_areas_all_included(self):
        sel = GreenShiftAreaViewSelect(make_collector(["Room A", "Room B", "Room C"]))
        sel._update_options()
        assert len(sel._attr_options) == 4  # All Areas + 3 rooms

    def test_only_no_area_returns_just_all_areas(self):
        sel = GreenShiftAreaViewSelect(make_collector(["No Area"]))
        sel._update_options()
        assert sel._attr_options == ["All Areas"]


class TestAreaViewSelectProperties:

    def test_current_option_returns_attr(self):
        sel = GreenShiftAreaViewSelect(make_collector(["Kitchen"]))
        sel._attr_current_option = "Kitchen"
        assert sel.current_option == "Kitchen"

    def test_current_option_returns_all_areas_by_default(self):
        sel = GreenShiftAreaViewSelect(make_collector())
        assert sel.current_option == "All Areas"


class TestAreaViewSelectSelectOption:

    @pytest.mark.asyncio
    async def test_select_existing_area(self):
        sel = GreenShiftAreaViewSelect(make_collector(["Kitchen"]))
        sel._update_options()
        await sel.async_select_option("Kitchen")
        assert sel._attr_current_option == "Kitchen"

    @pytest.mark.asyncio
    async def test_select_all_areas(self):
        sel = GreenShiftAreaViewSelect(make_collector(["Kitchen"]))
        sel._update_options()
        await sel.async_select_option("All Areas")
        assert sel._attr_current_option == "All Areas"

    @pytest.mark.asyncio
    async def test_select_changes_state(self):
        sel = GreenShiftAreaViewSelect(make_collector(["Kitchen", "Bedroom"]))
        sel._update_options()
        await sel.async_select_option("Bedroom")
        assert sel._attr_current_option == "Bedroom"


# -----------------------------------------------------------------------------
# GreenShiftNotificationSelect
# -----------------------------------------------------------------------------

class TestNotificationSelectInit:

    def test_initial_option_is_no_pending(self):
        sel = GreenShiftNotificationSelect(make_agent())
        assert sel._attr_current_option == "No pending notifications"

    def test_initial_options_list_has_no_pending_placeholder(self):
        sel = GreenShiftNotificationSelect(make_agent())
        assert sel._attr_options == ["No pending notifications"]


class TestNotificationSelectUpdateOptions:

    def test_empty_history_keeps_default(self):
        sel = GreenShiftNotificationSelect(make_agent([]))
        sel._update_options()
        assert sel._attr_options == ["No pending notifications"]
        assert sel._attr_current_option == "No pending notifications"

    def test_only_responded_notifications_keeps_default(self):
        agent = make_agent([responded_notif("n1"), responded_notif("n2")])
        sel = GreenShiftNotificationSelect(agent)
        sel._update_options()
        assert sel._attr_options == ["No pending notifications"]

    def test_pending_notification_appears_in_options(self):
        agent = make_agent([pending_notif("n1"), responded_notif("n2")])
        sel = GreenShiftNotificationSelect(agent)
        sel._update_options()
        assert "n1" in sel._attr_options
        assert "n2" not in sel._attr_options

    def test_multiple_pending_all_in_options(self):
        agent = make_agent([pending_notif("n1"), pending_notif("n2"), pending_notif("n3")])
        sel = GreenShiftNotificationSelect(agent)
        sel._update_options()
        assert sel._attr_options == ["n1", "n2", "n3"]

    def test_first_pending_selected_when_current_invalid(self):
        agent = make_agent([pending_notif("n1"), pending_notif("n2")])
        sel = GreenShiftNotificationSelect(agent)
        sel._attr_current_option = "old_invalid_id"
        sel._update_options()
        assert sel._attr_current_option == "n1"

    def test_current_selection_kept_when_still_pending(self):
        agent = make_agent([pending_notif("n1"), pending_notif("n2")])
        sel = GreenShiftNotificationSelect(agent)
        sel._attr_current_option = "n2"
        sel._update_options()
        assert sel._attr_current_option == "n2"

    def test_revert_to_default_when_all_notifications_responded(self):
        notif = {"notification_id": "n1", "responded": False}
        agent = make_agent([notif])
        sel = GreenShiftNotificationSelect(agent)
        sel._update_options()
        assert sel._attr_current_option == "n1"

        # All responded now
        notif["responded"] = True
        sel._update_options()
        assert sel._attr_current_option == "No pending notifications"


class TestNotificationSelectProperties:

    def test_current_option_property_returns_attr(self):
        sel = GreenShiftNotificationSelect(make_agent())
        sel._attr_current_option = "some_id"
        assert sel.current_option == "some_id"


class TestNotificationSelectSelectOption:

    @pytest.mark.asyncio
    async def test_select_valid_notification(self):
        agent = make_agent([pending_notif("n1"), pending_notif("n2")])
        sel = GreenShiftNotificationSelect(agent)
        sel._update_options()
        await sel.async_select_option("n2")
        assert sel._attr_current_option == "n2"

    @pytest.mark.asyncio
    async def test_reject_invalid_option_keeps_current(self):
        agent = make_agent([pending_notif("n1")])
        sel = GreenShiftNotificationSelect(agent)
        sel._update_options()
        assert sel._attr_current_option == "n1"
        await sel.async_select_option("nonexistent_id")
        assert sel._attr_current_option == "n1"

    @pytest.mark.asyncio
    async def test_select_changes_between_pending_notifications(self):
        agent = make_agent([pending_notif("n1"), pending_notif("n2"), pending_notif("n3")])
        sel = GreenShiftNotificationSelect(agent)
        sel._update_options()
        await sel.async_select_option("n3")
        assert sel._attr_current_option == "n3"
        await sel.async_select_option("n1")
        assert sel._attr_current_option == "n1"


# -----------------------------------------------------------------------------
# Callback methods: async_added_to_hass and _update_callback
# -----------------------------------------------------------------------------

class TestAreaViewSelectCallbacks:

    @pytest.mark.asyncio
    async def test_async_added_to_hass_calls_async_on_remove(self):
        """async_added_to_hass should register a dispatcher listener via async_on_remove."""
        sel = GreenShiftAreaViewSelect(make_collector(["Kitchen"]))
        sel.hass = MagicMock()
        sel.async_on_remove = MagicMock()
        sel.async_write_ha_state = MagicMock()
        await sel.async_added_to_hass()
        assert sel.async_on_remove.called

    @pytest.mark.asyncio
    async def test_async_added_to_hass_updates_options(self):
        """async_added_to_hass should call _update_options so initial list is correct."""
        sel = GreenShiftAreaViewSelect(make_collector(["Bedroom", "Kitchen"]))
        sel.hass = MagicMock()
        sel.async_on_remove = MagicMock()
        sel.async_write_ha_state = MagicMock()
        await sel.async_added_to_hass()
        assert "Bedroom" in sel._attr_options
        assert "Kitchen" in sel._attr_options

    def test_update_callback_writes_ha_state(self):
        """_update_callback should call async_write_ha_state."""
        sel = GreenShiftAreaViewSelect(make_collector(["Kitchen"]))
        sel.async_write_ha_state = MagicMock()
        sel._update_callback()
        sel.async_write_ha_state.assert_called_once()

    def test_update_callback_refreshes_options(self):
        """_update_callback should rebuild the options list."""
        collector = make_collector(["Old Room"])
        sel = GreenShiftAreaViewSelect(collector)
        sel._update_options()
        assert "Old Room" in sel._attr_options

        sel.async_write_ha_state = MagicMock()
        collector.get_all_areas.return_value = ["New Room"]
        sel._update_callback()
        assert "New Room" in sel._attr_options


class TestNotificationSelectCallbacks:

    @pytest.mark.asyncio
    async def test_async_added_to_hass_calls_async_on_remove(self):
        """async_added_to_hass should register a dispatcher listener."""
        sel = GreenShiftNotificationSelect(make_agent([pending_notif("n1")]))
        sel.hass = MagicMock()
        sel.async_on_remove = MagicMock()
        sel.async_write_ha_state = MagicMock()
        await sel.async_added_to_hass()
        assert sel.async_on_remove.called

    @pytest.mark.asyncio
    async def test_async_added_to_hass_updates_options(self):
        """async_added_to_hass should call _update_options so initial pending list is set."""
        sel = GreenShiftNotificationSelect(make_agent([pending_notif("n1"), pending_notif("n2")]))
        sel.hass = MagicMock()
        sel.async_on_remove = MagicMock()
        sel.async_write_ha_state = MagicMock()
        await sel.async_added_to_hass()
        assert "n1" in sel._attr_options
        assert "n2" in sel._attr_options

    def test_update_callback_writes_ha_state(self):
        """_update_callback should call async_write_ha_state."""
        sel = GreenShiftNotificationSelect(make_agent([pending_notif("n1")]))
        sel.async_write_ha_state = MagicMock()
        sel._update_callback()
        sel.async_write_ha_state.assert_called_once()

    def test_update_callback_refreshes_pending_list(self):
        """_update_callback rebuilds pending notifications from agent history."""
        notif = {"notification_id": "n1", "responded": False}
        sel = GreenShiftNotificationSelect(make_agent([notif]))
        sel.async_write_ha_state = MagicMock()
        sel._update_callback()
        assert "n1" in sel._attr_options

        notif["responded"] = True
        sel._update_callback()
        assert sel._attr_options == ["No pending notifications"]
