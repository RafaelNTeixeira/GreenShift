import logging
from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import DOMAIN, GS_AI_UPDATE_SIGNAL, GS_UPDATE_SIGNAL

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Setup of the select entities."""
    collector = hass.data[DOMAIN]["collector"]
    agent = hass.data[DOMAIN]["agent"]

    area_selector = GreenShiftAreaViewSelect(collector)
    notification_selector = GreenShiftNotificationSelect(agent)

    async_add_entities([area_selector, notification_selector])


class GreenShiftAreaViewSelect(SelectEntity):
    """Select entity to filter the dashboard area view."""

    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_translation_key = "area_filter"
    _attr_unique_id = f"{DOMAIN}_area_filter"
    _attr_icon = "mdi:filter-variant"

    def __init__(self, collector):
        self._collector = collector
        self._attr_current_option = "All Areas"
        self._attr_options = ["All Areas"]

    async def async_added_to_hass(self):
        """Update options when added."""
        await super().async_added_to_hass()
        self._update_options()
        # Listen for updates in case new areas are discovered
        self.async_on_remove(
            async_dispatcher_connect(self.hass, GS_UPDATE_SIGNAL, self._update_callback)
        )

    @callback
    def _update_callback(self):
        """Update options if areas change."""
        self._update_options()
        self.async_write_ha_state()

    def _update_options(self):
        """Rebuild the list of options based on discovered areas."""
        # Get areas from collector
        raw_areas = self._collector.get_all_areas()

        # Filter out 'No Area' and sort
        clean_areas = sorted([a for a in raw_areas if a != "No Area"])

        # Build options list
        new_options = ["All Areas"] + clean_areas

        # If options changed, update them
        if new_options != self._attr_options:
            self._attr_options = new_options
            # If current selection is no longer valid, reset to All
            if self._attr_current_option not in new_options:
                self._attr_current_option = "All Areas"

    @property
    def current_option(self) -> str:
        return self._attr_current_option

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        self._attr_current_option = option
        self.async_write_ha_state()

class GreenShiftNotificationSelect(SelectEntity):
    """Select entity to choose a pending notification ID."""

    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_translation_key = "notification_selector"
    _attr_unique_id = f"{DOMAIN}_notification_selector"
    _attr_icon = "mdi:message-badge"

    def __init__(self, agent):
        self._agent = agent
        self._attr_current_option = "No pending notifications"
        self._attr_options = ["No pending notifications"]

    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        self._update_options()
        # Update whenever AI state changes (new notification or feedback given)
        self.async_on_remove(
            async_dispatcher_connect(self.hass, GS_AI_UPDATE_SIGNAL, self._update_callback)
        )

    @callback
    def _update_callback(self):
        self._update_options()
        self.async_write_ha_state()

    def _update_options(self):
        """Fetch pending notification IDs from the agent."""
        pending = [
            n["notification_id"]
            for n in self._agent.notification_history
            if not n.get("responded", False)
        ]

        if not pending:
            self._attr_options = ["No pending notifications"]
            self._attr_current_option = "No pending notifications"
        else:
            self._attr_options = pending
            # If current selection is invalid, pick the first one
            if self._attr_current_option not in pending:
                self._attr_current_option = pending[0]

    @property
    def current_option(self) -> str:
        return self._attr_current_option

    async def async_select_option(self, option: str) -> None:
        if option in self._attr_options:
            self._attr_current_option = option
            self.async_write_ha_state()
