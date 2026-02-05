import logging
from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import DOMAIN, GS_UPDATE_SIGNAL

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Setup of the select entity."""
    collector = hass.data[DOMAIN]["collector"]
    
    selector = GreenShiftAreaViewSelect(collector)
    async_add_entities([selector])


class GreenShiftAreaViewSelect(SelectEntity):
    """Select entity to filter the dashboard area view."""

    _attr_should_poll = False
    _attr_name = "Dashboard Area Filter"
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