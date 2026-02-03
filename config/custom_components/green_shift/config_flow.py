import logging
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    EntitySelector,
    EntitySelectorConfig,
)

from . import async_discover_sensors
from .helpers import get_normalized_value
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

class GreenShiftConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for Green Shift integration."""

    VERSION = 1

    def __init__(self):
        """Initialize flow storage."""
        self.data = {}
        self.discovered_cache = {}

    async def async_step_user(self, user_input=None):
        """Step 1: Welcome Slide."""
        if user_input is not None:
            # Trigger discovery when the user moves past the first slide
            self.discovered_cache = await async_discover_sensors(self.hass)
            _LOGGER.debug("Discovered sensors: %s", self.discovered_cache)
            return await self.async_step_settings()

        return self.async_show_form(step_id="user", last_step=False)

    async def async_step_settings(self, user_input=None):
        """Step 2: Currency and energy cost configuration."""
        errors = {}

        if user_input is not None:
            self.data.update(user_input)
            return await self.async_step_sensor_confirmation()

        data_schema = vol.Schema({
            vol.Required("currency", default="EUR"): SelectSelector(
                SelectSelectorConfig(
                    options=["EUR", "USD", "GBP"],
                    mode=SelectSelectorMode.DROPDOWN,
                    translation_key="currency"
                )
            ),
            vol.Required("electricity_price", default=0.25): vol.Coerce(float),
        })

        return self.async_show_form(
            step_id="settings",
            data_schema=data_schema,
            errors=errors,
            last_step=False,
        )
    
    @callback
    def _get_sorted_entities(self, category: str):
        """Helper to sort entities by their current numeric state value."""
        entities = self.discovered_cache.get(category, [])
        if not entities:
            return []

        entity_values = []
        for entity_id in entities:
            state = self.hass.states.get(entity_id)
            
            # Use the helper to get a clean, normalized float (or None)
            val, _ = get_normalized_value(state, category)
            
            # Handle None if the helper returns it (e.g., unavailable sensor)
            if val is None:
                val = -1.0
                
            entity_values.append((entity_id, val))

        # Sort by value descending (highest first)
        entity_values.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in entity_values]
    
    async def async_step_sensor_confirmation(self, user_input=None):
        """Step 3: Multi-sensor confirmation slide. All fields are optional."""
        if user_input is not None:
            main_energy = user_input.get("main_total_energy_sensor") # Identify main energy sensor
            main_power = user_input.get("main_total_power_sensor") # Identify main power sensor

            self.data["main_total_energy_sensor"] = main_energy
            self.data["main_total_power_sensor"] = main_power

            confirmed_energy = user_input.get("confirmed_energy", [])
            confirmed_power = user_input.get("confirmed_power", [])

            # Append main energy sensor to energy sensors
            if main_energy and main_energy not in confirmed_energy:
                confirmed_energy.append(main_energy)
                _LOGGER.debug("Injected main energy sensor %s into energy list", main_energy)

            # Append main power sensor to power sensors
            if main_power and main_power not in confirmed_power:
                confirmed_power.append(main_power)
                _LOGGER.debug("Injected main power sensor %s into power list", main_power)
            
            # Map the confirmed sensors back to our internal data structure
            confirmed_sensors = {
                "energy": confirmed_energy,
                "power": confirmed_power,
                "temperature": user_input.get("confirmed_temp", []),
                "humidity": user_input.get("confirmed_hum", []),
                "illuminance": user_input.get("confirmed_lux", []),
                "occupancy": user_input.get("confirmed_occ", []),
            }
            
            self.data["discovered_sensors"] = confirmed_sensors
            return await self.async_step_intervention_info()

        # Prepare lists from cache for defaults
        sorted_energy = self._get_sorted_entities("energy")
        sorted_power = self._get_sorted_entities("power")

        _LOGGER.debug(
            "Sorted energy entities: %s, Sorted power entities: %s", 
            sorted_energy, 
            sorted_power
        )
        
        temp_list = self.discovered_cache.get("temperature", [])
        hum_list = self.discovered_cache.get("humidity", [])
        lux_list = self.discovered_cache.get("illuminance", [])
        occ_list = self.discovered_cache.get("occupancy", [])

        # Schema with everything as Optional to allow users to skip
        data_schema = vol.Schema({
            # Main energy sensor (Suggested highest current reading)
            vol.Optional(
                "main_total_energy_sensor", 
                description={"suggested_value": sorted_energy[0] if sorted_energy else None}
            ): EntitySelector(EntitySelectorConfig(domain="sensor", device_class="energy")),

            # Main power sensor (Suggested: Highest current reading)
            vol.Optional(
                "main_total_power_sensor", 
                description={"suggested_value": sorted_power[0] if sorted_power else None}
            ): EntitySelector(EntitySelectorConfig(domain="sensor", device_class="power")),

            # Other energy sensors (Optional/Multiple)
            vol.Optional("confirmed_energy", default=sorted_energy): EntitySelector(
                EntitySelectorConfig(domain="sensor", device_class="energy", multiple=True)
            ),

            # Power sensors (Optional/Multiple)
            vol.Optional("confirmed_power", default=sorted_power): EntitySelector(
                EntitySelectorConfig(domain="sensor", device_class="power", multiple=True)
            ),

            # Environment sensors (Optional/Multiple)
            vol.Optional("confirmed_temp", default=temp_list): EntitySelector(
                EntitySelectorConfig(domain="sensor", device_class="temperature", multiple=True)
            ),
            vol.Optional("confirmed_hum", default=hum_list): EntitySelector(
                EntitySelectorConfig(domain="sensor", device_class="humidity", multiple=True)
            ),
            vol.Optional("confirmed_lux", default=lux_list): EntitySelector(
                EntitySelectorConfig(domain="sensor", device_class="illuminance", multiple=True)
            ),

            # Occupancy sensors (Optional/Multiple)
            vol.Optional("confirmed_occ", default=occ_list): EntitySelector(
                EntitySelectorConfig(domain="binary_sensor", device_class="occupancy", multiple=True)
            ),
        })

        return self.async_show_form(
            step_id="sensor_confirmation",
            data_schema=data_schema,
            description_placeholders={
                "discovered_count": str(sum(len(v) for v in self.discovered_cache.values()))
            },
            last_step=False
        )

    async def async_step_intervention_info(self, user_input=None):
        """Step 4: Final informational slide."""
        if user_input is not None:
            return self.async_create_entry(
                title="Green Shift",
                data=self.data,
            )

        return self.async_show_form(
            step_id="intervention_info",
            last_step=True,
        )