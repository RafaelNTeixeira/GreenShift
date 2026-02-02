import logging
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    EntitySelector,
    EntitySelectorConfig,
)

from . import async_discover_sensors
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
            return await self.async_step_settings()

        return self.async_show_form(step_id="user", last_step=False)

    async def async_step_settings(self, user_input=None):
        """Step 2: Currency and energy cost configuration."""
        errors = {}

        if user_input is not None:
            self.data.update(user_input)
            return await self.async_step_sensor_confirmation()

        data_schema = vol.Schema({
            vol.Required("currency", default="EUR"): SelectSelector( # TODO: Need to set the input_select in HA as well to this picked value
                SelectSelectorConfig(
                    options=["EUR", "USD", "GBP"],
                    mode=SelectSelectorMode.DROPDOWN,
                    translation_key="currency"
                )
            ),
            vol.Required("electricity_price", default=0.25): vol.Coerce(float), # TODO: Need to set the input_number in HA as well to this picked value
        })

        return self.async_show_form(
            step_id="settings",
            data_schema=data_schema,
            errors=errors,
            last_step=False,
        )
    
    async def async_step_sensor_confirmation(self, user_input=None):
        """Step 3: Multi-sensor confirmation slide. All fields are optional."""
        if user_input is not None:
            self.data["main_total_energy_sensor"] = user_input.get("main_total_energy_sensor")
            
            # Map the confirmed sensors back to our internal data structure
            confirmed_sensors = {
                "energy": user_input.get("confirmed_energy", []),
                "power": user_input.get("confirmed_power", []),
                "temperature": user_input.get("confirmed_temp", []),
                "humidity": user_input.get("confirmed_hum", []),
                "illuminance": user_input.get("confirmed_lux", []),
                "occupancy": user_input.get("confirmed_occ", []),
            }
            
            self.data["discovered_sensors"] = confirmed_sensors
            return await self.async_step_intervention_info()

        # Prepare lists from cache for defaults
        energy_list = self.discovered_cache.get("energy", [])
        power_list = self.discovered_cache.get("power", [])
        temp_list = self.discovered_cache.get("temperature", [])
        hum_list = self.discovered_cache.get("humidity", [])
        lux_list = self.discovered_cache.get("illuminance", [])
        occ_list = self.discovered_cache.get("occupancy", [])

        # Schema with everything as Optional to allow users to skip
        data_schema = vol.Schema({
            # Main energy sensor (Optional)
            vol.Optional(
                "main_total_energy_sensor", 
                default=energy_list[0] if energy_list else None
            ): EntitySelector(EntitySelectorConfig(domain="sensor", device_class="energy")),

            # Other energy sensors (Optional/Multiple)
            vol.Optional("confirmed_energy", default=energy_list): EntitySelector(
                EntitySelectorConfig(domain="sensor", device_class="energy", multiple=True)
            ),

            # Power sensors (Optional/Multiple)
            vol.Optional("confirmed_power", default=power_list): EntitySelector(
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