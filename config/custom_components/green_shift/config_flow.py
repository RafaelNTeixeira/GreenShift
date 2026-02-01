import logging
from homeassistant import config_entries
from homeassistant.core import callback
import voluptuous as vol

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# TODO: Need to include pop-up displaying info about the intervention
class EnergyResearchConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for Green Shift integration."""
    
    VERSION = 1
    
    async def async_step_user(self, user_input=None):
        """Handle user-initiated setup."""
        if user_input is not None:
            return self.async_create_entry(
                title="Green Shift",
                data=user_input,
            )
        
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Optional("currency", default="EUR"): str,
                vol.Optional("electricity_price", default=0.25): vol.Coerce(float),
                vol.Optional("co2_factor", default=0.5): vol.Coerce(float),
            }),
        )
    
    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get options flow handler."""
        return EnergyResearchOptionsFlow(config_entry)


class EnergyResearchOptionsFlow(config_entries.OptionsFlow):
    """Options for Green Shift integration."""
    
    def __init__(self, config_entry):
        self.config_entry = config_entry
    
    async def async_step_init(self, user_input=None):
        """Manage options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)
        
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Optional(
                    "enable_normative",
                    default=self.config_entry.options.get("enable_normative", True),
                ): bool,
            }),
        )