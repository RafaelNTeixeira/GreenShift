import logging
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

class EnergyResearchConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for Green Shift integration."""

    VERSION = 1

    def __init__(self):
        """Initialize flow storage."""
        self.data = {}

    async def async_step_user(self, user_input=None):
        """Step 1: Welcome Slide."""
        if user_input is not None:
            return await self.async_step_settings()

        return self.async_show_form(
            step_id="user",
            last_step=False,
        )

    async def async_step_settings(self, user_input=None):
        """Step 2: Currency and energy cost configuration."""
        errors = {}

        if user_input is not None:
            self.data.update(user_input)
            return await self.async_step_intervention_info()

        # Define the schema using a Selector for the dropdown
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
            description_placeholders={
                "description": "Click the currency field to see the list and choose your local currency."
            },
            last_step=False,
        )

    async def async_step_intervention_info(self, user_input=None):
        """Step 3: Explanation of the 14-day trial and AI activation."""
        if user_input is not None:
            # Finalize the flow
            return self.async_create_entry(
                title="Green Shift",
                data=self.data,
            )

        return self.async_show_form(
            step_id="intervention_info",
            last_step=True,
        )
