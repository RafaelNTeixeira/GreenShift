import logging

_LOGGER = logging.getLogger(__name__)
DOMAIN = "green_shift"

async def async_setup(hass, config):
    """Set up the integration."""
    _LOGGER.info("Green Shift Integration is starting up!")
    return True