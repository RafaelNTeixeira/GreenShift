import logging
from datetime import datetime, timedelta
import numpy as np
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    DOMAIN,
    SENSOR_MAPPING,
    PHASE_BASELINE,
    PHASE_ACTIVE,
    BASELINE_DAYS,
    UPDATE_INTERVAL_SECONDS,
)
from .decision_agent import DecisionAgent

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["sensor"]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Setup of the component through config entry."""
    hass.data.setdefault(DOMAIN, {})
    
    # Auto-discovery of the sensors
    discovered_sensors = await async_discover_sensors(hass)
    
    # Initialize the decision agent
    agent = DecisionAgent(hass, discovered_sensors)
    hass.data[DOMAIN]["agent"] = agent
    hass.data[DOMAIN]["discovered_sensors"] = discovered_sensors
    hass.data[DOMAIN]["start_date"] = datetime.now()
    
    # Platform setup
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    # Periodic update task
    async def update_agent(now):
        """Update agent state periodically."""
        await agent.update_state()
        
        days_running = (datetime.now() - hass.data[DOMAIN]["start_date"]).days

        # During baseline phase: continuously update baseline_consumption
        if agent.phase == PHASE_BASELINE and len(agent.consumption_history) > 0:
            agent.baseline_consumption = np.mean(agent.consumption_history)
            _LOGGER.debug("Baseline consumption updated: %.2f W", agent.baseline_consumption)
        
        # Verify if the baseline phase is complete
        if days_running >= BASELINE_DAYS and agent.phase == PHASE_BASELINE:
            agent.phase = PHASE_ACTIVE
            # Freeze baseline_consumption and set fixed baseline for active phase
            agent.baseline_consumption_week = agent.baseline_consumption
            _LOGGER.info("System entered active phase after %d days with baseline: %.2f W", days_running, agent.baseline_consumption)
    
    hass.data[DOMAIN]["update_listener"] = async_track_time_interval(
        hass, update_agent, timedelta(seconds=UPDATE_INTERVAL_SECONDS)
    )
    
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload of the config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        hass.data[DOMAIN]["update_listener"]()
        hass.data.pop(DOMAIN)
    return unload_ok


async def async_discover_sensors(hass: HomeAssistant) -> dict:
    entity_reg = er.async_get(hass)
    device_reg = dr.async_get(hass)
    discovered = {cat: [] for cat in SENSOR_MAPPING}
    
    for entity in entity_reg.entities.values():
        if entity.platform == DOMAIN:
            continue

        if entity.device_id is None:
            continue

        device = device_reg.devices.get(entity.device_id)
        if device is None:
            continue

        if device.manufacturer == "Home Assistant": 
            continue

        entity_id = entity.entity_id
        state = hass.states.get(entity_id)
        
        device_class = entity.device_class or (state.attributes.get("device_class") if state else None)
        unit = entity.unit_of_measurement or (state.attributes.get("unit_of_measurement") if state else "")
        original_name = (entity.original_name or "").lower()

        for category, criteria in SENSOR_MAPPING.items():
            # 1. Check Device Class (First option)
            if device_class in criteria["classes"]:
                discovered[category].append(entity_id)
                break
            
            # 2. Check Units (Second option)
            if unit in criteria["units"]:
                discovered[category].append(entity_id)
                break

            # 3. Fallback to Keywords (Last option)
            if any(kw in entity_id.lower() or kw in original_name for kw in criteria["keywords"]):
                discovered[category].append(entity_id)
                break
            _LOGGER.debug("Entity %s did not match category %s", entity_id, category)
    
    _LOGGER.info("Discovered sensors: %s", discovered)
    return discovered