import logging
from datetime import datetime, timedelta
import numpy as np
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED

from .const import (
    DOMAIN,
    SENSOR_CATEGORIES,
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
    """
    Auto-Discovery of sensors in Home Assistant Entity Registry.
    Returns a dictionary with categories and lists of entity_ids.
    """
    entity_reg = er.async_get(hass)
    discovered = {cat: [] for cat in SENSOR_CATEGORIES}
    
    for entity in entity_reg.entities.values():
        entity_id = entity.entity_id
        original_name = entity.original_name or ""
        unit = entity.unit_of_measurement or ""
        
        # Verify each category
        for category, keywords in SENSOR_CATEGORIES.items():
            if any(kw in entity_id.lower() or 
                   kw in original_name.lower() or 
                   kw in unit.lower() for kw in keywords):
                discovered[category].append(entity_id)
                _LOGGER.debug("Discovered %s sensor: %s", category, entity_id)
                break
    
    # Log summary of discovered sensors
    for cat, entities in discovered.items():
        _LOGGER.info("Found %d %s sensors: %s", len(entities), cat, entities)
    
    return discovered