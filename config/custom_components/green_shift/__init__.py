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
from .data_collector import DataCollector
from .decision_agent import DecisionAgent

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["sensor"]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Setup of the component through config entry."""
    hass.data.setdefault(DOMAIN, {})
    
    # Use confirmed sensors from the user configuration
    discovered_sensors = entry.data.get("discovered_sensors")

    _LOGGER.debug("Setting up Green Shift with sensors: %s", discovered_sensors)

    # TODO: If no sensors were found, perform discovery again ?
    # if not discovered_sensors:
    #     discovered_sensors = await async_discover_sensors(hass)
    
    main_energy_sensor = entry.data.get("main_total_energy_sensor")
    main_power_sensor = entry.data.get("main_total_power_sensor")

    _LOGGER.info("Configuring Green Shift with main energy sensor: %s", main_energy_sensor)
    _LOGGER.info("Configuring Green Shift with main power sensor: %s", main_power_sensor)

    await sync_helper_entities(hass, entry)

    # Initialize the real-time data collector
    collector = DataCollector(hass, discovered_sensors, main_energy_sensor, main_power_sensor)
    await collector.setup()
    
    # Initialize the decision agent (AI)
    agent = DecisionAgent(hass, discovered_sensors, collector)
    
    hass.data[DOMAIN]["collector"] = collector
    hass.data[DOMAIN]["agent"] = agent
    hass.data[DOMAIN]["discovered_sensors"] = discovered_sensors
    hass.data[DOMAIN]["start_date"] = datetime.now()
    
    # Platform setup
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    # Periodic AI model update task (runs every UPDATE_INTERVAL_SECONDS)
    async def update_agent_ai_model(now):
        """Update agent AI model periodically - processes data collected by DataCollector."""
        _LOGGER.debug("Running AI model update cycle")
        
        # Run AI model processing
        await agent.process_ai_model()
        
        days_running = (datetime.now() - hass.data[DOMAIN]["start_date"]).days

        # During baseline phase: continuously update baseline_consumption
        if agent.phase == PHASE_BASELINE:
            power_history = collector.get_power_history()
            if len(power_history) > 0:
                agent.baseline_consumption = np.mean(power_history)
                _LOGGER.debug("Baseline consumption updated: %.2f kW", agent.baseline_consumption)
        
        # Verify if the baseline phase is complete
        if days_running >= BASELINE_DAYS and agent.phase == PHASE_BASELINE:
            agent.phase = PHASE_ACTIVE
            # Freeze baseline_consumption and set fixed baseline for active phase
            agent.baseline_consumption_week = agent.baseline_consumption
            _LOGGER.info("System entered active phase after %d days with baseline: %.2f kW", 
                        days_running, agent.baseline_consumption)
    
    hass.data[DOMAIN]["update_listener"] = async_track_time_interval(
        hass, update_agent_ai_model, timedelta(seconds=UPDATE_INTERVAL_SECONDS)
    )
    
    return True

async def sync_helper_entities(hass: HomeAssistant, entry: ConfigEntry):
    """Syncs the options chosen in the Config Flow to the corresponding helper entities in Home Assistant."""
    chosen_currency = entry.data.get("currency", "EUR")
    chosen_price = entry.data.get("electricity_price", 0.25)

    # Update currency (input_select)
    try:
        await hass.services.async_call(
            "input_select",
            "select_option",
            {"entity_id": "input_select.currency", "option": chosen_currency},
            blocking=False,
        )
        _LOGGER.debug("Synced currency helper to %s", chosen_currency)
    except Exception as e:
        _LOGGER.warning("Could not sync input_select.currency: %s", e)

    # Update electricity price (input_number)
    try:
        await hass.services.async_call(
            "input_number",
            "set_value",
            {"entity_id": "input_number.electricity_price", "value": chosen_price},
            blocking=False,
        )
        _LOGGER.debug("Synced electricity_price helper to %.2f", chosen_price)
    except Exception as e:
        _LOGGER.warning("Could not sync input_number.electricity_price: %s", e)

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
        if entity.platform == DOMAIN or entity.device_id is None:
            continue

        device = device_reg.devices.get(entity.device_id)
        if device is None or device.manufacturer == "Home Assistant": 
            continue

        entity_id = entity.entity_id
        state = hass.states.get(entity_id)
        
        device_class = entity.device_class or (state.attributes.get("device_class") if state else None)
        unit = entity.unit_of_measurement or (state.attributes.get("unit_of_measurement") if state else "")
        _LOGGER.debug("Evaluating entity %s: device_class=%s, unit=%s", entity_id, device_class, unit)
        original_name = (entity.original_name or "").lower()

        # Direct Matching by Device Class or Unit
        matched_category = None
        for category, criteria in SENSOR_MAPPING.items():
            if device_class in criteria["classes"] or (unit and unit in criteria["units"]):
                matched_category = category
                break # Found a definitive match
        
        # Fallback to Keyword Matching
        if not matched_category:
            for category, criteria in SENSOR_MAPPING.items():
                if any(kw in entity_id.lower() or kw in original_name for kw in criteria["keywords"]):
                    matched_category = category
                    break

        if matched_category:
            discovered[matched_category].append(entity_id)
            _LOGGER.debug("Entity %s classified as %s", entity_id, matched_category)
            
    _LOGGER.info("Discovery complete: %s", discovered)

    return discovered