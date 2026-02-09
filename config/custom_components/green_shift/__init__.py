import logging
from datetime import datetime, timedelta
import numpy as np
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.event import async_track_time_interval, async_track_state_change_event, async_track_time_change
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import (
    DOMAIN,
    GS_AI_UPDATE_SIGNAL,
    SENSOR_MAPPING,
    PHASE_BASELINE,
    PHASE_ACTIVE,
    BASELINE_DAYS,
    AI_FREQUENCY_SECONDS,
    TASK_GENERATION_TIME,
    VERIFY_TASKS_INTERVAL_MINUTES
)
from .data_collector import DataCollector
from .decision_agent import DecisionAgent
from .storage import StorageManager
from .task_manager import TaskManager

_LOGGER = logging.getLogger(__name__)
PLATFORMS = ["sensor", "select"]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Setup of the component through config entry."""
    hass.data.setdefault(DOMAIN, {})
    
    # Use sensors selected from the user configuration
    discovered_sensors = entry.data.get("discovered_sensors")

    _LOGGER.debug("Setting up Green Shift with sensors: %s", discovered_sensors)
    
    main_energy_sensor = entry.data.get("main_total_energy_sensor")
    main_power_sensor = entry.data.get("main_total_power_sensor")

    _LOGGER.info("Configuring Green Shift with main energy sensor: %s", main_energy_sensor)
    _LOGGER.info("Configuring Green Shift with main power sensor: %s", main_power_sensor)

    await sync_helper_entities(hass, entry)

    # Initialize storage manager (SQLite + JSON)
    storage = StorageManager(hass)
    await storage.setup()

    # Initialize the real-time data collector
    collector = DataCollector(hass, discovered_sensors, main_energy_sensor, main_power_sensor, storage)
    await collector.setup()
    
    # Initialize the decision agent (AI)
    agent = DecisionAgent(hass, discovered_sensors, collector, storage)
    await agent.setup()

    # Initialize task manager
    task_manager = TaskManager(hass, discovered_sensors, collector, storage)

    if not await storage.load_state():
        _LOGGER.debug("Fresh install detected: Saving initial start_date.")
        await agent._save_persistent_state()
    
    hass.data[DOMAIN]["storage"] = storage
    hass.data[DOMAIN]["collector"] = collector
    hass.data[DOMAIN]["agent"] = agent
    hass.data[DOMAIN]["task_manager"] = task_manager
    hass.data[DOMAIN]["discovered_sensors"] = discovered_sensors
    
    # Platform setup
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    await async_setup_services(hass)
    
    # Periodic AI model update task (runs every AI_FREQUENCY_SECONDS)
    async def update_agent_ai_model(now):
        """Update agent AI model periodically."""
        _LOGGER.debug("Running AI model update cycle")
        
        # Run AI model processing
        await agent.process_ai_model()
        
        days_running = (datetime.now() - agent.start_date).days

        days_running = 14 # TEMP: For testing purposes, simulate baseline phase completion after 14 days

        # During baseline phase: continuously update baseline_consumption
        if agent.phase == PHASE_BASELINE:
            power_history_data = await collector.get_power_history(days=days_running if days_running > 0 else None)
            power_values = [power for timestamp, power in power_history_data]

            if len(power_values) > 0:
                agent.baseline_consumption = np.mean(power_values)
                _LOGGER.debug("Baseline consumption updated: %.2f kW", agent.baseline_consumption)
        
        # Verify if the baseline phase is complete
        if days_running >= BASELINE_DAYS and agent.phase == PHASE_BASELINE:
            agent.phase = PHASE_ACTIVE
            _LOGGER.info("System entered active phase after %d days with baseline: %.2f kW", days_running, agent.baseline_consumption)
            # Trigger the new notification function
            await trigger_phase_transition_notification(hass, agent, collector)

            # Save phase transition to persistent storage
            if agent.storage:
                await agent._save_persistent_state()
            
        async_dispatcher_send(hass, GS_AI_UPDATE_SIGNAL)
    
    hass.data[DOMAIN]["update_listener"] = async_track_time_interval(
        hass, update_agent_ai_model, timedelta(seconds=AI_FREQUENCY_SECONDS)
    )

    # Listener for changes to the energy saving target slider
    async def target_changed(event):
        """Handle changes to the energy saving target slider."""
        _LOGGER.debug("Energy saving target changed, triggering update")
        async_dispatcher_send(hass, GS_AI_UPDATE_SIGNAL)
    
    hass.data[DOMAIN]["target_listener"] = async_track_state_change_event(
        hass, 
        ["input_number.energy_saving_target"],
        target_changed
    )

    # Daily task generation at TASK_GENERATION_TIME
    async def generate_daily_tasks_callback(now):
        """Generate daily tasks at TASK_GENERATION_TIME."""
        if agent.phase != PHASE_ACTIVE:
            _LOGGER.debug("Skipping task generation - system in %s phase", agent.phase)
            return
        
        _LOGGER.info("Generating daily tasks...")
        tasks = await task_manager.generate_daily_tasks()
        if tasks:
            _LOGGER.info("Generated %d tasks for today", len(tasks))
            async_dispatcher_send(hass, GS_AI_UPDATE_SIGNAL)
    
    hass.data[DOMAIN]["task_generation_listener"] = async_track_time_change(
        hass, generate_daily_tasks_callback, hour=TASK_GENERATION_TIME[0], minute=TASK_GENERATION_TIME[1], second=TASK_GENERATION_TIME[2]
    )

    async def verify_tasks_callback(now):
        """Verify tasks periodically every VERIFY_TASKS_INTERVAL_MINUTES minutes."""
        if agent.phase != PHASE_ACTIVE:
            _LOGGER.debug("Skipping task verification - system in %s phase", agent.phase)
            return
        
        results = await task_manager.verify_tasks()
        if any(results.values()):
            _LOGGER.info("Task verification completed: %s", results)
            async_dispatcher_send(hass, GS_AI_UPDATE_SIGNAL)
    
    hass.data[DOMAIN]["task_verification_listener"] = async_track_time_interval(
        hass, verify_tasks_callback, timedelta(minutes=VERIFY_TASKS_INTERVAL_MINUTES)
    )

    # Generate tasks immediately if none exist for today (only in active phase)
    if agent.phase == PHASE_ACTIVE:
        today_tasks = await storage.get_today_tasks()
        if not today_tasks:
            _LOGGER.info("No tasks found for today, generating now...")
            await task_manager.generate_daily_tasks()
    
    return True


async def async_setup_services(hass: HomeAssistant):
    """Setup services for task management."""
    
    async def submit_task_feedback(call: ServiceCall):
        """Service to submit task difficulty feedback."""
        task_index = call.data.get("task_index")
        feedback = call.data.get("feedback")
        
        if task_index is None or feedback is None:
            _LOGGER.error("Task index or feedback not provided")
            return
        
        if feedback not in ['too_easy', 'just_right', 'too_hard']:
            _LOGGER.error("Invalid feedback value: %s", feedback)
            return
        
        # Get today's tasks and find the task_id by index
        storage = hass.data[DOMAIN]["storage"]
        tasks = await storage.get_today_tasks()
        
        if not tasks or task_index >= len(tasks):
            _LOGGER.error("Invalid task index %d (only %d tasks available)", task_index, len(tasks) if tasks else 0)
            return
        
        task_id = tasks[task_index].get("task_id")
        if not task_id:
            _LOGGER.error("Could not find task_id for task index %d", task_index)
            return
        
        success = await storage.save_task_feedback(task_id, feedback)
        
        if success:
            _LOGGER.info("Feedback '%s' saved for task %s (index %d)", feedback, task_id, task_index)
            async_dispatcher_send(hass, GS_AI_UPDATE_SIGNAL)
        else:
            _LOGGER.error("Failed to save feedback for task %s", task_id)
    
    async def verify_tasks(call: ServiceCall):
        """Service to manually trigger task verification."""
        task_manager = hass.data[DOMAIN]["task_manager"]
        results = await task_manager.verify_tasks()
        
        _LOGGER.info("Manual task verification completed: %s", results)
        async_dispatcher_send(hass, GS_AI_UPDATE_SIGNAL)
    
    async def regenerate_tasks(call: ServiceCall):
        """Service to force regeneration of today's tasks (admin only)."""
        storage = hass.data[DOMAIN]["storage"]
        task_manager = hass.data[DOMAIN]["task_manager"]
        
        # Delete today's tasks first
        # today = datetime.now().strftime("%Y-%m-%d")
        # This would require a new storage method, for now just generate new ones
        
        tasks = await task_manager.generate_daily_tasks()
        _LOGGER.info("Tasks regenerated: %d tasks", len(tasks))
        async_dispatcher_send(hass, GS_AI_UPDATE_SIGNAL)
    
    # Register services
    hass.services.async_register(DOMAIN, "submit_task_feedback", submit_task_feedback)
    hass.services.async_register(DOMAIN, "verify_tasks", verify_tasks)
    hass.services.async_register(DOMAIN, "regenerate_tasks", regenerate_tasks)
    
    _LOGGER.info("Services registered successfully")


async def trigger_phase_transition_notification(hass, agent, collector):
    """Calculates baseline summary and sends the transition notification."""
    # Fetch data from collector
    summary = await collector.calculate_baseline_summary()
    impact = summary.get("impact", {})
    target = summary.get("target", 15)

    # Build the message
    notification_msg = (
        f"### Baseline Phase Complete! 🎉\n\n"
        f"**Daily Average:** {summary['avg_daily_kwh']} kWh\n"
        f"**Peak Usage:** {summary['peak_time']}\n"
    )

    if summary.get('top_area'):
        notification_msg += f"**Main Area:** {summary['top_area']}\n"

    notification_msg += (
        f"\n**Target:** We've set a **{target}%** reduction goal for you (you can change this in the Settings tab)\n"
        f"\n---\n"
        f"### Your Potential Impact 🌍\n"
        f"By hitting your **{summary['target']}% target**, in one year you would save:\n"
        f"* **{impact.get('co2_kg', 0)} kg** of CO₂\n"
        f"* The equivalent of planting **{impact.get('trees', 0)}** mature trees\n"
        f"* The carbon offset of **{impact.get('flights', 0)}** short-haul flights\n"
    )

    # Send the notification 
    await hass.services.async_call(
        "persistent_notification", "create",
        {
            "title": "Green Shift: Action Phase Started",
            "message": notification_msg,
            "notification_id": "gs_phase_transition"
        }
    )


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
    # Unregister services
    hass.services.async_remove(DOMAIN, "submit_task_feedback")
    hass.services.async_remove(DOMAIN, "verify_tasks")
    hass.services.async_remove(DOMAIN, "regenerate_tasks")

    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        hass.data[DOMAIN]["update_listener"]()

        # Cancel task listeners
        if "task_generation_listener" in hass.data[DOMAIN]:
            hass.data[DOMAIN]["task_generation_listener"]()
        
        if "task_verification_listener" in hass.data[DOMAIN]:
            hass.data[DOMAIN]["task_verification_listener"]()

        # Close storage connections
        storage = hass.data[DOMAIN].get("storage")
        if storage:
            await storage.close()
            
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