import logging
from datetime import datetime, timedelta
import numpy as np
import random
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

    # Initialize task manager (pass agent for phase access)
    task_manager = TaskManager(hass, discovered_sensors, collector, storage, agent)

    # Record initial phase if fresh install
    state = await storage.load_state()
    if not state:
        _LOGGER.debug("Fresh install detected: Recording initial baseline phase")
        await agent._save_persistent_state()
        await storage.record_phase_change(
            phase=PHASE_BASELINE,
            notes="Initial system setup"
        )
    
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

        # days_running = 14 # TEMP: For testing purposes, simulate baseline phase completion after 14 days

        # During baseline phase: continuously update baseline_consumption
        if agent.phase == PHASE_BASELINE:
            power_history_data = await collector.get_power_history(days=days_running if days_running > 0 else None)
            power_values = [power for timestamp, power in power_history_data]

            if len(power_values) > 0:
                agent.baseline_consumption = np.mean(power_values)
                _LOGGER.debug("Baseline consumption updated: %.2f W", agent.baseline_consumption)
        
        # Verify if the baseline phase is complete
        if days_running >= BASELINE_DAYS and agent.phase == PHASE_BASELINE:
            agent.phase = PHASE_ACTIVE
            _LOGGER.info("System entered active phase after %d days with baseline: %.2f W", days_running, agent.baseline_consumption)

            # Calculate area-specific baselines before entering active phase
            await agent.calculate_area_baselines()
            _LOGGER.info("Area baselines calculated for active phase")
            
            # Record phase change in research database
            if storage:
                await storage.record_phase_change(
                    phase=PHASE_ACTIVE,
                    baseline_consumption=agent.baseline_consumption,
                    notes=f"Transitioned after {days_running} days of baseline monitoring"
                )
            
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

    # Hourly aggregation for research data (keeps today's aggregate current)
    async def daily_aggregation_callback(now):
        """Compute/update daily aggregates for research analysis every hour."""
        _LOGGER.debug("Updating daily aggregates for research database...")
        today = datetime.now().strftime("%Y-%m-%d")
        
        try:
            await storage.compute_daily_aggregates(date=today, phase=agent.phase)
            await storage.compute_area_daily_aggregates(date=today, phase=agent.phase)
            _LOGGER.debug("Daily aggregates updated successfully for %s", today)
        except Exception as e:
            _LOGGER.error("Failed to update daily aggregates: %s", e)
    
    hass.data[DOMAIN]["daily_aggregation_listener"] = async_track_time_interval(
        hass, daily_aggregation_callback, timedelta(hours=1)
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
            # Also log to research database
            await storage.log_task_feedback(task_id, feedback)
            
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

    async def respond_to_selection(call: ServiceCall):
        """Service to respond to the notification currently selected in the dropdown."""
        decision = call.data.get("decision") # 'accept' or 'reject'
        
        # Get the state of the selector entity
        selector_state = hass.states.get("select.notification_selector")
        if not selector_state:
            _LOGGER.warning("Notification selector entity not found")
            return

        notification_id = selector_state.state
        
        if notification_id == "No pending notifications" or notification_id in ["unknown", "unavailable"]:
            _LOGGER.warning("No valid notification selected")
            return

        agent = hass.data[DOMAIN]["agent"]
        accepted = (decision == "accept")
        
        await agent._handle_notification_feedback(notification_id, accepted=accepted)
        _LOGGER.info("Notification %s marked as %s via selector", notification_id, decision)
        async_dispatcher_send(hass, GS_AI_UPDATE_SIGNAL)


    
    # ========================================
    # ========     FOR DEBUGGING     =========
    # ========================================

    async def force_ai_process(call: ServiceCall):
        """Force AI processing immediately (for testing)."""
        agent = hass.data[DOMAIN]["agent"]
        _LOGGER.info("Manual AI processing triggered")
        await agent.process_ai_model()
        _LOGGER.info("Manual AI processing complete")
    
    async def force_notification(call: ServiceCall):
        """Force notification decision, bypassing cooldowns (for testing)."""
        agent = hass.data[DOMAIN]["agent"]
        
        # Temporarily override cooldown
        original_time = agent.last_notification_time
        agent.last_notification_time = None
        
        _LOGGER.info("Forcing notification decision (cooldown bypassed)")
        await agent._decide_action()
        
        # Restore original time or set to now if notification was sent
        if len(agent.notification_history) > 0 and agent.last_notification_time is not None:
            # Notification was sent, keep the new time
            pass
        else:
            # No notification sent, restore original
            agent.last_notification_time = original_time
        
        _LOGGER.info("Forced notification decision complete")
    
    async def inject_test_data(call: ServiceCall):
        """Inject synthetic test data for testing."""
        import random
        from .const import UPDATE_INTERVAL_SECONDS
        
        hours = call.data.get("hours", 24)
        collector = hass.data[DOMAIN]["collector"]
        storage = hass.data[DOMAIN]["storage"]
        
        _LOGGER.info(f"Injecting {hours} hours of synthetic test data...")
        
        # Generate realistic data at the same frequency as real data collection
        now = datetime.now()
        data_points = int(hours * 3600 / UPDATE_INTERVAL_SECONDS)  # Every UPDATE_INTERVAL_SECONDS
        
        _LOGGER.info(f"Creating {data_points} data points (1 every {UPDATE_INTERVAL_SECONDS}s)")
        
        # Warn if this will take a while
        if data_points > 10000:
            _LOGGER.warning(f"Injecting {data_points} data points may take several seconds...")
        
        for i in range(data_points):
            timestamp = now - timedelta(seconds=i * UPDATE_INTERVAL_SECONDS)
            
            # Realistic pattern: higher during day, lower at night
            hour = timestamp.hour
            
            # Base power varies by time of day
            if 7 <= hour <= 9:  # Morning peak
                base_power = 1200
            elif 10 <= hour <= 16:  # Midday
                base_power = 800
            elif 17 <= hour <= 22:  # Evening peak
                base_power = 1500
            else:  # Night
                base_power = 300
            
            # Add realistic noise and variations
            noise = random.gauss(0, 150)
            power = max(50, base_power + noise)  # Never below 50W
            # power = 1500
            
            # Temperature varies slightly
            base_temp = 21
            temp_variation = random.gauss(0, 1.5)
            if hour < 6 or hour > 23:  # Cooler at night
                base_temp = 19
            temperature = base_temp + temp_variation
            
            # Humidity
            humidity = 50 + random.gauss(0, 8)
            humidity = max(30, min(70, humidity))
            
            # Illuminance (lights on during day if occupied, off at night)
            if 7 <= hour <= 22:
                illuminance = random.uniform(200, 600)
            else:
                illuminance = random.uniform(0, 50)
            
            # Occupancy
            occupancy = 7 <= hour <= 23 and random.random() > 0.3
            
            # Store data
            await storage.store_sensor_snapshot(
                timestamp=timestamp,
                power=power,
                temperature=temperature,
                humidity=humidity,
                illuminance=illuminance,
                occupancy=occupancy
            )
        
        _LOGGER.info(f"Successfully injected {data_points} data points ({hours} hours)")
    
    async def set_test_indices(call: ServiceCall):
        """Set AI indices for testing."""
        agent = hass.data[DOMAIN]["agent"]
        
        if "fatigue" in call.data:
            agent.fatigue_index = float(call.data["fatigue"])
            _LOGGER.info(f"Fatigue index set to: {agent.fatigue_index}")
        
        if "behavior" in call.data:
            agent.behaviour_index = float(call.data["behavior"])
            _LOGGER.info(f"Behavior index set to: {agent.behaviour_index}")
        
        if "anomaly" in call.data:
            agent.anomaly_index = float(call.data["anomaly"])
            _LOGGER.info(f"Anomaly index set to: {agent.anomaly_index}")
        
        # Trigger update
        async_dispatcher_send(hass, GS_AI_UPDATE_SIGNAL)
    
    # Register debug services
    hass.services.async_register(DOMAIN, "force_ai_process", force_ai_process)
    hass.services.async_register(DOMAIN, "force_notification", force_notification)
    hass.services.async_register(DOMAIN, "inject_test_data", inject_test_data)
    hass.services.async_register(DOMAIN, "set_test_indices", set_test_indices)

    # ===========================================================================================


    # Register services
    hass.services.async_register(DOMAIN, "submit_task_feedback", submit_task_feedback)
    hass.services.async_register(DOMAIN, "verify_tasks", verify_tasks)
    hass.services.async_register(DOMAIN, "regenerate_tasks", regenerate_tasks)
    hass.services.async_register(DOMAIN, "respond_to_selection", respond_to_selection)
    
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
        f"By hitting your **{target}%** reduction goal, in one year you would save:\n"
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
        
        if "daily_aggregation_listener" in hass.data[DOMAIN]:
            hass.data[DOMAIN]["daily_aggregation_listener"]()

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


