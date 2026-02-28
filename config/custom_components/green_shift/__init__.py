"""
File: __init__.py
Description: Main initialization and setup for the Green Shift Home Assistant component.
This module handles the setup of the component, including initializing the data collector, decision agent, task manager and backup manager. 
It also sets up periodic tasks for AI model updates, task generation and verification, daily aggregation for research data and automatic backups. 
Additionally, it defines services for submitting task feedback, verifying tasks, regenerating tasks, responding to notifications, and testing various functionalities.
"""

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
import sqlite3

from .const import (
    DOMAIN,
    GS_AI_UPDATE_SIGNAL,
    SENSOR_MAPPING,
    PHASE_BASELINE,
    PHASE_ACTIVE,
    BASELINE_DAYS,
    AI_FREQUENCY_SECONDS,
    TASK_GENERATION_TIME,
    VERIFY_TASKS_INTERVAL_MINUTES,
    BACKUP_INTERVAL_HOURS,
    KEEP_AUTO_BACKUPS,
    KEEP_STARTUP_BACKUPS,
    KEEP_SHUTDOWN_BACKUPS,
    ENVIRONMENT_OFFICE,
    UPDATE_INTERVAL_SECONDS,
    ACTIONS,
    GAMMA
)
from .data_collector import DataCollector
from .decision_agent import DecisionAgent
from .storage import StorageManager
from .task_manager import TaskManager
from .backup_manager import BackupManager
from .translations_runtime import get_language, get_phase_transition_template

_LOGGER = logging.getLogger(__name__)
PLATFORMS = ["sensor", "select"]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """
    Setup of the component through config entry.
    
    Args:
        hass (HomeAssistant): The Home Assistant instance.
        entry (ConfigEntry): The configuration entry containing user settings and discovered sensors.

    Returns:
        bool: True if setup was successful, False otherwise.
    """
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
    storage = StorageManager(hass, config_data=entry.data)
    await storage.setup()

    # Initialize backup manager
    backup_manager = BackupManager(hass.config.path("green_shift_data"))

    # Create initial backup on startup
    _LOGGER.info("Creating startup backup...")
    await backup_manager.create_backup(backup_type="startup")

    # Initialize the real-time data collector
    collector = DataCollector(hass, discovered_sensors, main_energy_sensor, main_power_sensor, storage, config_data=entry.data)
    await collector.setup()

    # Initialize the decision agent (AI)
    agent = DecisionAgent(hass, discovered_sensors, collector, storage, config_data=entry.data)
    await agent.setup()

    # Initialize task manager (pass agent for phase access)
    task_manager = TaskManager(hass, discovered_sensors, collector, storage, agent, config_data=entry.data)

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
    hass.data[DOMAIN]["backup_manager"] = backup_manager
    hass.data[DOMAIN]["discovered_sensors"] = discovered_sensors
    hass.data[DOMAIN]["config_data"] = entry.data  # Store config for working hours checks

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
            # In office mode, only use working hours data for baseline calculation
            is_office_mode = entry.data.get("environment_mode") == ENVIRONMENT_OFFICE
            working_hours_filter = True if is_office_mode else None

            power_history_data = await collector.get_power_history(
                days=max(1, days_running),
                working_hours_only=working_hours_filter
            )
            power_values = [power for timestamp, power in power_history_data]

            if len(power_values) > 0:
                agent.baseline_consumption = np.mean(power_values)
                _LOGGER.debug("Baseline consumption updated: %.2f W (office mode: %s, working hours only: %s)",
                             agent.baseline_consumption, is_office_mode, working_hours_filter is not None)

        # Verify if the baseline phase is complete
        if days_running >= BASELINE_DAYS and agent.phase == PHASE_BASELINE:
            agent.phase = PHASE_ACTIVE
            agent.active_since = datetime.now()
            _LOGGER.info("System entered active phase after %d days with baseline: %.2f W", days_running, agent.baseline_consumption)

            # Calculate area-specific baselines before entering active phase
            await agent.calculate_area_baselines()

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

    # Automatic backup every BACKUP_INTERVAL_HOURS hours
    async def auto_backup_callback(now):
        """Create automatic backup periodically."""
        _LOGGER.debug("Creating automatic backup...")
        try:
            success = await backup_manager.create_backup(backup_type="auto")
            if success:
                _LOGGER.info("Automatic backup created successfully")
                # Clean up old backups (manual backups are never auto-deleted)
                await backup_manager.cleanup_old_backups(
                    keep_auto=KEEP_AUTO_BACKUPS,
                    keep_startup=KEEP_STARTUP_BACKUPS,
                    keep_shutdown=KEEP_SHUTDOWN_BACKUPS
                )
            else:
                _LOGGER.warning("Automatic backup failed")
        except Exception as e:
            _LOGGER.error("Automatic backup error: %s", e)

    hass.data[DOMAIN]["auto_backup_listener"] = async_track_time_interval(
        hass, auto_backup_callback, timedelta(hours=BACKUP_INTERVAL_HOURS)
    )

    # Daily RL episode cleanup (runs at 3 AM)
    async def rl_cleanup_callback(now):
        """Clean up old RL episodes to manage database size."""
        _LOGGER.debug("Running RL episode cleanup...")
        try:
            await storage._cleanup_old_rl_episodes()

            # Update last cleanup timestamp in state
            current_state = await storage.load_state()
            current_state["last_rl_cleanup"] = datetime.now().isoformat()
            await storage.save_state(current_state)

            _LOGGER.info("RL episode cleanup completed successfully")
        except Exception as e:
            _LOGGER.error("RL episode cleanup error: %s", e)

    # Check if cleanup is needed on startup (in case HA was shut down for days)
    state = await storage.load_state()
    last_cleanup_str = state.get("last_rl_cleanup") if state else None

    if last_cleanup_str:
        try:
            last_cleanup = datetime.fromisoformat(last_cleanup_str)
            hours_since_cleanup = (datetime.now() - last_cleanup).total_seconds() / 3600

            if hours_since_cleanup >= 24:
                _LOGGER.info("RL cleanup overdue (%.1f hours since last run), running now...", hours_since_cleanup)
                await rl_cleanup_callback(None)
        except Exception as e:
            _LOGGER.warning("Could not parse last RL cleanup date: %s", e)
    else:
        # First time running, do cleanup
        _LOGGER.info("No previous RL cleanup found, running initial cleanup...")
        await rl_cleanup_callback(None)

    # Schedule daily cleanup at 3:00 AM
    hass.data[DOMAIN]["rl_cleanup_listener"] = async_track_time_change(
        hass, rl_cleanup_callback, hour=3, minute=0, second=0
    )

    # Daily sensor data cleanup at 3:30 AM (sensor_data.db rolling 14-day window)
    async def sensor_data_cleanup_callback(now):
        """Remove sensor data older than 14 days from sensor_data.db."""
        _LOGGER.debug("Running periodic sensor data cleanup...")
        try:
            await storage._cleanup_old_data()
            _LOGGER.info("Periodic sensor data cleanup completed successfully")
        except Exception as e:
            _LOGGER.error("Periodic sensor data cleanup error: %s", e)

    hass.data[DOMAIN]["sensor_cleanup_listener"] = async_track_time_change(
        hass, sensor_data_cleanup_callback, hour=3, minute=30, second=0
    )

    # Generate tasks immediately if none exist for today (only in active phase)
    if agent.phase == PHASE_ACTIVE:
        today_tasks = await storage.get_today_tasks()
        if not today_tasks:
            _LOGGER.info("No tasks found for today, generating now...")
            await task_manager.generate_daily_tasks()

    return True


async def async_setup_services(hass: HomeAssistant):
    """
    Setup services for task management.
    
    Args:
        hass (HomeAssistant): The Home Assistant instance.
    """

    async def submit_task_feedback(call: ServiceCall):
        """
        Service to submit task difficulty feedback.
        
        Args:
            call (ServiceCall): The service call containing 'task_index' and 'feedback' in call.data.
        """
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
        _LOGGER.info("Deleting today's tasks before regeneration")
        await storage.delete_today_tasks()

        # Generate new tasks
        tasks = await task_manager.generate_daily_tasks()
        _LOGGER.info("Tasks regenerated: %d tasks", len(tasks))
        async_dispatcher_send(hass, GS_AI_UPDATE_SIGNAL)

    async def respond_to_selection(call: ServiceCall):
        """
        Service to respond to the notification currently selected in the dropdown.
        
        Args:
            call (ServiceCall): The service call containing 'decision' in call.data, which should be 'accept' or 'reject'.
        """
        decision = call.data.get("decision") # 'accept' or 'reject'

        # Get the state of the selector entity
        selector_state = hass.states.get("select.notification_selector")
        if not selector_state:
            _LOGGER.warning("Notification selector entity not found")
            return

        notification_id = selector_state.state

        _LOGGER.debug("Responding to notification %s with decision: %s", notification_id, decision)

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
        """
        Inject synthetic test data for testing.
        
        Args:
             call (ServiceCall): The service call containing 'hours' in call.data, which specifies how many hours of data to inject (default 24).
        """
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

            # Cumulative energy for this day up to this timestamp (kWh)
            # Matches real collector: stores total kWh accumulated since midnight
            hours_since_midnight = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
            energy = (power / 1000.0) * hours_since_midnight  # kWh since midnight

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

            weekday = timestamp.weekday()
            working_hours = (weekday < 5 and 8 <= hour <= 18)

            # Store data
            await storage.store_sensor_snapshot(
                timestamp=timestamp,
                power=power,
                energy=energy,
                temperature=temperature,
                humidity=humidity,
                illuminance=illuminance,
                occupancy=occupancy,
                within_working_hours=working_hours
            )

        _LOGGER.info(f"Successfully injected {data_points} data points ({hours} hours)")

    async def set_test_indices(call: ServiceCall): 
        """
        Set AI indices for testing.
        
        Args:
            call (ServiceCall): The service call containing 'fatigue', 'behavior' and/or 'anomaly' in call.data to set the respective indices.
        """
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

    async def inspect_q_table(call: ServiceCall):
        """Inspect Q-table contents and state space usage."""
        agent = hass.data[DOMAIN]["agent"]

        _LOGGER.info("="*60)
        _LOGGER.info("Q-TABLE INSPECTION")
        _LOGGER.info("="*60)

        # Basic stats
        total_states = len(agent.q_table)
        _LOGGER.info(f"Total states in Q-table: {total_states}")
        _LOGGER.info(f"Theoretical max states: 5*4*3*2*4*2 = 960")
        _LOGGER.info(f"State space usage: {total_states/960*100:.2f}%")

        # Current state
        current_state_key = agent._discretize_state()
        anomaly_labels = ['none', 'low', 'medium', 'high']
        fatigue_labels = ['low', 'medium', 'high']
        time_labels = ['night', 'morning', 'afternoon', 'evening']

        _LOGGER.info(f"\nCurrent discretized state: {current_state_key}")
        _LOGGER.info(f"  power_bin={current_state_key[0]} (~{current_state_key[0]*100}-{(current_state_key[0]+1)*100}W)")
        _LOGGER.info(f"  anomaly_level={current_state_key[1]} ({anomaly_labels[current_state_key[1]]})")
        _LOGGER.info(f"  fatigue_level={current_state_key[2]} ({fatigue_labels[current_state_key[2]]})")
        _LOGGER.info(f"  has_area_anomaly={current_state_key[3]} ({'yes' if current_state_key[3] else 'no'})")
        _LOGGER.info(f"  time_period={current_state_key[4]} ({time_labels[current_state_key[4]]})")
        _LOGGER.info(f"  is_occupied={current_state_key[5]} ({'yes' if current_state_key[5] else 'no'})")

        # Q-values for current state
        if current_state_key in agent.q_table:
            _LOGGER.info(f"\nQ-values for current state:")
            for action, q_val in sorted(agent.q_table[current_state_key].items(), key=lambda x: x[1], reverse=True):
                action_name = [k for k, v in ACTIONS.items() if v == action][0]
                _LOGGER.info(f"  Action {action} ({action_name}): Q={q_val:.4f}")
        else:
            _LOGGER.info(f"\nCurrent state NOT in Q-table (will be initialized on next action)")

        # Top 10 states by max Q-value
        if agent.q_table:
            _LOGGER.info(f"\nTop 10 learned states (by max Q-value):")
            state_max_q = [(state, max(q_vals.values())) for state, q_vals in agent.q_table.items()]
            state_max_q.sort(key=lambda x: x[1], reverse=True)

            for i, (state, max_q) in enumerate(state_max_q[:10], 1):
                _LOGGER.info(f"  {i}. State {state}: max_Q={max_q:.4f}")

        # State distribution analysis
        if agent.q_table:
            _LOGGER.info(f"\nState component distribution:")
            power_bins = [s[0] for s in agent.q_table.keys()]
            anomaly_bins = [s[1] for s in agent.q_table.keys()]
            fatigue_bins = [s[2] for s in agent.q_table.keys()]
            area_bins = [s[3] for s in agent.q_table.keys()]
            time_bins = [s[4] for s in agent.q_table.keys()]
            occupancy_bins = [s[5] for s in agent.q_table.keys()]

            _LOGGER.info(f"  Power bins used: {len(set(power_bins))} / 51 (range: {min(power_bins) if power_bins else 0}-{max(power_bins) if power_bins else 0})")
            _LOGGER.info(f"  Anomaly levels used: {len(set(anomaly_bins))} / 4")
            _LOGGER.info(f"  Fatigue levels used: {len(set(fatigue_bins))} / 3")
            _LOGGER.info(f"  Area anomaly states used: {len(set(area_bins))} / 2")
            _LOGGER.info(f"  Time periods used: {len(set(time_bins))} / 4")
            _LOGGER.info(f"  Occupancy values used: {len(set(occupancy_bins))} / 2")

        # Learning progress
        _LOGGER.info(f"\nLearning parameters:")
        _LOGGER.info(f"  Episode number: {agent.episode_number}")
        _LOGGER.info(f"  Epsilon (exploration rate): {agent.epsilon}")
        _LOGGER.info(f"  Learning rate: {agent.learning_rate}")

        _LOGGER.info("="*60)

    async def test_q_learning(call: ServiceCall):
        """Test Q-learning update logic - tests both acceptance and rejection."""
        agent = hass.data[DOMAIN]["agent"]

        _LOGGER.info("="*60)
        _LOGGER.info("Q-LEARNING TEST - Testing both acceptance and rejection")
        _LOGGER.info("="*60)

        # Get current state
        state_before = agent._discretize_state()
        _LOGGER.info(f"Current state: {state_before}")

        # Check if state exists in Q-table
        if state_before not in agent.q_table:
            agent.q_table[state_before] = {a: 0.0 for a in ACTIONS.values()}
            _LOGGER.info(f"State initialized in Q-table")

        # Show Q-values before
        _LOGGER.info(f"\nQ-values BEFORE updates:")
        for action, q_val in agent.q_table[state_before].items():
            _LOGGER.info(f"  Action {action}: Q={q_val:.4f}")

        test_action = 1
        initial_q = agent.q_table[state_before].get(test_action, 0.0)

        # Get next state (same as current for this test)
        next_state = state_before
        if next_state not in agent.q_table:
            agent.q_table[next_state] = {a: 0.0 for a in ACTIONS.values()}

        max_next_q = max(agent.q_table[next_state].values())

        # TEST 1: REJECTION
        _LOGGER.info(f"\n{'─'*60}")
        _LOGGER.info(f"TEST 1: REJECTION (accepted=False)")
        _LOGGER.info(f"{'─'*60}")
        
        test_reward_reject = -0.25
        gamma_reject = 0.0  # Terminal state
        new_q_reject = initial_q + agent.learning_rate * (test_reward_reject + gamma_reject * max_next_q - initial_q)

        _LOGGER.info(f"Simulating: action={test_action}, reward={test_reward_reject:.4f}, gamma={gamma_reject}")
        _LOGGER.info(f"Calculation: {initial_q:.4f} + {agent.learning_rate} × ({test_reward_reject:.4f} + {gamma_reject}×{max_next_q:.4f} - {initial_q:.4f})")
        _LOGGER.info(f"Result: {initial_q:.4f} → {new_q_reject:.4f} (Δ = {new_q_reject - initial_q:+.4f})")

        # TEST 2: ACCEPTANCE
        _LOGGER.info(f"\n{'─'*60}")
        _LOGGER.info(f"TEST 2: ACCEPTANCE (accepted=True)")
        _LOGGER.info(f"{'─'*60}")

        test_reward_accept = 0.3
        gamma_accept = GAMMA  # Normal discounting (0.95)
        new_q_accept = initial_q + agent.learning_rate * (test_reward_accept + gamma_accept * max_next_q - initial_q)

        _LOGGER.info(f"Simulating: action={test_action}, reward={test_reward_accept:.4f}, gamma={gamma_accept}")
        _LOGGER.info(f"Calculation: {initial_q:.4f} + {agent.learning_rate} × ({test_reward_accept:.4f} + {gamma_accept}×{max_next_q:.4f} - {initial_q:.4f})")
        _LOGGER.info(f"Result: {initial_q:.4f} → {new_q_accept:.4f} (Δ = {new_q_accept - initial_q:+.4f})")

        # Apply rejection update (to show the fix works)
        agent.q_table[state_before][test_action] = new_q_reject

        _LOGGER.info(f"\n{'='*60}")
        _LOGGER.info(f"SUMMARY:")
        _LOGGER.info(f"  Rejection: Q goes {initial_q:.4f} → {new_q_reject:.4f} (gamma=0.0, terminal)")
        _LOGGER.info(f"  Acceptance: Q goes {initial_q:.4f} → {new_q_accept:.4f} (gamma={GAMMA}, normal)")
        _LOGGER.info(f"\n✓ Applied rejection update to Q-table (shows decrease)")
        _LOGGER.info("="*60)

    async def save_state(call: ServiceCall):
        """Immediately persist the AI agent state to JSON storage."""
        agent = hass.data[DOMAIN]["agent"]
        await agent._save_persistent_state()
        _LOGGER.info("AI state manually saved to persistent storage")

    # Register debug services
    hass.services.async_register(DOMAIN, "force_ai_process", force_ai_process)
    hass.services.async_register(DOMAIN, "force_notification", force_notification)
    hass.services.async_register(DOMAIN, "inject_test_data", inject_test_data)
    hass.services.async_register(DOMAIN, "set_test_indices", set_test_indices)
    hass.services.async_register(DOMAIN, "inspect_q_table", inspect_q_table)
    hass.services.async_register(DOMAIN, "test_q_learning", test_q_learning)
    hass.services.async_register(DOMAIN, "save_state", save_state)

    # ===========================================================================================

    # Backup and restore services
    async def create_backup(call: ServiceCall):
        """Service to manually create a backup."""
        backup_manager = hass.data[DOMAIN].get("backup_manager")
        if not backup_manager:
            _LOGGER.error("Backup manager not initialized")
            return

        _LOGGER.info("Manual backup requested")
        success = await backup_manager.create_backup(backup_type="manual")

        if success:
            _LOGGER.info("Manual backup created successfully")
        else:
            _LOGGER.error("Manual backup failed")

    async def restore_backup(call: ServiceCall):
        """
        Service to restore from a backup.
        
        Args:
            call (ServiceCall): The service call containing 'backup_name' in call.data, which specifies the name of the backup to restore.
        """
        backup_manager = hass.data[DOMAIN].get("backup_manager")
        if not backup_manager:
            _LOGGER.error("Backup manager not initialized")
            return

        backup_name = call.data.get("backup_name")
        if not backup_name:
            _LOGGER.error("No backup name provided")
            return

        _LOGGER.warning("Restoring from backup: %s - This will overwrite current data!", backup_name)
        success = await backup_manager.restore_from_backup(backup_name)

        if success:
            _LOGGER.info("Backup restored successfully. Please restart Home Assistant.")
        else:
            _LOGGER.error("Backup restoration failed")

    async def list_backups(call: ServiceCall):
        """Service to list available backups."""
        backup_manager = hass.data[DOMAIN].get("backup_manager")
        if not backup_manager:
            _LOGGER.error("Backup manager not initialized")
            return

        backups = backup_manager.list_backups()
        _LOGGER.info("="*60)
        _LOGGER.info("Available Backups (%d total):", len(backups))
        _LOGGER.info("="*60)

        for backup in backups:
            _LOGGER.info("  - %s", backup)

        _LOGGER.info("="*60)
        _LOGGER.info("Use 'green_shift.restore_backup' with backup_name to restore")


    async def test_data_retention(call: ServiceCall):
        """
        Service to test data retention mechanisms.
        
        Args:
            call (ServiceCall): The service call containing 'test_type' in call.data, which specifies the type of test to perform (e.g., 'status', 'inject_notifications', 'inject_old_episodes', 'set_overdue_cleanup').
        """
        test_type = call.data.get("test_type", "status")
        agent = hass.data[DOMAIN]["agent"]
        storage = hass.data[DOMAIN]["storage"]

        _LOGGER.info("="*60)
        _LOGGER.info("DATA RETENTION TEST - Type: %s", test_type)
        _LOGGER.info("="*60)

        if test_type == "status":
            # Show current status
            _LOGGER.info("\n📊 CURRENT STATUS:")

            # Check notification count in JSON
            state = await storage.load_state()
            notification_history = state.get("notification_history", [])
            _LOGGER.info(f"  Notifications in JSON: {len(notification_history)} (limit: 100)")

            # Check RL episodes in database
            def _count_episodes():
                conn = sqlite3.connect(str(storage.research_db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM research_rl_episodes")
                count, min_ts, max_ts = cursor.fetchone()
                conn.close()
                return count, min_ts, max_ts

            count, min_ts, max_ts = await hass.async_add_executor_job(_count_episodes)

            if count > 0:
                min_date = datetime.fromtimestamp(min_ts).strftime("%Y-%m-%d")
                max_date = datetime.fromtimestamp(max_ts).strftime("%Y-%m-%d")
                days_span = (max_ts - min_ts) / 86400
                _LOGGER.info(f"  RL Episodes in DB: {count:,}")
                _LOGGER.info(f"  Date range: {min_date} to {max_date} ({days_span:.1f} days)")
            else:
                _LOGGER.info(f"  RL Episodes in DB: 0")

            # Check last cleanup
            last_cleanup = state.get("last_rl_cleanup")
            if last_cleanup:
                last = datetime.fromisoformat(last_cleanup)
                hours_ago = (datetime.now() - last).total_seconds() / 3600
                _LOGGER.info(f"  Last RL cleanup: {hours_ago:.1f} hours ago ({last_cleanup})")
            else:
                _LOGGER.info(f"  Last RL cleanup: Never")

        elif test_type == "inject_notifications":
            # Inject 150 notifications to test 100-limit trimming
            count = call.data.get("count", 150)
            _LOGGER.info(f"\n🔧 Injecting {count} test notifications...")

            for i in range(count):
                agent.notification_history.append({
                    "notification_id": f"test_notification_{i}",
                    "timestamp": datetime.now().isoformat(),
                    "action_type": "test",
                    "accepted": None,
                    "responded": False
                })

            # Save and reload to trigger trimming
            await agent._save_persistent_state()

            state = await storage.load_state()
            notification_history = state.get("notification_history", [])
            _LOGGER.info(f"  ✓ Saved state with trimming")
            _LOGGER.info(f"  Notifications in memory: {len(agent.notification_history)}")
            _LOGGER.info(f"  Notifications in JSON: {len(notification_history)} (should be 100)")

        elif test_type == "inject_old_episodes":
            # Inject RL episodes from 5 months ago
            months = call.data.get("months", 5)
            episodes_per_day = call.data.get("episodes_per_day", 100)

            _LOGGER.info(f"\n🔧 Injecting episodes from {months} months ago ({episodes_per_day}/day)...")

            def _inject():
                conn = sqlite3.connect(str(storage.research_db_path))
                cursor = conn.cursor()

                days = months * 30
                total_inserted = 0

                for day in range(days):
                    date = datetime.now() - timedelta(days=days - day)

                    for ep in range(episodes_per_day):
                        timestamp = (date + timedelta(seconds=ep * 15)).timestamp()

                        cursor.execute("""
                            INSERT INTO research_rl_episodes
                            (timestamp, phase, state_vector, action, reward, opportunity_score, action_source)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            timestamp,
                            "active",
                            "0,0,0,0,0,0,0,0,0,0",
                            0,  # no-op
                            0.0,
                            0.0,
                            "test_injection"
                        ))
                        total_inserted += 1

                    if (day + 1) % 30 == 0:
                        conn.commit()
                        _LOGGER.info(f"  Inserted {total_inserted:,} episodes so far...")

                conn.commit()
                conn.close()
                return total_inserted

            total = await hass.async_add_executor_job(_inject)
            _LOGGER.info(f"  ✓ Injected {total:,} test episodes spanning {months} months")

        elif test_type == "set_overdue_cleanup":
            # Set last cleanup to 48 hours ago
            hours = call.data.get("hours_ago", 48)
            _LOGGER.info(f"\n🔧 Setting last cleanup to {hours} hours ago...")

            state = await storage.load_state()
            fake_cleanup_time = datetime.now() - timedelta(hours=hours)
            state["last_rl_cleanup"] = fake_cleanup_time.isoformat()
            await storage.save_state(state)

            _LOGGER.info(f"  ✓ Last cleanup set to: {fake_cleanup_time.isoformat()}")
            _LOGGER.info(f"  Restart HA to trigger overdue cleanup check")

        elif test_type == "run_cleanup":
            # Manually trigger cleanup
            _LOGGER.info(f"\n🔧 Running RL episode cleanup manually...")

            await storage._cleanup_old_rl_episodes()

            # Update timestamp
            state = await storage.load_state()
            state["last_rl_cleanup"] = datetime.now().isoformat()
            await storage.save_state(state)

            _LOGGER.info(f"  ✓ Cleanup completed - check logs above for deletion count")

        _LOGGER.info("="*60)

    # Register services
    hass.services.async_register(DOMAIN, "submit_task_feedback", submit_task_feedback)
    hass.services.async_register(DOMAIN, "verify_tasks", verify_tasks)
    hass.services.async_register(DOMAIN, "regenerate_tasks", regenerate_tasks)
    hass.services.async_register(DOMAIN, "respond_to_selection", respond_to_selection)
    hass.services.async_register(DOMAIN, "create_backup", create_backup)
    hass.services.async_register(DOMAIN, "restore_backup", restore_backup)
    hass.services.async_register(DOMAIN, "list_backups", list_backups)
    hass.services.async_register(DOMAIN, "test_data_retention", test_data_retention)

    _LOGGER.info("Services registered successfully")


async def trigger_phase_transition_notification(hass, agent, collector):
    """
    Calculates baseline summary and sends the transition notification.
    
    Args:
        hass: Home Assistant instance
        agent: The AI agent instance
        collector: The data collector instance
    """

    # Get user's language
    language = await get_language(hass)

    # Fetch data from collector
    summary = await collector.calculate_baseline_summary()
    impact = summary.get("impact", {})
    target = summary.get("target", 15)

    # Get translated template
    template = get_phase_transition_template(language)

    # Build top_area section if available
    top_area_section = ""
    if summary.get('top_area'):
        if language == "pt":
            top_area_section = f"**Área Principal:** {summary['top_area']}\n"
        else:
            top_area_section = f"**Main Area:** {summary['top_area']}\n"

    # Format the message with data
    notification_msg = template["message"].format(
        avg_daily_kwh=summary['avg_daily_kwh'],
        peak_time=summary['peak_time'],
        top_area_section=top_area_section,
        target=target,
        co2_kg=impact.get('co2_kg', 0),
        trees=impact.get('trees', 0),
        flights=impact.get('flights', 0)
    )

    # Send the notification
    await hass.services.async_call(
        "persistent_notification", "create",
        {
            "title": template["title"],
            "message": notification_msg,
            "notification_id": "gs_phase_transition"
        }
    )


async def sync_helper_entities(hass: HomeAssistant, entry: ConfigEntry):
    """
    Syncs the options chosen in the Config Flow to the corresponding helper entities in Home Assistant.
    
    Args:
        hass (HomeAssistant): The Home Assistant instance.
        entry (ConfigEntry): The configuration entry containing the user-selected options.
    """
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
    """
    Unload of the config entry.
    
    Args:
        hass (HomeAssistant): The Home Assistant instance.
        entry (ConfigEntry): The configuration entry to unload.

    Returns:
        bool: True if unload was successful, False otherwise.
    """
    _LOGGER.info("Green Shift shutting down - saving state and creating backup...")

    # Save AI state before shutdown
    agent = hass.data[DOMAIN].get("agent")
    if agent and agent.storage:
        try:
            await agent._save_persistent_state()
            _LOGGER.info("AI state saved successfully on shutdown")
        except Exception as e:
            _LOGGER.error("Failed to save AI state on shutdown: %s", e)

    # Create shutdown backup
    backup_manager = hass.data[DOMAIN].get("backup_manager")
    if backup_manager:
        try:
            await backup_manager.create_backup(backup_type="shutdown")
            _LOGGER.info("Shutdown backup created successfully")
        except Exception as e:
            _LOGGER.error("Failed to create shutdown backup: %s", e)

    # Unregister services
    hass.services.async_remove(DOMAIN, "submit_task_feedback")
    hass.services.async_remove(DOMAIN, "verify_tasks")
    hass.services.async_remove(DOMAIN, "regenerate_tasks")
    hass.services.async_remove(DOMAIN, "respond_to_selection")
    hass.services.async_remove(DOMAIN, "force_ai_process")
    hass.services.async_remove(DOMAIN, "force_notification")
    hass.services.async_remove(DOMAIN, "inject_test_data")
    hass.services.async_remove(DOMAIN, "set_test_indices")
    hass.services.async_remove(DOMAIN, "inspect_q_table")
    hass.services.async_remove(DOMAIN, "test_q_learning")
    hass.services.async_remove(DOMAIN, "create_backup")
    hass.services.async_remove(DOMAIN, "restore_backup")
    hass.services.async_remove(DOMAIN, "list_backups")
    hass.services.async_remove(DOMAIN, "test_data_retention")
    hass.services.async_remove(DOMAIN, "save_state")

    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        hass.data[DOMAIN]["update_listener"]()

        # Cancel target listener
        if "target_listener" in hass.data[DOMAIN]:
            hass.data[DOMAIN]["target_listener"]()

        # Cancel task listeners
        if "task_generation_listener" in hass.data[DOMAIN]:
            hass.data[DOMAIN]["task_generation_listener"]()

        if "task_verification_listener" in hass.data[DOMAIN]:
            hass.data[DOMAIN]["task_verification_listener"]()

        if "daily_aggregation_listener" in hass.data[DOMAIN]:
            hass.data[DOMAIN]["daily_aggregation_listener"]()

        # Cancel backup listener
        if "auto_backup_listener" in hass.data[DOMAIN]:
            hass.data[DOMAIN]["auto_backup_listener"]()

        # Cancel RL cleanup listener
        if "rl_cleanup_listener" in hass.data[DOMAIN]:
            hass.data[DOMAIN]["rl_cleanup_listener"]()

        # Cancel sensor data cleanup listener
        if "sensor_cleanup_listener" in hass.data[DOMAIN]:
            hass.data[DOMAIN]["sensor_cleanup_listener"]()

        # Close storage connections
        storage = hass.data[DOMAIN].get("storage")
        if storage:
            await storage.close()

        hass.data.pop(DOMAIN)

    _LOGGER.info("Green Shift unloaded successfully")
    return unload_ok


async def async_discover_sensors(hass: HomeAssistant) -> dict:
    """
    Discover relevant sensors based on device class, unit, and keywords.
    
    Args:
        hass (HomeAssistant): The Home Assistant instance.

    Returns:
        dict: A dictionary categorizing discovered sensor entity IDs by their respective categories.
    """
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
