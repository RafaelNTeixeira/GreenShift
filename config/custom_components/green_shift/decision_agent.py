import logging
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from homeassistant.core import HomeAssistant

from .storage import StorageManager
from .const import (
    UPDATE_INTERVAL_SECONDS,
    SAVE_STATE_INTERVAL_SECONDS,
    AI_FREQUENCY_SECONDS,
    ACTIONS,
    REWARD_WEIGHTS,
    PHASE_BASELINE,
    PHASE_ACTIVE,
    FATIGUE_THRESHOLD,
    GAMMA,
)

_LOGGER = logging.getLogger(f"{__name__}.ai_model")


class DecisionAgent:
    """
    AI Decision agent based on MDP: ⟨S, A, M, P, R, γ⟩:
    - S: State vector with sensor readings and indices
    - A: Action space (noop, specific, anomaly, behavioural, normative)
    - M: Action mask based on sensor availability and context
    - P: Transition probabilities (implicit in state updates)
    - R: Reward function based on energy savings and user engagement
    - γ: Discount factor for future rewards
    
    Note: This agent does NOT store sensor data. It reads data from DataCollector.
    """
    
    def __init__(self, hass: HomeAssistant, discovered_sensors: dict, data_collector, storage_manager: StorageManager = None):
        self.hass = hass
        self.sensors = discovered_sensors
        self.data_collector = data_collector  # Reference to DataCollector
        self.storage = storage_manager
        self.phase = PHASE_BASELINE 
        # self.phase = PHASE_ACTIVE # TEMP: For testing purposes, start in active phase
        
        # AI state
        self.state_vector = None
        self.action_mask = None
        self.baseline_consumption = 0.0 
        self.baseline_consumption_week = None  # Fixed baseline for challenges (set after each week)
        self.last_baseline_update_date = None  # Track weekly baseline updates
        
        # Engagement history
        self.engagement_history = deque(maxlen=100)
        self.notification_count_today = 0
        self.last_notification_date = None
        
        # Q-table simplified 
        self.q_table = {} # TODO: Update? Maybe to DQN
        self.learning_rate = 0.1
        self.epsilon = 0.2  # Exploration rate
        
        # Behaviour indices
        self.anomaly_index = 0.0 
        self.behaviour_index = 0.5 
        self.fatigue_index = 0.0 
        
        # Tasks and challenges
        self.daily_tasks = []
        self.target_percentage = 15 # 15% reduction goal # TODO: Needs to be updated based on percentage reduction target picked by the user
        self.last_task_generation_date = None
        self.tasks_completed_count = 0

    async def setup(self):
        """Initialize agent and load persistent state."""
        if self.storage:
            await self._load_persistent_state()
        _LOGGER.info("DecisionAgent initialized - phase: %s", self.phase)

    async def _load_persistent_state(self):
        """Load AI state from JSON storage."""
        if not self.storage:
            return
        
        state = await self.storage.load_state()
        
        # Load AI configuration
        if "phase" in state:
            self.phase = state["phase"]
            _LOGGER.info("Loaded phase: %s", self.phase)
        
        if "baseline_consumption" in state:
            self.baseline_consumption = state["baseline_consumption"]
            _LOGGER.info("Loaded baseline consumption: %.2f kW", self.baseline_consumption)
        
        if "baseline_consumption_week" in state:
            self.baseline_consumption_week = state["baseline_consumption_week"]
        
        # Load indices
        if "anomaly_index" in state:
            self.anomaly_index = state["anomaly_index"]
        
        if "behaviour_index" in state:
            self.behaviour_index = state["behaviour_index"]
        
        if "fatigue_index" in state:
            self.fatigue_index = state["fatigue_index"]
        
        # Load Q-table
        if "q_table" in state:
            # Convert string keys back to tuples if needed
            self.q_table = {eval(k) if isinstance(k, str) else k: v 
                           for k, v in state["q_table"].items()}
            _LOGGER.info("Loaded Q-table with %d entries", len(self.q_table))
        
        _LOGGER.info("Persistent AI state loaded successfully")

    async def _save_persistent_state(self):
        """Save AI state to JSON storage."""
        if not self.storage:
            return
        
        current_state = await self.storage.load_state()
        
        # Convert Q-table keys to strings for JSON serialization
        serializable_q_table = {str(k): v for k, v in self.q_table.items()}
        
        ai_state = {
            "phase": self.phase,
            "baseline_consumption": float(self.baseline_consumption),
            "baseline_consumption_week": float(self.baseline_consumption_week) if self.baseline_consumption_week else None,
            "anomaly_index": float(self.anomaly_index),
            "behaviour_index": float(self.behaviour_index),
            "fatigue_index": float(self.fatigue_index),
            "q_table": serializable_q_table,
        }

        current_state.update(ai_state)
        
        await self.storage.save_state(current_state)
        
    async def process_ai_model(self):
        """
        Process AI model and perform complex calculations.
        This method is called periodically (every UPDATE_INTERVAL_SECONDS).
        Reads data from DataCollector, does NOT store sensor data.
        """
        # _LOGGER.debug("Processing AI model...")
        
        # Counter reset of daily notifications
        today = datetime.now().date()
        if self.last_notification_date != today:
            self.notification_count_today = 0
            self.last_notification_date = today
        
        # Build state vector from DataCollector's current readings
        self._build_state_vector()
        
        # Calculate indices (anomaly, behaviour, fatigue)
        await self._update_anomaly_index()
        self._update_behaviour_index()
        self._update_fatigue_index()
        
        # Update action mask M_t
        await self._update_action_mask()
        
        # Decide action A_t if in active phase
        if self.phase == PHASE_ACTIVE:
            await self._decide_action()

        # Periodically save state (every ~10 minutes to avoid too many writes)
        # Save on every 40th call (40 * 15s = 600s = 10 min)
        calls_per_save = SAVE_STATE_INTERVAL_SECONDS // AI_FREQUENCY_SECONDS

        if not hasattr(self, '_process_count'):
            self._process_count = 0

        self._process_count += 1
        
        if self._process_count % calls_per_save == 0 and self.storage:
            await self._save_persistent_state()
        
        _LOGGER.debug("AI model processing complete")
    
    def _build_state_vector(self):
        """
        Builds state vector S_t from DataCollector's current sensor readings.
        Does NOT read from Home Assistant directly - gets data from DataCollector.
        """
        state = []
        
        # Get current readings from DataCollector
        current_state = self.data_collector.get_current_state()
        
        # E_total, F_total (Total Power Consumption)
        power_sensors = self.sensors.get("power", []) # TODO: This might be a single sensor with total power directly.
        if power_sensors:
            total_power = current_state["power"]
            state.extend([total_power, 1.0])
        else:
            state.extend([0.0, 0.0])
        # _LOGGER.debug("Total power consumption: %.2f kW", state[0])
        
        # E_app, F_app (Individual Appliance Power)
        # TODO: This needs to be updated when we separate smart plugs
        if len(power_sensors) > 1:
            app_power = current_state["power"]  # Simplified for now
            state.extend([app_power, 1.0])
        else:
            state.extend([0.0, 0.0])
        # _LOGGER.debug("Appliance power consumption: %.2f kW", state[2])
        
        # T_in, F_T (Temperature)
        temp_sensors = self.sensors.get("temperature", [])
        if temp_sensors:
            temp = current_state["temperature"]
            state.extend([temp, 1.0])
        else:
            state.extend([0.0, 0.0])
        # _LOGGER.debug("Indoor temperature: %.2f °C", state[4])
        
        # H_in, F_H (Humidity)
        hum_sensors = self.sensors.get("humidity", [])
        if hum_sensors:
            humidity = current_state["humidity"]
            state.extend([humidity, 1.0])
        else:
            state.extend([0.0, 0.0])
        # _LOGGER.debug("Indoor humidity: %.2f %%", state[6])
        
        # L_in, F_L (Luminosity)
        lux_sensors = self.sensors.get("illuminance", [])
        if lux_sensors:
            lux = current_state["illuminance"]
            state.extend([lux, 1.0])
        else:
            state.extend([0.0, 0.0])
        # _LOGGER.debug("Indoor luminosity: %.2f lx", state[8])
        
        # O_status, F_O (Occupancy)
        occ_sensors = self.sensors.get("occupancy", [])
        if occ_sensors:
            occupied = current_state["occupancy"]
            state.extend([float(occupied), 1.0])
        else:
            state.extend([0.0, 0.0])
        # _LOGGER.debug("Occupancy status: %s", "Occupied" if state[10] == 1.0 else "Unoccupied")
        
        # Add indices to state vector
        state.extend([self.anomaly_index, self.behaviour_index, self.fatigue_index])
        # _LOGGER.debug("State vector: %s", state)

        self.state_vector = np.array(state)
    
    async def _update_action_mask(self):
        """Generates binary action mask M_t."""
        mask = np.ones(len(ACTIONS))
        
        # noop: always available

        # specific: needs smart plugs
        if not self.sensors.get("power"):
            mask[ACTIONS["specific"]] = 0
        
        # anomaly: needs enough consumption history
        power_history_data = await self.data_collector.get_power_history(hours=1)

        power_values = [power for timestamp, power in power_history_data]

        if len(power_values) < 100:
            mask[ACTIONS["anomaly"]] = 0
        
        # behavioural: always available

        # normative: requires consumption data
        if self.baseline_consumption == 0.0:
            mask[ACTIONS["normative"]] = 0
        
        self.action_mask = mask
        _LOGGER.debug("Action mask: %s", mask)
    
    async def _decide_action(self):
        """Selects an action using epsilon-greedy policy."""  
        available_actions = [i for i, m in enumerate(self.action_mask) if m == 1]
        if not available_actions:
            _LOGGER.warning("No available actions in current state.")
            return
        
        state_key = self._discretize_state()
        
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.choice(available_actions)
            _LOGGER.debug("Random action selected: %d", action)
        else:
            q_values = {a: self.q_table.get((state_key, a), 0.0) for a in available_actions}
            action = max(q_values, key=q_values.get)
            _LOGGER.debug("Greedy action selected: %d with Q-value: %.2f", action, q_values[action])
        
        # Execute action
        await self._execute_action(action)
        
        # Update Q-table (simplified)
        reward = self._calculate_reward()
        old_q = self.q_table.get((state_key, action), 0.0)
        self.q_table[(state_key, action)] = old_q + self.learning_rate * (reward - old_q)
        _LOGGER.debug("Q-table updated: state=%s, action=%d, reward=%.2f", state_key, action, reward)
    
    async def _execute_action(self, action: int):
        """Executes selected action."""
        action_name = [k for k, v in ACTIONS.items() if v == action][0]

        messages = {
            "specific": "Tip: The heater is consuming more than normal.",
            "anomaly": "Anomaly detected in energy consumption.",
            "behavioural": "Try turning off devices in standby before going to bed.",
            "normative": "Your department is 15% above the weekly target.",
        }
        
        message = messages.get(action_name, "")
        if message:
            await self.hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": "Energy Nudge",
                    "message": message,
                    "notification_id": f"energy_nudge_{datetime.now().timestamp()}",
                },
            )
            self.notification_count_today += 1
            _LOGGER.info("Action executed: %s", action_name)
    
    async def _calculate_reward(self) -> float:
        """
        Calculates reward R_t based on energy savings and user engagement.
        Reads consumption history from DataCollector.
        """
        power_history_data = await self.data_collector.get_power_history(hours=1) # Last hour
        power_values = [power for timestamp, power in power_history_data]

        if len(power_values) < 10:
            return 0.0
        
        # Energy savings component
        current = power_values[-1]
        baseline = self.baseline_consumption if self.baseline_consumption > 0 else np.mean(power_values)
        energy_saving = max(0, (baseline - current) / baseline) if baseline > 0 else 0
        
        # User engagement component
        engagement = self.behaviour_index
        
        # Fatigue penalty
        fatigue_penalty = self.fatigue_index
        
        # Combined reward
        reward = (
            REWARD_WEIGHTS["alpha"] * energy_saving +
            REWARD_WEIGHTS["beta"] * engagement -
            REWARD_WEIGHTS["delta"] * fatigue_penalty
        )
        
        return reward
    
    async def _update_anomaly_index(self):
        """
        Detects anomalies in consumption using z-score.
        Reads consumption history from DataCollector.
        """
        power_history_data = await self.data_collector.get_power_history(hours=1)  # Last hour
        power_values = [power for timestamp, power in power_history_data]
        
        # Calculate anomaly index based on last hour of data
        readings_per_hour = int(3600 / UPDATE_INTERVAL_SECONDS)
        
        if len(power_values) < readings_per_hour:
            self.anomaly_index = 0.0
            return
        
        recent = power_values[-readings_per_hour:]
        mean = np.mean(recent)
        std = np.std(recent)
        current = recent[-1]
        
        # Z-score normalized to [0,1]
        if std > 0:
            z_score = abs((current - mean) / std)
            self.anomaly_index = min(z_score / 3.0, 1.0)
            # _LOGGER.debug("Anomaly index updated: %.2f", self.anomaly_index)
        else:
            self.anomaly_index = 0.0
    
    def _update_behaviour_index(self):
        """History of user engagement."""
        if len(self.engagement_history) > 0:
            self.behaviour_index = np.clip(np.mean(self.engagement_history), 0, 1)
            # _LOGGER.debug("Behaviour index updated: %.2f", self.behaviour_index)
    
    def _update_fatigue_index(self):
        """Risk of user fatigue from too many notifications."""
        # self.fatigue_index = self.notification_count_today / MAX_NOTIFICATIONS_PER_DAY # TODO: Cannot be based on a predefined max. Must be dynamic.
        # _LOGGER.debug("Fatigue index updated: %.2f", self.fatigue_index)
    
    def _discretize_state(self) -> tuple:
        """Converts continuous state vector to discrete tuple for Q-table."""
        if self.state_vector is None:
            return (0,)
        # Simplification: use only consumption, anomalies and fatigue
        power = int(self.state_vector[0] / 100) # Bins of 100W # TODO: Confirm this
        anomaly = int(self.anomaly_index * 10)
        fatigue = int(self.fatigue_index * 10)

        # _LOGGER.debug("State discretized: power=%d, anomaly=%d, fatigue=%d", power, anomaly, fatigue)

        return (power, anomaly, fatigue)
    
    def _generate_daily_tasks(self):
        """Generates 3 random daily tasks based on available sensors."""
        today = datetime.now().date()
        
        # Only generate once per day
        if self.last_task_generation_date == today and len(self.daily_tasks) > 0:
            return
        
        self.last_task_generation_date = today
        
        # Define available tasks based on sensor availability
        available_tasks = []
        
        # Temperature sensor tasks
        if self.sensors.get("temperature"):
            available_tasks.extend([
                {"title": "Lower heating by 1°C", "description": "Small temperature adjustments save energy", "category": "temperature"},
                {"title": "Use natural temperature control", "description": "Open windows during cool hours", "category": "temperature"},
            ])
        
        # Occupancy sensor tasks
        if self.sensors.get("occupancy"):
            available_tasks.extend([
                {"title": "Turn off lights when leaving", "description": "Ensure lights are off in empty rooms", "category": "occupancy"},
                {"title": "Manage room occupancy efficiently", "description": "Close doors to unoccupied areas", "category": "occupancy"},
            ])
        
        # Power sensor tasks
        if self.sensors.get("power"):
            available_tasks.extend([
                {"title": "Unplug unused devices", "description": "Eliminate standby power consumption", "category": "power"},
                {"title": "Use power strips for appliances", "description": "Group related devices for easier control", "category": "power"},
            ])
        
        # Humidity sensor tasks
        if self.sensors.get("humidity"):
            available_tasks.extend([
                {"title": "Use fan mode for cooling", "description": "Fans use less energy than AC", "category": "humidity"},
                {"title": "Improve air circulation", "description": "Better ventilation reduces HVAC load", "category": "humidity"},
            ])
        
        # Illuminance sensor tasks
        if self.sensors.get("illuminance"):
            available_tasks.extend([
                {"title": "Use natural light during day", "description": "Maximize sunlight hours", "category": "illuminance"},
                {"title": "Switch to energy-efficient lighting", "description": "Consider LED bulbs", "category": "illuminance"},
            ])
        
        # Default tasks (always available)
        available_tasks.extend([
            {"title": "Plan energy-intensive tasks", "description": "Run dishwasher/laundry during off-peak hours", "category": "general"},
            {"title": "Monitor energy usage", "description": "Check the dashboard for consumption patterns", "category": "general"},
        ])
        
        # Select 3 random tasks
        if len(available_tasks) >= 3:
            self.daily_tasks = list(np.random.choice(len(available_tasks), 3, replace=False))
            self.daily_tasks = [available_tasks[i] for i in self.daily_tasks]
        else:
            self.daily_tasks = available_tasks[:3]
        
        _LOGGER.info("Generated daily tasks: %s", [t["title"] for t in self.daily_tasks])
    
    async def _update_weekly_baseline(self):
        """Updates the fixed baseline once per week."""
        today = datetime.now().date()
        
        # Only update once per week (Monday)
        if self.last_baseline_update_date is None or \
           (today - self.last_baseline_update_date).days >= 7:
            self.last_baseline_update_date = today
            
            power_history_data = await self.data_collector.get_power_history(days=7) # Last week
            power_values = [power for timestamp, power in power_history_data] 

            if len(power_values) > 0:
                self.baseline_consumption_week = np.mean(power_values)
                _LOGGER.info("Weekly baseline updated: %.2f kW", self.baseline_consumption_week)
    
    async def get_weekly_challenge_status(self, target_percentage: float = 15.0) -> dict:
        """
        Calculates weekly challenge status (consumption reduction goal).
        Reads consumption history from DataCollector.
        """
        # Use fixed weekly baseline for consistent comparison
        if self.baseline_consumption_week is None or self.baseline_consumption_week == 0:
            return {"status": "pending", "current_avg": 0, "target_avg": 0, "progress": 0}
        
        power_history_data = await self.data_collector.get_power_history(days=7) # Last week
        power_values = [power for timestamp, power in power_history_data]
        
        # Calculate readings per day based on data collection interval
        day_in_seconds = 86400
        readings_per_day = int(day_in_seconds / UPDATE_INTERVAL_SECONDS)
        
        # Need at least 1 day of data
        if len(power_values) < readings_per_day:
            return {"status": "pending", "current_avg": 0, "target_avg": 0, "progress": 0}
        
        # Get last 7 days of consumption
        current_avg = np.mean(power_values) if power_values else 0

        reduction_multiplier = 1.0 - (target_percentage / 100.0)
        
        target_state = self.hass.states.get("input_number.energy_saving_target")
        try:
            user_target_pct = float(target_state.state) if target_state else 15.0
        except (ValueError, TypeError):
            user_target_pct = 15.0

        _LOGGER.debug("Updated target to: %d", target_state)

        self.target_percentage = user_target_pct

        # Convert percentage (e.g., 20) to multiplier (e.g., 0.80)
        reduction_multiplier = 1.0 - (user_target_pct / 100.0)
        
        # Calculate target average
        target_avg = self.baseline_consumption_week * reduction_multiplier
        
        # Calculate how close to target based on fixed baseline
        if self.baseline_consumption_week > 0:
            progress = (current_avg / self.baseline_consumption_week) * 100
        else:
            progress = 0
        
        status = "completed" if progress <= (100 - user_target_pct) else "in_progress"
        
        return {
            "status": status,
            "current_avg": round(current_avg, 2),
            "target_avg": round(target_avg, 2),
            "progress": round(progress, 1),
            "baseline": round(self.baseline_consumption_week, 2),
            "goal_percentage": user_target_pct
        }
    