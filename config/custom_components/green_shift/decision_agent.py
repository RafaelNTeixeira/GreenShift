import logging
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .storage import StorageManager
from .const import (
    UPDATE_INTERVAL_SECONDS,
    SAVE_STATE_INTERVAL_SECONDS,
    AI_FREQUENCY_SECONDS,
    GS_AI_UPDATE_SIGNAL,
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
        self.start_date = datetime.now()
        self._process_count = 0
        self.phase = PHASE_BASELINE 
        # self.phase = PHASE_ACTIVE # TEMP: For testing purposes, start in active phase
        
        # AI state
        self.state_vector = None
        self.action_mask = None
        self.baseline_consumption = 0.0
        
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
        
        # Challenges
        self.target_percentage = 15 # Default 15% reduction goal
        self.current_week_start_date = None  # Track when the current weekly challenge started 

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

        if "start_date" in state:
            try:
                self.start_date = datetime.fromisoformat(state["start_date"])
                _LOGGER.info("Loaded start date: %s", self.start_date)
            except ValueError:
                self.start_date = datetime.now()
        
        # Load AI configuration
        if "phase" in state:
            self.phase = state["phase"]
            _LOGGER.info("Loaded phase: %s", self.phase)
        
        if "baseline_consumption" in state:
            self.baseline_consumption = state["baseline_consumption"]
            _LOGGER.info("Loaded baseline consumption: %.2f kW", self.baseline_consumption)
        
        if "current_week_start_date" in state and state["current_week_start_date"]:
            try:
                self.current_week_start_date = datetime.fromisoformat(state["current_week_start_date"]).date()
            except (ValueError, AttributeError):
                self.current_week_start_date = None
        
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

        if "start_date" in current_state:
            safe_start_date = current_state["start_date"]
        else:
            safe_start_date = self.start_date.isoformat()
        
        # Convert Q-table keys to strings for JSON serialization
        serializable_q_table = {str(k): v for k, v in self.q_table.items()}
        
        ai_state = {
            "start_date": safe_start_date,
            "phase": self.phase,
            "baseline_consumption": float(self.baseline_consumption),
            "current_week_start_date": self.current_week_start_date.isoformat() if self.current_week_start_date else None,
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

        self._process_count += 1

        # Periodically save state (every ~10 minutes to avoid too many writes)
        # Save on every 40th call (40 * 15s = 600s = 10 min)
        calls_per_save = SAVE_STATE_INTERVAL_SECONDS // AI_FREQUENCY_SECONDS
        
        # Save if it's the first run or the periodic interval
        if self.storage and (self._process_count == 1 or self._process_count % calls_per_save == 0):
            _LOGGER.debug("Checkpoint: Saving AI state (Run #%d)", self._process_count)
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
        if len(power_history_data) < 100:
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
    
    async def get_weekly_challenge_status(self, target_percentage: float = 15.0) -> dict:
        """
        Calculates weekly challenge status (consumption reduction goal).
        Tracks energy from the start of the current week (Monday) and compares to baseline.
        Progress updates dynamically as new data comes in during the week.
        """
        # Use fixed baseline from baseline phase for consistent comparison
        if self.baseline_consumption is None or self.baseline_consumption == 0:
            _LOGGER.warning("Baseline consumption not set. Cannot calculate weekly challenge status.")
            return {"status": "pending", "current_avg": 0, "target_avg": 0, "progress": 0}
        
        # Initialize or reset current_week_start_date
        today = datetime.now().date()
        days_since_monday = today.weekday()  # 0 = Monday, 6 = Sunday
        current_week_monday = today - timedelta(days=days_since_monday)
        
        # Check if we need to initialize or if we've moved to a new week
        if self.current_week_start_date is None or self.current_week_start_date != current_week_monday:
            self.current_week_start_date = current_week_monday
            _LOGGER.info("Weekly challenge start date set to: %s", self.current_week_start_date)
            # Save immediately when week date changes
            if self.storage:
                await self._save_persistent_state()
        
        # Calculate days from start of week to now
        days_in_current_week = (today - self.current_week_start_date).days + 1  # +1 to include today
        
        # Get power history from the start of the current week
        power_history_data = await self.data_collector.get_power_history(days=days_in_current_week)
        power_values = [power for timestamp, power in power_history_data]
        
        # Calculate readings per day based on data collection interval
        day_in_seconds = 86400
        readings_per_day = int(day_in_seconds / UPDATE_INTERVAL_SECONDS)
        
        # Need at least 1 hour of data from this week
        min_readings = int(readings_per_day / 24) # 1 hour 

        if len(power_values) < min_readings:
            _LOGGER.warning("Not enough data for weekly challenge progress: %d readings (need at least %d)", len(power_values), min_readings)
            return {"status": "pending", "current_avg": 0, "target_avg": 0, "progress": 0}
        
        # Get current week's average (updates dynamically as the week progresses)
        current_avg = np.mean(power_values) if power_values else 0

        self.target_percentage = target_percentage

        # Convert percentage (e.g., 15) to multiplier (e.g., 0.85)
        reduction_multiplier = 1.0 - (target_percentage / 100.0)
        
        # Calculate target average based on fixed baseline
        target_avg = self.baseline_consumption * reduction_multiplier
        
        # Calculate progress as percentage of target consumption
        # Under 100% = SUCCESS (consuming less than target)
        # Over 100% = FAILURE (consuming more than target)
        if target_avg > 0:
            progress = (current_avg / target_avg) * 100
        else:
            progress = 0
        
        status = "completed" if progress < 100 else "in_progress"

        _LOGGER.info(
            "Weekly challenge calculation: %d readings, current_avg=%.2f, baseline=%.2f, target_pct=%.1f%%",
            len(power_values), np.mean(power_values), self.baseline_consumption, target_percentage
        )
        
        return {
            "status": status,
            "current_avg": round(current_avg, 2),
            "target_avg": round(target_avg, 2),
            "progress": round(progress, 1),
            "baseline": round(self.baseline_consumption, 2),
            "goal_percentage": target_percentage,
            "week_start": self.current_week_start_date.isoformat(),
            "days_in_week": days_in_current_week
        }
    