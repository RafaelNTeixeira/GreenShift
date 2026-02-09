import logging
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
import random

from .storage import StorageManager
from .helpers import get_friendly_name
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
    NOTIFICATION_TEMPLATES,
    MAX_NOTIFICATIONS_PER_DAY
)

_LOGGER = logging.getLogger(f"{__name__}.ai_model")


class DecisionAgent:
    """
    AI Decision agent based on MDP: ⟨S, A, M, P, R, γ⟩:
    - S: State vector with area-based sensor readings and indices
    - A: Action space (noop, specific, anomaly, behavioural, normative)
    - M: Action mask based on sensor availability and context
    - P: Transition probabilities (implicit in state updates)
    - R: Reward function based on energy savings and user engagement
    - γ: Discount factor for future rewards
    """
    
    def __init__(self, hass: HomeAssistant, discovered_sensors: dict, data_collector, storage_manager: StorageManager = None):
        self.hass = hass
        self.sensors = discovered_sensors
        self.data_collector = data_collector
        self.storage = storage_manager
        self.start_date = datetime.now()
        self._process_count = 0
        self.phase = PHASE_BASELINE 
        # self.phase = PHASE_ACTIVE # TEMP: For testing purposes, start in active phase
        
        # AI state
        self.state_vector = None
        self.action_mask = None
        self.baseline_consumption = 0.0

        # Area-based baselines for anomaly detection
        self.area_baselines = {}  # {area: {metric: baseline_value}}
        
        # Engagement history
        self.engagement_history = deque(maxlen=100)
        self.notification_history = deque(maxlen=50)  # Track recent notifications
        self.notification_count_today = 0
        self.last_notification_date = None
        self.last_notification_time = None
        
        # Q-table for reinforcement learning
        self.q_table = {}
        self.learning_rate = 0.1
        self.epsilon = 0.2  # Exploration rate
        
        # Behaviour indices
        self.anomaly_index = 0.0 
        self.behaviour_index = 0.5 
        self.fatigue_index = 0.0 

        # Area-specific anomaly tracking
        self.area_anomalies = {}  # {area: {metric: anomaly_score}}
        
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

        if "area_baselines" in state:
            self.area_baselines = state["area_baselines"]
            _LOGGER.info("Loaded %d area baselines", len(self.area_baselines))
        
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

        # Load notification history
        if "notification_history" in state:
            self.notification_history = deque(state["notification_history"], maxlen=50)
        
        # Load Q-table
        if "q_table" in state:
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
            "area_baselines": self.area_baselines,
            "current_week_start_date": self.current_week_start_date.isoformat() if self.current_week_start_date else None,
            "anomaly_index": float(self.anomaly_index),
            "behaviour_index": float(self.behaviour_index),
            "fatigue_index": float(self.fatigue_index),
            "notification_history": list(self.notification_history),
            "q_table": serializable_q_table,
        }

        current_state.update(ai_state)
        
        await self.storage.save_state(current_state)
        
    async def process_ai_model(self):
        """
        Process AI model and perform complex calculations.
        This method is called periodically (every AI_FREQUENCY_SECONDS).
        """
        # Counter reset of daily notifications
        today = datetime.now().date()
        if self.last_notification_date != today:
            self.notification_count_today = 0
            self.last_notification_date = today
        
        # Build state vector from DataCollector's current readings
        await self._build_state_vector()
        
        # Calculate indices (anomaly, behaviour, fatigue)
        await self._update_anomaly_index()
        await self._update_area_anomalies()
        self._update_behaviour_index()
        await self._update_fatigue_index()
        
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
    
    async def _build_state_vector(self):
        """
        Builds state vector S_t from DataCollector's current sensor readings.

        State components:
        1. Global power consumption + flag
        2. Individual appliance power (top consumer) + flag
        3. Global temperature + flag
        4. Global humidity + flag
        5. Global illuminance + flag
        6. Global occupancy + flag
        7. Anomaly index (0-1)
        8. Behaviour index (0-1)
        9. Fatigue index (0-1)
        10. Area anomaly count (number of areas with anomalies)
        11. Time of day (normalized 0-1)
        12. Day of week (normalized 0-1)
        """
        current_state = self.data_collector.get_current_state()

        state = []
        
        # 1. Global power consumption
        power = current_state.get("power", 0.0)
        state.extend([power, 1.0 if power > 0 else 0.0])
        _LOGGER.debug("Global power consumption: %.2f kW", state[0])
        
        # 2. Top power consumer (individual appliance)
        top_consumer = await self._get_top_power_consumer()
        state.extend([top_consumer, 1.0 if top_consumer > 0 else 0.0])
        _LOGGER.debug("Top power consumer: %.2f kW", state[2])
        
        # 3. Temperature
        temp = current_state.get("temperature", 0.0)
        state.extend([temp if temp is not None else 0.0, 1.0 if temp is not None else 0.0])
        _LOGGER.debug("Indoor temperature: %.2f °C", state[4])
        
        # 4. Humidity
        hum = current_state.get("humidity", 0.0)
        state.extend([hum if hum is not None else 0.0, 1.0 if hum is not None else 0.0])
        _LOGGER.debug("Indoor humidity: %.2f %%", state[6])
        
        # 5. Illuminance
        lux = current_state.get("illuminance", 0.0)
        state.extend([lux if lux is not None else 0.0, 1.0 if lux is not None else 0.0])
        _LOGGER.debug("Indoor illuminance: %.2f lx", state[8])
        
        # 6. Occupancy
        occ = 1.0 if current_state.get("occupancy", False) else 0.0
        state.extend([occ, 1.0])
        
        # 7-9. Indices
        state.extend([
            self.anomaly_index,
            self.behaviour_index,
            self.fatigue_index
        ])

        # 10. Area anomaly count (spatial awareness)
        area_anomaly_count = len([a for a in self.area_anomalies.values() if any(v > 0.3 for v in a.values())])
        state.append(area_anomaly_count)
        
        # 11. Time of day (normalized)
        now = datetime.now()
        time_of_day = (now.hour * 60 + now.minute) / (24 * 60)  # 0 to 1
        state.append(time_of_day)
        
        # 12. Day of week (normalized)
        day_of_week = now.weekday() / 6.0  # 0 (Monday) to 1 (Sunday)
        state.append(day_of_week)
        
        self.state_vector = state
        _LOGGER.debug("State vector built: %s", state)

    async def _get_top_power_consumer(self) -> float:
        """Get the power consumption of the highest consuming device."""
        power_sensors = self.sensors.get("power", [])
        if not power_sensors:
            return 0.0
        
        max_power = 0.0
        for entity_id in power_sensors:
            state = self.hass.states.get(entity_id)
            if state and state.state not in ['unknown', 'unavailable']:
                try:
                    power = float(state.state)
                    max_power = max(max_power, power)
                except (ValueError, TypeError):
                    continue
        
        return max_power
    
    async def _update_action_mask(self):
        """Updates the action mask M_t based on context and sensor availability."""
        mask = {action: False for action in ACTIONS.values()}
        
        # noop: always available
        mask[ACTIONS["noop"]] = True

        # specific: requires individual power sensors
        power_sensors = self.sensors.get("power", [])
        if len(power_sensors) > 0:
            mask[ACTIONS["specific"]] = True
        
        # anomaly: requires sufficient history and detected anomalies
        power_history = await self.data_collector.get_power_history(hours=1)
        has_area_anomalies = any(any(v > 0.3 for v in area_anomalies.values()) 
                                for area_anomalies in self.area_anomalies.values())
        if len(power_history) >= 100 and (self.anomaly_index > 0.3 or has_area_anomalies):
            mask[ACTIONS["anomaly"]] = True
        
        # behavioural: always available
        mask[ACTIONS["behavioural"]] = True

        # normative: requires consumption data
        if self.baseline_consumption > 0.0:
            mask[ACTIONS["normative"]] = True
        
        self.action_mask = mask
        _LOGGER.debug("Action mask updated: %s", {k: v for k, v in mask.items() if v})
    
    async def _decide_action(self):
        """Selects action A_t using epsilon-greedy policy with Q-learning.""" 
        # Check notification limits
        if self.notification_count_today >= MAX_NOTIFICATIONS_PER_DAY:
            _LOGGER.debug("Max notifications reached for today")
            return
        
        # Check fatigue threshold
        if self.fatigue_index > FATIGUE_THRESHOLD:
            _LOGGER.debug("User fatigue too high (%.2f), skipping notification", self.fatigue_index)
            return

        # Minimum time between notifications (1 hour)
        if self.last_notification_time:
            time_since_last = (datetime.now() - self.last_notification_time).total_seconds()
            if time_since_last < 3600:  # 1 hour
                return
            
        # Get current state
        state_key = self._discretize_state()

        # Available actions based on mask
        available_actions = [a for a, available in self.action_mask.items() if available and a != ACTIONS["noop"]]

        if not available_actions:
            return

       # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: random action
            action = random.choice(available_actions)
            _LOGGER.debug("Exploration: selected random action %d", action)
        else:
            # Exploitation: best known action
            if state_key not in self.q_table:
                self.q_table[state_key] = {a: 0.0 for a in ACTIONS.values()}
            
            # Choose best available action
            best_action = max(available_actions, key=lambda a: self.q_table[state_key].get(a, 0.0))
            action = best_action
            _LOGGER.debug("Exploitation: selected best action %d (Q=%.2f)", action, self.q_table[state_key].get(action, 0.0))
        
        # Execute action
        await self._execute_action(action)
        
        # Update Q-table
        reward = await self._calculate_reward()
        await self._update_q_table(state_key, action, reward)

    async def _update_q_table(self, state_key: tuple, action: int, reward: float):
        """
        Updates Q-table using Q-learning update rule:
        Q(s,a) ← Q(s,a) + α[R + γ max Q(s',a') - Q(s,a)]
        """
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in ACTIONS.values()}
        
        current_q = self.q_table[state_key].get(action, 0.0)
        
        # Get next state (after action execution)
        next_state_key = self._discretize_state()
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in ACTIONS.values()}
        
        max_next_q = max(self.q_table[next_state_key].values())
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + GAMMA * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        
        _LOGGER.debug("Q-table updated: state=%s, action=%d, reward=%.2f, Q: %.2f → %.2f", 
                     state_key[:3], action, reward, current_q, new_q)
    
    async def _execute_action(self, action: int):
        """Executes selected action by sending a notification."""
        action_name = [k for k, v in ACTIONS.items() if v == action][0]

        # Get appropriate notification template
        notification = await self._generate_notification(action_name)
        
        if notification:
            # Create actionable notification with feedback buttons
            notification_id = f"energy_nudge_{datetime.now().timestamp()}"
            
            await self.hass.services.async_call(
                "notify",
                "persistent_notification",
                {
                    "message": notification["message"],
                    "title": notification["title"],
                    "data": {
                        "notification_id": notification_id,
                        "actions": [
                            {
                                "action": f"accept_{notification_id}",
                                "title": "✓ Helpful"
                            },
                            {
                                "action": f"reject_{notification_id}",
                                "title": "✗ Not useful"
                            }
                        ]
                    }
                },
            )

            # Track notification
            self.notification_count_today += 1
            self.last_notification_time = datetime.now()
            self.notification_history.append({
                "timestamp": datetime.now().isoformat(),
                "action_type": action_name,
                "notification_id": notification_id,
                "responded": False
            })

            # Register notification response handler
            await self._register_notification_handler(notification_id, action_name)
            
            _LOGGER.info("Action executed: %s - %s", action_name, notification["title"])

    async def _generate_notification(self, action_type: str) -> dict:
        """
        Generates context-aware notification based on action type.
        """
        templates = NOTIFICATION_TEMPLATES.get(action_type, [])
        if not templates:
            return None
        
        # Select template based on context
        template = random.choice(templates)
        
        # Gather context for template
        context = await self._gather_notification_context(action_type)
        
        # Format message
        try:
            message = template["message"].format(**context)
            title = template["title"].format(**context)
        except KeyError as e:
            _LOGGER.error("Missing context key for notification: %s", e)
            return None
        
        return {
            "title": title,
            "message": message
        }
    
    async def _gather_notification_context(self, action_type: str) -> dict:
        """
        Gathers contextual information for notification templates.
        """
        context = {}
        
        # Current consumption
        current_state = self.data_collector.get_current_state()
        context["current_power"] = int(current_state.get("power", 0))
        context["baseline_power"] = int(self.baseline_consumption)
        
        # Calculate percentage difference
        if self.baseline_consumption > 0:
            diff = ((current_state.get("power", 0) - self.baseline_consumption) / self.baseline_consumption) * 100
            context["percent_above"] = int(abs(diff))
        else:
            context["percent_above"] = 0
        
        # Find top power consumer
        top_device, top_power = await self._find_top_consumer()
        context["device_name"] = top_device if top_device else "Unknown device"
        context["device_power"] = int(top_power)
        
        # Find area with highest anomaly
        anomaly_area, anomaly_metric = await self._find_highest_anomaly_area()
        context["area_name"] = anomaly_area if anomaly_area else "Living room"
        context["metric"] = anomaly_metric if anomaly_metric else "temperature"
        
        # Get area-specific temperature for comfort suggestions
        if anomaly_area:
            area_state = self.data_collector.get_area_state(anomaly_area)
            context["area_temp"] = int(area_state.get("temperature", 22) if area_state.get("temperature") is not None else 22)
        else:
            context["area_temp"] = int(current_state.get("temperature", 22) if current_state.get("temperature") is not None else 22)
        
        # Time-based context
        now = datetime.now()
        context["time_of_day"] = "evening" if 18 <= now.hour < 22 else "night" if 22 <= now.hour or now.hour < 6 else "day"
        
        return context
    
    async def _find_top_consumer(self) -> tuple:
        """Find the device consuming the most power."""
        power_sensors = self.sensors.get("power", [])
        if not power_sensors:
            return None, 0.0
        
        max_power = 0.0
        top_device = None
        
        for entity_id in power_sensors:
            state = self.hass.states.get(entity_id)
            if state and state.state not in ['unknown', 'unavailable']:
                try:
                    power = float(state.state)
                    if power > max_power:
                        max_power = power
                        top_device = get_friendly_name(self.hass, entity_id)
                except (ValueError, TypeError):
                    continue
        
        return top_device, max_power
    
    async def _find_highest_anomaly_area(self) -> tuple:
        """Find the area with the highest anomaly score."""
        if not self.area_anomalies:
            return None, None
        
        max_anomaly = 0.0
        anomaly_area = None
        anomaly_metric = None
        
        for area, metrics in self.area_anomalies.items():
            for metric, score in metrics.items():
                if score > max_anomaly:
                    max_anomaly = score
                    anomaly_area = area
                    anomaly_metric = metric
        
        return anomaly_area, anomaly_metric
    
    async def _register_notification_handler(self, notification_id: str, action_type: str):
        """Register handler for notification feedback."""
        async def handle_accept(call):
            """Handle acceptance of notification."""
            await self._handle_notification_feedback(notification_id, accepted=True)
        
        async def handle_reject(call):
            """Handle rejection of notification."""
            await self._handle_notification_feedback(notification_id, accepted=False)
        
        # Register temporary services for this notification
        self.hass.services.async_register(
            "green_shift",
            f"accept_{notification_id}",
            handle_accept
        )
        
        self.hass.services.async_register(
            "green_shift",
            f"reject_{notification_id}",
            handle_reject
        )

    async def _handle_notification_feedback(self, notification_id: str, accepted: bool):
        """
        Handle user feedback on notifications.
        Updates behaviour index and engagement history.
        """
        # Find notification in history
        for notif in self.notification_history:
            if notif["notification_id"] == notification_id:
                notif["responded"] = True
                notif["accepted"] = accepted
                break
        
        # Update engagement history
        engagement_score = 1.0 if accepted else -0.5
        self.engagement_history.append(engagement_score)
        
        # Update behaviour index
        self._update_behaviour_index()
        
        _LOGGER.info("Notification feedback received: %s - %s", notification_id, "accepted" if accepted else "rejected")
        
        # Save state
        if self.storage:
            await self._save_persistent_state()
    
    async def _calculate_reward(self) -> float:
        """
        Calculates reward R_t based on energy savings, engagement and fatigue.
        Formula: R_t = α·ΔE + β·I_engagement - δ·I_fatigue
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
        """
        power_history_data = await self.data_collector.get_power_history(hours=1) # Last hour
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
            _LOGGER.debug("Anomaly index updated: %.2f", self.anomaly_index)
        else:
            self.anomaly_index = 0.0
    
    async def _update_area_anomalies(self):
        """Detects anomalies in each area for spatial awareness."""
        areas = self.data_collector.get_all_areas()
        self.area_anomalies = {}
        
        for area in areas:
            if area == "No Area":
                continue
            
            area_anomalies = {}
            
            # Check temperature anomalies
            temp_history = await self.data_collector.get_area_history(area, "temperature", hours=2)
            if temp_history:
                temp_values = [val for ts, val in temp_history if val is not None]
                if len(temp_values) >= 10:
                    mean = np.mean(temp_values)
                    std = np.std(temp_values)
                    current = temp_values[-1]
                    
                    # TODO: Also check against baseline if available
                    baseline_temp = self.area_baselines.get(area, {}).get("temperature")
                    
                    if std > 0:
                        z_score = abs((current - mean) / std)
                        area_anomalies["temperature"] = min(z_score / 3.0, 1.0)
                    
                    # Additional check: extreme values
                    if current < 16 or current > 28:
                        area_anomalies["temperature"] = max(area_anomalies.get("temperature", 0), 0.8)
            
            # Check power anomalies per area
            power_history = await self.data_collector.get_area_history(area, "power", hours=2)
            if power_history:
                power_values = [val for ts, val in power_history if val is not None]
                if len(power_values) >= 10:
                    mean = np.mean(power_values)
                    std = np.std(power_values)
                    current = power_values[-1]
                    
                    if std > 0:
                        z_score = abs((current - mean) / std)
                        area_anomalies["power"] = min(z_score / 3.0, 1.0)
            
            # Check humidity anomalies
            hum_history = await self.data_collector.get_area_history(area, "humidity", hours=2)
            if hum_history:
                hum_values = [val for ts, val in hum_history if val is not None]
                if len(hum_values) >= 10:
                    current = hum_values[-1]
                    
                    # Check against comfortable range (30-60%)
                    if current < 30 or current > 60:
                        deviation = max(30 - current, current - 60, 0)
                        area_anomalies["humidity"] = min(deviation / 30.0, 1.0)
            
            if area_anomalies:
                self.area_anomalies[area] = area_anomalies
        
        _LOGGER.debug("Area anomalies updated: %d areas with anomalies", len(self.area_anomalies))

    def _update_behaviour_index(self):
        """
        Updates behaviour index based on user engagement history.
        Uses exponential moving average for recent behavior.
        """
        if len(self.engagement_history) > 0:
            # Weighted towards recent interactions
            weights = np.exp(np.linspace(-2, 0, len(self.engagement_history)))
            weights /= weights.sum()
            
            weighted_engagement = np.average(self.engagement_history, weights=weights)
            self.behaviour_index = np.clip(weighted_engagement, 0, 1)
            _LOGGER.debug("Behaviour index updated: %.2f", self.behaviour_index)
    
    async def _update_fatigue_index(self):
        """
        Calculates user fatigue based on recent notification patterns.
        Considers rejection rate, frequency of notifications and time since last interaction.
        """
        if len(self.notification_history) == 0:
            self.fatigue_index = 0.0
            return
        
        # Count responses in the last 10 notifications
        recent_notifs = list(self.notification_history)[-10:]
        responded = [n for n in recent_notifs if n.get("responded", False)]
        
        if not responded:
            # No responses = moderate fatigue
            self.fatigue_index = 0.4
            return
        
        rejected = [n for n in responded if not n.get("accepted", False)]
        rejection_rate = len(rejected) / len(responded) if responded else 0
        
        # Frequency component: high frequency in short time = higher fatigue
        if len(recent_notifs) >= 3:
            timestamps = [datetime.fromisoformat(n["timestamp"]) for n in recent_notifs[-3:]]
            time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
            
            if time_span > 0:
                frequency_factor = min(3 / time_span, 1.0)  # Ideal: 1 per hour
            else:
                frequency_factor = 1.0
        else:
            frequency_factor = 0.0
        
        # Combined fatigue score
        self.fatigue_index = np.clip(
            0.6 * rejection_rate + 0.4 * frequency_factor,
            0.0,
            1.0
        )
        
        _LOGGER.debug("Fatigue index updated: %.2f (rejection_rate=%.2f, freq=%.2f)", self.fatigue_index, rejection_rate, frequency_factor)
    
    def _discretize_state(self) -> tuple:
        """
        Converts continuous state vector to discrete tuple for Q-table.
        
        State components:
        - power_bin: Power consumption in 100W bins
        - anomaly_bin: Global anomaly level (0-10)
        - fatigue_bin: Fatigue level (0-10)
        - area_anomaly_bin: Number of areas with anomalies (0-5+)
        - time_bin: Time of day (morning/afternoon/evening/night)
        - occupancy: Occupied or not
        """
        if self.state_vector is None or len(self.state_vector) < 13:
            return (0, 0, 0, 0, 0, 0)
        
        # Power in 100W bins
        power = int(self.state_vector[0] / 100)
        power_bin = min(power, 50)  # Cap at 5000W for table size
        
        # Anomaly level (0-10)
        anomaly_bin = int(self.anomaly_index * 10)
        
        # Fatigue level (0-10)
        fatigue_bin = int(self.fatigue_index * 10)
        
        # Area anomaly count (capped at 5)
        area_anomaly_count = int(self.state_vector[10])
        area_anomaly_bin = min(area_anomaly_count, 5)
        
        # Time of day (4 bins: 0=night, 1=morning, 2=afternoon, 3=evening)
        time_of_day = self.state_vector[11]  # 0-1
        if time_of_day < 0.25:  # 0:00-6:00
            time_bin = 0
        elif time_of_day < 0.5:  # 6:00-12:00
            time_bin = 1
        elif time_of_day < 0.75:  # 12:00-18:00
            time_bin = 2
        else:  # 18:00-24:00
            time_bin = 3
        
        # Occupancy
        occupancy = int(self.state_vector[6])
        
        return (power_bin, anomaly_bin, fatigue_bin, area_anomaly_bin, time_bin, occupancy)
    
    async def calculate_area_baselines(self):
        """
        Calculate baseline values for each area during the baseline phase.
        Called at the end of the baseline phase.
        """
        areas = self.data_collector.get_all_areas()
        
        for area in areas:
            if area == "No Area":
                continue
            
            baselines = {}
            
            # Temperature baseline
            temp_history = await self.data_collector.get_area_history(area, "temperature", days=14)
            if temp_history:
                temp_values = [val for ts, val in temp_history if val is not None]
                if temp_values:
                    baselines["temperature"] = np.mean(temp_values)
            
            # Power baseline
            power_history = await self.data_collector.get_area_history(area, "power", days=14)
            if power_history:
                power_values = [val for ts, val in power_history if val is not None]
                if power_values:
                    baselines["power"] = np.mean(power_values)
            
            # Humidity baseline
            hum_history = await self.data_collector.get_area_history(area, "humidity", days=14)
            if hum_history:
                hum_values = [val for ts, val in hum_history if val is not None]
                if hum_values:
                    baselines["humidity"] = np.mean(hum_values)
            
            if baselines:
                self.area_baselines[area] = baselines
        
        _LOGGER.info("Calculated baselines for %d areas", len(self.area_baselines))
        
        # Save baselines
        if self.storage:
            await self._save_persistent_state()
    
    async def get_weekly_challenge_status(self, target_percentage: float = 15.0) -> dict:
        """
        Calculates weekly challenge status (consumption reduction goal).
        Tracks energy from the start of the current week (Monday) and compares to baseline.
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

        # _LOGGER.info(
        #     "Weekly challenge calculation: %d readings, current_avg=%.2f, baseline=%.2f, target_pct=%.1f%%",
        #     len(power_values), np.mean(power_values), self.baseline_consumption, target_percentage
        # )
        
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
    