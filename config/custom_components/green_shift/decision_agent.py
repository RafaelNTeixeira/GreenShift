"""Decision Agent com Reinforcement Learning."""
import logging
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from .const import (
    ACTIONS,
    REWARD_WEIGHTS,
    PHASE_BASELINE,
    PHASE_ACTIVE,
    MAX_NOTIFICATIONS_PER_DAY,
    FATIGUE_THRESHOLD,
    GAMMA,
)

_LOGGER = logging.getLogger(f"{__name__}.ai_model")


class DecisionAgent:
    """
    Decision agent based on MDP: ⟨S, A, M, P, R, γ⟩:
    - S: State vector with sensor readings and indices
    - A: Action space (noop, specific, anomaly, behavioural, normative)
    - M: Action mask based on sensor availability and context
    - P: Transition probabilities (implicit in state updates)
    - R: Reward function based on energy savings and user engagement
    - γ: Discount factor for future rewards
    """
    
    def __init__(self, hass: HomeAssistant, discovered_sensors: dict):
        self.hass = hass
        self.sensors = discovered_sensors
        self.phase = PHASE_BASELINE 
        # self.phase = PHASE_ACTIVE  # TEMP: For testing purposes, start in active phase
        
        # Internal state
        self.state_vector = None
        self.action_mask = None
        self.baseline_consumption = 0.0
        self.consumption_history = deque(maxlen=2016)  # 14 days * 144 (10min)
        
        # Engagement history
        self.engagement_history = deque(maxlen=100)
        self.notification_count_today = 0
        self.last_notification_date = None
        
        # Q-table simplified (for demonstration, in production use DQN)
        self.q_table = {}
        self.learning_rate = 0.1
        self.epsilon = 0.2  # Exploration rate
        
        # Behaviour indices
        self.anomaly_index = 0.0
        self.behaviour_index = 0.5
        self.fatigue_index = 0.0
        
    async def update_state(self):
        """Updates state vector S_t."""
        # Counter reset of daily notifications
        today = datetime.now().date()
        if self.last_notification_date != today:
            self.notification_count_today = 0
            self.last_notification_date = today
        
        # Collect sensor values with zero padding
        state = []

        # TODO: Might need to just retrieve a single main power sensor
        # E_total, F_total (Total Power Consumption)
        power_sensors = self.sensors.get("power", [])
        if power_sensors:
            total_power = await self._get_sensor_sum(power_sensors)
            state.extend([total_power, 1.0])
            self.consumption_history.append(total_power)
        else:
            state.extend([0.0, 0.0])
        _LOGGER.debug("Total power consumption: %.2f W", state[0])
        
        # TODO: Might need to update sensor value retrieval based on the TODO defined previously
        # E_app, F_app (Individual Appliance Power)
        if len(power_sensors) > 1:
            app_power = await self._get_sensor_value(power_sensors[0])
            state.extend([app_power, 1.0])
        else:
            state.extend([0.0, 0.0])
        _LOGGER.debug("Appliance power consumption: %.2f W", state[2])
        
        # T_in, F_T (Temperature)
        temp_sensors = self.sensors.get("temperature", [])
        if temp_sensors:
            temp = await self._get_sensor_value(temp_sensors[0])
            state.extend([temp, 1.0])
        else:
            state.extend([0.0, 0.0])
        _LOGGER.debug("Indoor temperature: %.2f °C", state[4])
        
        # H_in, F_H (Humidity)
        hum_sensors = self.sensors.get("humidity", [])
        if hum_sensors:
            humidity = await self._get_sensor_value(hum_sensors[0])
            state.extend([humidity, 1.0])
        else:
            state.extend([0.0, 0.0])
        _LOGGER.debug("Indoor humidity: %.2f %%", state[6])
        
        # L_in, F_L (Luminosity)
        lux_sensors = self.sensors.get("illuminance", [])
        if lux_sensors:
            lux = await self._get_sensor_value(lux_sensors[0])
            state.extend([lux, 1.0])
        else:
            state.extend([0.0, 0.0])
        _LOGGER.debug("Indoor luminosity: %.2f lx", state[8])
        
        # O_status, F_O (Occupancy)
        occ_sensors = self.sensors.get("occupancy", [])
        if occ_sensors:
            occupied = await self._get_binary_sensor(occ_sensors[0])
            state.extend([float(occupied), 1.0])
        else:
            state.extend([0.0, 0.0])
        _LOGGER.debug("Occupancy status: %s", "Occupied" if state[10] == 1.0 else "Unoccupied")
        
        # Calculate indices
        self._update_anomaly_index()
        self._update_behaviour_index()
        self._update_fatigue_index()
        
        state.extend([self.anomaly_index, self.behaviour_index, self.fatigue_index])
        _LOGGER.debug("State vector: %s", state)
        
        self.state_vector = np.array(state)
        
        # Update baseline consumption if in baseline phase
        if self.phase == PHASE_BASELINE and len(self.consumption_history) > 0:
            self.baseline_consumption = np.mean(self.consumption_history)
        
        # Update action mask M_t
        self._update_action_mask()
        
        # Decide action A_t if in active phase
        if self.phase == PHASE_ACTIVE:
            await self._decide_action()
    
    def _update_action_mask(self):
        """Generates binary action mask M_t."""
        mask = np.ones(len(ACTIONS))
        
        # noop always available

        # TODO: Might need to separate smart plugs from general power sensors
        # specific: needs smart plugs
        if not self.sensors.get("power"):
            mask[ACTIONS["specific"]] = 0
        
        # anomaly: needs enough consumption history
        if len(self.consumption_history) < 100:
            mask[ACTIONS["anomaly"]] = 0
        
        # behavioural: always available

        # normative: requires group data
        if self.baseline_consumption == 0.0:
            mask[ACTIONS["normative"]] = 0
        
        self.action_mask = mask
        _LOGGER.debug("Action mask: %s", mask)
    
    async def _decide_action(self):
        """Selects an action using epsilon-greedy policy."""
        if self.notification_count_today >= MAX_NOTIFICATIONS_PER_DAY:
            _LOGGER.info("Max daily notifications reached, no action taken.")
            return  # Notification limit reached
        
        # Discretize state for Q-table lookup
        state_key = self._discretize_state()
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            # Exploration: random valid action
            valid_actions = np.where(self.action_mask == 1)[0]
            action = np.random.choice(valid_actions)
        else:
            # Exploration: best known action
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(len(ACTIONS))
            q_values = self.q_table[state_key] * self.action_mask
            action = np.argmax(q_values)
        
        # Execute action
        if action != ACTIONS["noop"]:
            await self._execute_action(action)
            self.notification_count_today += 1
    
    async def _execute_action(self, action: int):
        """Fires a notification/service based on action."""
        action_name = [k for k, v in ACTIONS.items() if v == action][0]
        
        # TODO: Implement actual notification logic
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
            _LOGGER.info("Action executed: %s", action_name)
    
    # TODO: Implement reward update after user feedback
    def update_reward(self, feedback: str):
        """
        Updates Q-table with user feedback.
        feedback: 'positive', 'neutral', 'negative'
        """
        engagement_score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(
            feedback, 0.0
        )
        self.engagement_history.append(engagement_score)
        
        # Calculate reward R_t
        current_consumption = (
            self.consumption_history[-1] if self.consumption_history else 0
        )
        energy_saving = self.baseline_consumption - current_consumption
        
        avg_engagement = (
            np.mean(self.engagement_history) if self.engagement_history else 0
        )
        
        reward = (
            REWARD_WEIGHTS["alpha"] * energy_saving
            + REWARD_WEIGHTS["beta"] * avg_engagement
            - REWARD_WEIGHTS["delta"] * self.fatigue_index
        )
        
        # Update Q-table (simplified)
        state_key = self._discretize_state()
        if state_key in self.q_table:
            # Q-learning update: Q(s,a) <- Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
            action = np.argmax(self.q_table[state_key])
            old_q = self.q_table[state_key][action]
            max_future_q = np.max(self.q_table.get(state_key, np.zeros(len(ACTIONS))))
            new_q = old_q + self.learning_rate * (reward + GAMMA * max_future_q - old_q)
            self.q_table[state_key][action] = new_q
    
    def _update_anomaly_index(self):
        """Detects irregular consumption patterns."""
        if len(self.consumption_history) < 50:
            self.anomaly_index = 0.0
            return
        
        recent = list(self.consumption_history)[-50:]
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
    
    def _update_behaviour_index(self):
        """History of user engagement."""
        if len(self.engagement_history) > 0:
            self.behaviour_index = np.clip(np.mean(self.engagement_history), 0, 1)
            _LOGGER.debug("Behaviour index updated: %.2f", self.behaviour_index)
    
    def _update_fatigue_index(self):
        """Risk of user fatigue from too many notifications."""
        self.fatigue_index = self.notification_count_today / MAX_NOTIFICATIONS_PER_DAY
        _LOGGER.debug("Fatigue index updated: %.2f", self.fatigue_index)
    
    def _discretize_state(self) -> tuple:
        """Converts continuous state vector to discrete tuple for Q-table."""
        if self.state_vector is None:
            return (0,)
        # Simplification: use only consumption, anomalies e fatigue
        power = int(self.state_vector[0] / 100)  # Bins of 100W
        anomaly = int(self.anomaly_index * 10)
        fatigue = int(self.fatigue_index * 10)

        _LOGGER.debug("State discretized: power=%d, anomaly=%d, fatigue=%d", power, anomaly, fatigue)

        return (power, anomaly, fatigue)
    
    async def _get_sensor_value(self, entity_id: str) -> float:
        """Obtains value from a sensor entity."""
        state = self.hass.states.get(entity_id)
        if state is None:
            return 0.0
        try:
            return float(state.state)
        except (ValueError, TypeError):
            return 0.0
    
    async def _get_sensor_sum(self, entity_ids: list) -> float:
        """Sums values from multiple sensor entities."""
        total = 0.0
        for entity_id in entity_ids:
            total += await self._get_sensor_value(entity_id)
        return total
    
    async def _get_binary_sensor(self, entity_id: str) -> bool:
        """Obtains binary state (on/off) from a sensor entity."""
        state = self.hass.states.get(entity_id)
        if state is None:
            return False
        return state.state.lower() in ["on", "true", "detected"]