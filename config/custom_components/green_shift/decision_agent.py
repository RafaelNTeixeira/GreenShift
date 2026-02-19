import logging
import numpy as np
import ast
from datetime import datetime, timedelta
from collections import deque
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
import random

from .storage import StorageManager
from .helpers import get_friendly_name, should_ai_be_active
from .translations_runtime import get_language, get_notification_templates, get_time_of_day_name
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
    MAX_NOTIFICATIONS_PER_DAY,
    MIN_COOLDOWN_MINUTES,
    HIGH_OPPORTUNITY_THRESHOLD,
    CRITICAL_OPPORTUNITY_THRESHOLD,
    SHADOW_EXPLORATION_RATE,
    SHADOW_LEARNING_RATE,
    SHADOW_INTERVAL_MULTIPLIER,
    ENVIRONMENT_OFFICE
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
    
    def __init__(self, hass: HomeAssistant, discovered_sensors: dict, data_collector, storage_manager: StorageManager = None, config_data: dict = None):
        self.hass = hass
        self.sensors = discovered_sensors
        self.data_collector = data_collector
        self.storage = storage_manager
        self.config_data = config_data or {}
        self.start_date = datetime.now()
        self._process_count = 0
        self.phase = PHASE_BASELINE 
        
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
        self.episode_number = 0  # Track RL episodes for research logging
        self.shadow_episode_number = 0  # Track shadow RL episodes (baseline phase)
        
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
            _LOGGER.info("Loaded baseline consumption: %.2f W", self.baseline_consumption)

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

        # Load notification tracking
        if "notification_count_today" in state:
            self.notification_count_today = state["notification_count_today"]
        
        if "last_notification_date" in state and state["last_notification_date"]:
            try:
                self.last_notification_date = datetime.fromisoformat(state["last_notification_date"]).date()
            except (ValueError, AttributeError):
                self.last_notification_date = None
        
        if "last_notification_time" in state and state["last_notification_time"]:
            try:
                self.last_notification_time = datetime.fromisoformat(state["last_notification_time"])
            except (ValueError, AttributeError):
                self.last_notification_time = None

        # Load notification history
        if "notification_history" in state:
            self.notification_history = deque(state["notification_history"], maxlen=50)
        
        # Load Q-table (convert string keys back to tuples, and ensure action keys are ints)
        if "q_table" in state and state["q_table"]:
            try:
                loaded_count = 0
                self.q_table = {}
                for state_key, actions in state["q_table"].items():
                    try:
                        # Convert state key from string to tuple (safely)
                        parsed_key = ast.literal_eval(state_key)
                        if isinstance(parsed_key, tuple):
                            # Convert action keys from strings to ints
                            self.q_table[parsed_key] = {
                                int(action): float(q_value) 
                                for action, q_value in actions.items()
                            }
                            loaded_count += 1
                        else:
                            _LOGGER.warning("Invalid Q-table state key (not a tuple): %s", state_key)
                    except (ValueError, SyntaxError, TypeError) as e:
                        _LOGGER.warning("Failed to parse Q-table entry '%s': %s", state_key, e)
                        continue
                
                _LOGGER.info("Loaded Q-table with %d state entries", loaded_count)
            except Exception as e:
                _LOGGER.error("Failed to load Q-table: %s", e)
                self.q_table = {}

        # Load shadow learning episode counter
        if "shadow_episode_number" in state:
            self.shadow_episode_number = state["shadow_episode_number"]
            _LOGGER.info("Loaded shadow episode number: %d", self.shadow_episode_number)

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
        
        # Convert Q-table to JSON-serializable format
        # State keys (tuples) -> strings, Action keys (ints) -> remain as ints (JSON will convert)
        serializable_q_table = {
            str(state_key): {int(action): float(q_value) for action, q_value in actions.items()}
            for state_key, actions in self.q_table.items()
        }
        
        ai_state = {
            "start_date": safe_start_date,
            "phase": self.phase,
            "baseline_consumption": float(self.baseline_consumption),
            "area_baselines": self.area_baselines,
            "current_week_start_date": self.current_week_start_date.isoformat() if self.current_week_start_date else None,
            "anomaly_index": float(self.anomaly_index),
            "behaviour_index": float(self.behaviour_index),
            "fatigue_index": float(self.fatigue_index),
            "notification_count_today": self.notification_count_today,
            "last_notification_date": self.last_notification_date.isoformat() if self.last_notification_date else None,
            "last_notification_time": self.last_notification_time.isoformat() if self.last_notification_time else None,
            "notification_history": list(self.notification_history),
            "q_table": serializable_q_table,
            "shadow_episode_number": self.shadow_episode_number,
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
            old_count = self.notification_count_today
            self.notification_count_today = 0
            self.last_notification_date = today
            _LOGGER.info("Daily notification counter reset (was %d/%d)", old_count, MAX_NOTIFICATIONS_PER_DAY)
        
        # Build state vector from DataCollector's current readings
        await self._build_state_vector()
        
        # Calculate indices (anomaly, behaviour, fatigue)
        await self._update_anomaly_index()
        await self._update_area_anomalies()
        self._update_behaviour_index()
        await self._update_fatigue_index()
        
        # Update action mask M_t
        await self._update_action_mask()
        
        # Decide action A_t based on phase
        if self.phase == PHASE_ACTIVE:
            await self._decide_action()
        elif self.phase == PHASE_BASELINE:
            # Shadow learning: simulate decisions without executing actions
            # Runs at reduced frequency to avoid excessive Q-table noise
            if self._process_count % SHADOW_INTERVAL_MULTIPLIER == 0:
                await self._shadow_decide_action()

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
        _LOGGER.debug("Anomaly index: %.2f, Behaviour index: %.2f, Fatigue index: %.2f", self.anomaly_index, self.behaviour_index, self.fatigue_index)

        # 10. Area anomaly count (spatial awareness)
        area_anomaly_count = len([a for a in self.area_anomalies.values() if any(v > 0.3 for v in a.values())])
        state.append(area_anomaly_count)
        _LOGGER.debug("Area anomaly count: %d", area_anomaly_count)
        
        # 11. Time of day (normalized)
        now = datetime.now()
        time_of_day = (now.hour * 60 + now.minute) / (24 * 60)  # 0 to 1
        state.append(time_of_day)
        _LOGGER.debug("Time of day (normalized): %.2f", time_of_day)
        
        # 12. Day of week (normalized)
        day_of_week = now.weekday() / 6.0  # 0 (Monday) to 1 (Sunday)
        state.append(day_of_week)
        _LOGGER.debug("Day of week (normalized): %.2f", day_of_week)
        
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
        # Check if AI should be active (working hours for office mode)
        if not should_ai_be_active(self.config_data):
            _LOGGER.debug("Outside working hours - AI notifications paused")
            return
        
        # Check notification limits
        if self.notification_count_today >= MAX_NOTIFICATIONS_PER_DAY:
            _LOGGER.info("Max notifications reached for today (%d/%d)", self.notification_count_today, MAX_NOTIFICATIONS_PER_DAY)
            return
        
        # Calculate opportunity score BEFORE cooldown check
        opportunity_score = await self._calculate_opportunity_score()
        
        # Check adaptive cooldown with opportunity-based bypass
        if not await self._check_cooldown_with_opportunity(opportunity_score):
            return
        
        # Check fatigue threshold - but allow bypass for critical opportunities
        if self.fatigue_index > FATIGUE_THRESHOLD and opportunity_score < CRITICAL_OPPORTUNITY_THRESHOLD:
            _LOGGER.info("User fatigue too high (%.2f > %.2f) and opportunity not critical (%.2f), skipping notification", 
                        self.fatigue_index, FATIGUE_THRESHOLD, opportunity_score)
            await self._log_blocked_notification(
                reason="fatigue_threshold",
                opportunity_score=opportunity_score,
                time_since_last=None,
                required_cooldown=None,
                adaptive_cooldown=None,
                available_actions=[]
            )
            return
            
        # Get current state
        state_key = self._discretize_state()

        # Available actions based on mask (all non-noop actions)
        available_actions = [a for a, available in self.action_mask.items() if available]

        if not available_actions:
            _LOGGER.debug("No notification actions available based on current context")
            return

       # Epsilon-greedy action selection
        action_source = "explore"
        if random.random() < self.epsilon:
            # Exploration: random action
            action = random.choice(available_actions)
            action_source = "explore"
            _LOGGER.debug("Exploration: selected random action %d", action)
        else:
            # Exploitation: best known action
            action_source = "exploit"
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
        
        # Log RL episode for research analysis
        self.episode_number += 1
        await self._log_rl_episode(state_key, action, reward, action_source, opportunity_score)

    async def _shadow_decide_action(self):
        """
        Shadow learning: simulates RL decisions during baseline phase WITHOUT
        executing any actions (no notifications sent to the user).
        
        This allows the Q-table to be pre-trained on energy consumption patterns
        and temporal context so the agent is already warm-started when the active
        phase begins. Because no user interaction exists during baseline, the
        reward function uses only energy-pattern and context-based components.
        
        Key differences from _decide_action():
        - Higher exploration rate (SHADOW_EXPLORATION_RATE) since bad exploration has no cost
        - Lower learning rate (SHADOW_LEARNING_RATE) since rewards are estimated, not real
        - No notifications sent, no engagement/fatigue tracking modified
        - Episodes logged with action_source="shadow" for research differentiation
        """

        # Check if AI should be active (working hours for office mode)
        if not should_ai_be_active(self.config_data):
            _LOGGER.debug("Outside working hours - shadow learning paused")
            return
        
        # Get current state
        state_key = self._discretize_state()

        # Available actions based on mask
        available_actions = [a for a, available in self.action_mask.items() if available]

        if not available_actions:
            _LOGGER.debug("Shadow learning: no actions available based on current context")
            return

        # Epsilon-greedy selection with higher exploration rate
        action_source = "shadow_explore"
        if random.random() < SHADOW_EXPLORATION_RATE:
            # Exploration: random action
            action = random.choice(available_actions)
            action_source = "shadow_explore"
            _LOGGER.debug("Shadow learning (explore): selected random action %d", action)
        else:
            # Exploitation: best known action from Q-table
            action_source = "shadow_exploit"
            if state_key not in self.q_table:
                self.q_table[state_key] = {a: 0.0 for a in ACTIONS.values()}
            
            best_action = max(available_actions, key=lambda a: self.q_table[state_key].get(a, 0.0))
            action = best_action
            _LOGGER.debug("Shadow learning (exploit): selected best action %d (Q=%.2f)", action, self.q_table[state_key].get(action, 0.0))
        
        # Calculate shadow reward
        reward = await self._calculate_shadow_reward(action)
        
        # Update Q-table with shadow learning rate
        await self._shadow_update_q_table(state_key, action, reward)
        
        # Log shadow episode to research database
        self.shadow_episode_number += 1
        await self._log_rl_episode(state_key, action, reward, action_source)
        
        _LOGGER.debug("Shadow episode #%d: state=%s, action=%d, reward=%.4f",
                      self.shadow_episode_number, state_key[:3], action, reward)

    async def _calculate_shadow_reward(self, action: int) -> float:
        """
        Calculates an estimated reward during baseline without user interaction.
        
        Components:
        1. Energy savings potential: how much current consumption deviates above
           the running baseline mean (higher deviation = more opportunity for the
           action to be useful, higher reward for actions that address it)
        2. Action-context alignment: whether the selected action type matches the
           current environmental context (e.g., anomaly action when anomaly_index
           is high, specific action when a single device dominates)
        3. Temporal appropriateness: reward actions chosen at good times (occupied,
           reasonable hour) more than those at bad times
        
        Formula:
            R_shadow = α · savings_potential + β_ctx · context_alignment + δ_time · time_score
        
        The weights are intentionally conservative since these are estimated rewards.
        """
        power_history_data = await self.data_collector.get_power_history(hours=1)
        power_values = [power for timestamp, power in power_history_data]

        if len(power_values) < 10:
            return 0.0
        
        # Component 1: Energy savings potential 
        current_power = power_values[-1]
        running_mean = np.mean(power_values)
        
        if running_mean > 0:
            # Positive when consuming above mean (opportunity for savings)
            savings_potential = max(0, (current_power - running_mean) / running_mean)
            savings_potential = min(savings_potential, 1.0)  # Cap at 1.0
        else:
            savings_potential = 0.0
        
        # Component 2: Action–context alignment 
        # Reward higher when the chosen action type matches the situation
        context_alignment = 0.0
        action_name = [k for k, v in ACTIONS.items() if v == action][0]
        
        if action_name == "specific":
            # Specific actions are best when a single device dominates consumption
            top_consumer = await self._get_top_power_consumer()
            if running_mean > 0 and top_consumer > 0:
                # Higher reward if one device accounts for a large share
                dominance_ratio = top_consumer / max(current_power, 1.0)
                context_alignment = min(dominance_ratio, 1.0)
            
        elif action_name == "anomaly":
            # Anomaly actions are best when anomaly_index is high
            context_alignment = self.anomaly_index
            # Boost if area-specific anomalies are present
            area_anomaly_count = len([a for a in self.area_anomalies.values() 
                                      if any(v > 0.3 for v in a.values())])
            if area_anomaly_count > 0:
                context_alignment = min(context_alignment + 0.2, 1.0)
            
        elif action_name == "behavioural":
            # Behavioural nudges are best during typical usage patterns
            # Reward moderately as they're always somewhat applicable
            context_alignment = 0.4
            # Better when consumption is moderate
            if running_mean > 0:
                variability = np.std(power_values) / running_mean if running_mean > 0 else 0
                # Low variability = stable patterns = better for behavioural nudges
                context_alignment += 0.3 * max(0, 1.0 - variability)
            
        elif action_name == "normative":
            # Normative nudges work best when baseline exists and deviation is notable
            if self.baseline_consumption > 0:
                deviation = abs(current_power - self.baseline_consumption) / self.baseline_consumption
                context_alignment = min(deviation, 1.0)
            else:
                context_alignment = 0.2  # Still somewhat useful
        
        # Component 3: Temporal appropriateness
        now = datetime.now()
        hour = now.hour
        
        # Good notification times (morning, lunch, evening)
        if 7 <= hour <= 9 or 12 <= hour <= 14 or 18 <= hour <= 21:
            time_score = 1.0
        elif 22 <= hour or hour < 7:
            time_score = 0.2  # Bad time
        else:
            time_score = 0.6  # Neutral
        
        # Boost if building is occupied
        current_state = self.data_collector.get_current_state()
        if current_state.get("occupancy", False):
            time_score = min(time_score + 0.2, 1.0)
        
        # Weighted combination (conservative weights)
        reward = (
            0.5 * savings_potential +
            0.3 * context_alignment +
            0.2 * time_score
        )
        
        _LOGGER.debug(
            "Shadow reward: %.4f (savings=%.2f, alignment=%.2f [%s], time=%.2f)",
            reward, savings_potential, context_alignment, action_name, time_score
        )
        
        return reward

    async def _shadow_update_q_table(self, state_key: tuple, action: int, reward: float):
        """
        Updates Q-table with shadow learning rate (lower than active learning rate).
        Uses the same Q-learning formula but with SHADOW_LEARNING_RATE to avoid
        over-committing to estimated rewards.
        
        Q(s,a) ← Q(s,a) + α_shadow[R_shadow + γ max Q(s',a') - Q(s,a)]
        """
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in ACTIONS.values()}
        
        current_q = self.q_table[state_key].get(action, 0.0)
        
        # Next state is the same since no real action was executed
        next_state_key = self._discretize_state()
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in ACTIONS.values()}
        
        max_next_q = max(self.q_table[next_state_key].values())
        
        # Q-learning update with shadow learning rate
        new_q = current_q + SHADOW_LEARNING_RATE * (reward + GAMMA * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        
        _LOGGER.debug("Shadow Q-table updated: state=%s, action=%d, reward=%.2f, Q: %.2f → %.2f",
                      state_key[:3], action, reward, current_q, new_q)

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
        
        _LOGGER.debug("Q-table updated: state=%s, action=%d, reward=%.2f, Q: %.2f → %.2f", state_key[:3], action, reward, current_q, new_q)
    
    async def _execute_action(self, action: int):
        """Executes selected action by sending a notification."""
        action_name = [k for k, v in ACTIONS.items() if v == action][0]

        # Get appropriate notification template
        notification = await self._generate_notification(action_name)
        
        if notification:
            # Create notification ID for tracking
            notification_id = f"energy_nudge_{datetime.now().timestamp()}"
            
            # Track notification
            self.notification_count_today += 1
            self.last_notification_time = datetime.now()
            self.notification_history.append({
                "timestamp": datetime.now().isoformat(),
                "action_type": action_name,
                "notification_id": notification_id,
                "title": notification["title"],
                "message": notification["message"],
                "responded": False
            })
            
            # Log nudge to research database
            if self.storage:
                current_state = self.data_collector.get_current_state()
                await self.storage.log_nudge_sent({
                    "notification_id": notification_id,
                    "phase": self.phase,
                    "action_type": action_name,
                    "template_index": notification.get("template_index"),
                    "title": notification["title"],
                    "message": notification["message"],
                    "state_vector": self.state_vector.tolist() if self.state_vector is not None and hasattr(self.state_vector, 'tolist') else [],
                    "current_power": current_state.get("power", 0),
                    "anomaly_index": self.anomaly_index,
                    "behaviour_index": self.behaviour_index,
                    "fatigue_index": self.fatigue_index
                })
            
            _LOGGER.info("Nudge notification added to dashboard (%d/%d today): %s - %s", self.notification_count_today, MAX_NOTIFICATIONS_PER_DAY, action_name, notification["title"])

            async_dispatcher_send(self.hass, GS_AI_UPDATE_SIGNAL)

    async def _generate_notification(self, action_type: str) -> dict:
        """
        Generates context-aware notification based on action type.
        """
        # Get user's language
        language = await get_language(self.hass)
        
        # Get templates for user's language
        notification_templates = get_notification_templates(language)
        templates = notification_templates.get(action_type, [])
        if not templates:
            return None
        
        # Select template based on context
        template_index = random.randint(0, len(templates) - 1)
        template = templates[template_index]
        
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
            "message": message,
            "template_index": template_index
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
        language = await get_language(self.hass)
        
        # Determine time of day key
        if 6 <= now.hour < 12:
            time_key = "morning"
        elif 12 <= now.hour < 18:
            time_key = "afternoon"
        elif 18 <= now.hour < 22:
            time_key = "evening"
        else:
            time_key = "night"
        
        context["time_of_day"] = get_time_of_day_name(time_key, language)
        
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
        
        # Log response to research database
        if self.storage:
            await self.storage.log_nudge_response(notification_id, accepted)
        
        _LOGGER.info("Notification feedback received: %s - %s", notification_id, "accepted" if accepted else "rejected")
        
        # Save state
        if self.storage:
            await self._save_persistent_state()
    
    async def _log_blocked_notification(
        self,
        reason: str,
        opportunity_score: float,
        time_since_last: float = None,
        required_cooldown: float = None,
        adaptive_cooldown: float = None,
        available_actions: list = None
    ):
        """
        Log when a notification attempt was blocked in active phase.
        
        Args:
            reason: Reason for blocking (fatigue_threshold, no_available_actions)
            opportunity_score: Calculated opportunity score at time of block
            time_since_last: Minutes since last notification (for cooldown blocks)
            required_cooldown: Required cooldown that wasn't met (for cooldown blocks)
            adaptive_cooldown: Base adaptive cooldown calculated (for cooldown blocks)
            available_actions: List of available actions (empty if none)
        """
        if not self.storage or self.phase != PHASE_ACTIVE:
            # Only log in active phase
            return
        
        current_state = self.data_collector.get_current_state()
        now = datetime.now()
        
        # Serialize action mask
        action_mask_dict = {int(k): bool(v) for k, v in self.action_mask.items()}
        
        block_data = {
            "phase": self.phase,
            "block_reason": reason,
            "opportunity_score": opportunity_score,
            "current_power": current_state.get("power", 0),
            "anomaly_index": self.anomaly_index,
            "behaviour_index": self.behaviour_index,
            "fatigue_index": self.fatigue_index,
            "notification_count_today": self.notification_count_today,
            "time_since_last_notification_minutes": time_since_last,
            "required_cooldown_minutes": required_cooldown,
            "adaptive_cooldown_minutes": adaptive_cooldown,
            "available_action_count": len(available_actions) if available_actions else 0,
            "action_mask": action_mask_dict,
            "state_vector": self.state_vector.tolist() if self.state_vector is not None and hasattr(self.state_vector, 'tolist') else [],
            "time_of_day_hour": now.hour
        }
        
        await self.storage.log_blocked_notification(block_data)
        _LOGGER.debug("Blocked notification logged: reason=%s, opportunity=%.2f", reason, opportunity_score)
    
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
        _LOGGER.debug("Reward calculated: Energy saving=%.4f, Engagement=%.4f, Fatigue penalty=%.4f, Total reward=%.4f", 
                      energy_saving, engagement, fatigue_penalty, reward)
        
        return reward
    
    async def _log_rl_episode(self, state_key: tuple, action: int, reward: float, action_source: str, opportunity_score: float = None):
        """Log RL decision episode to research database for convergence analysis."""
        if not self.storage:
            return
        
        # Get action name
        action_name = [k for k, v in ACTIONS.items() if v == action][0]
        
        # Get current Q-values for this state
        q_values = self.q_table.get(state_key, {a: 0.0 for a in ACTIONS.values()})
        max_q = max(q_values.values()) if q_values else 0.0
        
        # Get current power
        current_state = self.data_collector.get_current_state()
        
        # Serialize action mask
        action_mask_dict = {int(k): bool(v) for k, v in self.action_mask.items()}
        
        # Get time of day
        now = datetime.now()
        time_of_day_hour = now.hour
        
        episode_data = {
            "episode": self.episode_number,
            "phase": self.phase,
            "state_vector": self.state_vector.tolist() if self.state_vector is not None and hasattr(self.state_vector, 'tolist') else [],
            "state_key": state_key,
            "action": action,
            "action_name": action_name,
            "action_source": action_source,
            "reward": reward,
            "q_values": {int(k): float(v) for k, v in q_values.items()},
            "max_q": max_q,
            "epsilon": self.epsilon,
            "action_mask": action_mask_dict,
            "power": current_state.get("power", 0),
            "anomaly_index": self.anomaly_index,
            "behaviour_index": self.behaviour_index,
            "fatigue_index": self.fatigue_index,
            "opportunity_score": opportunity_score,  # None for shadow episodes, calculated for active
            "time_of_day_hour": time_of_day_hour,
            "baseline_power_reference": self.baseline_consumption
        }
        
        await self.storage.log_rl_decision(episode_data)
    
    async def _update_anomaly_index(self):
        """
        Detects anomalies in consumption using z-score.
        """
        power_history_data = await self.data_collector.get_power_history(hours=1) # Last hour
        power_values = [power for timestamp, power in power_history_data]
        
        readings_per_hour = int((3600 / UPDATE_INTERVAL_SECONDS) * 0.8)  # Require at least 80% of expected readings for reliability
        
        _LOGGER.debug(f"Anomaly detection: {len(power_values)} readings available, {readings_per_hour} needed for 1 hour")
        
        if len(power_values) < readings_per_hour:
            self.anomaly_index = 0.0
            _LOGGER.debug(f"Not enough data for anomaly detection ({len(power_values)}/{readings_per_hour}), setting to 0")
            return

        recent = power_values[-readings_per_hour:]
        mean = np.mean(recent)
        std = np.std(recent)
        current = recent[-1]
        
        _LOGGER.debug(f"Anomaly stats: mean={mean:.2f}W, std={std:.2f}W, current={current:.2f}W")
        
        # Z-score normalized to [0,1]
        if std > 0:
            z_score = max((current - mean) / std, 0) # Consider only positive deviations (consumption above mean)
            self.anomaly_index = min(z_score / 3.0, 1.0)
            _LOGGER.debug("Anomaly index updated: %.2f (z-score: %.2f)", self.anomaly_index, z_score)
        else:
            self.anomaly_index = 0.0
            _LOGGER.debug("No variance in data, anomaly index set to 0")
    
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
        Uses exponential moving average for recent behavior with smoothing.
        """
        if len(self.engagement_history) == 0:
            return
        
        # Weighted towards recent interactions, but not too aggressively
        weights = np.exp(np.linspace(-1, 0, len(self.engagement_history)))
        weights /= weights.sum()
        
        weighted_engagement = np.average(self.engagement_history, weights=weights)
        
        # Smooth update to avoid volatility: new index is 40% new engagement, 60% old index
        new_index = np.clip(weighted_engagement, 0, 1)
        self.behaviour_index = 0.4 * new_index + 0.6 * self.behaviour_index
        
        _LOGGER.debug("Behaviour index updated: %.2f (raw: %.2f, history size: %d)", self.behaviour_index, weighted_engagement, len(self.engagement_history))
    
    async def _update_fatigue_index(self):
        """
        Calculates user fatigue based on recent notification patterns.
        Considers rejection rate, frequency of notifications, and time decay.
        Lower fatigue = more receptive to notifications.
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
            _LOGGER.debug("Timestamps of last 3 notifications for fatigue calculation: %s", timestamps)
            time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
            
            if time_span > 0:
                frequency_factor = min(3 / time_span, 1.0)  # Ideal: 1 per hour
            else:
                frequency_factor = 1.0
        else:
            frequency_factor = 0.0
        
        # Time decay component: fatigue naturally decreases over time
        # If more than 1 hour since last notification, reduce fatigue
        time_decay_factor = 1.0
        if self.last_notification_time:
            hours_since_last = (datetime.now() - self.last_notification_time).total_seconds() / 3600
            if hours_since_last > 1:
                # Decay factor: approaches 0.5 as time passes (never goes to 0)
                time_decay_factor = max(0.5, 1.0 - (hours_since_last - 1) * 0.1)
        
        # Combined fatigue score with time decay
        base_fatigue = 0.6 * rejection_rate + 0.4 * frequency_factor
        self.fatigue_index = np.clip(
            base_fatigue * time_decay_factor,
            0.0,
            1.0
        )
        
        _LOGGER.debug("Fatigue index updated: %.2f (rejection=%.2f, freq=%.2f, decay=%.2f)", 
                     self.fatigue_index, rejection_rate, frequency_factor, time_decay_factor)
    
    def _discretize_state(self) -> tuple:
        """
        Converts continuous state vector to discrete tuple for Q-table.
        
        State Space: ~19,584 possible states (101x4x3x2x4x2)
        
        State components:
        - power_bin: Power consumption in 100W bins (adaptive, 0-100)
        - anomaly_level: 0=none(<0.25), 1=low(0.25-0.5), 2=medium(0.5-0.75), 3=high(>0.75)
        - fatigue_level: 0=low(<0.33), 1=medium(0.33-0.66), 2=high(>0.66)
        - has_area_anomaly: 0=no area anomalies, 1=area anomalies present
        - time_period: 0=night, 1=morning, 2=afternoon, 3=evening
        - is_occupied: 0=not occupied, 1=occupied
        """
        if self.state_vector is None or len(self.state_vector) < 18:
            return (0, 0, 0, 0, 0, 0)
        
        # Power in 100W bins
        power = int(self.state_vector[0] / 100)
        power_bin = min(power, 100) # Cap at 10000W for table size
        
        # Anomaly level
        if self.anomaly_index < 0.25:
            anomaly_level = 0  # None
        elif self.anomaly_index < 0.5:
            anomaly_level = 1  # Low
        elif self.anomaly_index < 0.75:
            anomaly_level = 2  # Medium
        else:
            anomaly_level = 3  # High
        
        # Fatigue level
        if self.fatigue_index < 0.33:
            fatigue_level = 0  # Low
        elif self.fatigue_index < 0.66:
            fatigue_level = 1  # Medium
        else:
            fatigue_level = 2  # High
        
        # Area anomaly
        area_anomaly_count = int(self.state_vector[15])
        has_area_anomaly = 1 if area_anomaly_count > 0 else 0
        
        # Time of day (0=night, 1=morning, 2=afternoon, 3=evening)
        time_of_day = self.state_vector[16]  # 0-1 normalized
        if time_of_day < 0.25:  # 0:00-6:00
            time_period = 0
        elif time_of_day < 0.5:  # 6:00-12:00
            time_period = 1
        elif time_of_day < 0.75:  # 12:00-18:00
            time_period = 2
        else:  # 18:00-24:00
            time_period = 3
        
        # Occupancy (binary)
        is_occupied = int(self.state_vector[10])
        
        return (power_bin, anomaly_level, fatigue_level, has_area_anomaly, time_period, is_occupied)
    
    async def _calculate_opportunity_score(self) -> float:
        """
        Calculates opportunity score for sending a notification.
        Combines:
        - Energy savings potential (0-1)
        - Urgency/anomaly severity (0-1)
        - User receptiveness (0-1)
        - Context appropriateness (0-1)
        
        Returns: Combined score 0-1, higher = better opportunity
        """
        current_state = self.data_collector.get_current_state()
        current_power = current_state.get("power", 0)
        
        # Component 1: Energy savings potential
        if self.baseline_consumption > 0 and current_power > self.baseline_consumption:
            # Higher deviation = higher potential
            deviation_ratio = (current_power - self.baseline_consumption) / self.baseline_consumption
            savings_potential = min(deviation_ratio, 1.0)  # Cap at 100% deviation
        else:
            savings_potential = 0.0
        
        # Component 2: Urgency (anomaly + area anomalies)
        urgency = self.anomaly_index
        
        # Boost urgency if multiple areas have anomalies (spatial urgency)
        area_anomaly_count = len([a for a in self.area_anomalies.values() if any(v > 0.3 for v in a.values())])
        if area_anomaly_count > 0:
            urgency = min(urgency + (area_anomaly_count * 0.1), 1.0)
        
        # Component 3: User receptiveness (inverse of fatigue, boosted by good behaviour)
        receptiveness = (1.0 - self.fatigue_index) * self.behaviour_index
        
        # Component 4: Context appropriateness (time of day + occupancy)
        now = datetime.now()
        hour = now.hour
        
        # Better times: morning (7-9), lunch (12-14), evening (18-21)
        if 7 <= hour <= 9 or 12 <= hour <= 14 or 18 <= hour <= 21:
            time_score = 1.0
        elif 22 <= hour or hour < 7:  # Late night/early morning - poor time
            time_score = 0.3
        else:
            time_score = 0.7
        
        # Boost if occupied
        occupancy = 1.0 if current_state.get("occupancy", False) else 0.5
        context = (time_score + occupancy) / 2.0
        
        # Weighted combination
        opportunity_score = (
            0.35 * savings_potential +
            0.35 * urgency +
            0.20 * receptiveness +
            0.10 * context
        )
        
        _LOGGER.debug(
            "Opportunity score: %.2f (savings=%.2f, urgency=%.2f, receptive=%.2f, context=%.2f)",
            opportunity_score, savings_potential, urgency, receptiveness, context
        )
        
        return opportunity_score
    
    async def _check_cooldown_with_opportunity(self, opportunity_score: float) -> bool:
        """
        Checks if notification can be sent based on adaptive cooldown.
        Allows bypassing cooldown for high-opportunity situations.
        
        Returns: True if notification allowed, False if in cooldown
        """
        if self.last_notification_time is None:
            return True  # First notification always allowed
        
        time_since_last = (datetime.now() - self.last_notification_time).total_seconds() / 60 # minutes
        
        # Calculate adaptive cooldown based on fatigue and time of day
        base_cooldown = MIN_COOLDOWN_MINUTES
        
        # Increase cooldown if fatigue is high
        fatigue_multiplier = 1.0 + (self.fatigue_index * 2.0) # 1x to 3x based on fatigue
        
        # Decrease cooldown during peak energy usage hours (more opportunities)
        now = datetime.now()
        hour = now.hour
        if 17 <= hour <= 22:  # Evening peak
            time_multiplier = 0.7
        elif 7 <= hour <= 9:  # Morning peak
            time_multiplier = 0.8
        else:
            time_multiplier = 1.0
        
        adaptive_cooldown = base_cooldown * fatigue_multiplier * time_multiplier
        
        # Allow bypass if opportunity is exceptional
        if opportunity_score >= CRITICAL_OPPORTUNITY_THRESHOLD:
            # Critical opportunity - immediate notification allowed
            _LOGGER.info("Critical opportunity (%.2f) - bypassing cooldown (%.1f min since last)",
                opportunity_score, time_since_last
            )
            return True
        elif opportunity_score >= HIGH_OPPORTUNITY_THRESHOLD:
            # High opportunity - reduced cooldown (50% of adaptive)
            required_cooldown = adaptive_cooldown * 0.5
            if time_since_last >= required_cooldown:
                _LOGGER.info("High opportunity (%.2f) - reduced cooldown met (%.1f/%.1f min)",
                    opportunity_score, time_since_last, required_cooldown
                )
                return True
            else:
                _LOGGER.debug("High opportunity but cooldown not met: %.1f/%.1f min (opportunity=%.2f)",
                    time_since_last, required_cooldown, opportunity_score
                )
                return False
        else:
            # Normal opportunity - full adaptive cooldown required
            if time_since_last >= adaptive_cooldown:
                _LOGGER.debug("Standard cooldown met: %.1f/%.1f min (opportunity=%.2f)",
                    time_since_last, adaptive_cooldown, opportunity_score
                )
                return True
            else:
                _LOGGER.debug("In cooldown: %.1f/%.1f min (opportunity=%.2f, fatigue=%.2f)",
                    time_since_last, adaptive_cooldown, opportunity_score, self.fatigue_index
                )
                return False
    
    async def calculate_area_baselines(self):
        """
        Calculate baseline values for each area during the baseline phase.
        Called at the end of the baseline phase.
        
        In office mode, only uses working hours data to avoid weekend/off-hours bias.
        """
        areas = self.data_collector.get_all_areas()
        
        # Determine if we should filter to working hours only (office mode)
        is_office_mode = self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE
        working_hours_filter = True if is_office_mode else None
        
        if is_office_mode:
            _LOGGER.info("Office mode detected - calculating area baselines from working hours data only")
        
        for area in areas:
            if area == "No Area":
                continue
            
            baselines = {}
            
            # Temperature baseline
            temp_history = await self.data_collector.get_area_history(
                area, "temperature", days=14, working_hours_only=working_hours_filter
            )
            if temp_history:
                temp_values = [val for ts, val in temp_history if val is not None]
                if temp_values:
                    baselines["temperature"] = round(np.mean(temp_values), 2)
            
            # Power baseline
            power_history = await self.data_collector.get_area_history(
                area, "power", days=14, working_hours_only=working_hours_filter
            )
            if power_history:
                power_values = [val for ts, val in power_history if val is not None]
                if power_values:
                    baselines["power"] = round(np.mean(power_values), 2)
            
            # Humidity baseline
            hum_history = await self.data_collector.get_area_history(
                area, "humidity", days=14, working_hours_only=working_hours_filter
            )
            if hum_history:
                hum_values = [val for ts, val in hum_history if val is not None]
                if hum_values:
                    baselines["humidity"] = round(np.mean(hum_values), 2)
            
            if baselines:
                self.area_baselines[area] = baselines
        
        _LOGGER.info("Calculated baselines for %d areas (office mode: %s)", len(self.area_baselines), is_office_mode)
        
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
        # In office mode, only compare working hours to baseline (which was also working hours only)
        is_office_mode = self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE
        working_hours_filter = True if is_office_mode else None
        
        power_history_data = await self.data_collector.get_power_history(
            days=days_in_current_week,
            working_hours_only=working_hours_filter
        )
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

        # days_in_current_week = 7
        
        if today.weekday() == 6 and days_in_current_week >= 7:  # Sunday and full week complete
        # if days_in_current_week >= 7:
            # Check if we've already logged this week
            week_key = self.current_week_start_date.isoformat()
            if not hasattr(self, '_logged_weeks'):
                self._logged_weeks = set()
            
            if week_key not in self._logged_weeks and self.storage:
                success = progress < 100

                challenge_payload = {
                    'week_start_date': self.current_week_start_date.isoformat(),
                    'week_end_date': today.isoformat(),
                    'phase': PHASE_ACTIVE,
                    'target_percentage': target_percentage,
                    'baseline_W': self.baseline_consumption, 
                    'actual_W': current_avg,                 
                    'savings_W': self.baseline_consumption - current_avg,
                    'savings_percentage': progress,
                    'achieved': success                        
                }
                
                await self.storage.log_weekly_challenge(challenge_data=challenge_payload)

                self._logged_weeks.add(week_key)
                _LOGGER.info("Logged weekly challenge: week=%s, success=%s, progress=%.1f%%", week_key, success, progress)
        
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
    