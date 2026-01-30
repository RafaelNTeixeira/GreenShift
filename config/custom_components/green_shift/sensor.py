"""Entidades virtuais para Energy Research Platform."""
import logging
from datetime import datetime
from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, PHASE_BASELINE, BASELINE_DAYS

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Setup of virtual sensors."""
    agent = hass.data[DOMAIN]["agent"]
    start_date = hass.data[DOMAIN]["start_date"]
    
    sensors = [
        ResearchPhaseSensor(agent, start_date),
        EnergyBaselineSensor(agent),
        CurrentConsumptionSensor(agent),
        SavingsAccumulatedSensor(agent),
        CO2SavedSensor(agent),
        TasksCompletedSensor(),
        CollaborativeGoalSensor(agent),
        BehaviourIndexSensor(agent),
        FatigueIndexSensor(agent),
    ]
    
    async_add_entities(sensors)


class ResearchPhaseSensor(SensorEntity):
    """Sensor that indicates the current research phase."""
    
    def __init__(self, agent, start_date):
        self._agent = agent
        self._start_date = start_date
        self._attr_name = "Research Phase"
        self._attr_unique_id = f"{DOMAIN}_research_phase"
        self._attr_icon = "mdi:flask"
    
    @property
    def state(self):
        return self._agent.phase
    
    @property
    def extra_state_attributes(self):
        days_running = (datetime.now() - self._start_date).days
        days_remaining = max(0, BASELINE_DAYS - days_running)
        return {
            "days_running": days_running,
            "days_remaining": days_remaining,
            "baseline_complete": days_running >= BASELINE_DAYS,
        }


class EnergyBaselineSensor(SensorEntity):
    """Sensor with the learned energy baseline."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Energy Baseline"
        self._attr_unique_id = f"{DOMAIN}_baseline"
        self._attr_unit_of_measurement = "W"
        self._attr_device_class = "power"
        self._attr_icon = "mdi:chart-line"
    
    @property
    def state(self):
        return round(self._agent.baseline_consumption, 2)


class CurrentConsumptionSensor(SensorEntity):
    """Sensor with the current consumption."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Current Consumption"
        self._attr_unique_id = f"{DOMAIN}_current"
        self._attr_unit_of_measurement = "W"
        self._attr_device_class = "power"
    
    @property
    def state(self):
        if len(self._agent.consumption_history) > 0:
            return round(self._agent.consumption_history[-1], 2)
        return 0


class SavingsAccumulatedSensor(SensorEntity):
    """Sensor with the accumulated savings in EUR."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Savings Accumulated"
        self._attr_unique_id = f"{DOMAIN}_savings"
        self._attr_unit_of_measurement = "EUR"
        self._attr_icon = "mdi:currency-eur"
    
    @property
    def state(self):
        # Calculate savings: (baseline - avg_consumption) * kWh_price * hours
        if len(self._agent.consumption_history) < 10:
            return 0
        
        avg_consumption = sum(self._agent.consumption_history) / len(
            self._agent.consumption_history
        )
        saving_watts = self._agent.baseline_consumption - avg_consumption
        
        # Convert to kWh and multiply by price (€0.25/kWh estimated)
        hours = len(self._agent.consumption_history) / 6  # 10min intervals
        saving_kwh = (saving_watts * hours) / 1000
        savings_eur = saving_kwh * 0.25
        
        return round(max(0, savings_eur), 2)


class CO2SavedSensor(SensorEntity):
    """Sensor with the saved CO2 (kg)."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "CO2 Saved"
        self._attr_unique_id = f"{DOMAIN}_co2"
        self._attr_unit_of_measurement = "kg"
        self._attr_icon = "mdi:leaf"
    
    @property
    def state(self):
        # CO2: ~0.5 kg/kWh (mix energy Portugal)
        if len(self._agent.consumption_history) < 10:
            return 0
        
        avg_consumption = sum(self._agent.consumption_history) / len(
            self._agent.consumption_history
        )
        saving_watts = self._agent.baseline_consumption - avg_consumption
        
        hours = len(self._agent.consumption_history) / 6
        saving_kwh = (saving_watts * hours) / 1000
        co2_saved = saving_kwh * 0.5
        
        return round(max(0, co2_saved), 2)


class TasksCompletedSensor(SensorEntity):
    """Sensor with the number of completed tasks (simulated)."""
    
    def __init__(self):
        self._attr_name = "Tasks Completed"
        self._attr_unique_id = f"{DOMAIN}_tasks"
        self._attr_icon = "mdi:check-circle"
        self._tasks = 0
    
    @property
    def state(self):
        return self._tasks
    
    def increment_task(self):
        self._tasks += 1
        self.async_write_ha_state()


class CollaborativeGoalSensor(SensorEntity):
    """Sensor with the collaborative goal progress (% of limit)."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Collaborative Goal Progress"
        self._attr_unique_id = f"{DOMAIN}_collab_goal"
        self._attr_unit_of_measurement = "%"
        self._attr_icon = "mdi:account-group"
    
    @property
    def state(self):
        # Simulation: consumption of group vs. limit (85% of baseline)
        if len(self._agent.consumption_history) == 0:
            return 0
        
        current = self._agent.consumption_history[-1]
        limit = self._agent.baseline_consumption * 0.85  # Meta: -15%
        
        if limit > 0:
            progress = (current / limit) * 100
            return round(min(progress, 100), 1)
        return 0
    
    @property
    def extra_state_attributes(self):
        return {
            "status": "below_target" if self.state < 100 else "above_target",
            "limit_watts": round(self._agent.baseline_consumption * 0.85, 2),
        }


class BehaviourIndexSensor(SensorEntity):
    """Sensor with the agent's behaviour index."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Behaviour Index"
        self._attr_unique_id = f"{DOMAIN}_behaviour"
        self._attr_icon = "mdi:account-check"
    
    @property
    def state(self):
        return round(self._agent.behaviour_index, 2)


class FatigueIndexSensor(SensorEntity):
    """Sensor with the agent's fatigue index."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Fatigue Index"
        self._attr_unique_id = f"{DOMAIN}_fatigue"
        self._attr_icon = "mdi:alert-circle"
    
    @property
    def state(self):
        return round(self._agent.fatigue_index, 2)