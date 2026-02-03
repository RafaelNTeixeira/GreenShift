import logging
from datetime import datetime
from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import DOMAIN, GS_UPDATE_SIGNAL, BASELINE_DAYS, UPDATE_INTERVAL_SECONDS
from .helpers import get_normalized_value

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Setup of virtual sensors."""
    agent = hass.data[DOMAIN]["agent"]
    collector = hass.data[DOMAIN]["collector"]
    start_date = hass.data[DOMAIN]["start_date"]
    discovered_sensors = hass.data[DOMAIN]["discovered_sensors"]
    
    sensors = [
        HardwareSensorsSensor(hass, discovered_sensors, config_entry),
        ResearchPhaseSensor(agent, start_date),
        EnergyBaselineSensor(agent),
        CurrentConsumptionSensor(collector),
        CurrentCostConsumptionSensor(hass, collector), 
        DailyCostConsumptionSensor(hass, collector),
        DailyCO2EstimateSensor(hass, collector),
        SavingsAccumulatedSensor(agent, collector),
        CO2SavedSensor(agent, collector),
        TasksCompletedSensor(agent),
        DailyTasksSensor(agent),
        WeeklyChallengeSensor(agent),
        CollaborativeGoalSensor(agent, collector),
        BehaviourIndexSensor(agent),
        FatigueIndexSensor(agent),
    ]

    async_add_entities(sensors)


class GreenShiftBaseSensor(SensorEntity): # TODO: Maybe create one for the AI virtual sensors
    """Base class to handle updates."""
    _attr_should_poll = False 

    async def async_added_to_hass(self):
        """Register the listener when the entity is added to HA."""
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, 
                GS_UPDATE_SIGNAL,
                self._update_callback
            )
        )

    @callback
    def _update_callback(self):
        """Force the dashboard to update when the signal is received."""
        self.async_write_ha_state()

class HardwareSensorsSensor(GreenShiftBaseSensor):
    """Aggregates hardware sensors by category with live values."""

    def __init__(self, hass, discovered, config_entry):
        self.hass = hass
        self._discovered = discovered
        self.main_energy_sensor = config_entry.data.get("main_total_energy_sensor")
        self.main_power_sensor = config_entry.data.get("main_total_power_sensor")
        self._attr_name = "Hardware Sensors"
        self._attr_unique_id = f"{DOMAIN}_hardware_sensors"
        self._attr_icon = "mdi:database"

    @property
    def state(self):
        return "ok"

    @property
    def extra_state_attributes(self):
        data = {}
        to_exclude = {self.main_energy_sensor, self.main_power_sensor}

        for category, entities in self._discovered.items():
            data[category] = []

            for entity_id in entities:
                if entity_id in to_exclude:
                    continue
                
                state = self.hass.states.get(entity_id)
                if not state:
                    continue

                if category == "occupancy":
                    # Binary sensors don't have units and don't need float conversion
                    val = state.state
                    unit = None
                else:
                    # Numeric sensors (Power, Energy, Temp, Hum, Lux)
                    val, unit = get_normalized_value(state, category)

                if val is None:
                    continue

                data[category].append({
                    "entity_id": entity_id,
                    "name": state.attributes.get("friendly_name", entity_id),
                    "value": val,
                    "unit": unit,
                })

        return data
    

class ResearchPhaseSensor(GreenShiftBaseSensor):
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


class EnergyBaselineSensor(GreenShiftBaseSensor):
    """Sensor with the learned energy baseline from baseline phase."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Energy Baseline"
        self._attr_unique_id = f"{DOMAIN}_baseline"
        self._attr_unit_of_measurement = "kW"
        self._attr_device_class = "power"
        self._attr_icon = "mdi:chart-line"
    
    @property
    def state(self):
        # Show baseline_consumption (immutable baseline from intervention phase)
        return round(self._agent.baseline_consumption, 2)


class CurrentConsumptionSensor(GreenShiftBaseSensor):
    """Sensor with the current consumption from DataCollector."""
    
    def __init__(self, collector):
        self._collector = collector
        self._attr_name = "Current Consumption"
        self._attr_unique_id = f"{DOMAIN}_current"
        self._attr_unit_of_measurement = "W"
        self._attr_device_class = "power"
    
    @property
    def state(self):
        return round(self._collector.current_total_power, 3)
    

class CurrentCostConsumptionSensor(GreenShiftBaseSensor):
    """Sensor that calculates the current cost per hour based on consumption from DataCollector."""

    def __init__(self, hass, collector):
        self.hass = hass
        self._collector = collector
        self._attr_name = "Current Hourly Cost"
        self._attr_unique_id = f"{DOMAIN}_current_cost"
        self._attr_unit_of_measurement = "EUR/h"
        self._attr_icon = "mdi:cash-clock"

    @property
    def unit_of_measurement(self):
        """Dynamic unit based on input_select."""
        currency_state = self.hass.states.get("input_select.currency")
        
        return f"{currency_state.state}/h" if currency_state else "EUR/h" # Default to EUR/h if the input_select is missing

    @property
    def state(self):
        # Get electricity price from input_number (default to 0.25 if unavailable)
        price_state = self.hass.states.get("input_number.electricity_price")
        try:
            price_per_kwh = float(price_state.state) if price_state else 0.25
        except (ValueError, TypeError):
            price_per_kwh = 0.25

        power_kw = self._collector.current_total_power / 1000.0
        cost_hourly = power_kw * price_per_kwh
        
        return round(cost_hourly, 3)

    @property
    def extra_state_attributes(self):
        price_state = self.hass.states.get("input_number.electricity_price")
        currency_state = self.hass.states.get("input_select.currency")
        return {
            "current_load": round(self._collector.current_total_power, 3),
            "applied_price_per_kwh": price_state.state if price_state else "0.25 (default)",
            "currency": currency_state.state if currency_state else "EUR"
        }


class DailyCostConsumptionSensor(GreenShiftBaseSensor):
    """Sensor that calculates the daily cost based on consumption from DataCollector."""

    def __init__(self, hass, collector):
        self.hass = hass
        self._collector = collector
        self._attr_name = "Daily Cost"
        self._attr_unique_id = f"{DOMAIN}_daily_cost"
        self._attr_unit_of_measurement = "EUR"
        self._attr_icon = "mdi:cash-multiple"

    @property
    def unit_of_measurement(self):
        """Dynamic unit based on input_select."""
        currency_state = self.hass.states.get("input_select.currency")
        
        return f"{currency_state.state}" if currency_state else "EUR" # Default to EUR if the input_select is missing

    @property
    def state(self):
        # Price Logic
        price_state = self.hass.states.get("input_number.electricity_price")
        try:
            price_per_kwh = float(price_state.state) if price_state else 0.25
        except (ValueError, TypeError):
            price_per_kwh = 0.25

        # Get accurate daily kWh from the Odometer logic
        daily_kwh = self._collector.current_daily_energy
        
        # Calculate Cost
        cost = daily_kwh * price_per_kwh
        
        return round(cost, 2)

    @property
    def extra_state_attributes(self):
        price_state = self.hass.states.get("input_number.electricity_price")
        currency_state = self.hass.states.get("input_select.currency")
        return {
            "daily_kwh_accumulated": round(self._collector.current_daily_energy, 3),
            "applied_price": price_state.state if price_state else "0.25",
            "currency": currency_state.state if currency_state else "EUR"
        }
    

class DailyCO2EstimateSensor(GreenShiftBaseSensor):
    """Sensor that estimates daily CO2 emissions based on consumption from DataCollector."""

    def __init__(self, hass, collector):
        self.hass = hass
        self._collector = collector
        self._attr_name = "Daily CO2 Estimate"
        self._attr_unique_id = f"{DOMAIN}_daily_co2"
        self._attr_unit_of_measurement = "kg"
        self._attr_icon = "mdi:leaf-circle"

    @property
    def state(self):
        co2_factor_portugal = 0.097 # kg/kWh as of early 2026

        # Get accurate daily kWh from the Odometer logic
        daily_kwh = self._collector.current_daily_energy
        
        # Calculate CO2 Emissions (daily_kwh * kg/kWh)
        co2_emissions = daily_kwh * co2_factor_portugal
        
        return round(co2_emissions, 2)

    @property
    def extra_state_attributes(self):
        return {
            "daily_kwh_accumulated": round(self._collector.current_daily_energy, 3),
            "co2_factor": 0.097,
        }


class SavingsAccumulatedSensor(GreenShiftBaseSensor):
    """Sensor with the accumulated savings in EUR."""
    
    def __init__(self, agent, collector):
        self._agent = agent
        self._collector = collector
        self._attr_name = "Savings Accumulated"
        self._attr_unique_id = f"{DOMAIN}_savings"
        self._attr_unit_of_measurement = "EUR"
        self._attr_icon = "mdi:currency-eur"
    
    @property
    def state(self):
        # Calculate savings: (baseline - avg_consumption) * kWh_price * hours
        power_history = self._collector.get_power_history()
        if len(power_history) < 10:
            return 0
        
        avg_consumption = sum(power_history) / len(power_history)
        saving_kW = self._agent.baseline_consumption - avg_consumption
        
        # Convert to kWh and multiply by price (€0.25/kWh estimated)
        # 15-second intervals: 240 readings per hour
        seconds_in_an_hour = 3600
        hours = len(power_history) / (seconds_in_an_hour / UPDATE_INTERVAL_SECONDS)

        saving_kwh = (saving_kW * hours)
        savings_eur = saving_kwh * 0.25
        
        return round(max(0, savings_eur), 2)


class CO2SavedSensor(GreenShiftBaseSensor):
    """Sensor with the saved CO2 (kg)."""
    
    def __init__(self, agent, collector):
        self._agent = agent
        self._collector = collector
        self._attr_name = "CO2 Saved"
        self._attr_unique_id = f"{DOMAIN}_co2"
        self._attr_unit_of_measurement = "kg"
        self._attr_icon = "mdi:leaf"
    
    @property
    def state(self):
        # CO2: ~0.5 kg/kWh (mix energy Portugal)
        power_history = self._collector.get_power_history()
        if len(power_history) < 10:
            return 0
        
        avg_consumption = sum(power_history) / len(power_history)
        saving_watts = self._agent.baseline_consumption - avg_consumption
        
        # 15-second intervals: 240 readings per hour
        readings_per_hour = 3600 / UPDATE_INTERVAL_SECONDS
        hours = len(power_history) / readings_per_hour
        saving_kwh = (saving_watts * hours) / 1000
        co2_saved = saving_kwh * 0.5
        
        return round(max(0, co2_saved), 2)


class TasksCompletedSensor(GreenShiftBaseSensor):
    """Sensor with the number of completed tasks."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Tasks Completed"
        self._attr_unique_id = f"{DOMAIN}_tasks"
        self._attr_icon = "mdi:check-circle"
    
    @property
    def state(self):
        return self._agent.tasks_completed_count


class DailyTasksSensor(GreenShiftBaseSensor):
    """Sensor with today's random daily tasks."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Daily Tasks"
        self._attr_unique_id = f"{DOMAIN}_daily_tasks"
        self._attr_icon = "mdi:clipboard-list"
    
    @property
    def state(self):
        return len(self._agent.daily_tasks)
    
    @property
    def extra_state_attributes(self):
        return {
            "tasks": self._agent.daily_tasks,
        }


class WeeklyChallengeSensor(GreenShiftBaseSensor):
    """Sensor for the weekly energy reduction challenge."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Weekly Challenge"
        self._attr_unique_id = f"{DOMAIN}_weekly_challenge"
        self._attr_icon = "mdi:flag-checkered"
        self._attr_unit_of_measurement = "%"
    
    @property
    def state(self):
        challenge = self._agent.get_weekly_challenge_status()
        return challenge.get("progress", 0)
    
    @property
    def extra_state_attributes(self):
        challenge = self._agent.get_weekly_challenge_status()
        return {
            "status": challenge.get("status", "pending"),
            "current_avg_w": challenge.get("current_avg", 0),
            "target_avg_w": challenge.get("target_avg", 0),
            "baseline_w": challenge.get("baseline", 0),
            "goal": "Reduce consumption to 85% of baseline",
        }


class CollaborativeGoalSensor(GreenShiftBaseSensor):
    """Sensor with the collaborative goal progress (% of limit)."""
    
    def __init__(self, agent, collector):
        self._agent = agent
        self._collector = collector
        self._attr_name = "Collaborative Goal Progress"
        self._attr_unique_id = f"{DOMAIN}_collab_goal"
        self._attr_unit_of_measurement = "%"
        self._attr_icon = "mdi:account-group"
    
    @property
    def state(self):
        # Simulation: consumption of group vs. limit (85% of baseline)
        current = self._collector.current_total_power
        if current == 0:
            return 0
        
        # Use fixed baseline for consistent comparison during active phase
        baseline = self._agent.baseline_consumption_week or self._agent.baseline_consumption
        limit = baseline * 0.85  # Meta: -15%
        
        if limit > 0:
            progress = (current / limit) * 100
            return round(min(progress, 100), 1)
        return 0
    
    @property
    def extra_state_attributes(self):
        baseline = self._agent.baseline_consumption_week or self._agent.baseline_consumption
        return {
            "status": "below_target" if self.state < 100 else "above_target",
            "limit_watts": round(baseline * 0.85, 2),
        }


class BehaviourIndexSensor(GreenShiftBaseSensor):
    """Sensor with the agent's behaviour index."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Behaviour Index"
        self._attr_unique_id = f"{DOMAIN}_behaviour"
        self._attr_icon = "mdi:account-check"
    
    @property
    def state(self):
        return round(self._agent.behaviour_index, 2)


class FatigueIndexSensor(GreenShiftBaseSensor):
    """Sensor with the agent's fatigue index."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Fatigue Index"
        self._attr_unique_id = f"{DOMAIN}_fatigue"
        self._attr_icon = "mdi:alert-circle"
    
    @property
    def state(self):
        return round(self._agent.fatigue_index, 2)
