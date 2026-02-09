import logging
from datetime import datetime
from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import DOMAIN, GS_UPDATE_SIGNAL, GS_AI_UPDATE_SIGNAL, BASELINE_DAYS, UPDATE_INTERVAL_SECONDS
from .helpers import get_normalized_value, get_entity_area, get_environmental_impact

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Setup of virtual sensors."""
    agent = hass.data[DOMAIN]["agent"]
    collector = hass.data[DOMAIN]["collector"]
    storage = hass.data[DOMAIN]["storage"]
    discovered_sensors = hass.data[DOMAIN]["discovered_sensors"]
    
    sensors = [
        HardwareSensorsSensor(hass, discovered_sensors, config_entry),
        ResearchPhaseSensor(agent),
        EnergyBaselineSensor(agent),
        CurrentConsumptionSensor(collector),
        CurrentCostConsumptionSensor(hass, collector), 
        DailyCostConsumptionSensor(hass, collector),
        DailyCO2EstimateSensor(hass, collector),
        SavingsAccumulatedSensor(agent, collector),
        CO2SavedSensor(agent, collector),
        TasksCompletedSensor(storage),
        WeeklyChallengeSensor(agent),
        BehaviourIndexSensor(agent),
        FatigueIndexSensor(agent),
        DailyTasksSensor(storage),
    ]

    async_add_entities(sensors)


class GreenShiftBaseSensor(SensorEntity):
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

        # Perform an initial update if the sensor has an async update method
        self._update_callback()

    @callback
    def _update_callback(self):
        """Force the dashboard to update when the signal is received."""
        if hasattr(self, "_async_update_state"):
            # If the sensor defines an async update method (e.g., database call), run it in the background
            self.hass.async_create_task(self._async_update_and_write())
        else:
            # Standard synchronous update
            self.async_write_ha_state()

    async def _async_update_and_write(self):
        """Helper to await the update and then write state."""
        await self._async_update_state()
        self.async_write_ha_state()

class GreenShiftAISensor(SensorEntity):
    """Class to handle AI virtual sensor updates."""
    _attr_should_poll = False 

    async def async_added_to_hass(self):
        """Register the listener when the entity is added to HA."""
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, 
                GS_AI_UPDATE_SIGNAL,
                self._update_callback
            )
        )

        # Perform an initial update if the sensor has an async update method
        self._update_callback()

    @callback
    def _update_callback(self):
        """Force the dashboard to update when the signal is received."""
        if hasattr(self, "_async_update_state"):
            # If the sensor defines an async update method (e.g., database call), run it in the background
            self.hass.async_create_task(self._async_update_and_write())
        else:
            # Standard synchronous update
            self.async_write_ha_state()

    async def _async_update_and_write(self):
        """Helper to await the update and then write state."""
        await self._async_update_state()
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

                area = get_entity_area(self.hass, entity_id) or "No Area"

                data[category].append({
                    "entity_id": entity_id,
                    "name": state.attributes.get("friendly_name", entity_id),
                    "value": val,
                    "unit": unit,
                    "area": area,
                })

        return data
    

class ResearchPhaseSensor(GreenShiftAISensor):
    """Sensor that indicates the current research phase."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Research Phase"
        self._attr_unique_id = f"{DOMAIN}_research_phase"
        self._attr_icon = "mdi:flask"
    
    @property
    def state(self):
        return self._agent.phase
    
    @property
    def extra_state_attributes(self):
        days_running = (datetime.now() - self._agent.start_date).days
        days_remaining = max(0, BASELINE_DAYS - days_running)
        return {
            "days_running": days_running,
            "days_remaining": days_remaining,
            "baseline_complete": days_running >= BASELINE_DAYS,
        }


class EnergyBaselineSensor(GreenShiftAISensor):
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


class SavingsAccumulatedSensor(GreenShiftAISensor):
    """Sensor with the accumulated savings in EUR."""
    
    def __init__(self, agent, collector):
        self._agent = agent
        self._collector = collector
        self._attr_name = "Savings Accumulated"
        self._attr_unique_id = f"{DOMAIN}_savings"
        self._attr_icon = "mdi:cash-check"
        self._attr_native_value = 0

    @property
    def unit_of_measurement(self):
        """Dynamic unit based on input_select.currency."""
        currency_state = self.hass.states.get("input_select.currency")
        return currency_state.state if currency_state else "EUR"
    
    async def _async_update_state(self):
        """Fetch data asynchronously and calculate state."""
        # Await the async database call
        power_history_data = await self._collector.get_power_history()
        power_history = [power for timestamp, power in power_history_data]
        
        if len(power_history) < 10:
            self._attr_native_value = 0
            return
        
        price_state = self.hass.states.get("input_number.electricity_price")
        try:
            price_per_kwh = float(price_state.state) if price_state else 0.25
        except (ValueError, TypeError):
            price_per_kwh = 0.25
        
        avg_consumption = sum(power_history) / len(power_history)
        saving_watts = self._agent.baseline_consumption - avg_consumption
        
        # Calculate hours covered by history based on update interval
        readings_per_hour = 3600 / UPDATE_INTERVAL_SECONDS
        hours = len(power_history) / readings_per_hour

        # Convert W to kW then to kWh
        saving_kwh = (saving_watts / 1000.0) * hours
        savings_total = saving_kwh * price_per_kwh
        
        self._attr_native_value = round(max(0, savings_total), 2)


class CO2SavedSensor(GreenShiftAISensor):
    """Sensor with the saved CO2 (kg)."""
    
    def __init__(self, agent, collector):
        self._agent = agent
        self._collector = collector
        self._attr_name = "CO2 Saved"
        self._attr_unique_id = f"{DOMAIN}_co2"
        self._attr_unit_of_measurement = "kg"
        self._attr_icon = "mdi:leaf"
        self._attr_native_value = 0
        self._attr_extra_state_attributes = {}
    
    async def _async_update_state(self):
        power_history_data = await self._collector.get_power_history()

        power_history = [power for timestamp, power in power_history_data]
        
        if len(power_history) < 10:
            self._attr_native_value = 0
            return
        
        avg_consumption = sum(power_history) / len(power_history)
        saving_watts = self._agent.baseline_consumption - avg_consumption
        
        readings_per_hour = 3600 / UPDATE_INTERVAL_SECONDS
        hours = len(power_history) / readings_per_hour

        saving_kwh = (saving_watts * hours) / 1000

        impact = get_environmental_impact(max(0, saving_kwh))

        self._attr_native_value = impact["co2_kg"]
        
        self._attr_extra_state_attributes = {
            "trees": impact["trees"],
            "flights": impact["flights"],
            "car_km": impact["km"]
        }


class TasksCompletedSensor(GreenShiftAISensor):
    """Sensor with the number of completed tasks."""
    
    def __init__(self, storage):
        self._storage = storage
        self._attr_name = "Tasks Completed"
        self._attr_unique_id = f"{DOMAIN}_tasks"
        self._attr_icon = "mdi:check-circle"
        self._completed_count = 0
    
    @property
    def state(self):
        return self._completed_count

    async def _async_update_state(self):
        """Fetch total completed task count from storage."""
        self._completed_count = await self._storage.get_total_completed_tasks_count()

class WeeklyChallengeSensor(GreenShiftAISensor):
    """Sensor for the weekly energy reduction challenge."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Weekly Challenge"
        self._attr_unique_id = f"{DOMAIN}_weekly_challenge"
        self._attr_icon = "mdi:flag-checkered"
        self._attr_unit_of_measurement = "%"
        self._attr_native_value = 0
        self._attr_extra_state_attributes = {}

    def _get_target_percentage(self):
        """Get the current target percentage from input_number."""
        target_state = self.hass.states.get("input_number.energy_saving_target")
        try:
            return float(target_state.state) if target_state else 15.0
        except (ValueError, TypeError):
            return 15.0
    
    async def _async_update_state(self):
        current_target = self._get_target_percentage()

        challenge = await self._agent.get_weekly_challenge_status(target_percentage=current_target)
        
        self._attr_native_value = challenge.get("progress", 0)

        _LOGGER.debug(f"Weekly Challenge Update: Progress={self._attr_native_value}%, Details={challenge}")
        
        self._attr_extra_state_attributes = {
            "status": challenge.get("status", "pending"),
            "progress": self._attr_native_value,
            "current_avg_w": challenge.get("current_avg", 0),
            "target_avg_w": challenge.get("target_avg", 0),
            "baseline_w": challenge.get("baseline", 0),
            "goal": current_target,
            "week_start": challenge.get("week_start", None),
            "days_in_week": challenge.get("days_in_week", 0),
        }


class BehaviourIndexSensor(GreenShiftAISensor):
    """Sensor with the agent's behaviour index."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Behaviour Index"
        self._attr_unique_id = f"{DOMAIN}_behaviour"
        self._attr_icon = "mdi:account-check"
    
    @property
    def state(self):
        return round(self._agent.behaviour_index, 2)


class FatigueIndexSensor(GreenShiftAISensor):
    """Sensor with the agent's fatigue index."""
    
    def __init__(self, agent):
        self._agent = agent
        self._attr_name = "Fatigue Index"
        self._attr_unique_id = f"{DOMAIN}_fatigue"
        self._attr_icon = "mdi:alert-circle"
    
    @property
    def state(self):
        return round(self._agent.fatigue_index, 2)


class DailyTasksSensor(GreenShiftAISensor):
    """Sensor showing daily tasks with verification status and difficulty feedback."""
    
    _attr_should_poll = False
    
    def __init__(self, storage):
        self._storage = storage
        self._attr_name = "Daily Tasks"
        self._attr_unique_id = f"{DOMAIN}_daily_tasks"
        self._attr_icon = "mdi:clipboard-check-outline"
        self._tasks = []
    
    async def async_added_to_hass(self):
        """Register the listener when the entity is added to HA."""
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, 
                GS_AI_UPDATE_SIGNAL,
                self._update_callback
            )
        )
        # Initial load
        await self._async_update_state()
    
    @callback
    def _update_callback(self):
        """Force update when signal is received."""
        self.hass.async_create_task(self._async_update_and_write())
    
    async def _async_update_and_write(self):
        """Helper to await the update and then write state."""
        await self._async_update_state()
        self.async_write_ha_state()
    
    async def _async_update_state(self):
        """Fetch today's tasks from storage."""
        self._tasks = await self._storage.get_today_tasks()
    
    @property
    def state(self):
        """Return number of tasks."""
        return len(self._tasks)
    
    @property
    def extra_state_attributes(self):
        """Return task details with verification and feedback status."""
        if not self._tasks:
            return {
                "tasks": [],
                "completed_count": 0,
                "verified_count": 0,
            }
        
        completed_count = sum(1 for t in self._tasks if t['completed'])
        verified_count = sum(1 for t in self._tasks if t['verified'])
        
        # Format tasks for display
        tasks_display = []
        for task in self._tasks:
            task_info = {
                'task_id': task['task_id'],
                'title': task['title'],
                'description': task['description'],
                'target_value': task['target_value'],
                'target_unit': task['target_unit'],
                'baseline_value': task['baseline_value'],
                'difficulty_level': task['difficulty_level'],
                'completed': task['completed'],
                'verified': task['verified'],
                'user_feedback': task['user_feedback'],
                'area_name': task['area_name'],
            }
            
            # Add completion value if available
            if task['completion_value']:
                task_info['completion_value'] = task['completion_value']
            
            # Add status indicator
            if task['verified']:
                task_info['status'] = 'verified'
                task_info['status_emoji'] = '✅'
            elif task['completed']:
                task_info['status'] = 'completed'
                task_info['status_emoji'] = '⏳'
            else:
                task_info['status'] = 'pending'
                task_info['status_emoji'] = '🎯'
            
            # Add difficulty indicator
            difficulty_emojis = {1: '⭐', 2: '⭐⭐', 3: '⭐⭐⭐', 4: '⭐⭐⭐⭐', 5: '⭐⭐⭐⭐⭐'}
            task_info['difficulty_display'] = difficulty_emojis.get(task['difficulty_level'], '⭐⭐⭐')
            
            tasks_display.append(task_info)
        
        return {
            "tasks": tasks_display,
            "completed_count": completed_count,
            "verified_count": verified_count,
            "total_count": len(self._tasks),
        }