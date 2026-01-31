import logging
from datetime import datetime
from collections import deque
from homeassistant.core import HomeAssistant, callback, Event
from homeassistant.helpers.event import async_track_state_change_event

from .const import UPDATE_INTERVAL_SECONDS

_LOGGER = logging.getLogger(__name__)


class DataCollector:
    """
    Real-time data collection component.
    Continuously monitors sensors and stores readings instantly.
    Completely independent from AI processing.
    """
    
    def __init__(self, hass: HomeAssistant, discovered_sensors: dict):
        self.hass = hass
        self.sensors = discovered_sensors
        
        # Storage for historical data (14 days)
        days_to_store = 14
        day_in_seconds = 86400
        max_readings = int(days_to_store * day_in_seconds / UPDATE_INTERVAL_SECONDS)
        
        self.consumption_history = deque(maxlen=max_readings)
        self.temperature_history = deque(maxlen=max_readings)
        self.humidity_history = deque(maxlen=max_readings)
        self.illuminance_history = deque(maxlen=max_readings)
        self.occupancy_history = deque(maxlen=max_readings)
        
        # Current readings (latest values)
        self.current_total_power = 0.0
        self.current_temperature = 0.0
        self.current_humidity = 0.0
        self.current_illuminance = 0.0
        self.current_occupancy = False
        
        # Instant sensor cache
        self._power_sensor_cache = {}
        
        # Timestamp tracking
        self._last_history_update = None
        
    async def setup(self):
        """Setup real-time monitoring of all sensors."""
        await self._setup_power_monitoring()
        await self._setup_environment_monitoring()
        _LOGGER.info("DataCollector setup complete - real-time monitoring active")
    
    async def _setup_power_monitoring(self):
        """Setup instant monitoring for power sensors."""
        power_sensors = self.sensors.get("power", [])
        if not power_sensors:
            _LOGGER.warning("No power sensors found for monitoring")
            return
        
        @callback
        def handle_power_change(event: Event):
            """Handle power sensor state changes instantly."""
            entity_id = event.data.get("entity_id")
            new_state = event.data.get("new_state")
            
            if new_state is None or entity_id not in power_sensors:
                return
            
            # Update cache instantly
            try:
                value = float(new_state.state)
                self._power_sensor_cache[entity_id] = value
                self._update_total_power()
            except (ValueError, TypeError):
                _LOGGER.debug("Invalid power value for %s: %s", entity_id, new_state.state)
        
        async_track_state_change_event(self.hass, power_sensors, handle_power_change)
        _LOGGER.info("Real-time power monitoring active for %d sensors", len(power_sensors))
    
    async def _setup_environment_monitoring(self):
        """Setup instant monitoring for environmental sensors."""
        
        # Temperature
        temp_sensors = self.sensors.get("temperature", [])
        if temp_sensors:
            @callback
            def handle_temp_change(event: Event):
                new_state = event.data.get("new_state")
                if new_state:
                    try:
                        self.current_temperature = float(new_state.state)
                    except (ValueError, TypeError):
                        pass
            
            async_track_state_change_event(self.hass, temp_sensors, handle_temp_change)
        
        # Humidity
        hum_sensors = self.sensors.get("humidity", [])
        if hum_sensors:
            @callback
            def handle_hum_change(event: Event):
                new_state = event.data.get("new_state")
                if new_state:
                    try:
                        self.current_humidity = float(new_state.state)
                    except (ValueError, TypeError):
                        pass
            
            async_track_state_change_event(self.hass, hum_sensors, handle_hum_change)
        
        # Illuminance
        lux_sensors = self.sensors.get("illuminance", [])
        if lux_sensors:
            @callback
            def handle_lux_change(event: Event):
                new_state = event.data.get("new_state")
                if new_state:
                    try:
                        self.current_illuminance = float(new_state.state)
                    except (ValueError, TypeError):
                        pass
            
            async_track_state_change_event(self.hass, lux_sensors, handle_lux_change)
        
        # Occupancy
        occ_sensors = self.sensors.get("occupancy", [])
        if occ_sensors:
            @callback
            def handle_occ_change(event: Event):
                new_state = event.data.get("new_state")
                if new_state:
                    self.current_occupancy = new_state.state.lower() in ["on", "true", "detected"]
            
            async_track_state_change_event(self.hass, occ_sensors, handle_occ_change)
        
        _LOGGER.info("Real-time environmental monitoring active")
    
    def _update_total_power(self):
        """Calculate total power from cached sensor values and update history."""
        total = sum(self._power_sensor_cache.values())
        self.current_total_power = total
        
        # Add to history every UPDATE_INTERVAL_SECONDS to avoid too many entries
        now = datetime.now()
        if self._last_history_update is None or \
           (now - self._last_history_update).total_seconds() >= UPDATE_INTERVAL_SECONDS:
            self.consumption_history.append(total)
            self.temperature_history.append(self.current_temperature)
            self.humidity_history.append(self.current_humidity)
            self.illuminance_history.append(self.current_illuminance)
            self.occupancy_history.append(1.0 if self.current_occupancy else 0.0)
            self._last_history_update = now
            _LOGGER.debug("Data recorded - Power: %.2f kW, Temp: %.1f°C, Hum: %.1f%%, Lux: %.0f lx, Occ: %s",
                         total, self.current_temperature, self.current_humidity, 
                         self.current_illuminance, self.current_occupancy)
    
    def get_current_state(self) -> dict:
        """Get current sensor readings."""
        return {
            "power": self.current_total_power,
            "temperature": self.current_temperature,
            "humidity": self.current_humidity,
            "illuminance": self.current_illuminance,
            "occupancy": self.current_occupancy,
        }
    
    def get_consumption_history(self) -> list:
        """Get consumption history."""
        return list(self.consumption_history)
    
    def get_all_history(self) -> dict:
        """Get all historical data."""
        return {
            "consumption": list(self.consumption_history),
            "temperature": list(self.temperature_history),
            "humidity": list(self.humidity_history),
            "illuminance": list(self.illuminance_history),
            "occupancy": list(self.occupancy_history),
        }