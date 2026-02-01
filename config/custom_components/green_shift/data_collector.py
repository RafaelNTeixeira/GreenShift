import logging
from datetime import datetime
from collections import deque
from homeassistant.core import HomeAssistant, callback, Event
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_change

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
        self._energy_sensor_cache = {}
        self._energy_midnight_points = {}
        
        # Timestamp tracking
        self._last_history_update = None
        
    async def setup(self):
        """Setup real-time monitoring of all sensors."""
        await self._setup_power_monitoring()
        await self._setup_energy_monitoring()
        await self._setup_environment_monitoring()

        async_track_time_change(self.hass, self._reset_midnight_listener, hour=0, minute=0, second=0) # Schedule at midnight daily

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

    async def _setup_energy_monitoring(self):
        """Setup instant monitoring for energy sensors."""
        energy_sensors = self.sensors.get("energy", [])
        if not energy_sensors:
            _LOGGER.warning("No energy sensors found for monitoring")
            return
        
        @callback
        def handle_energy_change(event: Event):
            entity_id = event.data.get("entity_id") 
            new_state = event.data.get("new_state")
            
            if new_state and entity_id:
                try:
                    value = float(new_state.state)
                    self._energy_sensor_cache[entity_id] = value

                    if self._energy_midnight_points.get(entity_id) is None: # Initialize midnight point (since the setup will usually happen after midnight)
                        self._energy_midnight_points[entity_id] = value
                        _LOGGER.info("Initialized midnight baseline for %s: %.3f kWh", entity_id, value)
                    
                    _LOGGER.debug("Updated energy cache for %s: %.2f kWh", entity_id, value)
                    
                except (ValueError, TypeError):
                    pass
        
        async_track_state_change_event(self.hass, energy_sensors, handle_energy_change)
        _LOGGER.info("Real-time energy monitoring active for %d sensors", len(energy_sensors))
    
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
            _LOGGER.info("Real-time temperature monitoring active for %d sensors", len(temp_sensors))
        
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
            _LOGGER.info("Real-time humidity monitoring active for %d sensors", len(hum_sensors))
        
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
            _LOGGER.info("Real-time illuminance monitoring active for %d sensors", len(lux_sensors))
        
        # Occupancy
        occ_sensors = self.sensors.get("occupancy", [])
        if occ_sensors:
            @callback
            def handle_occ_change(event: Event):
                new_state = event.data.get("new_state")
                if new_state:
                    self.current_occupancy = new_state.state.lower() in ["on", "true", "detected"]
            
            async_track_state_change_event(self.hass, occ_sensors, handle_occ_change)
            _LOGGER.info("Real-time occupancy monitoring active for %d sensors", len(occ_sensors))
        
        _LOGGER.info("Real-time environmental monitoring active")

    @callback
    def _reset_midnight_listener(self, now):
        """Callback to reset the daily counter at midnight."""
        self.update_midnight_points()
        _LOGGER.info("Daily Cost Reset: Midnight snapshots taken.")

    @callback
    def update_midnight_points(self):
        """Snapshots current energy readings to establish the daily baseline."""
        energy_sensors = self.sensors.get("energy", [])
        
        for entity_id in energy_sensors:
            # Try cache first, then state machine
            val = self._energy_sensor_cache.get(entity_id)
            if val is None:
                state = self.hass.states.get(entity_id)
                if state and state.state not in ["unknown", "unavailable"]:
                    try:
                        val = float(state.state)
                        self._energy_sensor_cache[entity_id] = val
                    except (ValueError, TypeError):
                        continue
            
            if val is not None:
                self._energy_midnight_points[entity_id] = val
        
        _LOGGER.debug("Midnight snapshots updated: %s", self._energy_midnight_points)
    
    def _update_total_power(self):
        """Calculate total power from cached sensor values and update history."""
        total = sum(self._power_sensor_cache.values()) # TODO: Might have a single sensor that measures total power directly.
        self.current_total_power = total
        
        # Add to history every UPDATE_INTERVAL_SECONDS to avoid too many entries
        now = datetime.now()
        if self._last_history_update is None or \
           (now - self._last_history_update).total_seconds() >= UPDATE_INTERVAL_SECONDS:
            self.consumption_history.append((now, total))
            self.temperature_history.append((now, self.current_temperature))
            self.humidity_history.append((now, self.current_humidity))
            self.illuminance_history.append((now, self.current_illuminance))
            self.occupancy_history.append((now, 1.0 if self.current_occupancy else 0.0))
            self._last_history_update = now
            _LOGGER.debug("Data recorded - Power: %.2f kW, Temp: %.1f°C, Hum: %.1f%%, Lux: %.0f lx, Occ: %s",
                         total, self.current_temperature, self.current_humidity, 
                         self.current_illuminance, self.current_occupancy)
            
    # TODO: Might have a sensor that measures total energy directly.
    def get_daily_kwh(self) -> float:
        """
        Calculates total kWh consumed today.
        Self-healing: If midnight/startup baseline is missing, captures it immediately.
        """
        total_daily_kwh = 0.0
        energy_sensors = self.sensors.get("energy", [])
        
        for entity_id in energy_sensors:
            midnight_val = self._energy_midnight_points.get(entity_id)
            current_val = self._energy_sensor_cache.get(entity_id)
            
            # If cache is empty, read state directly
            if current_val is None:
                state = self.hass.states.get(entity_id)
                if state and state.state not in ["unknown", "unavailable"]:
                    try:
                        current_val = float(state.state)
                        # Update cache for next time
                        self._energy_sensor_cache[entity_id] = current_val
                    except ValueError:
                        continue
            
            # Proceed only if we have a valid current reading
            if current_val is not None:
                
                # If we missed the startup snapshot (sensor was unavailable), use the current value as the baseline.
                if midnight_val is None:
                    self._energy_midnight_points[entity_id] = current_val
                    midnight_val = current_val
                    _LOGGER.info("Late initialization: Set baseline for %s to %.3f", entity_id, midnight_val)

                if current_val < midnight_val:
                    # Sensor reset case (e.g., daily smart plug reset)
                    total_daily_kwh += current_val
                else:
                    # Standard odometer case
                    total_daily_kwh += (current_val - midnight_val)
        
        _LOGGER.debug("Total daily kWh calculated: %.3f kWh", total_daily_kwh)
        return total_daily_kwh
    
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
        """Returns only the power values (floats)."""
        # Return only the power values without timestamps
        return [v for _, v in self.consumption_history]
    
    def get_consumption_history_with_timestamps(self) -> list:
        """Get consumption history with timestamps."""
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