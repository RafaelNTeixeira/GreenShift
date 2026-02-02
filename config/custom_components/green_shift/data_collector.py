import logging
from collections import deque
from datetime import timedelta
from homeassistant.core import HomeAssistant, callback, Event
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_change, async_track_time_interval

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
        
        self.power_history = deque(maxlen=max_readings) # TODO: Store in SQLite database
        self.energy_history = deque(maxlen=max_readings) # TODO: Store in SQLite database
        self.temperature_history = deque(maxlen=max_readings) # TODO: Store in SQLite database
        self.humidity_history = deque(maxlen=max_readings) # TODO: Store in SQLite database
        self.illuminance_history = deque(maxlen=max_readings) # TODO: Store in SQLite database
        self.occupancy_history = deque(maxlen=max_readings) # TODO: Store in SQLite database
        
        # Current readings (latest values)
        self.current_total_power = 0.0
        self.current_daily_energy = 0.0
        self.current_temperature = 0.0
        self.current_humidity = 0.0
        self.current_illuminance = 0.0
        self.current_occupancy = False
        
        # Instant sensor cache
        self._power_sensor_cache = {}
        self._energy_sensor_cache = {}
        self._energy_midnight_points = {} # TODO: Store in persistent storage JSON
        
        # Timestamp tracking
        self._last_history_update = None
        
    async def setup(self):
        """Setup real-time monitoring of all sensors."""
        await self._setup_power_monitoring()
        await self._setup_energy_monitoring()
        await self._setup_environment_monitoring()

        async_track_time_change(self.hass, self._reset_midnight_listener, hour=0, minute=0, second=0) # Schedule at midnight daily

        async_track_time_interval(
            self.hass, 
            self._record_periodic_snapshot, 
            timedelta(seconds=UPDATE_INTERVAL_SECONDS)
        )

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
                _LOGGER.debug("Updated power cache for %s: %.2f W", entity_id, value)

                self._recalculate_total_power()
                _LOGGER.debug("Total power recalculated: %.2f W", self.current_total_power)
            except (ValueError, TypeError):
                _LOGGER.debug("Invalid power value for %s: %s", entity_id, new_state.state)
                pass
        
        async_track_state_change_event(self.hass, power_sensors, handle_power_change)
        _LOGGER.info("Real-time power monitoring active for %d sensors", len(power_sensors))

    def _recalculate_total_power(self):
        """Sum all power sensors for real-time display."""
        # Logic to sum cache or use main sensor
        total_power = sum(self._power_sensor_cache.values()) # TODO: Might have a sensor that measures total power directly.
        self.current_total_power = total_power

    async def _setup_energy_monitoring(self):
        """Setup instant monitoring for energy sensors."""
        energy_sensors = self.sensors.get("energy", [])
        if not energy_sensors:
            _LOGGER.warning("No energy sensors found for monitoring")
            return
        
        @callback
        def handle_energy_change(event: Event):
            """Handle energy sensor state changes instantly."""
            entity_id = event.data.get("entity_id") 
            new_state = event.data.get("new_state")
            
            if new_state is None or entity_id not in energy_sensors:
                return
            
            try:
                value = float(new_state.state)
                self._energy_sensor_cache[entity_id] = value
                _LOGGER.debug("Updated energy cache for %s: %.3f kWh", entity_id, value)

                if self._energy_midnight_points.get(entity_id) is None: # Initialize midnight point (since the setup will usually happen after midnight)
                    self._energy_midnight_points[entity_id] = value
                    _LOGGER.info("Initialized midnight baseline for %s: %.3f kWh", entity_id, value)

                self.get_daily_kwh()
                
            except (ValueError, TypeError):
                _LOGGER.debug("Invalid energy value for %s: %s", entity_id, new_state.state)
                pass
        
        async_track_state_change_event(self.hass, energy_sensors, handle_energy_change)
        _LOGGER.info("Real-time energy monitoring active for %d sensors", len(energy_sensors))

    # TODO: Might have a sensor that measures total energy directly.
    def get_daily_kwh(self):
        """
        Calculates total kWh consumed today.
        Sum up the differences based on cached values.
        """
        total_kwh = 0.0
        for entity_id, current_val in self._energy_sensor_cache.items():
            midnight_val = self._energy_midnight_points.get(entity_id, current_val)
            
            if current_val < midnight_val:
                # Handle sensor reset (odometer rolled over or reset to 0)
                total_kwh += current_val
            else:
                total_kwh += (current_val - midnight_val)
                
        _LOGGER.debug("Total daily kWh calculated: %.3f kWh", total_kwh)
        self.current_daily_energy = total_kwh

    
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
                        _LOGGER("Invalid temperature value: %s", new_state.state)
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
                        _LOGGER("Invalid humidity value: %s", new_state.state)
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
                        _LOGGER.debug("Invalid illuminance value: %s", new_state.state)
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
                    try:
                        self.current_occupancy = new_state.state.lower() in ["on", "true", "detected"]
                    except (ValueError, TypeError):
                        _LOGGER.debug("Invalid occupancy value: %s", new_state.state)
                        pass
            
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
                        _LOGGER.debug("Invalid energy value for %s during midnight reset: %s", entity_id, state.state)
                        continue
            
            if val is not None:
                self._energy_midnight_points[entity_id] = val
        
        _LOGGER.debug("Midnight snapshots updated: %s", self._energy_midnight_points)
    
    @callback
    def _record_periodic_snapshot(self, now):
        """Records a snapshot of all current readings."""
        self.power_history.append((now, self.current_total_power))
        self.energy_history.append((now, self.current_daily_energy))
        self.temperature_history.append((now, self.current_temperature))
        self.humidity_history.append((now, self.current_humidity))
        self.illuminance_history.append((now, self.current_illuminance))
        self.occupancy_history.append((now, 1.0 if self.current_occupancy else 0.0))
        
        _LOGGER.debug("History Snapshot Recorded: Power=%.2f kW | Energy=%.2f kWh", 
                      self.current_total_power, self.current_daily_energy)
    
    def get_current_state(self) -> dict:
        """Get current sensor readings."""
        return {
            "power": self.current_total_power,
            "energy": self.current_daily_energy,
            "temperature": self.current_temperature,
            "humidity": self.current_humidity,
            "illuminance": self.current_illuminance,
            "occupancy": self.current_occupancy,
        }
    
    def get_power_history(self) -> list:
        """Returns only the power values (floats)."""
        # Return only the power values without timestamps
        return [v for _, v in self.power_history]
    
    def get_power_history_with_timestamps(self) -> list:
        """Get consumption history with timestamps."""
        return list(self.power_history)
    
    def get_all_history(self) -> dict:
        """Get all historical data."""
        return {
            "consumption": list(self.power_history),
            "temperature": list(self.temperature_history),
            "humidity": list(self.humidity_history),
            "illuminance": list(self.illuminance_history),
            "occupancy": list(self.occupancy_history),
        }