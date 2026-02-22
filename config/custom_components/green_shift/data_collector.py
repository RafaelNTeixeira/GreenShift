"""
File: data_collector.py
Description: This module defines the DataCollector class, which is responsible for real-time monitoring and collection of sensor data in the Green Shift Home Assistant component.
The DataCollector continuously listens for state changes in relevant sensors (power, energy, temperature, humidity, illuminance, occupancy) and updates its internal state accordingly. 
It also organizes sensors by Home Assistant areas to provide area-specific insights.
"""

import logging
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from homeassistant.core import HomeAssistant, callback, Event
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_change, async_track_time_interval
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import UPDATE_INTERVAL_SECONDS, GS_UPDATE_SIGNAL, AREA_BASED_SENSORS, ENVIRONMENT_OFFICE
from .helpers import get_normalized_value, get_entity_area, group_sensors_by_area, get_environmental_impact, is_within_working_hours
from .storage import StorageManager

_LOGGER = logging.getLogger(__name__)


class DataCollector:
    """
    Real-time data collection component.
    Continuously monitors sensors and stores readings instantly.
    Completely independent from AI processing.
    """

    def __init__(self, hass: HomeAssistant, discovered_sensors: dict, main_energy_sensor: str = None, main_power_sensor: str = None, storage_manager: StorageManager = None, config_data: dict = None):
        self.hass = hass
        self.sensors = discovered_sensors
        self.main_energy_sensor = main_energy_sensor # Sensor that reads building energy consumption (kWh)
        self.main_power_sensor = main_power_sensor # Sensor that reads current building power consumption (kW)
        self.storage = storage_manager
        self.config_data = config_data or {}

        # Current readings (latest values - global aggregates)
        self.current_total_power = 0.0
        self.current_daily_energy = 0.0
        self.current_temperature = 0.0 # Global average
        self.current_humidity = 0.0 # Global average
        self.current_illuminance = 0.0 # Global average
        self.current_occupancy = False # Any area occupied

        # Instant sensor cache
        self._power_sensor_cache = {} # Stores the current readings of each power sensor (including the main one)
        self._energy_sensor_cache = {} # Stores the current readings of each energy sensor (including the main one)
        self._temperature_sensor_cache = {} # Stores the current readings of each temperature sensor
        self._humidity_sensor_cache = {} # Stores the current readings of each humidity sensor
        self._illuminance_sensor_cache = {} # Stores the current readings of each illuminance sensor
        self._occupancy_sensor_cache = {} # Stores the current readings of each occupancy sensor

        self._energy_midnight_points = {}

        self.area_sensors = {}  # Maps sensor_type -> area_name -> [entity_ids]
        self.area_data = {}  # Maps area_name -> {temperature, humidity, illuminance, occupancy}

    async def _load_persistent_data(self):
        """Load persistent data from JSON storage."""
        if not self.storage:
            return

        state = await self.storage.load_state()

        # Load energy midnight points
        if "energy_midnight_points" in state:
            self._energy_midnight_points = state["energy_midnight_points"]
            _LOGGER.info("Loaded %d midnight energy points from storage", len(self._energy_midnight_points))

        _LOGGER.info("Persistent data loaded successfully")

    async def setup(self):
        """Setup real-time monitoring of all sensors."""

        if self.storage:
            await self._load_persistent_data()

        await self._setup_area_grouping()
        await self._setup_power_monitoring()
        await self._setup_energy_monitoring()
        await self._setup_environment_monitoring()

        async_track_time_change(self.hass, self._reset_midnight_listener, hour=0, minute=0, second=0) # Schedule at midnight daily

        async_track_time_interval(
            self.hass,
            self._record_periodic_snapshot,
            timedelta(seconds=UPDATE_INTERVAL_SECONDS)
        )

        # _LOGGER.info("DataCollector setup complete - real-time monitoring active")

    async def _setup_area_grouping(self):
        """Group environmental sensors by Home Assistant areas."""
        for sensor_type in AREA_BASED_SENSORS:
            eids = self.sensors.get(sensor_type, [])

            to_exclude = {self.main_energy_sensor, self.main_power_sensor} # Exclude main consumption sensor readings from classification by area
            entity_ids = [e for e in eids if e not in to_exclude]

            if not entity_ids:
                continue

            grouped = group_sensors_by_area(self.hass, entity_ids)
            self.area_sensors[sensor_type] = grouped

            _LOGGER.info(
                "Grouped %d %s sensors into %d areas: %s",
                len(entity_ids),
                sensor_type,
                len(grouped),
                list(grouped.keys())
            )

        # Initialize area data structure
        all_areas = set()
        for grouped in self.area_sensors.values():
            all_areas.update(grouped.keys())

        for area in all_areas:
            self.area_data[area] = {
                "temperature": None,
                "humidity": None,
                "illuminance": None,
                "occupancy": False
            }

    async def _setup_power_monitoring(self):
        """Setup instant monitoring for power sensors."""
        power_sensors = self.sensors.get("power", [])
        if not power_sensors:
            _LOGGER.warning("No power sensors found for monitoring")
            return

        @callback
        def handle_power_change(event: Event):
            """
            Handle power sensor state changes instantly.
            
            Args:
                event: The state change event containing entity_id and new_state
            """
            entity_id = event.data.get("entity_id")
            new_state = event.data.get("new_state")

            if new_state is None or entity_id not in power_sensors:
                return

            # Update cache instantly
            try:
                value, _ = get_normalized_value(new_state, "power")

                if value is None:
                    return

                self._power_sensor_cache[entity_id] = value

                # Update area-specific data
                area = get_entity_area(self.hass, entity_id) or "No Area"
                if area in self.area_data:
                    # Sum all power sensors in this area
                    area_sensors = self.area_sensors.get("power", {}).get(area, [])
                    area_total = sum(
                        self._power_sensor_cache.get(eid, 0)
                        for eid in area_sensors
                    )
                    self.area_data[area]["power"] = round(area_total, 2)
                    # _LOGGER.debug("Area '%s' power: %.2f W", area, self.area_data[area]["power"])

                self._recalculate_total_power()

                async_dispatcher_send(self.hass, GS_UPDATE_SIGNAL)
            except (ValueError, TypeError):
                _LOGGER.debug("Invalid power value for %s: %s", entity_id, new_state.state)
                pass

        async_track_state_change_event(self.hass, power_sensors, handle_power_change)
        _LOGGER.info("Real-time power monitoring active for %d sensors", len(power_sensors))

    def _recalculate_total_power(self):
        """Calculates current power consumption."""

        # Case 1: Use main power sensor if available
        if self.main_power_sensor:
            main_val = self._power_sensor_cache.get(self.main_power_sensor)
            if main_val is not None:
                self.current_total_power = main_val
                _LOGGER.debug("Power updated from main sensor: %.2f W", main_val)
                return
        # Case 2: Fallback to summing all individual plugs
        total_power = 0.0
        for entity_id, val in self._power_sensor_cache.items():
            # Avoid double-counting if main_power_sensor is in the cache but currently None
            if self.main_power_sensor and entity_id == self.main_power_sensor:
                continue

            if val is not None:
                total_power += val

        self.current_total_power = total_power
        # _LOGGER.debug("Power recalculated by summing: %.2f W", total_power)

    async def _setup_energy_monitoring(self):
        """Setup instant monitoring for energy sensors."""
        energy_sensors = self.sensors.get("energy", [])
        if not energy_sensors:
            _LOGGER.warning("No energy sensors found for monitoring")
            return

        @callback
        def handle_energy_change(event: Event):
            """
            Handle energy sensor state changes instantly.
            
            Args:
                event: The state change event containing entity_id and new_state
            """
            entity_id = event.data.get("entity_id")
            new_state = event.data.get("new_state")

            if new_state is None or entity_id not in energy_sensors:
                return

            try:
                value, _ = get_normalized_value(new_state, "energy")

                if value is None:
                    return

                self._energy_sensor_cache[entity_id] = value
                # _LOGGER.debug("Updated energy cache for %s: %.3f kWh", entity_id, value)

                if self._energy_midnight_points.get(entity_id) is None: # Initialize midnight point (since the setup will usually happen after midnight)
                    self._energy_midnight_points[entity_id] = value
                    _LOGGER.info("Initialized midnight baseline for %s: %.3f kWh", entity_id, value)

                # Update area-specific data
                area = get_entity_area(self.hass, entity_id) or "No Area"
                if area in self.area_data:
                    # Calculate daily energy for sensors in this area
                    area_sensors = self.area_sensors.get("energy", {}).get(area, [])
                    area_daily = 0.0

                    for eid in area_sensors:
                        current_val = self._energy_sensor_cache.get(eid)
                        midnight_val = self._energy_midnight_points.get(eid)

                        if current_val is not None and midnight_val is not None:
                            if current_val < midnight_val:
                                area_daily += current_val  # Handle reset
                            else:
                                area_daily += (current_val - midnight_val)

                    self.area_data[area]["energy"] = round(area_daily, 3)
                    # _LOGGER.debug("Area '%s' daily energy: %.3f kWh", area, self.area_data[area]["energy"])

                self.get_daily_kwh()

            except (ValueError, TypeError):
                _LOGGER.debug("Invalid energy value for %s: %s", entity_id, new_state.state)
                pass

        async_track_state_change_event(self.hass, energy_sensors, handle_energy_change)
        _LOGGER.info("Real-time energy monitoring active for %d sensors", len(energy_sensors))

    def get_daily_kwh(self):
        """Calculates total kWh consumed today."""

        # Case 1: Use the Main Energy Sensor specifically if provided
        if self.main_energy_sensor:
            current_val = self._energy_sensor_cache.get(self.main_energy_sensor)
            midnight_val = self._energy_midnight_points.get(self.main_energy_sensor)

            if current_val is not None and midnight_val is not None:
                if current_val < midnight_val:
                    # Handle sensor reset
                    self.current_daily_energy = current_val
                else:
                    self.current_daily_energy = current_val - midnight_val

                # _LOGGER.debug("Daily energy from main sensor: %.3f kWh", self.current_daily_energy)
                return

        # Case 2: Fallback to summing all individual sensors
        total_kwh = 0.0
        for entity_id, current_val in self._energy_sensor_cache.items():
            if self.main_energy_sensor and entity_id == self.main_energy_sensor:
                continue

            midnight_val = self._energy_midnight_points.get(entity_id, current_val)

            if current_val < midnight_val:
                total_kwh += current_val # Handle sensor reset (odometer rolled over or reset to 0)
            else:
                total_kwh += (current_val - midnight_val)

        self.current_daily_energy = total_kwh
        # _LOGGER.debug("Total daily kWh calculated: %.3f kWh", total_kwh)

    async def _setup_environment_monitoring(self):
        """Setup instant monitoring for environmental sensors."""
        # Temperature
        temp_sensors = self.sensors.get("temperature", [])
        if temp_sensors:
            @callback
            def handle_temp_change(event: Event):
                """
                Handle temperature sensor state changes instantly.
                
                Args:
                    event: The state change event containing entity_id and new_state
                """
                entity_id = event.data.get("entity_id")
                new_state = event.data.get("new_state")

                if new_state is None or new_state.state in ["unavailable", "unknown"]:
                    return

                try:
                    val = float(new_state.state)
                    self._temperature_sensor_cache[entity_id] = val
                    # _LOGGER.debug("Temperature value: %.2f", val)

                    # Update area-specific data
                    area = get_entity_area(self.hass, entity_id) or "No Area"
                    if area in self.area_data:
                        # Average all temperature sensors in this area
                        area_sensors = self.area_sensors.get("temperature", {}).get(area, [])
                        area_values = [self._temperature_sensor_cache[eid] for eid in area_sensors if eid in self._temperature_sensor_cache]
                        if area_values:
                            self.area_data[area]["temperature"] = round(sum(area_values) / len(area_values), 1)
                            # _LOGGER.debug("Area '%s' temperature: %.1f°C", area, self.area_data[area]["temperature"])

                    # Calculate Average of all valid cache entries
                    if self._temperature_sensor_cache:
                        avg = sum(self._temperature_sensor_cache.values()) / len(self._temperature_sensor_cache)
                        self.current_temperature = round(avg, 1)

                except (ValueError, TypeError):
                    _LOGGER.debug("Invalid temperature value for %s: %s", entity_id, new_state.state)

            async_track_state_change_event(self.hass, temp_sensors, handle_temp_change)
            _LOGGER.info("Real-time temperature monitoring active for %d sensors", len(temp_sensors))

        # Humidity
        hum_sensors = self.sensors.get("humidity", [])
        if hum_sensors:
            @callback
            def handle_hum_change(event: Event):
                """
                Handle humidity sensor state changes instantly.
                
                Args:
                    event: The state change event containing entity_id and new_state
                """
                entity_id = event.data.get("entity_id")
                new_state = event.data.get("new_state")

                if new_state is None or new_state.state in ["unavailable", "unknown"]:
                    return

                try:
                    val = float(new_state.state)
                    self._humidity_sensor_cache[entity_id] = val

                    # _LOGGER.debug("Hum value: %.2f", val)

                    # Update area-specific data
                    area = get_entity_area(self.hass, entity_id) or "No Area"
                    if area in self.area_data:
                        area_sensors = self.area_sensors.get("humidity", {}).get(area, [])
                        area_values = [self._humidity_sensor_cache[eid] for eid in area_sensors if eid in self._humidity_sensor_cache]
                        if area_values:
                            self.area_data[area]["humidity"] = round(sum(area_values) / len(area_values), 1)
                            # _LOGGER.debug("Area '%s' humidity: %.1f%%", area, self.area_data[area]["humidity"])

                    if self._humidity_sensor_cache:
                        avg = sum(self._humidity_sensor_cache.values()) / len(self._humidity_sensor_cache)
                        self.current_humidity = round(avg, 1)

                except (ValueError, TypeError):
                    pass

            async_track_state_change_event(self.hass, hum_sensors, handle_hum_change)
            _LOGGER.info("Real-time humidity monitoring active for %d sensors", len(hum_sensors))

        # Illuminance
        lux_sensors = self.sensors.get("illuminance", [])
        if lux_sensors:
            @callback
            def handle_lux_change(event: Event):
                """
                Handle illuminance sensor state changes instantly.
                
                Args:
                    event: The state change event containing entity_id and new_state
                """
                entity_id = event.data.get("entity_id")
                new_state = event.data.get("new_state")

                if new_state is None or new_state.state in ["unavailable", "unknown"]:
                    return

                try:
                    val = float(new_state.state)
                    self._illuminance_sensor_cache[entity_id] = val

                    # _LOGGER.debug("Illum value: %.2f", val)

                    # Update area-specific data
                    area = get_entity_area(self.hass, entity_id) or "No Area"
                    if area in self.area_data:
                        area_sensors = self.area_sensors.get("illuminance", {}).get(area, [])
                        area_values = [self._illuminance_sensor_cache[eid] for eid in area_sensors if eid in self._illuminance_sensor_cache]
                        if area_values:
                            self.area_data[area]["illuminance"] = round(sum(area_values) / len(area_values), 1)
                            # _LOGGER.debug("Area '%s' illuminance: %.1f lx", area, self.area_data[area]["illuminance"])

                    if self._illuminance_sensor_cache:
                        avg = sum(self._illuminance_sensor_cache.values()) / len(self._illuminance_sensor_cache)
                        self.current_illuminance = round(avg, 1)

                except (ValueError, TypeError):
                    pass

            async_track_state_change_event(self.hass, lux_sensors, handle_lux_change)
            _LOGGER.info("Real-time illuminance monitoring active for %d sensors", len(lux_sensors))

        # Occupancy
        occ_sensors = self.sensors.get("occupancy", [])
        if occ_sensors:
            @callback
            def handle_occ_change(event: Event):
                """
                Handle occupancy sensor state changes instantly.
                
                Args:
                    event: The state change event containing entity_id and new_state
                """
                entity_id = event.data.get("entity_id")
                new_state = event.data.get("new_state")

                if new_state is None or new_state.state in ["unavailable", "unknown"]:
                    return

                try:
                    # Determine boolean state
                    is_on = new_state.state.lower() in ["on", "true", "detected"]
                    self._occupancy_sensor_cache[entity_id] = is_on

                    # _LOGGER.debug("Occupancy value: %s", is_on)

                    # Update area-specific data
                    area = get_entity_area(self.hass, entity_id) or "No Area"
                    if area in self.area_data:
                        area_sensors = self.area_sensors.get("occupancy", {}).get(area, [])
                        # Area is occupied if ANY sensor in the area is True
                        area_occupied = any(self._occupancy_sensor_cache.get(eid, False) for eid in area_sensors)
                        self.area_data[area]["occupancy"] = area_occupied
                        # _LOGGER.debug("Area '%s' occupancy: %s", area, area_occupied)

                    # If ANY sensor in the cache is True, the building is occupied
                    self.current_occupancy = any(self._occupancy_sensor_cache.values())

                    async_dispatcher_send(self.hass, GS_UPDATE_SIGNAL)

                except (ValueError, TypeError):
                    pass

            async_track_state_change_event(self.hass, occ_sensors, handle_occ_change)
            _LOGGER.info("Real-time occupancy monitoring active for %d sensors", len(occ_sensors))

        _LOGGER.info("Real-time environmental monitoring active")

    @callback
    def _reset_midnight_listener(self, now):
        """Callback to reset the daily counter at midnight."""
        self.update_midnight_points()

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
                        val, _ = get_normalized_value(state, "energy")
                        if val is None:
                            continue # To process next sensors inside for loop
                        self._energy_sensor_cache[entity_id] = val
                    except (ValueError, TypeError):
                        _LOGGER.debug("Invalid energy value for %s during midnight reset: %s", entity_id, state.state)
                        continue

            if val is not None:
                self._energy_midnight_points[entity_id] = val

        # Save to persistent storage
        if self.storage:
            self.hass.async_create_task(
                self.storage.update_state_field("energy_midnight_points", self._energy_midnight_points)
            )

        _LOGGER.debug("Midnight snapshots updated: %s", self._energy_midnight_points)

    @callback
    def _record_periodic_snapshot(self, now):
        """
        Records snapshots of all current readings to SQLite.
        
        Args:
            now: The current datetime when the snapshot is taken
        """
        if not self.storage:
            _LOGGER.warning("Storage not available - snapshot not saved")
            return

        # Check if within working hours (for office mode filtering)
        within_working_hours = is_within_working_hours(self.config_data, now)

        # Store global aggregate snapshot
        self.hass.async_create_task(
            self.storage.store_sensor_snapshot(
                timestamp=now,
                power=self.current_total_power,
                energy=self.current_daily_energy,
                temperature=self.current_temperature,
                humidity=self.current_humidity,
                illuminance=self.current_illuminance,
                occupancy=self.current_occupancy,
                within_working_hours=within_working_hours
            )
        )

       # Store area-specific snapshots
        for area_name, data in self.area_data.items():
            self.hass.async_create_task(
                self.storage.store_area_snapshot(
                    timestamp=now,
                    area_name=area_name,
                    power=data.get("power"),
                    energy=data.get("energy"),
                    temperature=data.get("temperature"),
                    humidity=data.get("humidity"),
                    illuminance=data.get("illuminance"),
                    occupancy=data.get("occupancy"),
                    within_working_hours=within_working_hours
                )
            )

        _LOGGER.debug(
            "Snapshot stored: Power=%.2f W | Energy=%.2f kWh | %d areas | Working hours: %s",
            self.current_total_power,
            self.current_daily_energy,
            len(self.area_data),
            within_working_hours
        )

    def get_current_state(self) -> dict:
        """
        Get current sensor readings.
        
        Returns:
            dict:
                - power: Current total power consumption (kW)
                - energy: Current daily energy consumption (kWh)
                - temperature: Current average temperature (°C)
                - humidity: Current average humidity (%)
                - illuminance: Current average illuminance (lx)
                - occupancy: Current occupancy status (True/False)
        """
        return {
            "power": self.current_total_power,
            "energy": self.current_daily_energy,
            "temperature": self.current_temperature,
            "humidity": self.current_humidity,
            "illuminance": self.current_illuminance,
            "occupancy": self.current_occupancy,
        }

    def get_area_state(self, area_name: str) -> dict:
        """
        Get current sensor readings for a specific area.
        
        Args:
            area_name: Name of the area to retrieve data for

        Returns:
            dict:
                - temperature: Current average temperature in the area (°C)
                - humidity: Current average humidity in the area (%)
                - illuminance: Current average illuminance in the area (lx)
                - occupancy: Current occupancy status in the area (True/False)
        """
        return self.area_data.get(area_name, {
            "temperature": None,
            "humidity": None,
            "illuminance": None,
            "occupancy": False
        })

    def get_all_areas(self) -> list:
        """
        Get list of all tracked areas.
        
        Returns:
            list: List of area names
        """
        return list(self.area_data.keys())

    async def get_area_history(self, area_name: str, metric: str, hours: int = None, days: int = None, working_hours_only: bool = None) -> list:
        """
        Get historical data for a specific area.

        Args:
            area_name: Name of the area
            metric: Metric to retrieve ('temperature', 'humidity', 'illuminance', 'occupancy', 'power', 'energy')
            hours: Number of hours to retrieve
            days: Number of days to retrieve
            working_hours_only: Filter to only working hours (True), non-working hours (False), or all (None)

        Returns:
            List of (datetime, value) tuples
        """
        if not self.storage:
            return []
        return await self.storage.get_area_history(area_name, metric, hours=hours, days=days, working_hours_only=working_hours_only)

    async def get_power_history(self, hours: int = None, days: int = None, working_hours_only: bool = None) -> list:
        """
        Get power history with timestamps from SQLite.

        Args:
            hours: Number of hours to retrieve
            days: Number of days to retrieve
            working_hours_only: Filter to only working hours (True), non-working hours (False), or all (None)

        Returns:
            List of (datetime, value) tuples
        """
        if not self.storage:
            return []

        # Returns [(2026-02-04 10:00:00, power), ...]
        return await self.storage.get_history("power", hours=hours, days=days, working_hours_only=working_hours_only)

    async def get_energy_history(self, hours: int = None, days: int = None, working_hours_only: bool = None) -> list:
        """
        Get energy history with timestamps from SQLite.

        Args:
            hours: Number of hours to retrieve
            days: Number of days to retrieve
            working_hours_only: Filter to only working hours (True), non-working hours (False), or all (None)

        Returns:
            List of (datetime, value) tuples
        """
        if not self.storage:
            return []

        # Returns [(2026-02-04 10:00:00, energy), ...]
        return await self.storage.get_history("energy", hours=hours, days=days, working_hours_only=working_hours_only)

    async def get_temperature_history(self, hours: int = None, days: int = None, working_hours_only: bool = None) -> list:
        """
        Get temperature history with timestamps from SQLite.

        Args:
            hours: Number of hours to retrieve
            days: Number of days to retrieve
            working_hours_only: Filter to only working hours (True), non-working hours (False), or all (None)

        Returns:
            List of (datetime, value) tuples
        """
        if not self.storage:
            return []

        # Returns [(2026-02-04 10:00:00, temperature), ...]
        return await self.storage.get_history("temperature", hours=hours, days=days, working_hours_only=working_hours_only)

    async def get_humidity_history(self, hours: int = None, days: int = None, working_hours_only: bool = None) -> list:
        """
        Get humidity history with timestamps from SQLite.

        Args:
            hours: Number of hours to retrieve
            days: Number of days to retrieve
            working_hours_only: Filter to only working hours (True), non-working hours (False), or all (None)

        Returns:
            List of (datetime, value) tuples
        """
        if not self.storage:
            return []

        # Returns [(2026-02-04 10:00:00, humidity), ...]
        return await self.storage.get_history("humidity", hours=hours, days=days, working_hours_only=working_hours_only)

    async def get_illuminance_history(self, hours: int = None, days: int = None, working_hours_only: bool = None) -> list:
        """
        Get illuminance history with timestamps from SQLite.

        Args:
            hours: Number of hours to retrieve
            days: Number of days to retrieve
            working_hours_only: Filter to only working hours (True), non-working hours (False), or all (None)

        Returns:
            List of (datetime, value) tuples
        """
        if not self.storage:
            return []

        # Returns [(2026-02-04 10:00:00, illuminance), ...]
        return await self.storage.get_history("illuminance", hours=hours, days=days, working_hours_only=working_hours_only)

    async def get_occupancy_history(self, hours: int = None, days: int = None, working_hours_only: bool = None) -> list:
        """
        Get occupancy history with timestamps from SQLite.

        Args:
            hours: Number of hours to retrieve
            days: Number of days to retrieve
            working_hours_only: Filter to only working hours (True), non-working hours (False), or all (None)

        Returns:
            List of (datetime, value) tuples
        """
        if not self.storage:
            return []

        # Returns [(2026-02-04 10:00:00, occupancy), ...]
        return await self.storage.get_history("occupancy", hours=hours, days=days, working_hours_only=working_hours_only)

    async def get_all_history(self, hours: int = None, days: int = None) -> dict:
        """
        Get all historical data from SQLite.

        Args:
            hours: Number of hours to retrieve (optional)
            days: Number of days to retrieve (optional)

        Returns:
            Dictionary with all sensor histories
        """
        if not self.storage:
            _LOGGER.warning("Storage not available - returning empty histories")
            return {
                "power": [],
                "energy": [],
                "temperature": [],
                "humidity": [],
                "illuminance": [],
                "occupancy": [],
            }

        return {
            "power": await self.storage.get_history("power", hours=hours, days=days),
            "energy": await self.storage.get_history("energy", hours=hours, days=days),
            "temperature": await self.storage.get_history("temperature", hours=hours, days=days),
            "humidity": await self.storage.get_history("humidity", hours=hours, days=days),
            "illuminance": await self.storage.get_history("illuminance", hours=hours, days=days),
            "occupancy": await self.storage.get_history("occupancy", hours=hours, days=days),
        }

    async def calculate_baseline_summary(self) -> dict:
        """Calculates summary stats for the baseline phase.

        For office mode, only uses working hours data to avoid weekend/off-hours bias.

        Returns:
            dict:
                - avg_daily_kwh: Average daily energy consumption in kWh
                - peak_time: Time interval with highest average power consumption (e.g. "18:00 - 19:00")
                - top_area: Area with highest average power consumption
                - target: Percentage reduction target (e.g. 15 for 15%)
                - impact: Estimated environmental impact of hitting the target (e.g. "X kg CO2 saved annually")
        """
        if not self.storage:
            return {}

        # Determine if we should filter to working hours only (office mode)
        is_office_mode = self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE
        working_hours_filter = True if is_office_mode else None

        if is_office_mode:
            _LOGGER.info("Office mode detected - baseline calculations will use working hours data only")

        # Avg Daily Usage (kWh)
        # Query the total energy history for the last 14 days
        energy_history = await self.get_energy_history(days=14, working_hours_only=working_hours_filter)
        energy_values = [val for ts, val in energy_history]
        avg_daily = np.mean(energy_values) if energy_values else 0.0

        # Peak Time Interval
        # Analyze power history to find which hour of the day has the highest average load
        power_history = await self.get_power_history(days=14, working_hours_only=working_hours_filter)
        peak_time = "Unknown"
        if power_history:
            hourly_buckets = {}
            for ts, val in power_history:
                hour = ts.hour
                hourly_buckets.setdefault(hour, []).append(val)

            if hourly_buckets:
                peak_hour = max(hourly_buckets, key=lambda k: np.mean(hourly_buckets[k]))
                peak_time = f"{peak_hour:02d}:00 - {peak_hour+1:02d}:00"

        # Top Area
        all_areas = await self.storage.get_all_areas()
        top_area = None
        max_area_avg = -1.0
        for area in all_areas:
            if area == "No Area":
                continue
            # Use the existing area history method from StorageManager
            area_stats = await self.storage.get_area_stats(area, "power", days=14, working_hours_only=working_hours_filter)
            if area_stats["mean"] > max_area_avg:
                max_area_avg = area_stats["mean"]
                top_area = area

        # Impact Calculation (hitting the 15% target)
        target_percent = 15
        yearly_savings_kwh = (avg_daily * (target_percent / 100)) * 365
        impact = get_environmental_impact(yearly_savings_kwh)

        return {
            "avg_daily_kwh": round(avg_daily, 2),
            "peak_time": peak_time,
            "top_area": top_area,
            "target": target_percent,
            "impact": impact
        }
