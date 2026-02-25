"""
File: const.py
Description: This module defines constants used throughout the Green Shift Home Assistant component.
It includes domain definitions, signal names, environment modes, working hours defaults, system phases, update intervals, backup settings, data retention settings, sensor categories for auto-discovery, RL action spaces, reward function weights, notification settings and shadow learning parameters. 
These constants help maintain consistency across the component and make it easier to manage configuration and behavior in a centralized manner.
"""

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.components.binary_sensor import BinarySensorDeviceClass

DOMAIN = "green_shift"
GS_UPDATE_SIGNAL = "green_shift_update"
GS_AI_UPDATE_SIGNAL = "green_shift_ai_update"

# Environment modes
ENVIRONMENT_HOME = "home"
ENVIRONMENT_OFFICE = "office"

# Working hours (for office environments)
DEFAULT_WORKING_DAYS = [0, 1, 2, 3, 4]  # Monday=0 to Friday=4
DEFAULT_WORKING_START = "08:00"
DEFAULT_WORKING_END = "18:00"

# System phases
PHASE_BASELINE = "baseline"
PHASE_ACTIVE = "active"
BASELINE_DAYS = 14

# Update intervals and scheduling
UPDATE_INTERVAL_SECONDS = 15       # Frequency to store data
AI_FREQUENCY_SECONDS = 15          # Update interval for agent state
SAVE_STATE_INTERVAL_SECONDS = 600  # Frequency to save AI state to avoid many writes (600 seconds = 10 minutes)
TASK_GENERATION_TIME = (6, 0, 0)   # (hour, minute, second) for daily task generation
VERIFY_TASKS_INTERVAL_MINUTES = 15 # Frequency to verify tasks in minutes

# Backup settings
BACKUP_INTERVAL_HOURS = 6  # Create automatic backup every X hours
KEEP_AUTO_BACKUPS = 10     # Keep last N auto backups (~2.5 days with 6h interval)
KEEP_STARTUP_BACKUPS = 3   # Keep last N startup backups (prevents accumulation)
KEEP_SHUTDOWN_BACKUPS = 3  # Keep last N shutdown backups (prevents accumulation)

# Data retention settings
RL_EPISODE_RETENTION_DAYS = 120   # Keep RL episodes for 120 days (4 months)
NOTIFICATION_HISTORY_LIMIT = 100  # Keep last N notifications in JSON state

# Sensor categories and keywords for auto-discovery
SENSOR_MAPPING = {
    "power": {
        "classes": [SensorDeviceClass.POWER],
        "units": ["W", "kW"],
        "keywords": ["power", "consumption", "current_load"]
    },
    "energy": {
        "classes": [SensorDeviceClass.ENERGY],
        "units": ["kWh", "Wh"],
        "keywords": ["energy", "total_consumption", "meter"]
    },
    "temperature": {
        "classes": [SensorDeviceClass.TEMPERATURE],
        "units": ["°C", "°F", "K"],
        "keywords": ["temperature", "temp"]
    },
    "humidity": {
        "classes": [SensorDeviceClass.HUMIDITY],
        "units": ["%"],
        "keywords": ["humidity", "hum"]
    },
    "illuminance": {
        "classes": [SensorDeviceClass.ILLUMINANCE],
        "units": ["lx", "lux"],
        "keywords": ["illuminance", "light_level"]
    },
    "occupancy": {
        "classes": [
            BinarySensorDeviceClass.OCCUPANCY,
            BinarySensorDeviceClass.MOTION,
            BinarySensorDeviceClass.PRESENCE
        ],
        "units": [],
        "keywords": ["occupancy", "motion", "presence"]
    }
}

# Environmental sensor types that benefit from area-based tracking
AREA_BASED_SENSORS = ["power", "energy", "temperature", "humidity", "illuminance", "occupancy"]

# RL Action Spaces
ACTIONS = {
    "specific": 1,
    "anomaly": 2,
    "behavioural": 3,
    "normative": 4,
}

# Rewards function weights
REWARD_WEIGHTS = {
    "alpha": 1.0,  # Energy Savings
    "beta": 0.5,   # User Engagement
    "delta": 0.3,  # Penalization for excessive notifications
}

# Notification settings
MAX_NOTIFICATIONS_PER_DAY = 10       # Maximum notifications per day to prevent fatigue
FATIGUE_THRESHOLD = 0.7              # User fatigue threshold (0 to 1) where 1 means fully fatigued
MIN_COOLDOWN_MINUTES = 30            # Base cooldown between notifications
HIGH_OPPORTUNITY_THRESHOLD = 0.6     # Score needed to bypass cooldown
CRITICAL_OPPORTUNITY_THRESHOLD = 0.8 # Score for immediate notification

# Shadow learning (offline RL during baseline phase)
SHADOW_EXPLORATION_RATE = 0.5  # Higher exploration during baseline (no cost to bad exploration)
SHADOW_LEARNING_RATE = 0.05    # Lower learning rate for shadow episodes (noisier signal)
SHADOW_INTERVAL_MULTIPLIER = 4 # Shadow episode every N AI cycles (e.g., if AI_FREQUENCY_SECONDS=15s, happens every 4 * 15s = 60s)

# Discount factor for RL
GAMMA = 0.95

# Weather-related constants
BASE_TEMPERATURE = 18.0
WEATHER_ENTITIES = [
    "weather.home",
    "weather.forecast_home",
    "sensor.outdoor_temperature",
    "sensor.outside_temperature",
    "sensor.outdoor_temp"
]

CO2_FACTOR = 0.1 # kg CO2/kWh as of early 2026 for Portugal