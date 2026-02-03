from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.components.binary_sensor import BinarySensorDeviceClass

DOMAIN = "green_shift"
GS_UPDATE_SIGNAL = "green_shift_update"

# System phases
PHASE_BASELINE = "baseline"
PHASE_ACTIVE = "active"
BASELINE_DAYS = 14

UPDATE_INTERVAL_SECONDS = 5 # Frequency to store data
AI_FREQUENCY_SECONDS = 15 # Update interval for agent state

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

# RL Action Spaces
ACTIONS = {
    "noop": 0,
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

# Motification settings
FATIGUE_THRESHOLD = 0.7

# Discount factor for RL
GAMMA = 0.95