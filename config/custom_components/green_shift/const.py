from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.components.binary_sensor import BinarySensorDeviceClass

DOMAIN = "green_shift"

# System phases
PHASE_BASELINE = "baseline"
PHASE_ACTIVE = "active"
BASELINE_DAYS = 14

# Update interval for agent state
UPDATE_INTERVAL_SECONDS = 15

# Sensor categories and keywords for auto-discovery
SENSOR_MAPPING = {
    "power": {
        "classes": [SensorDeviceClass.POWER, SensorDeviceClass.ENERGY],
        "units": ["W", "kW", "Wh", "kWh"],
        "keywords": ["power", "watt", "energy"]
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
MAX_NOTIFICATIONS_PER_DAY = 3
FATIGUE_THRESHOLD = 0.7

# Discount factor for RL
GAMMA = 0.95