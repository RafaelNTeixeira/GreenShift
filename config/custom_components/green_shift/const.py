from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.components.binary_sensor import BinarySensorDeviceClass

DOMAIN = "green_shift"
GS_UPDATE_SIGNAL = "green_shift_update"
GS_AI_UPDATE_SIGNAL = "green_shift_ai_update"

# System phases
PHASE_BASELINE = "baseline"
PHASE_ACTIVE = "active"
BASELINE_DAYS = 14

# Update intervals and scheduling
UPDATE_INTERVAL_SECONDS = 5 # Frequency to store data
AI_FREQUENCY_SECONDS = 15 # Update interval for agent state
SAVE_STATE_INTERVAL_SECONDS = 600 # Frequency to save AI state to avoid many writes (600 seconds = 10 minutes)
TASK_GENERATION_TIME = (6, 0, 0) # (hour, minute, second) for daily task generation
VERIFY_TASKS_INTERVAL_MINUTES = 15 # Frequency to verify tasks in minutes

# Backup settings
BACKUP_INTERVAL_HOURS = 4 # Create automatic backup every 4 hours
BACKUP_DAILY_TIME = (3, 0, 0) # (hour, minute, second) for daily backup
BACKUP_WEEKLY_DAY = 0 # Day of week for weekly backup (0 = Monday)
BACKUP_WEEKLY_TIME = (3, 30, 0) # (hour, minute, second) for weekly backup
KEEP_AUTO_BACKUPS = 12 # Keep last 2 days (12 * 4 hours = 48 hours) of automatic backups
KEEP_DAILY_BACKUPS = 7 # Keep last 7 daily backups (1 week)
KEEP_WEEKLY_BACKUPS = 4 # Keep last 4 weekly backups (1 month)

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
MAX_NOTIFICATIONS_PER_DAY = 10
FATIGUE_THRESHOLD = 0.7
MIN_COOLDOWN_MINUTES = 30  # Base cooldown between notifications
HIGH_OPPORTUNITY_THRESHOLD = 0.6  # Score needed to bypass cooldown
CRITICAL_OPPORTUNITY_THRESHOLD = 0.8  # Score for immediate notification

# Shadow learning (offline RL during baseline phase)
SHADOW_EXPLORATION_RATE = 0.5  # Higher exploration during baseline (no cost to bad exploration)
SHADOW_LEARNING_RATE = 0.05   # Lower learning rate for shadow episodes (noisier signal)
SHADOW_INTERVAL_MULTIPLIER = 4 # Shadow episode every N AI cycles (e.g., if AI_FREQUENCY_SECONDS=15s, happens every 4 * 15s = 60s)

# Discount factor for RL
GAMMA = 0.95
