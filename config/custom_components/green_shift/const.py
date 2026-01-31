DOMAIN = "green_shift"

# System phases
PHASE_BASELINE = "baseline"
PHASE_ACTIVE = "active"
BASELINE_DAYS = 14

# Update interval for agent state
UPDATE_INTERVAL_SECONDS = 15

# Sensor categories and keywords for auto-discovery
SENSOR_CATEGORIES = {
    "power": ["power", "watt", "energy", "kwh"],
    "temperature": ["temperature", "temp"],
    "humidity": ["humidity"],
    "illuminance": ["illuminance", "lux", "light_level"],
    "occupancy": ["occupancy", "motion", "presence", "binary_sensor"],
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