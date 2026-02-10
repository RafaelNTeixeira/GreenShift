from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.components.binary_sensor import BinarySensorDeviceClass

DOMAIN = "green_shift"
GS_UPDATE_SIGNAL = "green_shift_update"
GS_AI_UPDATE_SIGNAL = "green_shift_ai_update"

# System phases
PHASE_BASELINE = "baseline"
PHASE_ACTIVE = "active"
BASELINE_DAYS = 14

UPDATE_INTERVAL_SECONDS = 5 # Frequency to store data
AI_FREQUENCY_SECONDS = 15 # Update interval for agent state
SAVE_STATE_INTERVAL_SECONDS = 600 # Frequency to save AI state to avoid many writes (600 seconds = 10 minutes)
TASK_GENERATION_TIME = (6, 0, 0) # (hour, minute, second) for daily task generation
VERIFY_TASKS_INTERVAL_MINUTES = 15 # Frequency to verify tasks in minutes

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

# Notification settings
MAX_NOTIFICATIONS_PER_DAY = 5
FATIGUE_THRESHOLD = 0.7

# Discount factor for RL
GAMMA = 0.95

# Notification Templates
NOTIFICATION_TEMPLATES = {
    "specific": [
        {
            "title": "High Consumption Alert",
            "message": "🔌 {device_name} is currently using {device_power}W, which is higher than usual. Consider turning it off when not in use."
        },
        {
            "title": "Device Energy Tip",
            "message": "💡 {device_name} has been running continuously and is consuming {device_power}W. A quick power cycle might help optimize its efficiency."
        },
        {
            "title": "Appliance Usage Notice",
            "message": "⚡ Your {device_name} is drawing {device_power}W right now. If you're not actively using it, switching it off could save energy."
        },
        {
            "title": "Smart Plug Alert",
            "message": "🔋 {device_name} appears to be in standby mode but still consuming {device_power}W. Unplugging when idle could reduce phantom load."
        },
        {
            "title": "Energy Efficiency Tip",
            "message": "🌟 {device_name} is using {device_power}W. Consider using it during off-peak hours or reducing usage duration for better efficiency."
        }
    ],
    
    "anomaly": [
        {
            "title": "Unusual Consumption Pattern",
            "message": "📊 Your current power usage ({current_power}W) is {percent_above}% higher than your typical baseline ({baseline_power}W). Check if any devices were left on accidentally."
        },
        {
            "title": "Energy Anomaly Detected",
            "message": "🔍 We've detected unusual energy consumption in your {area_name}. The {metric} levels are outside the normal range. Worth investigating?"
        },
        {
            "title": "Consumption Spike Alert",
            "message": "⚠️ Your energy use just spiked to {current_power}W (normal: {baseline_power}W). This could indicate an appliance malfunction or unusual activity."
        },
        {
            "title": "Area Anomaly Notice",
            "message": "🏠 {area_name} is showing unusual patterns - {metric} readings are significantly different from normal. Everything okay there?"
        },
        {
            "title": "Pattern Change Detected",
            "message": "📈 Your consumption pattern has changed significantly. Current usage is {percent_above}% above normal. New devices recently added?"
        }
    ],
    
    "behavioural": [
        {
            "title": "Bedtime Energy Tip",
            "message": "🌙 It's {time_of_day} - remember to turn off devices in standby mode before bed. Small actions like this can save up to 10% on your energy bill."
        },
        {
            "title": "Smart Habit Suggestion",
            "message": "💚 Try unplugging chargers when not in use. They consume power even when devices aren't connected - a simple habit that adds up over time."
        },
        {
            "title": "Energy-Saving Routine",
            "message": "⏰ Consider setting a routine to power down non-essential devices during {time_of_day}. Consistency in small actions leads to big savings."
        },
        {
            "title": "Comfort & Efficiency Tip",
            "message": "🌡️ Your {area_name} is at {area_temp}°C. Adjusting by just 1-2 degrees can save significant energy while maintaining comfort."
        },
        {
            "title": "Lighting Optimization",
            "message": "💡 Natural light is available during the day. Consider opening blinds instead of using artificial lighting when possible."
        },
        {
            "title": "Weekend Energy Habits",
            "message": "🏡 Weekends are great for reviewing your energy habits. Check which devices are always on and consider smarter usage patterns."
        },
        {
            "title": "Phantom Load Awareness",
            "message": "👻 Did you know? Devices in standby mode can account for 5-10% of household energy use. A power strip makes it easy to cut phantom loads."
        },
        {
            "title": "Seasonal Energy Tip",
            "message": "🍂 As seasons change, so should energy habits. Review your heating/cooling settings to match the current weather patterns."
        }
    ],
    
    "normative": [
        {
            "title": "Weekly Goal Update",
            "message": "🎯 Your consumption this week is {percent_above}% above target. You're close to achieving your {baseline_power}W goal - keep it up!"
        },
        {
            "title": "Community Comparison",
            "message": "📊 Similar households in your area are averaging {baseline_power}W. You're currently at {current_power}W. Small adjustments can make a big difference."
        },
        {
            "title": "Progress Check-In",
            "message": "🏆 You've saved energy before - your best week showed {baseline_power}W average. Current week: {current_power}W. You can do it again!"
        },
        {
            "title": "Benchmark Update",
            "message": "📈 Your current energy use ({current_power}W) is {percent_above}% above your personal best. Let's work together to improve this week."
        },
        {
            "title": "Target Achievement",
            "message": "🌟 Great progress! You're {percent_above}% away from your weekly reduction target. A few small changes could close the gap."
        },
        {
            "title": "Department Comparison",
            "message": "🏢 Your area's energy consumption is {percent_above}% above the organizational average. Together we can optimize our collective impact."
        }
    ]
}