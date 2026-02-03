import paho.mqtt.client as mqtt
import json
import time
import random

# --- SELECTION VARIABLE ---
# Change this to 1, 2, 3, or 4 to switch environments
ENVIRONMENT_SELECTION = 1

# --- CONFIGURATION ---
BROKER_IP = "127.0.0.1"
PORT = 1883

# Device Metadata Templates
HUB_DEV = {
    "identifiers": ["hub_01"], 
    "name": "Environment Hub", 
    "manufacturer": "Custom-Labs", 
    "model": "Multi-Sensor-v1"
}
PLUG_FRIDGE = {
    "identifiers": ["plug_fridge"], 
    "name": "Fridge Smart Plug", 
    "model": "Power-Meter-v1"
}
PLUG_COFFEE = {
    "identifiers": ["plug_coffee"], 
    "name": "Coffee Machine Smart Plug", 
    "model": "Power-Meter-v1"
}

# Definition of Environments
ENVIRONMENTS = {
    1: {
        "name": "Fraunhofer Lab Environment",
        "sensors": [
            # (dev_id, name, class, unit, dev_info, value_key, state_class, sensor_type)
            ("hub_01", "Overall Energy Consumption", "energy", "kWh", HUB_DEV, "energy", "total_increasing", "sensor"),
            ("hub_01", "Ambient Temperature", "temperature", "°C", HUB_DEV, "temp1", "measurement", "sensor"),
            ("hub_01", "Ambient Temperature 1", "temperature", "°C", HUB_DEV, "temp2", "measurement", "sensor"),
            ("hub_01", "Relative Humidity", "humidity", "%", HUB_DEV, "hum1", "measurement", "sensor"),
            ("hub_01", "Relative Humidity 1", "humidity", "%", HUB_DEV, "hum2", "measurement", "sensor"),
            ("hub_01", "Luminosity", "illuminance", "lx", HUB_DEV, "lux1", "measurement", "sensor"),
            ("hub_01", "Luminosity 1", "illuminance", "lx", HUB_DEV, "lux2", "measurement", "sensor"),
            ("hub_01", "Presence", "occupancy", None, HUB_DEV, "presence1", None, "binary_sensor"),
            ("hub_01", "Presence 1", "occupancy", None, HUB_DEV, "presence2", None, "binary_sensor"),
            ("plug_a", "Plug Alpha Power", "power", "W", {"identifiers":["pa"], "name":"Plug Monitor Alpha"}, "power", "measurement", "sensor"),
            ("plug_b", "Plug Beta Power", "power", "W", {"identifiers":["pb"], "name":"Plug Monitor Beta"}, "power", "measurement", "sensor"),
            ("plug_c", "Plug Charlie Power", "power", "W", {"identifiers":["pc"], "name":"Plug Monitor Charlie"}, "power", "measurement", "sensor"),
        ]
    },
    2: {
        "name": "FEUP Lab Environment",
        "sensors": [
            ("hub_01", "Ambient Temperature", "temperature", "°C", HUB_DEV, "temp", "measurement", "sensor"),
            ("hub_01", "Presence", "occupancy", None, HUB_DEV, "presence", None, "binary_sensor"),
            ("plug_fridge", "Fridge Energy", "energy", "kWh", PLUG_FRIDGE, "energy", "total_increasing", "sensor"),
            ("plug_coffee", "Coffee Machine Energy", "energy", "kWh", PLUG_COFFEE, "energy", "total_increasing", "sensor")
        ]
    },
    3: {
        "name": "Smart Home F",
        "sensors": [
            # Add sensors
        ]
    },
    4: {
        "name": "Smart Home B",
        "sensors": [
            # Add sensors
        ]
    }
}

active_env = ENVIRONMENTS.get(ENVIRONMENT_SELECTION, ENVIRONMENTS[1])
current_values = {}

# --- MQTT SETUP ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)

def publish_discovery():
    print(f"Announcing discovery for: {active_env['name']}")

    for dev_id, name, d_class, unit, dev_info, v_key, s_class, s_type in active_env["sensors"]:
        config_topic = f"homeassistant/{s_type}/{dev_id}/{v_key}/config"

        payload = {
            "name": name,
            "unique_id": f"{dev_id}_{v_key}",
            "state_topic": f"homeassistant/{s_type}/{dev_id}/state",
            "value_template": f"{{{{ value_json.{v_key} }}}}",
            "device": dev_info
        }
        if d_class: payload["device_class"] = d_class
        if unit: payload["unit_of_measurement"] = unit
        if s_class: payload["state_class"] = s_class
        
        client.publish(config_topic, json.dumps(payload), retain=True)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker.")
        publish_discovery()

client.on_connect = on_connect

def get_random_walk(current, min_val, max_val, step):
    return max(min(current + random.uniform(-step, step), max_val), min_val)

# --- EXECUTION ---
client.connect(BROKER_IP, PORT, 60)
client.loop_start()

# Initialize random values for active sensors
for _, _, _, _, _, v_key, _, _ in active_env["sensors"]:
    if "temp" in v_key: current_values[v_key] = 22.0
    elif "hum" in v_key: current_values[v_key] = 50.0
    elif "lux" in v_key: current_values[v_key] = 300.0
    elif "energy" in v_key: current_values[v_key] = 100.0
    elif "power" in v_key: current_values[v_key] = 50.0
    elif "presence" in v_key: current_values[v_key] = "OFF"

try:
    while True:
        # Get unique device IDs
        devices_to_update = list(set([s[0] for s in active_env["sensors"]]))
        
        for dev_id in devices_to_update:
            # Get all sensors belonging to THIS specific device
            dev_sensors = [s for s in active_env["sensors"] if s[0] == dev_id]

            # Simulate all values for this device first
            for _, _, _, _, _, v_key, _, s_type in dev_sensors:
                if "temp" in v_key:
                    current_values[v_key] = round(get_random_walk(current_values[v_key], 15, 30, 0.2), 2)
                elif "hum" in v_key:
                    current_values[v_key] = round(get_random_walk(current_values[v_key], 30, 70, 0.5), 2)
                elif "lux" in v_key:
                    current_values[v_key] = round(get_random_walk(current_values[v_key], 0, 1000, 10), 2)
                elif "power" in v_key:
                    current_values[v_key] = round(random.uniform(5, 150), 2)
                elif "energy" in v_key:
                    current_values[v_key] += round(random.uniform(0.001, 0.01), 2)
                elif "presence" in v_key:
                    if random.random() < 0.05:
                        current_values[v_key] = "ON" if current_values[v_key] == "OFF" else "OFF"

            # Group sensors by s_type (sensor vs binary_sensor).
            type_groups = {}
            for _, _, _, _, _, v_key, _, s_type in dev_sensors:
                type_groups.setdefault(s_type, {})[v_key] = (
                    round(current_values[v_key], 4)
                    if isinstance(current_values[v_key], float)
                    else current_values[v_key]
                )

            # Publish one payload per s_type so every topic receives its data
            for s_type, payload in type_groups.items():
                client.publish(f"homeassistant/{s_type}/{dev_id}/state", json.dumps(payload))
                print(f"[{active_env['name']}] Sent for {dev_id} ({s_type}): {payload}")
            
        time.sleep(5)

except KeyboardInterrupt:
    print("Exiting...")
    client.loop_stop()
    client.disconnect()