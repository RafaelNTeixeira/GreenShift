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
HUB_DEV = {"identifiers": ["hub_01"], "name": "Environment Hub", "manufacturer": "Custom-Labs", "model": "Multi-Sensor-v1"}
PLUG_FRIDGE = {"identifiers": ["plug_fridge"], "name": "Fridge Smart Plug", "model": "Power-Meter-v1"}
PLUG_COFFEE = {"identifiers": ["plug_coffee"], "name": "Coffee Machine Smart Plug", "model": "Power-Meter-v1"}

# Definition of Environments
ENVIRONMENTS = {
    1: {
        "name": "Fraunhofer Lab Environment",
        "sensors": [
            # (dev_id, name, class, unit, dev_info, value_key, state_class, sensor_type)
            ("hub_01", "Ambient Temperature", "temperature", "°C", HUB_DEV, "temp", "measurement", "sensor"),
            ("hub_01", "Relative Humidity", "humidity", "%", HUB_DEV, "hum", "measurement", "sensor"),
            ("hub_01", "Luminosity", "illuminance", "lx", HUB_DEV, "lux", "measurement", "sensor"),
            ("hub_01", "Presence", "occupancy", None, HUB_DEV, "presence", None, "binary_sensor"),
            ("plug_a", "Plug Alpha", "energy", "kWh", {"identifiers":["pa"], "name":"Plug Alpha"}, "energy", "total_increasing", "sensor")
        ]
    },
    2: {
        "name": "FEUP Lab Environment",
        "sensors": [
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
    current_values[v_key] = 20.0 if "temp" in v_key else 0.0
    if "presence" in v_key: current_values[v_key] = "OFF"

try:
    while True:
        # Update and Publish for each unique device in the active environment
        devices_to_update = list(set([s[0] for s in active_env["sensors"]]))
        
        for dev_id in devices_to_update:
            payload = {}
            # Get all sensors belonging to this device
            dev_sensors = [s for s in active_env["sensors"] if s[0] == dev_id]
            
            for _, _, _, _, _, v_key, _, s_type in dev_sensors:
                # Simulation Logic based on key name
                if "temp" in v_key:
                    current_values[v_key] = get_random_walk(current_values[v_key], 15, 30, 0.2)
                elif "energy" in v_key or "energy" in v_key:
                    current_values[v_key] += random.uniform(0.001, 0.01)
                elif "presence" in v_key:
                    if random.random() < 0.1:
                        current_values[v_key] = "ON" if current_values[v_key] == "OFF" else "OFF"
                
                payload[v_key] = round(current_values[v_key], 4) if isinstance(current_values[v_key], float) else current_values[v_key]

            # Publish to the specific device state topic
            # Using the first sensor type found for the topic path
            s_type = dev_sensors[0][7]
            client.publish(f"homeassistant/{s_type}/{dev_id}/state", json.dumps(payload))
        
        print(f"Published update for {active_env['name']}: {payload}")
        time.sleep(5)

except KeyboardInterrupt:
    client.loop_stop()
    client.disconnect()