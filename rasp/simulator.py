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
HUB_BASE = {
    "name": "Environment Hub", 
    "manufacturer": "Custom-Labs", 
    "model": "Multi-Sensor-v1"
}
PLUG_DEV = {
    "manufacturer": "Custom-Labs", 
    "model": "Power-Meter-v1"
}

# Definition of Environments
# Format: (dev_id, name, class, unit, dev_info, value_key, state_class, sensor_type, area)
ENVIRONMENTS = {
    1: {
        "name": "Fraunhofer Lab Environment",
        "sensors": [
            ("hub_main", "Overall Energy Consumption", "energy", "kWh", HUB_BASE, "energy", "total_increasing", "sensor", "Main Lab"),
            ("hub_main", "Ambient Temperature", "temperature", "°C", HUB_BASE, "temp1", "measurement", "sensor", "Main Lab"),
            ("hub_server", "Ambient Temperature 1", "temperature", "°C", HUB_BASE, "temp2", "measurement", "sensor", "Server Room"),
            ("hub_main", "Relative Humidity", "humidity", "%", HUB_BASE, "hum1", "measurement", "sensor", "Main Lab"),
            ("hub_server", "Relative Humidity 1", "humidity", "%", HUB_BASE, "hum2", "measurement", "sensor", "Server Room"),
            ("hub_main", "Luminosity", "illuminance", "lx", HUB_BASE, "lux1", "measurement", "sensor", "Main Lab"),
            ("hub_main", "Luminosity 1", "illuminance", "lx", HUB_BASE, "lux2", "measurement", "sensor", "Server Room"),
            ("hub_entrance", "Presence", "occupancy", None, HUB_BASE, "presence1", None, "binary_sensor", "Entrance"),
            ("hub_main", "Presence 1", "occupancy", None, HUB_BASE, "presence2", None, "binary_sensor", "Main Lab"),
            ("plug_alpha", "Plug Alpha Power", "power", "W", {"name": "Plug Monitor Alpha", **PLUG_DEV}, "power", "measurement", "sensor", "Main Lab"), # Use ** to unpack dict
            ("plug_beta", "Plug Beta Power", "power", "W", {"name": "Plug Monitor Beta", **PLUG_DEV}, "power", "measurement", "sensor", "Main Lab"),
            ("plug_charlie", "Plug Charlie Power", "power", "W", {"name": "Plug Monitor Charlie", **PLUG_DEV}, "power", "measurement", "sensor", "Main Lab"),
        ]
    },
    2: {
        "name": "FEUP Lab Environment",
        "sensors": [
            ("hub_feup", "Ambient Temperature", "temperature", "°C", HUB_BASE, "temp", "measurement", "sensor", "No Area"),
            ("hub_feup", "Presence", "occupancy", None, HUB_BASE, "presence", None, "binary_sensor", "No Area"),
            ("plug_fridge", "Fridge Energy", "energy", "kWh", {"name": "Fridge Smart Plug", **PLUG_DEV}, "energy", "total_increasing", "sensor", "No Area"),
            ("plug_coffee", "Coffee Machine Energy", "energy", "kWh", {"name": "Coffee Machine Smart Plug", **PLUG_DEV}, "energy", "total_increasing", "sensor", "No Area")
        ]
    }
}

active_env = ENVIRONMENTS.get(ENVIRONMENT_SELECTION, ENVIRONMENTS[1])
current_values = {}

# --- MQTT SETUP ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)

def clear_legacy_configs():
    """Wipes all possible old discovery topics to ensure a clean registry in HA."""
    print("Initiating full cleanup of MQTT discovery topics...")
    # Comprehensive list of old and current IDs to ensure nothing is left behind
    target_ids = [
        "hub_main", "hub_server", "hub_entrance", "hub_feup",
        "plug_alpha", "plug_beta", "plug_charlie",
        "plug_fridge", "plug_coffee"
    ]
    
    # Common keys used across all versions of the script
    keys_to_clear = [
        "energy", "power", "presence", "presence1", "presence2", 
        "temp", "temp1", "temp2", "hum1", "hum2", "lux1", "lux2"
    ]
    
    for dev in target_ids:
        for s_type in ["sensor", "binary_sensor"]:
            for key in keys_to_clear:
                topic = f"homeassistant/{s_type}/{dev}/{key}/config"
                client.publish(topic, "", retain=True)
    print("Cleanup complete.")

def publish_discovery():
    print(f"Announcing discovery for: {active_env['name']}")

    for dev_id, name, d_class, unit, dev_info, v_key, s_class, s_type, area in active_env["sensors"]:
        config_topic = f"homeassistant/{s_type}/{dev_id}/{v_key}/config"

        device_payload = dev_info.copy()
        device_payload["identifiers"] = [dev_id]
        
        if area and area != "No Area":
            device_payload["suggested_area"] = area

        payload = {
            "name": name,
            "unique_id": f"{dev_id}_{v_key}",
            "state_topic": f"homeassistant/{s_type}/{dev_id}/state",
            "value_template": f"{{{{ value_json.{v_key} }}}}",
            "device": device_payload
        }
        
        if d_class: payload["device_class"] = d_class
        if unit: payload["unit_of_measurement"] = unit
        if s_class: payload["state_class"] = s_class
        
        client.publish(config_topic, json.dumps(payload), retain=True)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker.")
        clear_legacy_configs()
        time.sleep(1) 
        publish_discovery()

client.on_connect = on_connect

def get_random_walk(current, min_val, max_val, step):
    return max(min(current + random.uniform(-step, step), max_val), min_val)

# --- EXECUTION ---
client.connect(BROKER_IP, PORT, 60)
client.loop_start()

for sensor in active_env["sensors"]:
    v_key = sensor[5]
    if "temp" in v_key: current_values[v_key] = 22.0
    elif "hum" in v_key: current_values[v_key] = 50.0
    elif "lux" in v_key: current_values[v_key] = 300.0
    elif "energy" in v_key: current_values[v_key] = 100.0
    elif "power" in v_key: current_values[v_key] = 50.0
    elif "presence" in v_key: current_values[v_key] = "OFF"

try:
    while True:
        devices_to_update = list(set([s[0] for s in active_env["sensors"]]))
        
        for dev_id in devices_to_update:
            dev_sensors = [s for s in active_env["sensors"] if s[0] == dev_id]

            for s in dev_sensors:
                v_key = s[5]
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

            type_groups = {}
            for s in dev_sensors:
                v_key, s_type = s[5], s[7]
                type_groups.setdefault(s_type, {})[v_key] = current_values[v_key]

            for s_type, payload in type_groups.items():
                client.publish(f"homeassistant/{s_type}/{dev_id}/state", json.dumps(payload))
            
        time.sleep(5)

except KeyboardInterrupt:
    print("Exiting...")
    client.loop_stop()
    client.disconnect()