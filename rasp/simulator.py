import paho.mqtt.client as mqtt
import json
import time
import random

# --- CONFIGURATION ---
BROKER_IP = "127.0.0.1"
PORT = 1883

HUB_ID = "environment_monitor_hub"
PLUG_A_ID = "smart_plug_alpha"
PLUG_B_ID = "smart_plug_beta"
PLUG_C_ID = "smart_plug_charlie"

# Initial sensor values
current_values = {
    "temp": 21.5,      # Range: 18-26°C
    "hum": 45.0,       # Range: 35-65%
    "lux": 300.0,      # Range: 0-1000 lx
    "total_e": 120.0,  # Total House Energy (kWh)
    "plug_a_e": 15.0,  # Plug A Energy (kWh)
    "plug_b_e": 8.0,   # Plug B Energy (kWh)
    "plug_c_w": 45.0,  # Plug C Power (W)
    "presence": "OFF"
}

# --- MQTT SETUP ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker successfully.")
        publish_discovery()
    else:
        print(f"Connection failed with code {rc}")

client.on_connect = on_connect

def publish_discovery():
    """Announces devices to Home Assistant."""
    
    # 1. MAIN HUB DEVICE INFO
    hub_device = {
        "identifiers": [HUB_ID],
        "name": "Environmental Data Hub",
        "manufacturer": "Custom-Labs",
        "model": "Multi-Sensor-v1"
    }

    # 2. PLUG DEVICE INFOS
    plug_a_device = {"identifiers": [PLUG_A_ID], "name": "Smart Plug Alpha", "model": "Power-Meter-v1"}
    plug_b_device = {"identifiers": [PLUG_B_ID], "name": "Smart Plug Beta", "model": "Power-Meter-v1"}
    plug_c_device = {"identifiers": [PLUG_C_ID], "name": "Smart Plug Charlie", "model": "Power-Meter-v1"}

    # --- SENSOR CONFIGURATIONS ---
    # format: (id, name, class, unit, parent_device, value_key, state_class)
    sensor_configs = [
        (HUB_ID, "Ambient Temperature", "temperature", "°C", hub_device, "temp", "measurement"),
        (HUB_ID, "Relative Humidity", "humidity", "%", hub_device, "hum", "measurement"),
        (HUB_ID, "Luminosity", "illuminance", "lx", hub_device, "lux", "measurement"),
        (HUB_ID, "Overall Consumption", "energy", "kWh", hub_device, "total_e", "total_increasing"),
        (PLUG_A_ID, "Plug Alpha Consumption", "energy", "kWh", plug_a_device, "energy", "total_increasing"),
        (PLUG_B_ID, "Plug Beta Consumption", "energy", "kWh", plug_b_device, "energy", "total_increasing"),
        (PLUG_C_ID, "Plug Charlie Power", "power", "W", plug_c_device, "power", "measurement")
    ]

    for dev_id, name, d_class, unit, dev_info, v_key, s_class in sensor_configs:
        config_payload = {
            "name": name,
            "unique_id": f"{dev_id}_{v_key}",
            "device_class": d_class,
            "state_topic": f"homeassistant/sensor/{dev_id}/state",
            "unit_of_measurement": unit,
            "value_template": f"{{{{ value_json.{v_key} }}}}",
            "device": dev_info,
            "state_class": s_class
        }
        client.publish(f"homeassistant/sensor/{dev_id}/{v_key}/config", json.dumps(config_payload), retain=True)

    # 3. BINARY SENSOR (Presence)
    presence_config = {
        "name": "Presence Detection",
        "unique_id": f"{HUB_ID}_presence",
        "device_class": "occupancy",
        "state_topic": f"homeassistant/binary_sensor/{HUB_ID}/state",
        "value_template": "{{ value_json.presence }}",
        "device": hub_device
    }
    client.publish(f"homeassistant/binary_sensor/{HUB_ID}/presence/config", json.dumps(presence_config), retain=True)

    print("Discovery payloads sent to Home Assistant.")

def get_random_walk(current, min_val, max_val, step):
    """Nudges value within a considerate threshold."""
    new_val = current + random.uniform(-step, step)
    return max(min(new_val, max_val), min_val)

# --- EXECUTION ---
client.connect(BROKER_IP, PORT, 60)
client.loop_start()

try:
    while True:
        # Update values using random walk (real-world simulation)
        current_values["temp"] = get_random_walk(current_values["temp"], 18.0, 26.0, 0.2)
        current_values["hum"] = get_random_walk(current_values["hum"], 35.0, 65.0, 0.5)
        current_values["lux"] = get_random_walk(current_values["lux"], 0.0, 1000.0, 15.0)
        
        # Energy values (must always increase)
        current_values["total_e"] += random.uniform(0.005, 0.02)
        current_values["plug_a_e"] += random.uniform(0.001, 0.005)
        current_values["plug_b_e"] += random.uniform(0.001, 0.003)

        # Randomly walk the wattage between 5W and 150W
        current_values["plug_c_w"] = get_random_walk(current_values["plug_c_w"], 5.0, 150.0, 10.0)
        
        # Presence (5% chance to flip state)
        if random.random() < 0.05:
            current_values["presence"] = "ON" if current_values["presence"] == "OFF" else "OFF"

        # Prepare Payloads
        hub_data = {
            "temp": round(current_values["temp"], 1),
            "hum": round(current_values["hum"], 1),
            "lux": round(current_values["lux"], 0),
            "total_e": round(current_values["total_e"], 4),
            "presence": current_values["presence"]
        }
        
        # Publish Data
        client.publish(f"homeassistant/sensor/{HUB_ID}/state", json.dumps(hub_data))
        client.publish(f"homeassistant/binary_sensor/{HUB_ID}/state", json.dumps(hub_data))
        client.publish(f"homeassistant/sensor/{PLUG_A_ID}/state", json.dumps({"energy": round(current_values["plug_a_e"], 4)}))
        client.publish(f"homeassistant/sensor/{PLUG_B_ID}/state", json.dumps({"energy": round(current_values["plug_b_e"], 4)}))
        client.publish(f"homeassistant/sensor/{PLUG_C_ID}/state", json.dumps({"power": round(current_values["plug_c_w"], 1)}))

        print(f"Update: Temp {hub_data['temp']}°C | Energy Total {hub_data['total_e']} kWh | Presence {hub_data['presence']} | Plug A {round(current_values['plug_a_e'],4)} kWh | Plug B {round(current_values['plug_b_e'],4)} kWh | Plug C {round(current_values['plug_c_w'],1)} W")
        
        # Every 5 seconds
        time.sleep(5) 

except KeyboardInterrupt:
    print("Stopping simulator...")
    client.loop_stop()
    client.disconnect()