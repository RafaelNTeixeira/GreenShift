# Green Shift - Energy Consumption Management System

## 🌱 About Green Shift

**Green Shift** is a Home Assistant custom component that uses **Reinforcement Learning** to help households and organizations optimize their energy consumption. The system learns your consumption patterns and provides personalized, behaviour-changing notifications to reduce energy use while minimizing user fatigue.

### Key Features

- 🤖 **AI-Powered Decisions**: Uses a Markov Decision Process (MDP) with reinforcement learning
- 🎮 **Gamification:** Daily tasks and a weekly challenge with adaptive difficulty to keep you entertained
- 🔍 **Auto-Discovery**: Automatically detects all sensors and areas in your Home Assistant setup
- 📊 **Real-Time Monitoring**: Tracks power consumption, temperature, humidity and occupancy
- 💡 **Smart Recommendations**: Suggests contextual actions based on consumption anomalies
- 📈 **Impact Tracking**: Shows savings in CO2 avoided and your picked currency
- 😊 **Feedback Integration**: Learns from your reactions to improve future recommendations
- ⚙️ **Zero Configuration**: Helpers and dashboard created automatically
- 🎯 **Phased Approach**: 14-day learning phase, then active engagement

---

## 📦 Installation

### Step 1: Copy Component Files

```bash
# SSH into Home Assistant or use Terminal & SSH add-on

# Navigate to custom_components
cd /config/custom_components/

# If green_shift folder doesn't exist, create it
mkdir -p green_shift

# Copy all files from the Green Shift repository:
# - __init__.py
# - config_flow.py
# - const.py
# - decision_agent.py
# - manifest.json
# - sensor.py
```

### Step 2: Restart Home Assistant

```bash
# Via Home Assistant UI
Settings → System → Restart Home Assistant

# Or via terminal
ha core restart
```

### Step 3: Add the Integration

1. Go to **Settings** → **Devices & Services**
2. Click **+ ADD INTEGRATION**
3. Search for **"Green Shift"**
4. Configure basic settings:
   - Currency: EUR (or USD, GBP)
   - Electricity Price: 0.25 €/kWh
   - CO2 Factor: 0.5 kg/kWh

✅ **That's it!** Helpers and sensors are created automatically.

---

## 🎨 Dashboard Setup

### Automatic Setup

The dashboard is automatically created when the integration loads. Access it at:

**Settings** → **Dashboards** → **lovelace-green-shift**

### Manual Access to Dashboard Configuration

If you need to edit or import the dashboard configuration:

1. Go to **Settings** → **Dashboards**
2. Find **"Energy Research Platform"**
3. Click the three dots (**⋮**) → **Edit Dashboard**
4. Click **Edit** → **Edit in YAML**

The dashboard configuration is defined in `ui-lovelace.yaml`.

---

## 🧪 How It Works

### Phase 1: Baseline (Days 0-14)

During the initial 14 days, the system is in **learning mode**:

- ✅ Observes your consumption patterns
- ✅ Learns the "normal" baseline (`E_baseline`)
- ❌ Does NOT send notifications or challenges
- 📊 Visible tabs: **Devices**, **Dashboard**, **Profile**, **Settings**
- 📋 Banner shows: "Calibration Mode: X days remaining"

**State Vector Components** collected:
- Total power consumption (W)
- Individual appliance power (W)
- Temperature (°C)
- Humidity (%)
- Illuminance (lux)
- Occupancy status (on/off)
- Anomaly, behaviour, and Fatigue indices

### Phase 2: Active (Day 15+)

After baseline learning, the system becomes **active**:

- 🎯 Starts suggesting energy-saving actions
- 📢 Sends up to 10 notifications per day
- 🎮 **Tasks** and **Collaborative Goal** tabs unlock
- 😊 Learns from your feedback via reinforcement learning
- ⚖️ Adapts behaviour to minimize user fatigue

**Action Types**:
1. **noop**: No action
2. **specific**: Appliance-specific tip (e.g., "Heater consuming more than normal")
3. **anomaly**: Unusual consumption pattern detected
4. **behavioural**: Habit-change suggestion (e.g., "Turn off standby")
5. **normative**: Social/department comparison ("Your group is 15% above target")

---

## 🔍 Automatic Sensor Discovery

Green Shift **automatically discovers** sensors based on keywords:

| Category | Keywords |
|----------|----------|
| **Power** | power, watt, energy, kwh |
| **Temperature** | temperature, temp |
| **Humidity** | humidity |
| **Illuminance** | illuminance, lux, light_level |
| **Occupancy** | occupancy, motion, presence, binary_sensor |

**No manual configuration needed!** Simply add sensors to Home Assistant and Green Shift finds them automatically.

---

## 📊 Created Entities

Green Shift creates the following sensors:

| Entity | Description | Unit |
|--------|-------------|------|
| `sensor.research_phase` | Current phase (baseline/active) | text |
| `sensor.energy_baseline` | Learned average consumption | W |
| `sensor.current_consumption` | Real-time power draw | W |
| `sensor.savings_accumulated` | Total savings | EUR |
| `sensor.co2_saved` | CO2 avoided | kg |
| `sensor.tasks_completed` | Completed tasks count | number |
| `sensor.collaborative_goal_progress` | Group goal progress | % |
| `sensor.behaviour_index` | User adherence score | 0-1 |
| `sensor.fatigue_index` | Notification fatigue risk | 0-1 |

### Helper Entities (Auto-Created)

| Entity | Type | Purpose |
|--------|------|---------|
| `input_number.energy_saving_target` | Number | Target savings % |
| `input_number.electricity_price` | Number | Local electricity cost |
| `input_select.currency` | Select | Display currency (EUR/USD/GBP) |
| `input_select.task_difficulty` | Select | Rate task difficulty |
| `input_boolean.task_1/2/3` | Boolean | Daily task completion |

---

## 🎓 Technical Architecture

### State-Action-Reward Loop

Green Shift implements a **Markov Decision Process** (MDP):

$$\langle S, A, M, P, R, \gamma \rangle$$

**S - State Vector** (12 components):
1. Global power consumption + existence flag
2. Top appliance power + existence flag  
3. Temperature + existence flag
4. Humidity + existence flag
5. Illuminance + existence flag
6. Occupancy + existence flag
7. Anomaly index (0-1)
8. Behaviour index (0-1)
9. Fatigue index (0-1)
10. Area anomaly count (spatial awareness)
11. Time of day (normalized 0-1)
12. Day of week (normalized 0-1)

**A - Action Space** (5 discrete actions):
```python
ACTIONS = {
    "noop": 0,          # No action
    "specific": 1,      # Appliance-specific tip
    "anomaly": 2,       # Anomaly alert
    "behavioural": 3,   # behaviour change suggestion
    "normative": 4,     # Social/group comparison
}
```

**M - Action Mask** (context-dependent availability):
- `noop`: Always available
- `specific`: Requires power sensors (smart plugs)
- `anomaly`: Requires 100+ historical samples
- `behavioural`: Always available (if in active phase)
- `normative`: Requires non-zero baseline

**R - Reward Function**:
$$R_t = \alpha \cdot \Delta E - \beta \cdot I_{fatigue} + \gamma \cdot I_{engagement}$$

Where:
- **α = 1.0**: Energy savings weight
- **β = 0.5**: Engagement weight (user feedback)
- **δ = 0.3**: Fatigue penalty weight

### Q-Learning Implementation

**Update Rule:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max Q(s',a') - Q(s,a)]$$

**Parameters:**
- Learning rate (α): 0.1
- Discount factor (γ): 0.95
- Exploration rate (ε): 0.2

**Epsilon-Greedy Policy:**
- 20% exploration (random available action)
- 80% exploitation (best known action from Q-table)


### Area-Based Learning

During baseline phase, the system calculates **area-specific baselines**:

```python
area_baselines = {
    "Living Room": {
        "temperature": 21.5,  # °C
        "power": 120.0,       # W
        "humidity": 45.0      # %
    },
    "Bedroom": {
        "temperature": 19.0,
        "power": 40.0,
        "humidity": 50.0
    }
}
```

During active phase, **area anomalies** are detected:

```python
area_anomalies = {
    "Living Room": {
        "temperature": 0.8,  # High anomaly (0-1)
        "power": 0.2         # Low anomaly
    }
}
```

## 💬 User Feedback Integration

### Through Actionable Notifications

When you receive a recommendation from Green Shift:

1. **Positive** ✅ "Helpful": 
   - Engagement score: +1.0
   - Behaviour index increases
   - Q-table: Positive reward for (state, action) pair
   
2. **Negative** ✗ "Not useful":
   - Engagement score: -0.5
   - Behaviour index decreases  
   - Fatigue index increases
   - Q-table: Negative reward for (state, action) pair

**Feedback Processing:**
- Exponentially weighted moving average (recent feedback weighted more)
- Updates behaviour index: `I_behaviour ∈ [0, 1]`
- Influences future action selection via Q-learning
- Prevents notification fatigue via adaptive fatigue index

### Task Difficulty Rating

In the **Challenges** tab, rate each task:

- **Too Easy** ↗️: System increases challenge complexity
- **Just Right** ➡️: System maintains current difficulty
- **Too Hard** ↘️: System simplifies future suggestions

This feedback is stored and analyzed to personalize future task generation.

---

## ⚙️ Configuration

### Settings Tab

Configure these parameters:

- **Savings Target (%)**: Your desired energy reduction (default: 15%)
- **Electricity Price (€/kWh)**: Local electricity cost for savings calculation (default: 0.25)
- **Currency**: Display currency for savings (EUR/USD/GBP)

### Notification Limits

- **Max notifications per day**: 10 (prevents user fatigue)
- **Min time between notifications**: 1 hour
- **Fatigue threshold**: 0.7 (notifications pause above this)
- **Baseline days**: 14 (calibration period before recommendations)

---

## 🤝 Contributing

This is an academic research project. For issues, feature requests or improvements:

1. **Report Issues**: Open a GitHub issue with detailed logs
2. **Submit Changes**: Create a pull request with clear descriptions
3. **Contact**: rafael2003t.18@gmail.com

---

**Happy Energy Saving! 🌱⚡**
