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
- 🌍 **Multilingual**: Integration for different languages

### Supported Languages

Green Shift is **fully translated** including dynamic content (AI notifications and tasks):

- 🇬🇧 **English** - Default
- 🇵🇹 **Português** - Complete translation

**To switch languages:**

1. **Update your Home Assistant system language**:
   - Go to **Settings → System → General**
   - Scroll to **Language**
   - Select your language (e.g., Português)
   - Click **Save**

   ⚠️ **Note**: The Profile language (Settings → Profile → Language) only changes the Home Assistant UI language, not the integration language. You must change the **System Language** for Green Shift to detect it.

2. **Update [`configuration.yaml`](./config/configuration.yaml)**:
   ```yaml
   homeassistant:
     customize: !include locales/customize_pt.yaml  # or locales/customize_en.yaml

   lovelace:
     dashboards:
       lovelace-green-shift:
         mode: yaml
         filename: locales/ui-lovelace-pt.yaml  # or locales/ui-lovelace-en.yaml
         title: Green Shift
         icon: mdi:leaf
   ```
3. **Restart Home Assistant**

📚 **[Translation Guide](./docs/TRANSLATIONS.md)** - Help us add more languages!

---

## 📦 Installation

### Step 1: Copy Component Files

```bash
# SSH into Home Assistant or use Terminal & SSH add-on

# Navigate to custom_components
cd /config/custom_components/

# If green_shift folder doesn't exist, create it
mkdir -p green_shift

# Copy all files from config/custom_components/green_shift/:
# - __init__.py
# - backup_manager.py
# - config_flow.py
# - const.py
# - data_collector.py
# - decision_agent.py
# - helpers.py
# - manifest.json
# - select.py
# - sensor.py
# - services.yaml
# - storage.py
# - task_manager.py
# - translations_runtime.py
# - translations/ (folder with en.json and pt.json)
```

### Step 2: Copy Configuration Files

```bash
# Navigate to config directory
cd /config/

# Copy the locales folder (for UI and dashboards)
# This includes:
# - locales/customize_en.yaml
# - locales/customize_pt.yaml
# - locales/ui-lovelace-en.yaml
# - locales/ui-lovelace-pt.yaml

# Copy helper configuration files to /config/
# - input_numbers.yaml
# - input_selects.yaml
# - input_booleans.yaml (can be empty)
```

### Step 3: Update configuration.yaml

Use one of the following approaches, depending on your current setup.

#### Option A: Fresh/minimal `configuration.yaml`

If these sections are not defined yet in your file, you can add this block directly:

```yaml
homeassistant:
  # Choose your language (en or pt)
  customize: !include locales/customize_en.yaml

# Lovelace dashboard configuration
lovelace:
  dashboards:
    lovelace-green-shift:
      mode: yaml
      filename: locales/ui-lovelace-en.yaml # Choose your language (en or pt)
      title: Green Shift
      icon: mdi:leaf

# Include helper files
input_number: !include input_numbers.yaml
input_select: !include input_selects.yaml
input_boolean: !include input_booleans.yaml
```

#### Option B: Existing/populated `configuration.yaml`

If your file already has `homeassistant`, `lovelace`, `input_number`, `input_select` or `input_boolean`, do **not** duplicate those top-level keys. Merge Green Shift entries into your existing sections.

Example (merge into existing structure):

```yaml
homeassistant:
  name: My Home
  customize: !include locales/customize_en.yaml

lovelace:
  dashboards:
    lovelace-green-shift:
      mode: yaml
      filename: locales/ui-lovelace-en.yaml
      title: Green Shift
      icon: mdi:leaf

# Keep your existing input_* definitions strategy:
# - If they already use !include, merge Green Shift helpers into those included files
# - If they are inline maps, add Green Shift helper entities inline
```

The key rule is simple: each top-level key should exist only once in `configuration.yaml`.

### Step 4: Configure Workday integration

Workday is required if you want Green Shift to correctly treat public holidays as non-working days in office mode.

1. Go to **Settings** -> **Devices & Services**
2. Click **+ ADD INTEGRATION**
3. Search for **Workday**
4. Configure your country/region and holiday options
5. Make sure the entity exists as `binary_sensor.workday_sensor`

> Green Shift uses `binary_sensor.workday_sensor` to avoid generating office-mode activity on holidays.

### Step 5: Restart Home Assistant

```bash
# Via Home Assistant UI
Settings → System → Restart Home Assistant

# Or via terminal
ha core restart
```

### Step 6: Add the Integration

1. Go to **Settings** → **Devices & Services**
2. Click **+ ADD INTEGRATION**
3. Search for **"Green Shift"**
4. Follow the configuration wizard:
   - Configure currency and environment settings
   - Select your sensors (energy, power, temperature, etc.)
   - **Optional**: Select a weather entity for Heating/Cooling Degree Days analysis (see note below)
   - Assign areas to sensors

> **Weather Entity (optional but recommended)**
>
> Providing a weather entity (e.g. `weather.home`) unlocks several improvements:
>
> - **Climate-aware task difficulty**: on extremely hot or cold days the temperature tasks are automatically scaled down so users are not penalised for conditions they cannot control.
> - **Degree-day research data**: daily HDD/CDD values are recorded alongside consumption data for richer analysis.
>
> The integration searches common weather entity names automatically, but you can select any `weather.*` or `sensor.*` (outdoor temperature) entity during setup.
> If you skip this field the integration works normally: weather-dependent features are simply disabled.

✅ **That's it!** The integration will now start collecting baseline data.

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

## 🧠 How It Works

### Phase 1: Baseline (Days 0-14)

During the initial 14 days, the system is in **learning mode**:

- ✅ Observes your consumption patterns
- ✅ Learns the "normal" baseline (`E_baseline`)
- ❌ Does NOT send notifications or challenges
- 📊 Visible tabs: **Devices**, **Dashboard**, **Settings**
- 📋 Banner shows: "Calibration Mode: X days remaining"

**State Vector Components** collected:
- Total power consumption (W)
- Individual appliance power (W)
- Temperature (°C)
- Humidity (%)
- Illuminance (lux)
- Occupancy status (on/off)
- Anomaly, Behaviour and Fatigue indices

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

**A - Action Space** (4 discrete actions):
```python
ACTIONS = {
    "noop": 0,          # No intervention
    "specific": 1,      # Appliance-specific tip
    "anomaly": 2,       # Anomaly alert
    "behavioural": 3,   # Behaviour change suggestion
    "normative": 4,     # Social/group comparison
}
```

**M - Action Mask** (context-dependent availability):
- `noop`: Always available in active phase
- `specific`: Requires individual power sensors
- `anomaly`: Requires 100+ historical samples
- `behavioural`: Always available in active phase
- `normative`: Requires non-zero baseline consumption

**R - Reward Function** (called after user responds - delayed Q-learning):
$$R_t = \alpha \cdot \Delta E + \beta \cdot f_{feedback} - \delta \cdot I_{fatigue}$$

Where:
- **α = 1.0**: Energy savings weight ($\Delta E$ = normalised power drop vs. baseline)
- **β = 0.5**: Feedback signal weight ($f_{feedback}$ = +1.0 accept, −0.5 reject)
- **δ = 0.3**: Fatigue penalty weight ($I_{fatigue}$ = current fatigue index at response time)

### Q-Learning Implementation

**Update Rule (dynamic γ based on user response):**
$$Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma_{dyn} \max Q(s',a') - Q(s,a)]$$

$$\gamma_{dyn} = \begin{cases} 0.95 & \text{(accepted)} \\ 0.0 & \text{(rejected -- terminal)} \end{cases}$$

Rejection is treated as a terminal state: future value estimation is disabled, so the agent learns solely from the negative immediate reward without being partially offset by future Q-values.

**Parameters:**
- Learning rate (α): 0.1
- Discount factor (γ): 0.95 (accept) / 0.0 (reject)
- Exploration rate (ε): 0.2
- Shadow exploration rate: 0.5
- Shadow learning rate: 0.05

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

### Task Validation Windows

Daily-average tasks are not finalized immediately after generation.

- Home environment: validation starts at 20:00.
- Office environment: validation starts 2 hours before the configured `working_end` time.
- Before the minimum validation time, task status stays pending and the UI reason shows the expected validation time.
- Before cutoff, the UI still receives current measured values (running average) and the reason includes the current average vs target.

This avoids false early wins (for example, very low standby consumption right after 06:00 generation) and prevents streak/reward credit before a representative day window is observed.
---

## ⚙️ Configuration

### Settings Tab

Configure these parameters:

- **Savings Target (%)**: Your desired energy reduction (default: 15%)
- **Electricity Price (€/kWh)**: Local electricity cost for savings calculation (default: 0.25)
- **Currency**: Display currency for savings (EUR/USD/GBP)

### Notification Limits

- **Max notifications per day**: 10 (prevents user fatigue)
- **Min time between notifications**: 30 minutes (base cooldown)
- **Fatigue threshold**: 0.7 (notifications pause above this)
- **High opportunity bypass threshold**: 0.6 (can bypass standard cooldown)
- **Critical opportunity threshold**: 0.8 (can bypass fatigue block)
- **Baseline days**: 14 (calibration period before recommendations)

---

## 🛡️ Data Safety & Backup

Green Shift includes comprehensive data protection to ensure your energy data and AI learning state are never lost:

### Automatic Protection
- ✅ **Write-Ahead Logging (WAL)**: Protects against crashes and power failures
- ✅ **Automatic Backups**: Every 6 hours, keeps last 10 (~2.5 days of protection)
- ✅ **Atomic Writes**: State files never partially written (no corruption)
- ✅ **Startup/Shutdown Backups**: Snapshots before and after restarts

### Recovery Services

If something goes wrong, use these services:

```yaml
# List all available backups
service: green_shift.list_backups

# Create a manual backup (before major changes)
service: green_shift.create_backup

# Restore from a backup
service: green_shift.restore_backup
data:
  backup_name: "auto/20260218_100000"  # or just "20260218_100000"
```

### Backup Storage

All data is stored in: `config/green_shift_data/`
- `sensor_data.db` - Last 14 days of sensor readings
- `research_data.db` - Permanent research and analytics data
- `state.json` - AI model state (Q-table, indices)
- `backups/` - Organized backup snapshots:
  - `auto/` - Every 6 hours, keeps last 10 (~2.5 days)
  - `startup/` - On integration startup, keeps last 2
  - `shutdown/` - On integration shutdown, keeps last 2
  - `manual/` - User-created backups (never auto-deleted)
  - `pre_restore/` - On restoring database (saves current database state before changing it)

---

## 🧪 Testing

Green Shift includes currently **1196 comprehensive tests** covering AI logic, backup systems, configuration and utility functions - with **100% total code coverage**.

### Quick Start

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests with coverage (from workspace root)
python3 -m pytest -n auto tests/

# Run without coverage (faster for quick checks)
python3 -m pytest -n auto tests/ --no-cov

# Run specific test file
python3 -m pytest -n auto tests/test_decision_agent.py -v

# View HTML coverage report
# After running tests, open tests/htmlcov/index.html in your browser
```

**Coverage is automatically generated** in `tests/htmlcov/` when you run pytest.

**Test Coverage:**
- ✅ **46 tests** - Backup management (100%)
- ✅ **77 tests** - Config flow & sensor discovery (100%)
- ✅ **97 tests** - Real-time data collection & energy tracking (100%)
- ✅ **351 tests** - AI decision agent & Q-learning (100%)
- ✅ **58 tests** - Helper functions & conversions (100%)
- ✅ **57 tests** - Integration setup/services/unload/discovery (100%)
- ✅ **133 tests** - Database operations & persistence (100%)
- ✅ **129 tests** - Sensor entities (100%)
- ✅ **37 tests** - Select entities (100%)
- ✅ **143 tests** - Task generation & verification (100%)
- ✅ **68 tests** - Multilingual support & translations (100%)

### Pre-Commit Hooks

To automatically run tests before every commit, install pre-commit hooks:

```bash
# Install pre-commit (already in requirements.txt)
pip install -r requirements.txt

# Install the git hook scripts
pre-commit install

# (Optional) Run against all files manually
pre-commit run --all-files
```

Once installed, tests will automatically run before each commit. If tests fail, the commit will be blocked until issues are fixed. This ensures code quality and prevents broken code from being committed.

📚 **[Full Testing Documentation](./tests/TESTING.md)** - Detailed test structure and CI/CD information

---

# 🤝 Contributing

We welcome contributions! Whether you want to:
- 🌍 Add translations for your language
- 🐛 Report bugs or suggest features
- 💻 Submit code improvements
- 📚 Improve documentation

**See [CONTRIBUTING.md](./docs/CONTRIBUTING.md) for detailed guidelines.**

Quick links:
- **Report Issues**: [GitHub Issues](https://github.com/RafaelNTeixeira/GreenShift/issues)
- **Translation Guide**: [docs/TRANSLATIONS.md](./docs/TRANSLATIONS.md)
- **Contact**: rafael2003t.18@gmail.com

---

**Happy Energy Saving! 🌱⚡**
