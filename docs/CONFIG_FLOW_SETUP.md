# Green Shift - Config Flow Setup Guide

This guide explains every screen and field shown when adding Green Shift in Home Assistant.

## Before you start

Complete these first:

1. Install Green Shift files and update `configuration.yaml`
2. Configure weather integration (recommended: Met.no)
3. Configure Workday integration (for office mode)
4. Create and organize your Home Assistant areas (Settings -> Areas, Labels & Zones -> Areas)
5. Restart Home Assistant

Then go to **Settings -> Devices & Services -> Add Integration -> Green Shift**.

---

## Step 1 - Welcome (`user`)

This is an informational screen.

- Action required: click **Next/Submit** to continue
- What happens: Green Shift starts automatic sensor discovery in your Home Assistant instance

---

## Step 2 - Settings (`settings`)

### `currency`

Currency used in savings/impact displays.

- Recommended: your local currency
- Allowed values: `EUR`, `USD`, `GBP`

### `electricity_price`

Energy price per kWh.

- Format: decimal number (example: `0.25`)
- Validation: must be `>= 0`
- Tip: use your contract average price if you have variable tariffs

### `environment_mode`

Defines which behaviour model to use.

- `home`: normal home usage profile
- `office`: office profile with working schedule support

If you choose `office`, the next screen (`working_hours`) will be shown.

### `has_ac`

Whether the space has air conditioning.

- `true`: enable temperature/comfort recommendations that assume AC is available
- `false`: avoid recommendations that require AC control

---

## Step 2.5 - Working Hours (`working_hours`) [office mode only]

Only shown when `environment_mode = office`.

### `working_start` and `working_end`

Office operating schedule.

- Required format: `HH:MM` (24h), example `09:00`
- Valid range: `00:00` to `23:59`

### Weekday checkboxes

Fields:

- `working_monday`
- `working_tuesday`
- `working_wednesday`
- `working_thursday`
- `working_friday`
- `working_saturday`
- `working_sunday`

Rule: at least one day must be selected.

Why this matters:

- Controls when office-related tasks can be generated
- Works together with `binary_sensor.workday_sensor` for holiday handling

---

## Step 3 - Sensor Confirmation (`sensor_confirmation`)

This is where you confirm discovered sensors and choose the main references.

All fields are optional, but better selections improve AI quality.

### Weather and outdoor temperature

#### `weather_entity` (recommended)

- Select a `weather.*` entity (for example `weather.home` from Met.no)
- Used for climate-aware task difficulty and degree-day research metrics (HDD/CDD)

#### `outdoor_temp_sensor` (optional fallback)

- Select a physical `sensor.*` temperature entity if available
- Useful when weather provider is unavailable or unstable

### Main aggregate sensors

#### `main_total_energy_sensor`

- Main whole-building energy sensor
- Measures cumulative energy (`kWh`)

#### `main_total_power_sensor`

- Main whole-building power sensor
- Measures instantaneous power (`W`)

Tip: choose the sensor that best represents the total property load.

### Multi-select sensor groups

- `confirmed_energy`: Sensors for appliance-level cumulative energy (`kWh`) readings
- `confirmed_power`: Sensors for appliance-level instantaneous power (`W`) readings
- `confirmed_temp`: Sensors for room temperature readings
- `confirmed_hum`: Sensors for room humidity readings
- `confirmed_lux`: Sensors for room luminosity readings
- `confirmed_occ`: Sensors for room occupation readings

Guidance:

- Keep sensors that are reliable and relevant
- Remove noisy/duplicated sensors when possible

---

## Step 4 - Area Assignment (`area_assignment`)

Assign selected sensors to Home Assistant areas.

- Main total energy/power sensors are excluded from area assignment (they represent whole-building scope)
- For other sensors, choose area where each sensor is physically located

Why this matters:

- Enables room/area-aware analysis
- Improves anomaly context and recommendation quality

---

## Step 5 - Final Info (`intervention_info`)

Informational confirmation step.

- Action required: submit to create entry
- Result: Green Shift entry is saved and baseline collection starts
