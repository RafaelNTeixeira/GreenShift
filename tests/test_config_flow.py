"""
Tests for config_flow.py

Covers:
- Multi-step config flow navigation
- Environment mode branching (home vs office)
- Sensor discovery and sorting
- Area assignment
- Input validation"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import types
import importlib
import pathlib

# Load real voluptuous and save the key callables BEFORE the mock loop can
# overwrite them on the same module object.
import importlib as _real_importlib
_real_vol = _real_importlib.import_module("voluptuous")
_real_Schema  = _real_vol.Schema
_real_All     = _real_vol.All
_real_Coerce  = _real_vol.Coerce
_real_Range   = _real_vol.Range
_real_Invalid = _real_vol.Invalid

# Mock Home Assistant modules before importing config_flow
for mod in [
    "homeassistant",
    "homeassistant.core",
    "homeassistant.config_entries",
    "homeassistant.helpers",
    "homeassistant.helpers.entity_registry",
    "homeassistant.helpers.device_registry",
    "homeassistant.helpers.area_registry",
    "homeassistant.helpers.selector",
    "voluptuous",
]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

# Mock voluptuous classes
vol_mock = sys.modules["voluptuous"]
vol_mock.Schema = lambda x: x
vol_mock.Required = lambda key, **kwargs: key
vol_mock.Optional = lambda key, **kwargs: key
vol_mock.Coerce = lambda x: x
vol_mock.All = lambda *args: args[-1]   # pass-through: returns last validator
vol_mock.Range = lambda **kwargs: None  # no-op range in test context

# Mock selector classes
selector_mock = sys.modules["homeassistant.helpers.selector"]
selector_mock.SelectSelector = MagicMock
selector_mock.SelectSelectorConfig = MagicMock
selector_mock.SelectSelectorMode = MagicMock()
selector_mock.SelectSelectorMode.DROPDOWN = "dropdown"
selector_mock.EntitySelector = MagicMock
selector_mock.EntitySelectorConfig = MagicMock
selector_mock.AreaSelector = MagicMock
selector_mock.AreaSelectorConfig = MagicMock

# Import const module
const_spec = importlib.util.spec_from_file_location(
    "custom_components.green_shift.const",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "const.py"
)
const_mod = importlib.util.module_from_spec(const_spec)
const_mod.__package__ = "custom_components.green_shift"
const_spec.loader.exec_module(const_mod)
sys.modules["custom_components.green_shift.const"] = const_mod

# Import helpers module
helpers_spec = importlib.util.spec_from_file_location(
    "custom_components.green_shift.helpers",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "helpers.py"
)
helpers_mod = importlib.util.module_from_spec(helpers_spec)
helpers_mod.__package__ = "custom_components.green_shift"
helpers_spec.loader.exec_module(helpers_mod)
sys.modules["custom_components.green_shift.helpers"] = helpers_mod

# Create a mock __init__ module for the package
init_mock = types.ModuleType("custom_components.green_shift")
init_mock.__package__ = "custom_components.green_shift"
init_mock.async_discover_sensors = AsyncMock(return_value={})  # Mock the discovery function
sys.modules["custom_components.green_shift"] = init_mock

# Import config_flow
config_flow_spec = importlib.util.spec_from_file_location(
    "custom_components.green_shift.config_flow",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "config_flow.py"
)
config_flow_mod = importlib.util.module_from_spec(config_flow_spec)
config_flow_mod.__package__ = "custom_components.green_shift"
config_flow_spec.loader.exec_module(config_flow_mod)
sys.modules["custom_components.green_shift.config_flow"] = config_flow_mod

GreenShiftConfigFlow = config_flow_mod.GreenShiftConfigFlow


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.states = MagicMock()
    hass.data = {}
    return hass


@pytest.fixture
def mock_entity_registry():
    """Create a mock entity registry."""
    registry = MagicMock()
    registry.async_get = MagicMock(return_value=None)
    registry.async_update_entity = MagicMock()
    return registry


@pytest.fixture
def discovered_sensors():
    """Sample discovered sensors."""
    return {
        "energy": ["sensor.energy_total", "sensor.energy_room1"],
        "power": ["sensor.power_total", "sensor.power_room1", "sensor.power_room2"],
        "temperature": ["sensor.temp_living", "sensor.temp_bedroom"],
        "humidity": ["sensor.hum_living"],
        "illuminance": ["sensor.lux_living"],
        "occupancy": ["binary_sensor.motion_living"],
    }


@pytest.fixture
def config_flow(mock_hass):
    """Create a config flow instance."""
    flow = GreenShiftConfigFlow()
    flow.hass = mock_hass
    return flow


# ============================================================================
# Step 1: User (Welcome)
# ============================================================================

class TestAsyncStepUser:
    """Test the initial welcome step."""

    async def test_shows_form_on_first_load(self, config_flow):
        """First visit shows the welcome form."""
        result = await config_flow.async_step_user()

        assert result["type"] == "form"
        assert result["step_id"] == "user"
        assert result["last_step"] is False

    async def test_proceeds_to_settings_after_submit(self, config_flow, discovered_sensors):
        """Submitting welcome proceeds to settings step."""
        with patch.object(config_flow_mod, "async_discover_sensors",
                         new_callable=AsyncMock, return_value=discovered_sensors):
            result = await config_flow.async_step_user(user_input={})

            assert result["type"] == "form"
            assert result["step_id"] == "settings"
            assert config_flow.discovered_cache == discovered_sensors


# ============================================================================
# Step 2: Settings
# ============================================================================

class TestAsyncStepSettings:
    """Test the settings configuration step."""

    async def test_shows_settings_form(self, config_flow):
        """Settings step shows form with currency and environment options."""
        result = await config_flow.async_step_settings()

        assert result["type"] == "form"
        assert result["step_id"] == "settings"
        assert result["last_step"] is False

    async def test_home_mode_skips_working_hours(self, config_flow):
        """Home mode skips working hours and goes to sensor confirmation."""
        user_input = {
            "currency": "EUR",
            "electricity_price": 0.25,
            "environment_mode": "home"
        }

        result = await config_flow.async_step_settings(user_input)

        assert result["type"] == "form"
        assert result["step_id"] == "sensor_confirmation"
        assert config_flow.data["environment_mode"] == "home"

    async def test_office_mode_proceeds_to_working_hours(self, config_flow):
        """Office mode proceeds to working hours configuration."""
        user_input = {
            "currency": "USD",
            "electricity_price": 0.30,
            "environment_mode": "office"
        }

        result = await config_flow.async_step_settings(user_input)

        assert result["type"] == "form"
        assert result["step_id"] == "working_hours"
        assert config_flow.data["environment_mode"] == "office"

    async def test_stores_user_input_in_data(self, config_flow):
        """Settings are stored in flow data."""
        user_input = {
            "currency": "GBP",
            "electricity_price": 0.28,
            "environment_mode": "home"
        }

        await config_flow.async_step_settings(user_input)

        assert config_flow.data["currency"] == "GBP"
        assert config_flow.data["electricity_price"] == 0.28

    # ---- electricity_price must be >= 0 ----
    async def test_zero_electricity_price_is_accepted(self, config_flow):
        """Price of 0.0 is a valid edge case and must proceed normally."""
        await config_flow.async_step_settings({
            "currency": "EUR",
            "electricity_price": 0.0,
            "environment_mode": "home",
        })
        # If 0.0 was accepted, the flow data must have been updated.
        # (Rejected inputs leave data untouched and re-show the form.)
        assert config_flow.data.get("electricity_price") == 0.0, (
            "electricity_price=0.0 should be accepted and persisted to flow data"
        )

    async def test_positive_electricity_price_is_accepted(self, config_flow):
        """A valid positive price must proceed to the next step."""
        result = await config_flow.async_step_settings({
            "currency": "EUR",
            "electricity_price": 0.25,
            "environment_mode": "home",
        })
        assert result["step_id"] != "settings"


# ============================================================================
# Step 2.5: Working Hours (Office Mode Only)
# ============================================================================

class TestAsyncStepWorkingHours:
    """Test working hours configuration for office mode."""

    async def test_shows_working_hours_form(self, config_flow):
        """Working hours step shows form with time and day options."""
        result = await config_flow.async_step_working_hours()

        assert result["type"] == "form"
        assert result["step_id"] == "working_hours"
        assert result["last_step"] is False

    async def test_stores_working_hours_configuration(self, config_flow):
        """Working hours input is stored correctly."""
        user_input = {
            "working_start": "09:00",
            "working_end": "17:00",
            "working_monday": True,
            "working_tuesday": True,
            "working_wednesday": True,
            "working_thursday": True,
            "working_friday": True,
            "working_saturday": False,
            "working_sunday": False,
        }

        result = await config_flow.async_step_working_hours(user_input)

        assert config_flow.data["working_start"] == "09:00"
        assert config_flow.data["working_end"] == "17:00"
        assert config_flow.data["working_monday"] is True
        assert config_flow.data["working_saturday"] is False

    async def test_proceeds_to_sensor_confirmation(self, config_flow):
        """After working hours, proceeds to sensor confirmation."""
        user_input = {
            "working_start": "08:00",
            "working_end": "18:00",
            "working_monday": True,
            "working_tuesday": True,
            "working_wednesday": True,
            "working_thursday": True,
            "working_friday": True,
            "working_saturday": False,
            "working_sunday": False,
        }

        result = await config_flow.async_step_working_hours(user_input)

        assert result["type"] == "form"
        assert result["step_id"] == "sensor_confirmation"

    # -- Bug fix #4: working hours time format validation ------------------

    async def test_invalid_working_start_format_returns_error(self, config_flow):
        """'8am' and other non-HH:MM strings must produce a form error, not proceed."""
        user_input = {
            "working_start": "8am",
            "working_end": "18:00",
            "working_monday": True,
            "working_tuesday": True,
            "working_wednesday": True,
            "working_thursday": True,
            "working_friday": True,
            "working_saturday": False,
            "working_sunday": False,
        }

        result = await config_flow.async_step_working_hours(user_input)

        assert result["type"] == "form", "Invalid time format must re-show the form"
        assert result["step_id"] == "working_hours"
        assert "working_start" in result.get("errors", {}), (
            "Expected 'working_start' error for non-HH:MM input '8am'"
        )

    async def test_invalid_working_end_format_returns_error(self, config_flow):
        """'6pm' must produce a form error for working_end."""
        user_input = {
            "working_start": "08:00",
            "working_end": "6pm",
            "working_monday": True,
            "working_tuesday": True,
            "working_wednesday": True,
            "working_thursday": True,
            "working_friday": True,
            "working_saturday": False,
            "working_sunday": False,
        }

        result = await config_flow.async_step_working_hours(user_input)

        assert result["type"] == "form"
        assert "working_end" in result.get("errors", {})

    async def test_valid_hhmm_format_is_accepted(self, config_flow):
        """HH:MM format must be accepted without errors."""
        for start, end in [("08:00", "18:00"), ("00:00", "23:59"), ("09:30", "17:45")]:
            result = await config_flow.async_step_working_hours({
                "working_start": start,
                "working_end": end,
                "working_monday": True,
                "working_tuesday": False,
                "working_wednesday": False,
                "working_thursday": False,
                "working_friday": False,
                "working_saturday": False,
                "working_sunday": False,
            })
            # HA async_show_form returns errors=None when none are set;
            # a truthy errors dict means real validation errors exist.
            assert not result.get("errors"), (
                f"HH:MM values '{start}'/'{end}' must not produce errors, got: {result.get('errors')}"
            )

    async def test_invalid_time_does_not_update_flow_data(self, config_flow):
        """When time format is invalid, flow data must not be updated."""
        config_flow.data = {}
        user_input = {
            "working_start": "not-a-time",
            "working_end": "18:00",
            "working_monday": True,
            "working_tuesday": True,
            "working_wednesday": True,
            "working_thursday": True,
            "working_friday": True,
            "working_saturday": False,
            "working_sunday": False,
        }

        await config_flow.async_step_working_hours(user_input)

        assert "working_start" not in config_flow.data, (
            "Invalid working_start must not be persisted to flow data"
        )

    async def test_no_working_days_selected_returns_error(self, config_flow):
        """Submitting with all days unchecked must re-show the form with a no_working_days error."""
        user_input = {
            "working_start": "09:00",
            "working_end": "17:00",
            "working_monday": False,
            "working_tuesday": False,
            "working_wednesday": False,
            "working_thursday": False,
            "working_friday": False,
            "working_saturday": False,
            "working_sunday": False,
        }

        result = await config_flow.async_step_working_hours(user_input)

        assert result["type"] == "form", "No working days must re-show the form"
        assert result["step_id"] == "working_hours"
        assert "base" in result.get("errors", {}), (
            "Expected 'base' error when all day checkboxes are False"
        )
        assert result["errors"]["base"] == "no_working_days"

    async def test_single_working_day_is_accepted(self, config_flow):
        """A single working day selected must pass validation and proceed."""
        user_input = {
            "working_start": "09:00",
            "working_end": "17:00",
            "working_monday": True,
            "working_tuesday": False,
            "working_wednesday": False,
            "working_thursday": False,
            "working_friday": False,
            "working_saturday": False,
            "working_sunday": False,
        }

        result = await config_flow.async_step_working_hours(user_input)

        assert "base" not in (result.get("errors") or {}), (
            "A single selected day must not trigger the no_working_days error"
        )

    # -- Improved time regex rejects impossible hour/minute values --

    @pytest.mark.parametrize("bad_time", [
        "25:00",   # hour > 23
        "24:00",   # hour > 23
        "00:60",   # minute > 59
        "23:60",   # minute > 59
        "99:99",   # both invalid
    ])
    async def test_impossible_time_values_rejected(self, config_flow, bad_time):
        """Syntactically valid-looking but semantically invalid times must be rejected."""
        user_input = {
            "working_start": bad_time,
            "working_end": "18:00",
            "working_monday": True,
            "working_tuesday": False,
            "working_wednesday": False,
            "working_thursday": False,
            "working_friday": False,
            "working_saturday": False,
            "working_sunday": False,
        }
        result = await config_flow.async_step_working_hours(user_input)
        assert result["type"] == "form", f"Time '{bad_time}' should be rejected"
        assert "working_start" in result.get("errors", {}), (
            f"Expected 'working_start' error for impossible time '{bad_time}'"
        )

    @pytest.mark.parametrize("good_time", [
        "00:00",
        "23:59",
        "09:30",
        "12:00",
        "20:45",
    ])
    async def test_valid_boundary_times_accepted(self, config_flow, good_time):
        """Valid boundary times must not produce an invalid_time_format error."""
        user_input = {
            "working_start": good_time,
            "working_end": "23:59",
            "working_monday": True,
            "working_tuesday": False,
            "working_wednesday": False,
            "working_thursday": False,
            "working_friday": False,
            "working_saturday": False,
            "working_sunday": False,
        }
        result = await config_flow.async_step_working_hours(user_input)
        assert "working_start" not in (result.get("errors") or {}), (
            f"Valid time '{good_time}' must not produce a format error"
        )


# ============================================================================
# Step 3: Sensor Confirmation
# ============================================================================

class TestAsyncStepSensorConfirmation:
    """Test sensor discovery and confirmation."""

    async def test_shows_sensor_confirmation_form(self, config_flow, discovered_sensors):
        """Sensor confirmation step shows form with discovered sensors."""
        config_flow.discovered_cache = discovered_sensors

        result = await config_flow.async_step_sensor_confirmation()

        assert result["type"] == "form"
        assert result["step_id"] == "sensor_confirmation"
        assert result["last_step"] is False

    async def test_sorts_energy_sensors_by_value(self, config_flow, mock_hass):
        """Energy sensors are sorted by current value (highest first)."""
        config_flow.hass = mock_hass
        config_flow.discovered_cache = {
            "energy": ["sensor.energy_low", "sensor.energy_high", "sensor.energy_mid"]
        }

        # Mock sensor states with different values
        def get_state(entity_id):
            state = MagicMock()
            state.state = "10.0" if "low" in entity_id else "100.0" if "high" in entity_id else "50.0"
            state.attributes = {"unit_of_measurement": "kWh"}
            return state

        mock_hass.states.get = get_state

        sorted_entities = config_flow._get_sorted_entities("energy")

        assert sorted_entities[0] == "sensor.energy_high"
        assert sorted_entities[1] == "sensor.energy_mid"
        assert sorted_entities[2] == "sensor.energy_low"

    async def test_sorts_power_sensors_by_value(self, config_flow, mock_hass):
        """Power sensors are sorted by current value (highest first)."""
        config_flow.hass = mock_hass
        config_flow.discovered_cache = {
            "power": ["sensor.power_100w", "sensor.power_500w", "sensor.power_200w"]
        }

        def get_state(entity_id):
            state = MagicMock()
            state.state = "100" if "100w" in entity_id else "500" if "500w" in entity_id else "200"
            state.attributes = {"unit_of_measurement": "W"}
            return state

        mock_hass.states.get = get_state

        sorted_entities = config_flow._get_sorted_entities("power")

        assert sorted_entities[0] == "sensor.power_500w"
        assert sorted_entities[1] == "sensor.power_200w"
        assert sorted_entities[2] == "sensor.power_100w"

    async def test_handles_unavailable_sensors_in_sorting(self, config_flow, mock_hass):
        """Unavailable sensors are placed at end of sorted list."""
        config_flow.hass = mock_hass
        config_flow.discovered_cache = {
            "power": ["sensor.available", "sensor.unavailable"]
        }

        def get_state(entity_id):
            state = MagicMock()
            if "unavailable" in entity_id:
                state.state = "unavailable"
            else:
                state.state = "100"
            state.attributes = {"unit_of_measurement": "W"}
            return state

        mock_hass.states.get = get_state

        sorted_entities = config_flow._get_sorted_entities("power")

        # Available sensor should be first
        assert sorted_entities[0] == "sensor.available"

    async def test_stores_confirmed_sensors(self, config_flow):
        """User-confirmed sensors are stored in flow data."""
        user_input = {
            "main_total_energy_sensor": "sensor.energy_total",
            "main_total_power_sensor": "sensor.power_total",
            "confirmed_energy": ["sensor.energy_total", "sensor.energy_room1"],
            "confirmed_power": ["sensor.power_total"],
            "confirmed_temp": ["sensor.temp_living"],
            "confirmed_hum": [],
            "confirmed_lux": [],
            "confirmed_occ": ["binary_sensor.motion_living"],
        }

        result = await config_flow.async_step_sensor_confirmation(user_input)

        assert config_flow.data["main_total_energy_sensor"] == "sensor.energy_total"
        assert config_flow.data["main_total_power_sensor"] == "sensor.power_total"
        assert "sensor.energy_total" in config_flow.data["discovered_sensors"]["energy"]
        assert "sensor.temp_living" in config_flow.data["discovered_sensors"]["temperature"]

    async def test_injects_main_sensors_into_lists(self, config_flow):
        """Main sensors are automatically added to respective lists if missing."""
        user_input = {
            "main_total_energy_sensor": "sensor.energy_main",
            "main_total_power_sensor": "sensor.power_main",
            "confirmed_energy": ["sensor.energy_other"],  # Main not in list
            "confirmed_power": [],  # Empty list
            "confirmed_temp": [],
            "confirmed_hum": [],
            "confirmed_lux": [],
            "confirmed_occ": [],
        }

        await config_flow.async_step_sensor_confirmation(user_input)

        # Main sensors should be injected
        assert "sensor.energy_main" in config_flow.data["discovered_sensors"]["energy"]
        assert "sensor.power_main" in config_flow.data["discovered_sensors"]["power"]

    async def test_proceeds_to_area_assignment(self, config_flow):
        """After sensor confirmation, proceeds to area assignment."""
        user_input = {
            "confirmed_energy": ["sensor.energy_total"],
            "confirmed_power": [],
            "confirmed_temp": [],
            "confirmed_hum": [],
            "confirmed_lux": [],
            "confirmed_occ": [],
        }

        result = await config_flow.async_step_sensor_confirmation(user_input)

        assert result["type"] == "form"
        assert result["step_id"] == "area_assignment"


# ============================================================================
# Step 4: Area Assignment
# ============================================================================

class TestAsyncStepAreaAssignment:
    """Test area assignment for sensors."""

    async def test_excludes_main_sensors_from_area_assignment(self, config_flow, mock_entity_registry):
        """Main energy and power sensors should not be assigned to areas."""
        config_flow.data = {
            "main_total_energy_sensor": "sensor.energy_main",
            "main_total_power_sensor": "sensor.power_main",
            "discovered_sensors": {
                "energy": ["sensor.energy_main", "sensor.energy_room"],
                "power": ["sensor.power_main", "sensor.power_room"],
                "temperature": [],
                "humidity": [],
                "illuminance": [],
                "occupancy": [],
            }
        }

        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get", return_value=mock_entity_registry):
            with patch.object(helpers_mod, "get_entity_area_id", return_value=None):
                result = await config_flow.async_step_area_assignment()

                # Should only show non-main sensors in schema
                assert result["type"] == "form"
                assert result["step_id"] == "area_assignment"

    async def test_updates_entity_registry_with_areas(self, config_flow, mock_entity_registry):
        """Area selections update the entity registry."""
        config_flow.data = {
            "discovered_sensors": {
                "energy": ["sensor.energy_room"],
                "power": [],
                "temperature": [],
                "humidity": [],
                "illuminance": [],
                "occupancy": [],
            }
        }

        user_input = {
            "sensor.energy_room": "area_living_room"
        }

        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get", return_value=mock_entity_registry):
            with patch.object(helpers_mod, "get_entity_area_id", return_value=None):
                await config_flow.async_step_area_assignment(user_input)

        # Verify entity registry was updated
        mock_entity_registry.async_update_entity.assert_called_with(
            "sensor.energy_room",
            area_id="area_living_room"
        )

    async def test_skips_empty_area_selections(self, config_flow, mock_entity_registry):
        """Empty area selections are not applied."""
        config_flow.data = {
            "discovered_sensors": {
                "power": ["sensor.power_device"],
                "energy": [],
                "temperature": [],
                "humidity": [],
                "illuminance": [],
                "occupancy": [],
            }
        }

        user_input = {
            "sensor.power_device": None  # No area selected
        }

        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get", return_value=mock_entity_registry):
            with patch.object(helpers_mod, "get_entity_area_id", return_value=None):
                await config_flow.async_step_area_assignment(user_input)

        # Should not update registry for None values
        mock_entity_registry.async_update_entity.assert_not_called()

    async def test_proceeds_to_intervention_info(self, config_flow):
        """After area assignment, proceeds to final info step."""
        config_flow.data = {
            "discovered_sensors": {
                "energy": [],
                "power": [],
                "temperature": [],
                "humidity": [],
                "illuminance": [],
                "occupancy": [],
            }
        }

        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get", return_value=MagicMock()):
            with patch.object(helpers_mod, "get_entity_area_id", return_value=None):
                result = await config_flow.async_step_area_assignment({})

        assert result["type"] == "form"
        assert result["step_id"] == "intervention_info"


# ============================================================================
# Step 5: Intervention Info (Final)
# ============================================================================

class TestAsyncStepInterventionInfo:
    """Test the final information step."""

    async def test_shows_info_form(self, config_flow):
        """Info step shows final form."""
        result = await config_flow.async_step_intervention_info()

        assert result["type"] == "form"
        assert result["step_id"] == "intervention_info"
        assert result["last_step"] is True

    async def test_creates_config_entry(self, config_flow):
        """Submitting info step creates the config entry."""
        config_flow.data = {
            "currency": "EUR",
            "electricity_price": 0.25,
            "environment_mode": "home",
            "discovered_sensors": {"energy": [], "power": []},
        }

        result = await config_flow.async_step_intervention_info(user_input={})

        assert result["type"] == "create_entry"
        assert result["title"] == "Green Shift"
        assert result["data"] == config_flow.data


# ============================================================================
# Integration Tests
# ============================================================================

class TestConfigFlowIntegration:
    """Test complete flows from start to finish."""

    async def test_complete_home_mode_flow(self, mock_hass, discovered_sensors):
        """Complete config flow for home mode."""
        flow = GreenShiftConfigFlow()
        flow.hass = mock_hass

        # Step 1: Welcome
        with patch.object(config_flow_mod, "async_discover_sensors",
                         new_callable=AsyncMock, return_value=discovered_sensors):
            result = await flow.async_step_user(user_input={})

        # Step 2: Settings (home mode)
        result = await flow.async_step_settings({
            "currency": "EUR",
            "electricity_price": 0.25,
            "environment_mode": "home"
        })

        # Should skip working hours and go to sensor confirmation
        assert result["step_id"] == "sensor_confirmation"

        # Step 3: Sensor confirmation
        result = await flow.async_step_sensor_confirmation({
            "confirmed_energy": ["sensor.energy_total"],
            "confirmed_power": ["sensor.power_total"],
            "confirmed_temp": [],
            "confirmed_hum": [],
            "confirmed_lux": [],
            "confirmed_occ": [],
        })

        # Step 4: Area assignment
        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get", return_value=MagicMock()):
            with patch.object(helpers_mod, "get_entity_area_id", return_value=None):
                result = await flow.async_step_area_assignment({})

        # Step 5: Final info
        result = await flow.async_step_intervention_info({})

        assert result["type"] == "create_entry"
        assert flow.data["environment_mode"] == "home"

    async def test_complete_office_mode_flow(self, mock_hass, discovered_sensors):
        """Complete config flow for office mode with working hours."""
        flow = GreenShiftConfigFlow()
        flow.hass = mock_hass

        # Step 1: Welcome
        with patch.object(config_flow_mod, "async_discover_sensors",
                         new_callable=AsyncMock, return_value=discovered_sensors):
            result = await flow.async_step_user(user_input={})

        # Step 2: Settings (office mode)
        result = await flow.async_step_settings({
            "currency": "USD",
            "electricity_price": 0.30,
            "environment_mode": "office"
        })

        # Should proceed to working hours
        assert result["step_id"] == "working_hours"

        # Step 2.5: Working hours
        result = await flow.async_step_working_hours({
            "working_start": "09:00",
            "working_end": "17:00",
            "working_monday": True,
            "working_tuesday": True,
            "working_wednesday": True,
            "working_thursday": True,
            "working_friday": True,
            "working_saturday": False,
            "working_sunday": False,
        })

        # Should proceed to sensor confirmation
        assert result["step_id"] == "sensor_confirmation"

        # Step 3: Sensor confirmation
        result = await flow.async_step_sensor_confirmation({
            "confirmed_energy": [],
            "confirmed_power": [],
            "confirmed_temp": [],
            "confirmed_hum": [],
            "confirmed_lux": [],
            "confirmed_occ": [],
        })

        # Step 4: Area assignment
        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get", return_value=MagicMock()):
            with patch.object(helpers_mod, "get_entity_area_id", return_value=None):
                result = await flow.async_step_area_assignment({})

        # Step 5: Final info
        result = await flow.async_step_intervention_info({})

        assert result["type"] == "create_entry"
        assert flow.data["environment_mode"] == "office"
        assert flow.data["working_start"] == "09:00"
        assert flow.data["working_friday"] is True


# ============================================================================
# Additional edge-case tests for uncovered lines
# ============================================================================

class TestElectricityPriceValidation:
    """Test that Voluptuous schema rejects invalid electricity price values.

    In production HA validates user_input against the vol.Schema defined in
    async_step_settings before the method body runs.  The schema uses
    vol.All(vol.Coerce(float), vol.Range(min=0)) which raises vol.Invalid for
    non-numeric or negative inputs before async_step_settings is ever called.
    These tests use _real_Schema/_real_Invalid etc. saved before the mock
    overwrites them on the shared module object.
    """

    def test_zero_is_on_the_acceptable_boundary(self, config_flow):
        """Sanity check: 0.0 is valid (== min=0 in Range)."""
        schema = _real_Schema(_real_All(_real_Coerce(float), _real_Range(min=0)))
        assert schema(0.0) == 0.0

    def test_negative_price_is_below_range_minimum(self, config_flow):
        """vol.Range(min=0) must reject values below 0."""
        schema = _real_Schema(_real_All(_real_Coerce(float), _real_Range(min=0)))
        with pytest.raises(_real_Invalid):
            schema(-0.10)

    def test_non_numeric_electricity_price_rejected_by_schema(self, config_flow):
        """A non-numeric string cannot be coerced to float; vol.Coerce raises Invalid."""
        schema = _real_Schema(_real_All(_real_Coerce(float), _real_Range(min=0)))
        with pytest.raises(_real_Invalid):
            schema("free")

    def test_none_electricity_price_rejected_by_schema(self, config_flow):
        """None cannot be coerced to float; vol.Coerce raises Invalid."""
        schema = _real_Schema(_real_All(_real_Coerce(float), _real_Range(min=0)))
        with pytest.raises((_real_Invalid, TypeError)):
            schema(None)



    """Test _get_weather_entities helper."""

    def test_returns_list_of_weather_entity_ids(self, config_flow, mock_hass):
        """Should iterate hass.states.async_all('weather') and collect entity IDs."""
        mock_state_1 = MagicMock()
        mock_state_1.entity_id = "weather.home"
        mock_state_2 = MagicMock()
        mock_state_2.entity_id = "weather.office"

        mock_hass.states.async_all = MagicMock(return_value=[mock_state_1, mock_state_2])
        config_flow.hass = mock_hass

        result = config_flow._get_weather_entities()

        assert "weather.home" in result
        assert "weather.office" in result
        assert len(result) == 2

    def test_returns_empty_list_when_no_weather_entities(self, config_flow, mock_hass):
        """When there are no weather entities, must return an empty list."""
        mock_hass.states.async_all = MagicMock(return_value=[])
        config_flow.hass = mock_hass

        result = config_flow._get_weather_entities()

        assert result == []


class TestAreaAssignmentEdgeCases:
    """Edge cases for area assignment step."""

    async def test_area_assignment_proceeds_despite_update_entity_exception(self, config_flow, mock_entity_registry):
        """If async_update_entity raises, the exception is logged but flow still proceeds."""
        config_flow.data = {
            "discovered_sensors": {
                "power": ["sensor.power_room"],
                "energy": [], "temperature": [], "humidity": [],
                "illuminance": [], "occupancy": [],
            }
        }

        # Make update_entity raise on every call
        mock_entity_registry.async_update_entity = MagicMock(side_effect=Exception("registry error"))

        user_input = {"sensor.power_room": "area_bedroom"}

        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get",
                          return_value=mock_entity_registry):
            with patch.object(helpers_mod, "get_entity_area_id", return_value=None):
                result = await config_flow.async_step_area_assignment(user_input)

        # Despite the error, flow proceeds to the next step
        assert result["type"] == "form"
        assert result["step_id"] == "intervention_info"

    async def test_area_assignment_shows_existing_area_as_default(self, config_flow, mock_entity_registry):
        """Sensors that already have an area should show it as default in the schema."""
        config_flow.data = {
            "discovered_sensors": {
                "power": ["sensor.power_room"],
                "energy": [], "temperature": [], "humidity": [],
                "illuminance": [], "occupancy": [],
            }
        }

        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get",
                          return_value=mock_entity_registry):
            # Sensor already assigned to "area_living_room"
            with patch.object(helpers_mod, "get_entity_area_id", return_value="area_living_room"):
                result = await config_flow.async_step_area_assignment()

        # Form is shown without error
        assert result["type"] == "form"
        assert result["step_id"] == "area_assignment"

    async def test_area_assignment_skips_entity_with_schema_exception(self, config_flow, mock_entity_registry):
        """If building a sensor's schema entry raises, it logs and skips that entity."""
        config_flow.data = {
            "discovered_sensors": {
                "power": ["sensor.bad_entity"],
                "energy": [], "temperature": [], "humidity": [],
                "illuminance": [], "occupancy": [],
            }
        }

        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get",
                          return_value=mock_entity_registry):
            with patch.object(helpers_mod, "get_entity_area_id",
                               side_effect=Exception("registry crash")):
                # Should not raise; bad entity is skipped
                result = await config_flow.async_step_area_assignment()

        assert result["type"] == "form"
        assert result["step_id"] == "area_assignment"

    async def test_area_assignment_builds_optional_selector_without_default(self, config_flow, mock_entity_registry):
        """When no current area exists, schema must still include an optional selector entry."""
        config_flow.data = {
            "discovered_sensors": {
                "power": ["sensor.power_room"],
                "energy": [], "temperature": [], "humidity": [],
                "illuminance": [], "occupancy": [],
            }
        }

        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get",
                          return_value=mock_entity_registry):
            with patch.object(config_flow_mod, "get_entity_area_id", return_value=None):
                result = await config_flow.async_step_area_assignment()

        assert result["type"] == "form"
        assert result["step_id"] == "area_assignment"

    async def test_area_assignment_builds_optional_selector_with_default(self, config_flow, mock_entity_registry):
        """When an entity already has area_id, schema entry should be created with default."""
        config_flow.data = {
            "discovered_sensors": {
                "power": ["sensor.power_room"],
                "energy": [], "temperature": [], "humidity": [],
                "illuminance": [], "occupancy": [],
            }
        }

        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get",
                          return_value=mock_entity_registry):
            with patch.object(config_flow_mod, "get_entity_area_id", return_value="area_kitchen"):
                result = await config_flow.async_step_area_assignment()

        assert result["type"] == "form"
        assert result["step_id"] == "area_assignment"


# ============================================================================
# Outdoor Temperature Sensor Field
# ============================================================================

class TestOutdoorTempSensorField:
    """Tests for the outdoor_temp_sensor field added to sensor_confirmation."""

    async def test_outdoor_temp_sensor_saved_when_provided(self, config_flow):
        """outdoor_temp_sensor is persisted in flow data when user provides it.""" 
        user_input = {
            "outdoor_temp_sensor": "sensor.outside_temp",
            "confirmed_energy": [],
            "confirmed_power": [],
            "confirmed_temp": [],
            "confirmed_hum": [],
            "confirmed_lux": [],
            "confirmed_occ": [],
        }

        await config_flow.async_step_sensor_confirmation(user_input)

        assert config_flow.data["outdoor_temp_sensor"] == "sensor.outside_temp"

    async def test_outdoor_temp_sensor_is_none_when_not_provided(self, config_flow):
        """outdoor_temp_sensor defaults to None when the user leaves it empty."""
        user_input = {
            "confirmed_energy": [],
            "confirmed_power": [],
            "confirmed_temp": [],
            "confirmed_hum": [],
            "confirmed_lux": [],
            "confirmed_occ": [],
        }

        await config_flow.async_step_sensor_confirmation(user_input)

        assert config_flow.data.get("outdoor_temp_sensor") is None

    async def test_outdoor_temp_sensor_field_appears_in_form(self, config_flow, discovered_sensors):
        """The sensor_confirmation form is shown without error even when no sensors are cached."""
        config_flow.discovered_cache = discovered_sensors
        config_flow.hass.states.get = MagicMock(return_value=None)

        result = await config_flow.async_step_sensor_confirmation()

        assert result["type"] == "form"
        assert result["step_id"] == "sensor_confirmation"

    async def test_outdoor_temp_sensor_independent_of_weather_entity(self, config_flow):
        """outdoor_temp_sensor and weather_entity can both be set independently."""
        user_input = {
            "weather_entity": "weather.home",
            "outdoor_temp_sensor": "sensor.balcony_temp",
            "confirmed_energy": [],
            "confirmed_power": [],
            "confirmed_temp": [],
            "confirmed_hum": [],
            "confirmed_lux": [],
            "confirmed_occ": [],
        }

        await config_flow.async_step_sensor_confirmation(user_input)

        assert config_flow.data["weather_entity"] == "weather.home"
        assert config_flow.data["outdoor_temp_sensor"] == "sensor.balcony_temp"

    async def test_outdoor_temp_sensor_not_added_to_indoor_temperature_list(self, config_flow):
        """outdoor_temp_sensor must NOT be injected into the indoor temperature sensor list."""
        user_input = {
            "outdoor_temp_sensor": "sensor.outside_temp",
            "confirmed_energy": [],
            "confirmed_power": [],
            "confirmed_temp": ["sensor.living_room_temp"],
            "confirmed_hum": [],
            "confirmed_lux": [],
            "confirmed_occ": [],
        }

        await config_flow.async_step_sensor_confirmation(user_input)

        indoor_temps = config_flow.data["discovered_sensors"]["temperature"]
        assert "sensor.outside_temp" not in indoor_temps
        assert "sensor.living_room_temp" in indoor_temps


# ============================================================================
# Options Flow (Post-setup sensor/area management)
# ============================================================================

class TestOptionsFlowHelpers:
    """Cover helper utilities and options flow bootstrap."""

    def test_merge_unique_preserves_order_and_deduplicates(self):
        merged = config_flow_mod._merge_unique(["a", "b", "a"], ["b", "c", None, ""])
        assert merged == ["a", "b", "c"]

    def test_async_get_options_flow_returns_options_flow_instance(self, config_flow):
        entry = MagicMock()
        entry.data = {}
        entry.options = {}

        options_flow = config_flow.async_get_options_flow(entry)

        assert isinstance(options_flow, config_flow_mod.GreenShiftOptionsFlow)


class TestOptionsFlowSteps:
    """Cover options flow steps and branches."""

    @pytest.fixture
    def options_flow(self, mock_hass):
        entry = MagicMock()
        entry.data = {
            "main_total_energy_sensor": "sensor.energy_main",
            "main_total_power_sensor": "sensor.power_main",
            "weather_entity": "weather.home",
            "outdoor_temp_sensor": "sensor.outdoor_temp",
            "discovered_sensors": {
                "energy": ["sensor.energy_main", "sensor.energy_old"],
                "power": ["sensor.power_main", "sensor.power_old"],
                "temperature": ["sensor.temp_old"],
                "humidity": ["sensor.hum_old"],
                "illuminance": ["sensor.lux_old"],
                "occupancy": ["binary_sensor.occ_old"],
            },
        }
        entry.options = {
            "discovered_sensors": {
                "energy": ["sensor.energy_opt"],
                "power": ["sensor.power_opt"],
                "temperature": [],
                "humidity": [],
                "illuminance": [],
                "occupancy": [],
            }
        }

        flow = config_flow_mod.GreenShiftOptionsFlow(entry)
        flow.hass = mock_hass
        return flow

    async def test_step_init_discovers_and_merges_sensors(self, options_flow, discovered_sensors):
        with patch.object(config_flow_mod, "async_discover_sensors", new_callable=AsyncMock, return_value=discovered_sensors):
            result = await options_flow.async_step_init()

        assert result["type"] == "form"
        assert result["step_id"] == "sensor_management"
        # Keeps previous sensors and includes discovered ones
        assert "sensor.energy_opt" in options_flow.discovered_cache["energy"]
        assert "sensor.energy_total" in options_flow.discovered_cache["energy"]

    async def test_sorted_entities_returns_empty_for_unknown_category(self, options_flow):
        options_flow.discovered_cache = {"power": []}
        assert options_flow._get_sorted_entities("energy") == []

    async def test_sorted_entities_orders_values_descending(self, options_flow, mock_hass):
        options_flow.hass = mock_hass
        options_flow.discovered_cache = {"power": ["sensor.p1", "sensor.p2"]}

        def get_state(entity_id):
            state = MagicMock()
            state.state = "10" if entity_id == "sensor.p1" else "100"
            state.attributes = {"unit_of_measurement": "W"}
            return state

        mock_hass.states.get = get_state
        assert options_flow._get_sorted_entities("power") == ["sensor.p2", "sensor.p1"]

    async def test_get_weather_entities_from_hass_state_machine(self, options_flow, mock_hass):
        state_1 = MagicMock()
        state_1.entity_id = "weather.home"
        state_2 = MagicMock()
        state_2.entity_id = "weather.office"
        mock_hass.states.async_all = MagicMock(return_value=[state_1, state_2])

        result = options_flow._get_weather_entities()
        assert result == ["weather.home", "weather.office"]

    async def test_sensor_management_show_form_uses_existing_defaults(self, options_flow, mock_hass):
        mock_hass.states.async_all = MagicMock(return_value=[])
        mock_hass.states.get = MagicMock(return_value=None)
        options_flow.discovered_cache = {
            "energy": ["sensor.energy_main"],
            "power": ["sensor.power_main"],
            "temperature": ["sensor.temp_old"],
            "humidity": ["sensor.hum_old"],
            "illuminance": ["sensor.lux_old"],
            "occupancy": ["binary_sensor.occ_old"],
        }

        result = await options_flow.async_step_sensor_management()

        assert result["type"] == "form"
        assert result["step_id"] == "sensor_management"

    async def test_sensor_management_submit_builds_options_and_injects_main_sensors(self, options_flow):
        user_input = {
            "main_total_energy_sensor": "sensor.energy_main",
            "main_total_power_sensor": "sensor.power_main",
            "weather_entity": "weather.home",
            "outdoor_temp_sensor": "sensor.outdoor_temp",
            "confirmed_energy": ["sensor.energy_room"],
            "confirmed_power": [],
            "confirmed_temp": ["sensor.temp_room"],
            "confirmed_hum": [],
            "confirmed_lux": [],
            "confirmed_occ": [],
        }

        result = await options_flow.async_step_sensor_management(user_input)

        assert result["type"] == "form"
        assert result["step_id"] == "area_assignment"
        assert "sensor.energy_main" in options_flow.options_data["discovered_sensors"]["energy"]
        assert "sensor.power_main" in options_flow.options_data["discovered_sensors"]["power"]

    async def test_options_area_assignment_show_form_handles_defaults_and_exceptions(self, options_flow, mock_entity_registry):
        options_flow.options_data = {
            "main_total_energy_sensor": "sensor.energy_main",
            "main_total_power_sensor": "sensor.power_main",
            "discovered_sensors": {
                "energy": ["sensor.energy_main", "sensor.energy_room"],
                "power": ["sensor.power_main", "sensor.power_room"],
                "temperature": ["sensor.bad_entity"],
                "humidity": [],
                "illuminance": [],
                "occupancy": [],
            },
        }

        def area_side_effect(hass, entity_id):
            if entity_id == "sensor.bad_entity":
                raise Exception("boom")
            if entity_id == "sensor.energy_room":
                return "area_kitchen"
            return None

        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get", return_value=mock_entity_registry):
            with patch.object(config_flow_mod, "get_entity_area_id", side_effect=area_side_effect):
                result = await options_flow.async_step_area_assignment()

        assert result["type"] == "form"
        assert result["step_id"] == "area_assignment"

    async def test_options_area_assignment_submit_updates_registry_and_creates_entry(self, options_flow, mock_entity_registry):
        options_flow.options_data = {
            "main_total_energy_sensor": "sensor.energy_main",
            "main_total_power_sensor": "sensor.power_main",
            "discovered_sensors": {
                "energy": ["sensor.energy_main", "sensor.energy_room"],
                "power": ["sensor.power_main", "sensor.power_room"],
                "temperature": [],
                "humidity": [],
                "illuminance": [],
                "occupancy": [],
            },
        }

        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get", return_value=mock_entity_registry):
            result = await options_flow.async_step_area_assignment(
                {
                    "sensor.energy_room": "area_living",
                    "sensor.power_room": None,
                }
            )

        assert result["type"] == "create_entry"
        mock_entity_registry.async_update_entity.assert_called_once_with("sensor.energy_room", area_id="area_living")

    async def test_options_area_assignment_submit_logs_update_exception_and_still_creates_entry(self, options_flow, mock_entity_registry):
        options_flow.options_data = {
            "main_total_energy_sensor": None,
            "main_total_power_sensor": None,
            "discovered_sensors": {
                "energy": ["sensor.energy_room"],
                "power": [],
                "temperature": [],
                "humidity": [],
                "illuminance": [],
                "occupancy": [],
            },
        }
        mock_entity_registry.async_update_entity = MagicMock(side_effect=Exception("registry failed"))

        with patch.object(sys.modules["homeassistant.helpers.entity_registry"], "async_get", return_value=mock_entity_registry):
            result = await options_flow.async_step_area_assignment({"sensor.energy_room": "area_x"})

        assert result["type"] == "create_entry"
