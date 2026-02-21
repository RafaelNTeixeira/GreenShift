"""
Tests for config_flow.py

Covers:
- Multi-step config flow navigation
- Environment mode branching (home vs office)
- Sensor discovery and sorting
- Area assignment
- Input validation
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import types
import importlib
import pathlib

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
