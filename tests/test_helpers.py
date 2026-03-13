"""
Tests for helpers.py

Covers:
- get_normalized_value   : unit conversions (power kW->W, energy Wh->kWh)
- get_environmental_impact: CO2 / metaphor calculations
- get_working_days_from_config: config dict -> list of day indices
- is_within_working_hours: home always active; office respects days & times
- should_ai_be_active    : thin wrapper check
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

# We import only the pure helpers that have no mandatory HA dependency
import sys, types

# Provide lightweight stubs for the HA imports inside helpers.py
for mod in [
    "homeassistant",
    "homeassistant.core",
    "homeassistant.helpers",
    "homeassistant.helpers.area_registry",
    "homeassistant.helpers.entity_registry",
    "homeassistant.helpers.device_registry",
]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

# Import real const module instead of stubbing
import importlib, pathlib
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
    "helpers",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "helpers.py"
)
helpers = importlib.util.module_from_spec(helpers_spec)
# Patch the relative import that helpers.py uses
helpers.__package__ = "custom_components.green_shift"
helpers_spec.loader.exec_module(helpers)

get_normalized_value = helpers.get_normalized_value
get_environmental_impact = helpers.get_environmental_impact
get_working_days_from_config = helpers.get_working_days_from_config
is_within_working_hours = helpers.is_within_working_hours
should_ai_be_active = helpers.should_ai_be_active


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_state(value, unit):
    s = MagicMock()
    s.state = str(value)
    s.attributes = {"unit_of_measurement": unit}
    return s


@pytest.fixture
def office_cfg():
    return {
        "environment_mode": "office",
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


@pytest.fixture
def home_cfg():
    return {"environment_mode": "home"}


# ─────────────────────────────────────────────────────────────────────────────
# get_normalized_value
# ─────────────────────────────────────────────────────────────────────────────

class TestGetNormalizedValue:

    def test_power_watts_passthrough(self):
        val, unit = get_normalized_value(make_state(500, "W"), "power")
        assert val == 500.0
        assert unit == "W"

    def test_power_kw_converted_to_watts(self):
        val, unit = get_normalized_value(make_state(1.5, "kW"), "power")
        assert val == pytest.approx(1500.0)
        assert unit == "W"

    def test_energy_kwh_passthrough(self):
        val, unit = get_normalized_value(make_state(12.5, "kWh"), "energy")
        assert val == pytest.approx(12.5)
        assert unit == "kWh"

    def test_energy_wh_converted_to_kwh(self):
        val, unit = get_normalized_value(make_state(2500, "Wh"), "energy")
        assert val == pytest.approx(2.5)
        assert unit == "kWh"

    def test_none_state_returns_none(self):
        val, unit = get_normalized_value(None, "power")
        assert val is None
        assert unit is None

    def test_non_numeric_state_returns_none(self):
        val, unit = get_normalized_value(make_state("unavailable", "W"), "power")
        assert val is None

    def test_temperature_passthrough(self):
        val, unit = get_normalized_value(make_state(22.3, "°C"), "temperature")
        assert val == pytest.approx(22.3)


# ─────────────────────────────────────────────────────────────────────────────
# get_environmental_impact
# ─────────────────────────────────────────────────────────────────────────────

class TestGetEnvironmentalImpact:

    def test_zero_savings(self):
        impact = get_environmental_impact(0)
        assert impact["co2_kg"] == 0.0
        assert impact["trees"] == 0.0
        assert impact["flights"] == 0.0
        assert impact["km"] == 0.0

    def test_positive_savings_co2(self):
        # 10 kWh * 0.1 kg/kWh = 1 kg CO2
        impact = get_environmental_impact(10)
        assert impact["co2_kg"] == pytest.approx(1.0)

    def test_tree_equivalence(self):
        # 22 kg CO2 saved = 1 tree (22 kg / 22 kg per tree)
        # Need 220 kWh * 0.1 = 22 kg CO2
        impact = get_environmental_impact(220)
        assert impact["trees"] == pytest.approx(1.0)

    def test_all_keys_present(self):
        impact = get_environmental_impact(100)
        assert set(impact.keys()) == {"co2_kg", "trees", "flights", "km"}

    def test_values_are_non_negative(self):
        impact = get_environmental_impact(50)
        for v in impact.values():
            assert v >= 0


# ─────────────────────────────────────────────────────────────────────────────
# get_working_days_from_config
# ─────────────────────────────────────────────────────────────────────────────

class TestGetWorkingDaysFromConfig:

    def test_mon_to_fri_standard(self, office_cfg):
        days = get_working_days_from_config(office_cfg)
        assert sorted(days) == [0, 1, 2, 3, 4]

    def test_empty_config_returns_empty(self):
        days = get_working_days_from_config({})
        assert days == []

    def test_weekend_only(self):
        cfg = {"working_saturday": True, "working_sunday": True}
        days = get_working_days_from_config(cfg)
        assert sorted(days) == [5, 6]

    def test_all_days(self):
        cfg = {f"working_{d}": True for d in
               ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]}
        days = get_working_days_from_config(cfg)
        assert sorted(days) == [0, 1, 2, 3, 4, 5, 6]


# ─────────────────────────────────────────────────────────────────────────────
# is_within_working_hours
# ─────────────────────────────────────────────────────────────────────────────

class TestIsWithinWorkingHours:

    # --- Home mode: always active ---

    def test_home_mode_always_true(self, home_cfg):
        # Saturday midnight — still active in home mode
        t = datetime(2026, 2, 21, 0, 30)  # Saturday
        assert is_within_working_hours(home_cfg, t) is True

    # --- Office mode: working day, inside hours ---

    def test_office_weekday_inside_hours(self, office_cfg):
        # Wednesday 10:00
        t = datetime(2026, 2, 18, 10, 0)
        assert is_within_working_hours(office_cfg, t) is True

    def test_office_weekday_on_boundary_start(self, office_cfg):
        # Exactly 08:00: boundary is inclusive
        t = datetime(2026, 2, 18, 8, 0)
        assert is_within_working_hours(office_cfg, t) is True

    def test_office_weekday_on_boundary_end(self, office_cfg):
        # Exactly 18:00: boundary is exclusive (end time is not included)
        t = datetime(2026, 2, 18, 18, 0)
        assert is_within_working_hours(office_cfg, t) is False

    def test_office_weekday_before_hours(self, office_cfg):
        # Tuesday 07:59
        t = datetime(2026, 2, 17, 7, 59)
        assert is_within_working_hours(office_cfg, t) is False

    def test_office_weekday_after_hours(self, office_cfg):
        # Thursday 19:00
        t = datetime(2026, 2, 19, 19, 0)
        assert is_within_working_hours(office_cfg, t) is False

    def test_office_saturday_is_inactive(self, office_cfg):
        # Saturday 10:00 — not a working day
        t = datetime(2026, 2, 21, 10, 0)
        assert is_within_working_hours(office_cfg, t) is False

    def test_office_sunday_is_inactive(self, office_cfg):
        t = datetime(2026, 2, 22, 12, 0)
        assert is_within_working_hours(office_cfg, t) is False

    def test_office_bad_time_format_defaults_to_true(self, office_cfg):
        bad_cfg = {**office_cfg, "working_start": "bad", "working_end": "time"}
        t = datetime(2026, 2, 18, 10, 0)
        # Bad time strings -> fallback returns True (safe default)
        assert is_within_working_hours(bad_cfg, t) is True

    def test_office_uses_datetime_now_when_check_time_missing(self, office_cfg):
        fake_now = datetime(2026, 2, 18, 9, 30)  # Wednesday inside office hours
        with patch.object(helpers, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.strptime = datetime.strptime
            assert is_within_working_hours(office_cfg, None) is True


# ─────────────────────────────────────────────────────────────────────────────
# should_ai_be_active (wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class TestShouldAiBeActive:

    def test_home_mode_always_active(self, home_cfg):
        assert should_ai_be_active(home_cfg) is True

    def test_office_mode_weekend_inactive(self, office_cfg):
        t = datetime(2026, 2, 21, 12, 0)  # Saturday
        assert should_ai_be_active(office_cfg, t) is False

    def test_office_mode_weekday_active(self, office_cfg):
        t = datetime(2026, 2, 18, 14, 0)  # Wednesday afternoon
        assert should_ai_be_active(office_cfg, t) is True


# ─────────────────────────────────────────────────────────────────────────────
# HA-registry functions  (get_entity_area, get_entity_area_id, group_sensors_by_area, get_friendly_name)
# ─────────────────────────────────────────────────────────────────────────────

# Aliases for the HA-stub modules so tests can set mock async_get on them
_er_stub = sys.modules["homeassistant.helpers.entity_registry"]
_ar_stub = sys.modules["homeassistant.helpers.area_registry"]
_dr_stub = sys.modules["homeassistant.helpers.device_registry"]

get_entity_area = helpers.get_entity_area
get_entity_area_id = helpers.get_entity_area_id
group_sensors_by_area = helpers.group_sensors_by_area
get_friendly_name = helpers.get_friendly_name
get_daily_working_hours = helpers.get_daily_working_hours


class TestGetEntityArea:

    def test_returns_none_when_entity_not_found(self):
        hass = MagicMock()
        mock_entity_reg = MagicMock()
        mock_entity_reg.async_get.return_value = None
        _er_stub.async_get = MagicMock(return_value=mock_entity_reg)
        _ar_stub.async_get = MagicMock(return_value=MagicMock())
        assert get_entity_area(hass, "sensor.unknown") is None

    def test_returns_area_from_entity_area_id(self):
        hass = MagicMock()
        mock_entity = MagicMock()
        mock_entity.area_id = "area_kitchen"
        mock_entity.device_id = None
        mock_entity_reg = MagicMock()
        mock_entity_reg.async_get.return_value = mock_entity

        mock_area = MagicMock()
        mock_area.name = "Kitchen"
        mock_area_reg = MagicMock()
        mock_area_reg.async_get_area.return_value = mock_area

        _er_stub.async_get = MagicMock(return_value=mock_entity_reg)
        _ar_stub.async_get = MagicMock(return_value=mock_area_reg)
        assert get_entity_area(hass, "sensor.kitchen_temp") == "Kitchen"

    def test_returns_area_via_device_when_no_entity_area(self):
        hass = MagicMock()
        mock_entity = MagicMock()
        mock_entity.area_id = None
        mock_entity.device_id = "dev_1"
        mock_entity_reg = MagicMock()
        mock_entity_reg.async_get.return_value = mock_entity

        mock_device = MagicMock()
        mock_device.area_id = "area_living"
        mock_device_reg = MagicMock()
        mock_device_reg.async_get.return_value = mock_device

        mock_area = MagicMock()
        mock_area.name = "Living Room"
        mock_area_reg = MagicMock()
        mock_area_reg.async_get_area.return_value = mock_area

        _er_stub.async_get = MagicMock(return_value=mock_entity_reg)
        _ar_stub.async_get = MagicMock(return_value=mock_area_reg)
        _dr_stub.async_get = MagicMock(return_value=mock_device_reg)
        assert get_entity_area(hass, "sensor.living_power") == "Living Room"

    def test_returns_none_when_no_area_and_no_device(self):
        hass = MagicMock()
        mock_entity = MagicMock()
        mock_entity.area_id = None
        mock_entity.device_id = None
        mock_entity_reg = MagicMock()
        mock_entity_reg.async_get.return_value = mock_entity

        _er_stub.async_get = MagicMock(return_value=mock_entity_reg)
        _ar_stub.async_get = MagicMock(return_value=MagicMock())
        assert get_entity_area(hass, "sensor.orphan") is None


class TestGetEntityAreaId:

    def test_returns_none_when_entity_not_found(self):
        hass = MagicMock()
        mock_entity_reg = MagicMock()
        mock_entity_reg.async_get.return_value = None
        _er_stub.async_get = MagicMock(return_value=mock_entity_reg)
        _ar_stub.async_get = MagicMock(return_value=MagicMock())
        assert get_entity_area_id(hass, "sensor.unknown") is None

    def test_returns_entity_area_id_directly(self):
        hass = MagicMock()
        mock_entity = MagicMock()
        mock_entity.area_id = "area_xyz"
        mock_entity.device_id = None
        mock_entity_reg = MagicMock()
        mock_entity_reg.async_get.return_value = mock_entity

        _er_stub.async_get = MagicMock(return_value=mock_entity_reg)
        _ar_stub.async_get = MagicMock(return_value=MagicMock())
        assert get_entity_area_id(hass, "sensor.test") == "area_xyz"

    def test_returns_device_area_id_when_entity_has_no_area(self):
        hass = MagicMock()
        mock_entity = MagicMock()
        mock_entity.area_id = None
        mock_entity.device_id = "dev_abc"
        mock_entity_reg = MagicMock()
        mock_entity_reg.async_get.return_value = mock_entity

        mock_device = MagicMock()
        mock_device.area_id = "area_device_zone"
        mock_device_reg = MagicMock()
        mock_device_reg.async_get.return_value = mock_device

        _er_stub.async_get = MagicMock(return_value=mock_entity_reg)
        _ar_stub.async_get = MagicMock(return_value=MagicMock())
        _dr_stub.async_get = MagicMock(return_value=mock_device_reg)
        assert get_entity_area_id(hass, "sensor.device_entity") == "area_device_zone"

    def test_returns_none_when_no_area_at_all(self):
        hass = MagicMock()
        mock_entity = MagicMock()
        mock_entity.area_id = None
        mock_entity.device_id = None
        mock_entity_reg = MagicMock()
        mock_entity_reg.async_get.return_value = mock_entity

        _er_stub.async_get = MagicMock(return_value=mock_entity_reg)
        _ar_stub.async_get = MagicMock(return_value=MagicMock())
        assert get_entity_area_id(hass, "sensor.orphan") is None


class TestGroupSensorsByArea:

    def test_groups_entities_by_named_areas(self):
        hass = MagicMock()
        mock_entity_kitchen = MagicMock()
        mock_entity_kitchen.area_id = "kitchen"
        mock_entity_kitchen.device_id = None

        mock_entity_living = MagicMock()
        mock_entity_living.area_id = "living"
        mock_entity_living.device_id = None

        mock_entity_reg = MagicMock()
        mock_entity_reg.async_get.side_effect = lambda eid: {
            "sensor.k1": mock_entity_kitchen,
            "sensor.l1": mock_entity_living,
        }.get(eid)

        mock_area_k = MagicMock()
        mock_area_k.name = "Kitchen"
        mock_area_l = MagicMock()
        mock_area_l.name = "Living Room"
        mock_area_reg = MagicMock()
        mock_area_reg.async_get_area.side_effect = lambda aid: {
            "kitchen": mock_area_k,
            "living": mock_area_l,
        }.get(aid)

        _er_stub.async_get = MagicMock(return_value=mock_entity_reg)
        _ar_stub.async_get = MagicMock(return_value=mock_area_reg)

        result = group_sensors_by_area(hass, ["sensor.k1", "sensor.l1"])
        assert result["Kitchen"] == ["sensor.k1"]
        assert result["Living Room"] == ["sensor.l1"]

    def test_groups_unknown_entities_under_no_area(self):
        hass = MagicMock()
        mock_entity_reg = MagicMock()
        mock_entity_reg.async_get.return_value = None  # not found -> no area
        _er_stub.async_get = MagicMock(return_value=mock_entity_reg)
        _ar_stub.async_get = MagicMock(return_value=MagicMock())

        result = group_sensors_by_area(hass, ["sensor.orphan"])
        assert "No Area" in result
        assert "sensor.orphan" in result["No Area"]


class TestGetFriendlyName:

    def test_returns_friendly_name_from_state_attributes(self):
        hass = MagicMock()
        state = MagicMock()
        state.attributes = {"friendly_name": "My Power Sensor"}
        hass.states.get.return_value = state
        _er_stub.async_get = MagicMock(return_value=MagicMock())
        assert get_friendly_name(hass, "sensor.power") == "My Power Sensor"

    def test_returns_original_name_from_entity_registry(self):
        hass = MagicMock()
        hass.states.get.return_value = None

        mock_entity = MagicMock()
        mock_entity.original_name = "Original Power"
        mock_entity_reg = MagicMock()
        mock_entity_reg.async_get.return_value = mock_entity
        _er_stub.async_get = MagicMock(return_value=mock_entity_reg)

        assert get_friendly_name(hass, "sensor.power") == "Original Power"

    def test_falls_back_to_entity_id(self):
        hass = MagicMock()
        hass.states.get.return_value = None

        mock_entity = MagicMock()
        mock_entity.original_name = None
        mock_entity_reg = MagicMock()
        mock_entity_reg.async_get.return_value = mock_entity
        _er_stub.async_get = MagicMock(return_value=mock_entity_reg)

        assert get_friendly_name(hass, "sensor.power_meter") == "sensor.power_meter"


# ─────────────────────────────────────────────────────────────────────────────
# is_within_working_hours: timezone-aware datetime path
# ─────────────────────────────────────────────────────────────────────────────

class TestIsWithinWorkingHoursTzAware:

    def test_tz_aware_datetime_does_not_raise(self, office_cfg):
        """Timezone-aware datetime triggers the tzinfo branch without error."""
        from datetime import timezone as tz
        t = datetime(2026, 2, 18, 10, 0, tzinfo=tz.utc)  # Wednesday 10:00 UTC
        result = is_within_working_hours(office_cfg, t)
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# get_daily_working_hours
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDailyWorkingHours:

    def test_standard_10_hour_shift(self):
        cfg = {"working_start": "08:00", "working_end": "18:00"}
        assert get_daily_working_hours(cfg) == pytest.approx(10.0)

    def test_custom_4_hour_shift(self):
        cfg = {"working_start": "09:00", "working_end": "13:00"}
        assert get_daily_working_hours(cfg) == pytest.approx(4.0)

    def test_bad_time_format_returns_10_as_fallback(self):
        cfg = {"working_start": "bad", "working_end": "format"}
        assert get_daily_working_hours(cfg) == pytest.approx(10.0)

    def test_midnight_crossing_shift(self):
        """Night shift: end before start wraps around -> positive hours."""
        cfg = {"working_start": "22:00", "working_end": "06:00"}
        result = get_daily_working_hours(cfg)
        assert result == pytest.approx(8.0)

