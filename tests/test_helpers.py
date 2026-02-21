"""
Tests for helpers.py

Covers:
- get_normalized_value   : unit conversions (power kW→W, energy Wh→kWh)
- get_environmental_impact: CO2 / metaphor calculations
- get_working_days_from_config: config dict → list of day indices
- is_within_working_hours: home always active; office respects days & times
- should_ai_be_active    : thin wrapper check
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock

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
        # Exactly 08:00 — boundary is inclusive
        t = datetime(2026, 2, 18, 8, 0)
        assert is_within_working_hours(office_cfg, t) is True

    def test_office_weekday_on_boundary_end(self, office_cfg):
        # Exactly 18:00 — boundary is inclusive
        t = datetime(2026, 2, 18, 18, 0)
        assert is_within_working_hours(office_cfg, t) is True

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
        # Bad time strings → fallback returns True (safe default)
        assert is_within_working_hours(bad_cfg, t) is True


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
