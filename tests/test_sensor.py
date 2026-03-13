"""
Tests for sensor.py

Covers (pure property/logic tests — no HA event loop required):
- HardwareSensorsSensor: state always "ok", extra_state_attributes excludes main sensors
- ResearchPhaseSensor: state reflects agent.phase, extra_state_attributes calculates days
- EnergyBaselineSensor: state rounds baseline_consumption
- CurrentConsumptionSensor: state rounds current_total_power
- CurrentCostConsumptionSensor: state calculates cost, unit_of_measurement reads currency
- DailyCostConsumptionSensor: state calculates cost, unit_of_measurement reads currency
- DailyCO2EstimateSensor: state calculates CO2, extra_state_attributes contains co2_factor
- BehaviourIndexSensor: state rounds behaviour_index
- FatigueIndexSensor: state rounds fatigue_index
- TaskStreakSensor: state, extra_state_attributes with date
- WeeklyStreakSensor: state, extra_state_attributes
- ActiveNotificationsSensor: state counts unresponded, extra_state_attributes aggregates,
  _get_time_ago formats correctly
- DailyTasksSensor: state = len(tasks), extra_state_attributes with status flags
"""

import sys
import types
import pathlib
import importlib.util
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch

# -- Minimal HA stubs ---------------------------------------------------------

for mod_name in [
    "homeassistant",
    "homeassistant.components",
    "homeassistant.components.sensor",
    "homeassistant.config_entries",
    "homeassistant.core",
    "homeassistant.helpers",
    "homeassistant.helpers.entity_platform",
    "homeassistant.helpers.dispatcher",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)


# Minimal SensorEntity stub so subclasses can inherit without importing real HA
class _SensorEntityBase:
    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_native_value = None
    _attr_extra_state_attributes = {}
    hass = None

    async def async_added_to_hass(self):
        pass

    def async_write_ha_state(self):
        pass

    def async_on_remove(self, unsub):
        pass

    def async_create_task(self, coro):
        pass


sys.modules["homeassistant.components.sensor"].SensorEntity = _SensorEntityBase
sys.modules["homeassistant.core"].callback = lambda f: f
sys.modules["homeassistant.helpers.dispatcher"].async_dispatcher_connect = MagicMock(
    return_value=MagicMock()
)

# -- Real const module ---------------------------------------------------------

const_spec = importlib.util.spec_from_file_location(
    "custom_components.green_shift.const",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "const.py",
)
const_mod = importlib.util.module_from_spec(const_spec)
const_mod.__package__ = "custom_components.green_shift"
const_spec.loader.exec_module(const_mod)
sys.modules["custom_components.green_shift.const"] = const_mod

CO2_FACTOR = const_mod.CO2_FACTOR
DOMAIN = const_mod.DOMAIN
BASELINE_DAYS = const_mod.BASELINE_DAYS

# -- Stub helpers --------------------------------------------------------------

helpers_stub = types.ModuleType("custom_components.green_shift.helpers")
helpers_stub.get_normalized_value = MagicMock(return_value=(100.0, "W"))
helpers_stub.get_entity_area = MagicMock(return_value="Living Room")
helpers_stub.get_environmental_impact = MagicMock(
    return_value={"co2_kg": 0.05, "trees": 0.002, "flights": 0.0003, "km": 0.29}
)
sys.modules["custom_components.green_shift.helpers"] = helpers_stub

# -- Load sensor module --------------------------------------------------------

sensor_spec = importlib.util.spec_from_file_location(
    "gs_sensor",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "sensor.py",
)
sensor_mod = importlib.util.module_from_spec(sensor_spec)
sensor_mod.__package__ = "custom_components.green_shift"
sensor_spec.loader.exec_module(sensor_mod)

HardwareSensorsSensor = sensor_mod.HardwareSensorsSensor
ResearchPhaseSensor = sensor_mod.ResearchPhaseSensor
EnergyBaselineSensor = sensor_mod.EnergyBaselineSensor
CurrentConsumptionSensor = sensor_mod.CurrentConsumptionSensor
CurrentCostConsumptionSensor = sensor_mod.CurrentCostConsumptionSensor
DailyCostConsumptionSensor = sensor_mod.DailyCostConsumptionSensor
DailyCO2EstimateSensor = sensor_mod.DailyCO2EstimateSensor
SavingsAccumulatedSensor = sensor_mod.SavingsAccumulatedSensor
CO2SavedSensor = sensor_mod.CO2SavedSensor
TasksCompletedSensor = sensor_mod.TasksCompletedSensor
WeeklyChallengeSensor = sensor_mod.WeeklyChallengeSensor
SavingsAccumulatedSensor = sensor_mod.SavingsAccumulatedSensor
CO2SavedSensor = sensor_mod.CO2SavedSensor
TasksCompletedSensor = sensor_mod.TasksCompletedSensor
WeeklyChallengeSensor = sensor_mod.WeeklyChallengeSensor
BehaviourIndexSensor = sensor_mod.BehaviourIndexSensor
FatigueIndexSensor = sensor_mod.FatigueIndexSensor
TaskStreakSensor = sensor_mod.TaskStreakSensor
WeeklyStreakSensor = sensor_mod.WeeklyStreakSensor
ActiveNotificationsSensor = sensor_mod.ActiveNotificationsSensor
DailyTasksSensor = sensor_mod.DailyTasksSensor


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_agent(**kwargs):
    agent = MagicMock()
    agent.phase = kwargs.get("phase", "baseline")
    agent.start_date = kwargs.get("start_date", datetime.now() - timedelta(days=5))
    agent.baseline_consumption = kwargs.get("baseline_consumption", 1000.0)
    agent.active_since = kwargs.get("active_since", None)
    agent.behaviour_index = kwargs.get("behaviour_index", 0.75)
    agent.fatigue_index = kwargs.get("fatigue_index", 0.3)
    agent.task_streak = kwargs.get("task_streak", 5)
    agent.task_streak_last_date = kwargs.get("task_streak_last_date", datetime.now().date())
    agent.weekly_streak = kwargs.get("weekly_streak", 2)
    agent.weekly_streak_last_week = kwargs.get("weekly_streak_last_week", "2026-W09")
    agent.notification_history = kwargs.get("notification_history", [])
    return agent


def make_collector(**kwargs):
    collector = MagicMock()
    collector.current_total_power = kwargs.get("power", 500.0)
    collector.current_daily_energy = kwargs.get("energy", 2.5)
    return collector


def make_hass(states=None):
    """Return a minimal hass mock with optional state overrides."""
    hass = MagicMock()
    state_dict = states or {}

    def get_state(entity_id):
        if entity_id in state_dict:
            s = MagicMock()
            s.state = str(state_dict[entity_id])
            return s
        return None

    hass.states.get = MagicMock(side_effect=get_state)
    return hass


def make_notification(nid, responded=False, accepted=None, minutes_ago=60):
    ts = (datetime.now() - timedelta(minutes=minutes_ago)).isoformat()
    notif = {
        "notification_id": nid,
        "action_type": "specific",
        "title": "Test Notification",
        "message": "Save energy now",
        "timestamp": ts,
        "responded": responded,
    }
    if accepted is not None:
        notif["accepted"] = accepted
    return notif


def make_task(task_id, completed=False, verified=False, target_unit="W", target_value=500):
    return {
        "task_id": task_id,
        "title": f"Task {task_id}",
        "description": "Do something energy-efficient",
        "target_value": target_value,
        "target_unit": target_unit,
        "baseline_value": 600,
        "difficulty_level": 3,
        "completed": completed,
        "verified": verified,
        "user_feedback": None,
        "area_name": "Living Room",
        "completion_value": None,
    }


# -----------------------------------------------------------------------------
# HardwareSensorsSensor
# -----------------------------------------------------------------------------

class TestHardwareSensorsSensor:

    def _make_sensor(self, discovered=None, states=None, main_power=None, main_energy=None):
        hass = make_hass(states)
        state = MagicMock()
        state.state = "150"
        state.attributes = {"unit_of_measurement": "W", "friendly_name": "Device"}
        hass.states.get = MagicMock(return_value=state)

        config_entry = MagicMock()
        config_entry.data = {
            "main_total_power_sensor": main_power,
            "main_total_energy_sensor": main_energy,
        }
        return HardwareSensorsSensor(hass, discovered or {}, config_entry)

    def test_state_is_always_ok(self):
        sensor = self._make_sensor()
        assert sensor.state == "ok"

    def test_extra_state_attributes_excludes_main_power_sensor(self):
        helpers_stub.get_normalized_value.return_value = (150.0, "W")
        sensor = self._make_sensor(
            discovered={"power": ["sensor.main_pwr", "sensor.device_pwr"]},
            main_power="sensor.main_pwr",
        )
        attrs = sensor.extra_state_attributes
        entity_ids = [e["entity_id"] for e in attrs.get("power", [])]
        assert "sensor.main_pwr" not in entity_ids

    def test_extra_state_attributes_excludes_main_energy_sensor(self):
        helpers_stub.get_normalized_value.return_value = (10.0, "kWh")
        sensor = self._make_sensor(
            discovered={"energy": ["sensor.main_energy", "sensor.sub_energy"]},
            main_energy="sensor.main_energy",
        )
        attrs = sensor.extra_state_attributes
        entity_ids = [e["entity_id"] for e in attrs.get("energy", [])]
        assert "sensor.main_energy" not in entity_ids

    def test_extra_state_attributes_returns_all_categories(self):
        sensor = self._make_sensor(
            discovered={"power": [], "energy": [], "temperature": []}
        )
        attrs = sensor.extra_state_attributes
        assert set(attrs.keys()) == {"power", "energy", "temperature"}

    def test_extra_state_attributes_skips_states_returning_none(self):
        hass = MagicMock()
        hass.states.get = MagicMock(return_value=None)
        config_entry = MagicMock()
        config_entry.data = {}
        sensor = HardwareSensorsSensor(hass, {"power": ["sensor.power"]}, config_entry)
        attrs = sensor.extra_state_attributes
        assert attrs["power"] == []


# -----------------------------------------------------------------------------
# ResearchPhaseSensor
# -----------------------------------------------------------------------------

class TestResearchPhaseSensor:

    def test_state_returns_baseline(self):
        sensor = ResearchPhaseSensor(make_agent(phase="baseline"))
        assert sensor.state == "baseline"

    def test_state_returns_active(self):
        sensor = ResearchPhaseSensor(make_agent(phase="active"))
        assert sensor.state == "active"

    def test_extra_attributes_days_running(self):
        start = datetime.now() - timedelta(days=5)
        sensor = ResearchPhaseSensor(make_agent(start_date=start))
        assert sensor.extra_state_attributes["days_running"] == 5

    def test_extra_attributes_days_remaining(self):
        start = datetime.now() - timedelta(days=5)
        sensor = ResearchPhaseSensor(make_agent(start_date=start))
        # BASELINE_DAYS = 14 → 14 - 5 = 9
        assert sensor.extra_state_attributes["days_remaining"] == BASELINE_DAYS - 5

    def test_extra_attributes_baseline_not_complete(self):
        start = datetime.now() - timedelta(days=5)
        sensor = ResearchPhaseSensor(make_agent(start_date=start))
        assert sensor.extra_state_attributes["baseline_complete"] is False

    def test_extra_attributes_baseline_complete_after_14_days(self):
        start = datetime.now() - timedelta(days=14)
        sensor = ResearchPhaseSensor(make_agent(start_date=start))
        assert sensor.extra_state_attributes["baseline_complete"] is True

    def test_extra_attributes_days_remaining_never_negative(self):
        start = datetime.now() - timedelta(days=20)
        sensor = ResearchPhaseSensor(make_agent(start_date=start))
        assert sensor.extra_state_attributes["days_remaining"] == 0


# -----------------------------------------------------------------------------
# EnergyBaselineSensor
# -----------------------------------------------------------------------------

class TestEnergyBaselineSensor:

    def test_state_returns_rounded_value(self):
        sensor = EnergyBaselineSensor(make_agent(baseline_consumption=1234.5678))
        assert sensor.state == pytest.approx(1234.57)

    def test_state_zero_when_no_baseline(self):
        sensor = EnergyBaselineSensor(make_agent(baseline_consumption=0.0))
        assert sensor.state == 0.0

    def test_state_is_rounded_to_two_decimals(self):
        sensor = EnergyBaselineSensor(make_agent(baseline_consumption=999.999))
        assert sensor.state == pytest.approx(1000.0)


# -----------------------------------------------------------------------------
# CurrentConsumptionSensor
# -----------------------------------------------------------------------------

class TestCurrentConsumptionSensor:

    def test_state_rounds_to_3_decimals(self):
        sensor = CurrentConsumptionSensor(make_collector(power=750.123456))
        assert sensor.state == pytest.approx(750.123)

    def test_state_zero_when_no_consumption(self):
        sensor = CurrentConsumptionSensor(make_collector(power=0.0))
        assert sensor.state == 0.0

    def test_state_large_value(self):
        sensor = CurrentConsumptionSensor(make_collector(power=5000.0))
        assert sensor.state == pytest.approx(5000.0)


# -----------------------------------------------------------------------------
# CurrentCostConsumptionSensor
# -----------------------------------------------------------------------------

class TestCurrentCostConsumptionSensor:

    def test_state_calculates_cost_correctly(self):
        # 2 kW * 0.20 EUR/kWh = 0.40 EUR/h
        hass = make_hass({"input_number.electricity_price": "0.20"})
        sensor = CurrentCostConsumptionSensor(hass, make_collector(power=2000.0))
        assert sensor.state == pytest.approx(0.4, abs=0.001)

    def test_state_uses_default_price_when_missing(self):
        # No price state → 0.25; 1 kW * 0.25 = 0.25 EUR/h
        hass = make_hass()
        sensor = CurrentCostConsumptionSensor(hass, make_collector(power=1000.0))
        assert sensor.state == pytest.approx(0.25, abs=0.001)

    def test_state_handles_invalid_price_string(self):
        hass = make_hass({"input_number.electricity_price": "invalid"})
        sensor = CurrentCostConsumptionSensor(hass, make_collector(power=1000.0))
        # Falls back to default 0.25
        assert sensor.state == pytest.approx(0.25, abs=0.001)

    def test_unit_of_measurement_uses_currency_select(self):
        hass = make_hass({"input_select.currency": "USD"})
        sensor = CurrentCostConsumptionSensor(hass, make_collector())
        assert sensor.unit_of_measurement == "USD/h"

    def test_unit_of_measurement_defaults_to_eur_per_hour(self):
        hass = make_hass()
        sensor = CurrentCostConsumptionSensor(hass, make_collector())
        assert sensor.unit_of_measurement == "EUR/h"

    def test_state_zero_power_gives_zero_cost(self):
        hass = make_hass({"input_number.electricity_price": "0.25"})
        sensor = CurrentCostConsumptionSensor(hass, make_collector(power=0.0))
        assert sensor.state == pytest.approx(0.0)

    def test_extra_state_attributes_has_required_keys(self):
        hass = make_hass({
            "input_number.electricity_price": "0.20",
            "input_select.currency": "EUR",
        })
        sensor = CurrentCostConsumptionSensor(hass, make_collector(power=500.0))
        attrs = sensor.extra_state_attributes
        assert "current_load" in attrs
        assert "applied_price_per_kwh" in attrs
        assert "currency" in attrs


# -----------------------------------------------------------------------------
# DailyCostConsumptionSensor
# -----------------------------------------------------------------------------

class TestDailyCostConsumptionSensor:

    def test_state_calculates_daily_cost(self):
        # 10 kWh * 0.25 EUR/kWh = 2.50 EUR
        hass = make_hass({"input_number.electricity_price": "0.25"})
        sensor = DailyCostConsumptionSensor(hass, make_collector(energy=10.0))
        assert sensor.state == pytest.approx(2.5, abs=0.01)

    def test_state_uses_default_price_when_missing(self):
        # 4 kWh * 0.25 = 1.0 EUR
        hass = make_hass()
        sensor = DailyCostConsumptionSensor(hass, make_collector(energy=4.0))
        assert sensor.state == pytest.approx(1.0, abs=0.01)

    def test_state_handles_invalid_price(self):
        hass = make_hass({"input_number.electricity_price": "bad"})
        sensor = DailyCostConsumptionSensor(hass, make_collector(energy=2.0))
        # Falls back to 0.25; 2 kWh * 0.25 = 0.50
        assert sensor.state == pytest.approx(0.5, abs=0.01)

    def test_unit_of_measurement_uses_currency(self):
        hass = make_hass({"input_select.currency": "GBP"})
        sensor = DailyCostConsumptionSensor(hass, make_collector())
        assert sensor.unit_of_measurement == "GBP"

    def test_unit_of_measurement_defaults_to_eur(self):
        hass = make_hass()
        sensor = DailyCostConsumptionSensor(hass, make_collector())
        assert sensor.unit_of_measurement == "EUR"

    def test_extra_attributes_has_daily_kwh_and_price(self):
        hass = make_hass()
        sensor = DailyCostConsumptionSensor(hass, make_collector(energy=3.0))
        attrs = sensor.extra_state_attributes
        assert "daily_kwh_accumulated" in attrs
        assert "applied_price" in attrs


# -----------------------------------------------------------------------------
# DailyCO2EstimateSensor
# -----------------------------------------------------------------------------

class TestDailyCO2EstimateSensor:

    def test_state_calculates_co2_correctly(self):
        # 10 kWh * 0.1 kg/kWh = 1.0 kg
        hass = make_hass()
        sensor = DailyCO2EstimateSensor(hass, make_collector(energy=10.0))
        assert sensor.state == pytest.approx(1.0, abs=0.01)

    def test_state_zero_when_no_energy(self):
        hass = make_hass()
        sensor = DailyCO2EstimateSensor(hass, make_collector(energy=0.0))
        assert sensor.state == 0.0

    def test_state_scales_with_energy(self):
        hass = make_hass()
        sensor5 = DailyCO2EstimateSensor(hass, make_collector(energy=5.0))
        sensor10 = DailyCO2EstimateSensor(hass, make_collector(energy=10.0))
        assert sensor10.state == pytest.approx(sensor5.state * 2, abs=0.01)

    def test_extra_attributes_contains_co2_factor(self):
        hass = make_hass()
        sensor = DailyCO2EstimateSensor(hass, make_collector())
        attrs = sensor.extra_state_attributes
        assert "co2_factor" in attrs
        assert attrs["co2_factor"] == CO2_FACTOR

    def test_extra_attributes_contains_daily_kwh(self):
        hass = make_hass()
        sensor = DailyCO2EstimateSensor(hass, make_collector(energy=3.5))
        attrs = sensor.extra_state_attributes
        assert "daily_kwh_accumulated" in attrs
        assert attrs["daily_kwh_accumulated"] == pytest.approx(3.5, abs=0.001)


# -----------------------------------------------------------------------------
# BehaviourIndexSensor
# -----------------------------------------------------------------------------

class TestBehaviourIndexSensor:

    def test_state_rounds_to_2_decimals(self):
        sensor = BehaviourIndexSensor(make_agent(behaviour_index=0.7654))
        assert sensor.state == pytest.approx(0.77)

    def test_state_zero(self):
        sensor = BehaviourIndexSensor(make_agent(behaviour_index=0.0))
        assert sensor.state == 0.0

    def test_state_one(self):
        sensor = BehaviourIndexSensor(make_agent(behaviour_index=1.0))
        assert sensor.state == pytest.approx(1.0)

    def test_state_mid_range(self):
        sensor = BehaviourIndexSensor(make_agent(behaviour_index=0.5))
        assert sensor.state == pytest.approx(0.5)


# -----------------------------------------------------------------------------
# FatigueIndexSensor
# -----------------------------------------------------------------------------

class TestFatigueIndexSensor:

    def test_state_rounds_to_2_decimals(self):
        sensor = FatigueIndexSensor(make_agent(fatigue_index=0.3456))
        assert sensor.state == pytest.approx(0.35)

    def test_state_zero(self):
        sensor = FatigueIndexSensor(make_agent(fatigue_index=0.0))
        assert sensor.state == 0.0

    def test_state_one(self):
        sensor = FatigueIndexSensor(make_agent(fatigue_index=1.0))
        assert sensor.state == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# TaskStreakSensor
# -----------------------------------------------------------------------------

class TestTaskStreakSensor:

    def test_state_returns_agent_streak(self):
        sensor = TaskStreakSensor(make_agent(task_streak=7))
        assert sensor.state == 7

    def test_state_zero(self):
        sensor = TaskStreakSensor(make_agent(task_streak=0))
        assert sensor.state == 0

    def test_state_large_streak(self):
        sensor = TaskStreakSensor(make_agent(task_streak=30))
        assert sensor.state == 30

    def test_extra_attributes_last_credited_date(self):
        today = datetime.now().date()
        sensor = TaskStreakSensor(make_agent(task_streak_last_date=today))
        attrs = sensor.extra_state_attributes
        assert attrs["last_credited_date"] == today.isoformat()

    def test_extra_attributes_none_when_no_date(self):
        sensor = TaskStreakSensor(make_agent(task_streak_last_date=None))
        attrs = sensor.extra_state_attributes
        assert attrs["last_credited_date"] is None


# -----------------------------------------------------------------------------
# WeeklyStreakSensor
# -----------------------------------------------------------------------------

class TestWeeklyStreakSensor:

    def test_state_returns_agent_weekly_streak(self):
        sensor = WeeklyStreakSensor(make_agent(weekly_streak=3))
        assert sensor.state == 3

    def test_state_zero(self):
        sensor = WeeklyStreakSensor(make_agent(weekly_streak=0))
        assert sensor.state == 0

    def test_extra_attributes_last_credited_week(self):
        sensor = WeeklyStreakSensor(make_agent(weekly_streak_last_week="2026-W09"))
        attrs = sensor.extra_state_attributes
        assert attrs["last_credited_week"] == "2026-W09"

    def test_extra_attributes_none_last_week(self):
        sensor = WeeklyStreakSensor(make_agent(weekly_streak_last_week=None))
        attrs = sensor.extra_state_attributes
        assert attrs["last_credited_week"] is None


# -----------------------------------------------------------------------------
# ActiveNotificationsSensor
# -----------------------------------------------------------------------------

class TestActiveNotificationsSensorState:

    def test_state_counts_only_unresponded(self):
        agent = make_agent(notification_history=[
            make_notification("n1", responded=False),
            make_notification("n2", responded=True),
            make_notification("n3", responded=False),
        ])
        sensor = ActiveNotificationsSensor(agent)
        assert sensor.state == 2

    def test_state_zero_when_all_responded(self):
        agent = make_agent(notification_history=[
            make_notification("n1", responded=True, accepted=True),
            make_notification("n2", responded=True, accepted=False),
        ])
        sensor = ActiveNotificationsSensor(agent)
        assert sensor.state == 0

    def test_state_zero_with_empty_history(self):
        sensor = ActiveNotificationsSensor(make_agent(notification_history=[]))
        assert sensor.state == 0

    def test_state_all_pending(self):
        agent = make_agent(notification_history=[
            make_notification("n1"),
            make_notification("n2"),
            make_notification("n3"),
        ])
        sensor = ActiveNotificationsSensor(agent)
        assert sensor.state == 3


class TestActiveNotificationsSensorAttributes:

    def test_total_count_matches_all_notifications(self):
        agent = make_agent(notification_history=[
            make_notification("n1", responded=False),
            make_notification("n2", responded=True, accepted=True),
        ])
        sensor = ActiveNotificationsSensor(agent)
        attrs = sensor.extra_state_attributes
        assert attrs["total_count"] == 2

    def test_pending_count(self):
        agent = make_agent(notification_history=[
            make_notification("n1", responded=False),
            make_notification("n2", responded=True, accepted=True),
            make_notification("n3", responded=False),
        ])
        sensor = ActiveNotificationsSensor(agent)
        assert sensor.extra_state_attributes["pending_count"] == 2

    def test_accepted_and_rejected_counts(self):
        agent = make_agent(notification_history=[
            make_notification("n1", responded=True, accepted=True),
            make_notification("n2", responded=True, accepted=False),
            make_notification("n3", responded=True, accepted=False),
        ])
        sensor = ActiveNotificationsSensor(agent)
        attrs = sensor.extra_state_attributes
        assert attrs["accepted_count"] == 1
        assert attrs["rejected_count"] == 2

    def test_acceptance_rate_50_percent(self):
        agent = make_agent(notification_history=[
            make_notification("n1", responded=True, accepted=True),
            make_notification("n2", responded=True, accepted=False),
        ])
        sensor = ActiveNotificationsSensor(agent)
        assert sensor.extra_state_attributes["acceptance_rate"] == pytest.approx(50.0)

    def test_acceptance_rate_zero_when_no_responded(self):
        agent = make_agent(notification_history=[
            make_notification("n1", responded=False),
        ])
        sensor = ActiveNotificationsSensor(agent)
        assert sensor.extra_state_attributes["acceptance_rate"] == 0.0

    def test_notifications_sorted_newest_first(self):
        agent = make_agent(notification_history=[
            make_notification("old", responded=False, minutes_ago=120),
            make_notification("new", responded=False, minutes_ago=5),
        ])
        sensor = ActiveNotificationsSensor(agent)
        nids = [n["notification_id"] for n in sensor.extra_state_attributes["notifications"]]
        assert nids[0] == "new"
        assert nids[1] == "old"

    def test_status_emoji_for_pending(self):
        agent = make_agent(notification_history=[make_notification("n1", responded=False)])
        sensor = ActiveNotificationsSensor(agent)
        notifs = sensor.extra_state_attributes["notifications"]
        assert notifs[0]["status_emoji"] == "⏳"

    def test_status_emoji_for_accepted(self):
        agent = make_agent(notification_history=[
            make_notification("n1", responded=True, accepted=True)
        ])
        sensor = ActiveNotificationsSensor(agent)
        notifs = sensor.extra_state_attributes["notifications"]
        assert notifs[0]["status_emoji"] == "✅"

    def test_status_emoji_for_rejected(self):
        agent = make_agent(notification_history=[
            make_notification("n1", responded=True, accepted=False)
        ])
        sensor = ActiveNotificationsSensor(agent)
        notifs = sensor.extra_state_attributes["notifications"]
        assert notifs[0]["status_emoji"] == "❌"


class TestGetTimeAgo:

    def test_just_now_for_recent_timestamps(self):
        sensor = ActiveNotificationsSensor(make_agent())
        ts = datetime.now() - timedelta(seconds=30)
        assert sensor._get_time_ago(ts) == "Just now"

    def test_minutes_ago_format(self):
        sensor = ActiveNotificationsSensor(make_agent())
        ts = datetime.now() - timedelta(minutes=45)
        result = sensor._get_time_ago(ts)
        assert "m ago" in result
        assert "45" in result

    def test_hours_ago_format(self):
        sensor = ActiveNotificationsSensor(make_agent())
        ts = datetime.now() - timedelta(hours=3)
        result = sensor._get_time_ago(ts)
        assert "h ago" in result
        assert "3" in result

    def test_days_ago_format(self):
        sensor = ActiveNotificationsSensor(make_agent())
        ts = datetime.now() - timedelta(days=2)
        result = sensor._get_time_ago(ts)
        assert "d ago" in result
        assert "2" in result

    def test_tz_aware_timestamp_handled(self):
        """Timezone-aware timestamps should not cause errors."""
        sensor = ActiveNotificationsSensor(make_agent())
        ts = datetime.now(timezone.utc) - timedelta(minutes=5)
        result = sensor._get_time_ago(ts)
        assert result  # Some valid string


# -----------------------------------------------------------------------------
# DailyTasksSensor
# -----------------------------------------------------------------------------

class TestDailyTasksSensorState:

    def test_state_returns_task_count(self):
        sensor = DailyTasksSensor(AsyncMock())
        sensor._tasks = [make_task("t1"), make_task("t2")]
        assert sensor.state == 2

    def test_state_zero_with_no_tasks(self):
        sensor = DailyTasksSensor(AsyncMock())
        sensor._tasks = []
        assert sensor.state == 0

    def test_state_single_task(self):
        sensor = DailyTasksSensor(AsyncMock())
        sensor._tasks = [make_task("t1")]
        assert sensor.state == 1


class TestDailyTasksSensorAttributes:

    def test_empty_tasks_returns_zero_counts(self):
        sensor = DailyTasksSensor(AsyncMock())
        sensor._tasks = []
        attrs = sensor.extra_state_attributes
        assert attrs["total_count"] == 0
        assert attrs["completed_count"] == 0
        assert attrs["verified_count"] == 0
        assert attrs["tasks"] == []

    def test_completed_count(self):
        sensor = DailyTasksSensor(AsyncMock())
        sensor._tasks = [
            make_task("t1", completed=True),
            make_task("t2", completed=False),
            make_task("t3", completed=True),
        ]
        assert sensor.extra_state_attributes["completed_count"] == 2

    def test_verified_count(self):
        sensor = DailyTasksSensor(AsyncMock())
        sensor._tasks = [
            make_task("t1", verified=True),
            make_task("t2", verified=False),
        ]
        assert sensor.extra_state_attributes["verified_count"] == 1

    def test_target_value_formatted_as_int_for_watts(self):
        sensor = DailyTasksSensor(AsyncMock())
        sensor._tasks = [make_task("t1", target_unit="W", target_value=500)]
        attrs = sensor.extra_state_attributes
        assert isinstance(attrs["tasks"][0]["target_value"], int)

    def test_target_value_formatted_as_float_for_celsius(self):
        sensor = DailyTasksSensor(AsyncMock())
        sensor._tasks = [make_task("t1", target_unit="°C", target_value="21.5")]
        attrs = sensor.extra_state_attributes
        assert isinstance(attrs["tasks"][0]["target_value"], float)

    def test_task_status_pending_when_not_completed(self):
        sensor = DailyTasksSensor(AsyncMock())
        sensor._tasks = [make_task("t1", completed=False, verified=False)]
        attrs = sensor.extra_state_attributes
        assert attrs["tasks"][0]["status"] == "pending"
        assert attrs["tasks"][0]["status_emoji"] == "🎯"

    def test_task_status_completed_when_completed_not_verified(self):
        sensor = DailyTasksSensor(AsyncMock())
        sensor._tasks = [make_task("t1", completed=True, verified=False)]
        attrs = sensor.extra_state_attributes
        assert attrs["tasks"][0]["status"] == "completed"
        assert attrs["tasks"][0]["status_emoji"] == "⏳"

    def test_task_status_verified_when_verified(self):
        sensor = DailyTasksSensor(AsyncMock())
        sensor._tasks = [make_task("t1", completed=True, verified=True)]
        attrs = sensor.extra_state_attributes
        assert attrs["tasks"][0]["status"] == "verified"
        assert attrs["tasks"][0]["status_emoji"] == "✅"

    def test_difficulty_display_format(self):
        sensor = DailyTasksSensor(AsyncMock())
        sensor._tasks = [make_task("t1")]
        sensor._tasks[0]["difficulty_level"] = 3
        attrs = sensor.extra_state_attributes
        assert "⭐" in attrs["tasks"][0]["difficulty_display"]

    def test_task_not_checked_when_no_task_manager(self):
        sensor = DailyTasksSensor(AsyncMock(), task_manager=None)
        sensor._tasks = [make_task("t1")]
        attrs = sensor.extra_state_attributes
        assert attrs["tasks"][0]["check_result"] == "not_checked"

    def test_last_check_info_from_task_manager(self):
        task_manager = MagicMock()
        checked_at = datetime.now() - timedelta(minutes=5)
        task_manager._last_verification_results = {
            "t1": {
                "verified": True,
                "failed": False,
                "pending": False,
                "checked_at": checked_at,
                "reason": "Target achieved",
            }
        }
        sensor = DailyTasksSensor(AsyncMock(), task_manager=task_manager)
        sensor._tasks = [make_task("t1")]
        attrs = sensor.extra_state_attributes
        assert attrs["tasks"][0]["check_result"] == "verified"
        assert attrs["tasks"][0]["check_reason"] == "Target achieved"

    def test_task_with_completion_value_included_in_attrs(self):
        sensor = DailyTasksSensor(AsyncMock())
        task = make_task("t1", completed=True)
        task["completion_value"] = 450.0
        sensor._tasks = [task]
        attrs = sensor.extra_state_attributes
        assert attrs["tasks"][0]["completion_value"] == 450.0

    def test_task_manager_with_none_checked_at(self):
        task_manager = MagicMock()
        task_manager._last_verification_results = {
            "t1": {
                "verified": False,
                "failed": True,
                "pending": False,
                "checked_at": None,
                "reason": "Target missed",
            }
        }
        sensor = DailyTasksSensor(AsyncMock(), task_manager=task_manager)
        sensor._tasks = [make_task("t1")]
        attrs = sensor.extra_state_attributes
        assert attrs["tasks"][0]["check_result"] == "failed"
        assert attrs["tasks"][0]["last_check_minutes_ago"] is None

    def test_task_pending_check_result(self):
        task_manager = MagicMock()
        task_manager._last_verification_results = {
            "t1": {
                "verified": False,
                "failed": False,
                "pending": True,
                "checked_at": datetime.now() - timedelta(minutes=2),
                "reason": "Peak hour not yet reached",
            }
        }
        sensor = DailyTasksSensor(AsyncMock(), task_manager=task_manager)
        sensor._tasks = [make_task("t1")]
        attrs = sensor.extra_state_attributes
        assert attrs["tasks"][0]["check_result"] == "pending"


# -----------------------------------------------------------------------------
# HardwareSensorsSensor: extra branches
# -----------------------------------------------------------------------------

class TestHardwareSensorsSensorExtraBranches:

    def test_occupancy_sensor_uses_raw_state(self):
        """Occupancy sensors should use state.state directly (no float conversion)."""
        hass = MagicMock()
        occ_state = MagicMock()
        occ_state.state = "on"
        occ_state.attributes = {"friendly_name": "Motion Sensor"}
        hass.states.get = MagicMock(return_value=occ_state)

        helpers_stub.get_entity_area.return_value = "Living Room"

        config_entry = MagicMock()
        config_entry.data = {}
        sensor = HardwareSensorsSensor(hass, {"occupancy": ["binary_sensor.motion"]}, config_entry)
        attrs = sensor.extra_state_attributes
        assert attrs["occupancy"][0]["value"] == "on"
        assert attrs["occupancy"][0]["unit"] is None

    def test_normalized_value_none_skips_entity(self):
        """When get_normalized_value returns (None, None) for a sensor, it should be excluded."""
        hass = MagicMock()
        power_state = MagicMock()
        power_state.state = "unavailable"
        power_state.attributes = {"unit_of_measurement": "W"}
        hass.states.get = MagicMock(return_value=power_state)

        helpers_stub.get_normalized_value.return_value = (None, None)

        config_entry = MagicMock()
        config_entry.data = {}
        sensor = HardwareSensorsSensor(hass, {"power": ["sensor.pwr"]}, config_entry)
        attrs = sensor.extra_state_attributes
        assert attrs["power"] == []

        # Restore stub default for other tests
        helpers_stub.get_normalized_value.return_value = (100.0, "W")


# -----------------------------------------------------------------------------
# GreenShiftBaseSensor / GreenShiftAISensor: callback methods
# -----------------------------------------------------------------------------

class TestGreenShiftBaseSensorCallbacks:

    def test_update_callback_writes_state_for_sync_sensor(self):
        """Sensors without _async_update_state call async_write_ha_state directly."""
        sensor = CurrentConsumptionSensor(make_collector())
        sensor.async_write_ha_state = MagicMock()
        sensor._update_callback()
        sensor.async_write_ha_state.assert_called_once()

    def test_update_callback_creates_task_for_async_sensor(self):
        """Sensors with _async_update_state use hass.async_create_task."""
        storage = AsyncMock()
        agent = make_agent()
        sensor = SavingsAccumulatedSensor(agent, make_collector(), storage)
        sensor.hass = MagicMock()
        sensor._update_callback()
        sensor.hass.async_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_added_to_hass_registers_listener(self):
        """async_added_to_hass should call async_on_remove with a dispatcher unsub."""
        sensor = CurrentConsumptionSensor(make_collector())
        sensor.hass = MagicMock()
        sensor.async_on_remove = MagicMock()
        sensor.async_write_ha_state = MagicMock()
        await sensor.async_added_to_hass()
        assert sensor.async_on_remove.called

    @pytest.mark.asyncio
    async def test_async_update_and_write_calls_update_then_write(self):
        """_async_update_and_write awaits _async_update_state then writes HA state."""
        storage = AsyncMock()
        storage.get_total_completed_tasks_count = AsyncMock(return_value=7)
        sensor = TasksCompletedSensor(storage)
        sensor.async_write_ha_state = MagicMock()
        await sensor._async_update_and_write()
        assert sensor._completed_count == 7
        sensor.async_write_ha_state.assert_called_once()


class TestGreenShiftAISensorCallbacks:

    def test_update_callback_creates_task_for_async_sensor(self):
        """AI sensor with _async_update_state creates an async task via hass."""
        storage = AsyncMock()
        agent = make_agent()
        sensor = WeeklyChallengeSensor(agent)
        sensor.hass = MagicMock()
        sensor._update_callback()
        sensor.hass.async_create_task.assert_called_once()

    def test_update_callback_writes_state_for_sync_sensor(self):
        """AI sensor without _async_update_state calls async_write_ha_state directly."""
        sensor = BehaviourIndexSensor(make_agent())
        sensor.async_write_ha_state = MagicMock()
        sensor._update_callback()
        sensor.async_write_ha_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_added_to_hass_registers_ai_listener(self):
        """AI sensor async_added_to_hass subscribes to GS_AI_UPDATE_SIGNAL."""
        sensor = BehaviourIndexSensor(make_agent())
        sensor.hass = MagicMock()
        sensor.async_on_remove = MagicMock()
        sensor.async_write_ha_state = MagicMock()
        await sensor.async_added_to_hass()
        assert sensor.async_on_remove.called


# -----------------------------------------------------------------------------
# SavingsAccumulatedSensor
# -----------------------------------------------------------------------------

class TestSavingsAccumulatedSensor:

    def _make_sensor(self, phase="baseline", active_since=None, baseline=1000.0, states=None):
        agent = make_agent(phase=phase, active_since=active_since, baseline_consumption=baseline)
        storage = AsyncMock()
        collector = make_collector(power=700.0)
        collector.get_current_state = MagicMock(return_value={"power": 700.0})
        sensor = SavingsAccumulatedSensor(agent, collector, storage)
        sensor.hass = make_hass(states or {})
        return sensor, agent, storage

    @pytest.mark.asyncio
    async def test_baseline_phase_native_value_is_zero(self):
        sensor, _, _ = self._make_sensor(phase="baseline")
        await sensor._async_update_state()
        assert sensor._attr_native_value == 0

    @pytest.mark.asyncio
    async def test_baseline_phase_note_present(self):
        sensor, _, _ = self._make_sensor(phase="baseline")
        await sensor._async_update_state()
        assert "Waiting for active phase" in sensor._attr_extra_state_attributes.get("note", "")

    @pytest.mark.asyncio
    async def test_active_phase_no_active_since_returns_zero(self):
        sensor, _, _ = self._make_sensor(phase="active", active_since=None)
        await sensor._async_update_state()
        assert sensor._attr_native_value == 0

    @pytest.mark.asyncio
    async def test_active_no_daily_data_uses_provisional_estimate(self):
        active_since = datetime(2026, 2, 1)
        sensor, _, storage = self._make_sensor(
            phase="active", active_since=active_since, baseline=1000.0
        )
        storage.get_active_phase_savings = AsyncMock(return_value={
            "days_with_data": 0,
            "total_savings_kwh": 0.0,
            "overall_avg_power_w": 0.0,
        })
        await sensor._async_update_state()
        assert "Provisional estimate" in sensor._attr_extra_state_attributes.get("note", "")

    @pytest.mark.asyncio
    async def test_active_with_data_calculates_savings(self):
        active_since = datetime(2026, 2, 1)
        sensor, _, storage = self._make_sensor(
            phase="active", active_since=active_since, baseline=1000.0,
            states={"input_number.electricity_price": "0.25"}
        )
        storage.get_active_phase_savings = AsyncMock(return_value={
            "days_with_data": 5,
            "total_savings_kwh": 2.0,
            "overall_avg_power_w": 800.0,
        })
        await sensor._async_update_state()
        # 2.0 kWh * 0.25 EUR/kWh = 0.50 EUR
        assert sensor._attr_native_value == pytest.approx(0.50, abs=0.01)

    @pytest.mark.asyncio
    async def test_active_with_data_attrs_contain_days_tracked(self):
        active_since = datetime(2026, 2, 1)
        sensor, _, storage = self._make_sensor(
            phase="active", active_since=active_since
        )
        storage.get_active_phase_savings = AsyncMock(return_value={
            "days_with_data": 10,
            "total_savings_kwh": 5.0,
            "overall_avg_power_w": 900.0,
        })
        await sensor._async_update_state()
        assert sensor._attr_extra_state_attributes.get("days_tracked") == 10

    def test_unit_of_measurement_uses_currency_input_select(self):
        sensor, _, _ = self._make_sensor()
        sensor.hass = make_hass({"input_select.currency": "USD"})
        assert sensor.unit_of_measurement == "USD"

    def test_unit_of_measurement_defaults_to_eur(self):
        sensor, _, _ = self._make_sensor()
        assert sensor.unit_of_measurement == "EUR"

    def test_extra_attributes_returns_stored_dict(self):
        sensor, _, _ = self._make_sensor()
        sensor._attr_extra_state_attributes = {"key": "value"}
        assert sensor.extra_state_attributes == {"key": "value"}


# -----------------------------------------------------------------------------
# CO2SavedSensor
# -----------------------------------------------------------------------------

class TestCO2SavedSensor:

    def _make_sensor(self, phase="baseline", active_since=None, baseline=1000.0):
        agent = make_agent(phase=phase, active_since=active_since, baseline_consumption=baseline)
        storage = AsyncMock()
        collector = make_collector(power=700.0)
        collector.get_current_state = MagicMock(return_value={"power": 700.0})
        sensor = CO2SavedSensor(agent, collector, storage)
        sensor.hass = make_hass({})
        return sensor, agent, storage

    @pytest.mark.asyncio
    async def test_baseline_phase_native_value_is_zero(self):
        sensor, _, _ = self._make_sensor(phase="baseline")
        await sensor._async_update_state()
        assert sensor._attr_native_value == 0
        assert sensor._attr_extra_state_attributes == {}

    @pytest.mark.asyncio
    async def test_active_no_active_since_returns_zero(self):
        sensor, _, _ = self._make_sensor(phase="active", active_since=None)
        await sensor._async_update_state()
        assert sensor._attr_native_value == 0

    @pytest.mark.asyncio
    async def test_active_no_daily_data_provisional_estimate(self):
        active_since = datetime(2026, 2, 1)
        sensor, _, storage = self._make_sensor(
            phase="active", active_since=active_since, baseline=1000.0
        )
        storage.get_active_phase_savings = AsyncMock(return_value={
            "days_with_data": 0,
            "total_savings_kwh": 0.0,
            "overall_avg_power_w": 0.0,
        })
        # helpers_stub.get_environmental_impact returns co2_kg=0.05
        await sensor._async_update_state()
        assert "Provisional estimate" in sensor._attr_extra_state_attributes.get("note", "")

    @pytest.mark.asyncio
    async def test_active_with_data_returns_co2_value(self):
        active_since = datetime(2026, 2, 1)
        sensor, _, storage = self._make_sensor(
            phase="active", active_since=active_since
        )
        storage.get_active_phase_savings = AsyncMock(return_value={
            "days_with_data": 3,
            "total_savings_kwh": 5.0,
            "overall_avg_power_w": 900.0,
        })
        helpers_stub.get_environmental_impact.return_value = {
            "co2_kg": 0.5, "trees": 0.02, "flights": 0.003, "km": 2.9
        }
        await sensor._async_update_state()
        assert sensor._attr_native_value == pytest.approx(0.5, abs=0.01)
        assert sensor._attr_extra_state_attributes["trees"] == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_active_with_data_attrs_contain_kwh(self):
        active_since = datetime(2026, 2, 1)
        sensor, _, storage = self._make_sensor(
            phase="active", active_since=active_since
        )
        storage.get_active_phase_savings = AsyncMock(return_value={
            "days_with_data": 2,
            "total_savings_kwh": 3.5,
            "overall_avg_power_w": 850.0,
        })
        helpers_stub.get_environmental_impact.return_value = {
            "co2_kg": 0.35, "trees": 0.01, "flights": 0.002, "km": 1.5
        }
        await sensor._async_update_state()
        assert sensor._attr_extra_state_attributes["total_savings_kwh"] == pytest.approx(3.5)


# -----------------------------------------------------------------------------
# TasksCompletedSensor
# -----------------------------------------------------------------------------

class TestTasksCompletedSensor:

    @pytest.mark.asyncio
    async def test_async_update_state_fetches_count_from_storage(self):
        storage = AsyncMock()
        storage.get_total_completed_tasks_count = AsyncMock(return_value=15)
        sensor = TasksCompletedSensor(storage)
        await sensor._async_update_state()
        assert sensor._completed_count == 15

    @pytest.mark.asyncio
    async def test_async_update_state_zero_count(self):
        storage = AsyncMock()
        storage.get_total_completed_tasks_count = AsyncMock(return_value=0)
        sensor = TasksCompletedSensor(storage)
        await sensor._async_update_state()
        assert sensor._completed_count == 0

    def test_state_returns_completed_count(self):
        storage = AsyncMock()
        sensor = TasksCompletedSensor(storage)
        sensor._completed_count = 7
        assert sensor.state == 7


# -----------------------------------------------------------------------------
# WeeklyChallengeSensor
# -----------------------------------------------------------------------------

class TestWeeklyChallengeSensor:

    def _make_sensor(self, phase="active", states=None):
        agent = make_agent(phase=phase)
        agent.get_weekly_challenge_status = AsyncMock(return_value={
            "progress": 65.0,
            "status": "on_track",
            "current_avg": 850.0,
            "target_avg": 800.0,
            "baseline": 1000.0,
            "week_start": "2026-02-02",
            "days_in_week": 3,
        })
        sensor = WeeklyChallengeSensor(agent)
        sensor.hass = make_hass(states or {})
        return sensor, agent

    def test_get_target_percentage_from_hass_state(self):
        sensor, _ = self._make_sensor(states={"input_number.energy_saving_target": "20"})
        assert sensor._get_target_percentage() == pytest.approx(20.0)

    def test_get_target_percentage_defaults_to_15(self):
        sensor, _ = self._make_sensor()
        assert sensor._get_target_percentage() == pytest.approx(15.0)

    def test_get_target_percentage_handles_invalid_state(self):
        sensor, _ = self._make_sensor(states={"input_number.energy_saving_target": "invalid"})
        assert sensor._get_target_percentage() == pytest.approx(15.0)

    @pytest.mark.asyncio
    async def test_async_update_state_baseline_phase_returns_early(self):
        sensor, _ = self._make_sensor(phase="baseline")
        sensor._attr_native_value = 42
        await sensor._async_update_state()
        # No update should have happened (early return)
        assert sensor._attr_native_value == 42

    @pytest.mark.asyncio
    async def test_async_update_state_active_sets_progress(self):
        sensor, agent = self._make_sensor(phase="active")
        await sensor._async_update_state()
        assert sensor._attr_native_value == pytest.approx(65.0)

    @pytest.mark.asyncio
    async def test_async_update_state_attrs_contain_status(self):
        sensor, agent = self._make_sensor(phase="active")
        await sensor._async_update_state()
        attrs = sensor._attr_extra_state_attributes
        assert attrs["status"] == "on_track"
        assert attrs["days_in_week"] == 3
        assert attrs["goal"] == pytest.approx(15.0)


class TestSensorAdditionalCoverage:

    @pytest.mark.asyncio
    async def test_sensor_async_setup_entry_adds_entities(self):
        hass = MagicMock()
        hass.data = {
            const_mod.DOMAIN: {
                "agent": make_agent(),
                "collector": make_collector(),
                "storage": AsyncMock(),
                "discovered_sensors": {"power": ["sensor.p1"]},
                "task_manager": MagicMock(),
            }
        }
        async_add_entities = MagicMock()
        config_entry = MagicMock()
        config_entry.data = {}

        await sensor_mod.async_setup_entry(hass, config_entry, async_add_entities)
        async_add_entities.assert_called_once()
        assert len(async_add_entities.call_args.args[0]) == 17

    @pytest.mark.asyncio
    async def test_base_sensor_async_update_and_write_path(self):
        class DummyBase(sensor_mod.GreenShiftBaseSensor):
            async def _async_update_state(self):
                self.updated = True

        sensor = DummyBase()
        sensor.updated = False
        sensor.async_write_ha_state = MagicMock()
        await sensor._async_update_and_write()
        assert sensor.updated is True
        sensor.async_write_ha_state.assert_called_once()

    def test_base_sensor_update_callback_async_path(self):
        class DummyBase(sensor_mod.GreenShiftBaseSensor):
            async def _async_update_state(self):
                return None

        sensor = DummyBase()
        sensor.hass = MagicMock()
        sensor._update_callback()
        sensor.hass.async_create_task.assert_called_once()

    def test_static_unit_properties(self):
        assert EnergyBaselineSensor(make_agent()).unit_of_measurement == "W"
        assert CurrentConsumptionSensor(make_collector()).unit_of_measurement == "W"
        assert DailyCO2EstimateSensor(make_hass(), make_collector()).unit_of_measurement == "kg"
        assert CO2SavedSensor(make_agent(), make_collector(), AsyncMock()).unit_of_measurement == "kg"

    @pytest.mark.asyncio
    async def test_savings_active_invalid_price_uses_default(self):
        active_since = datetime(2026, 2, 1)
        agent = make_agent(phase="active", active_since=active_since, baseline_consumption=1000.0)
        storage = AsyncMock()
        storage.get_active_phase_savings = AsyncMock(return_value={
            "days_with_data": 1,
            "total_savings_kwh": 2.0,
            "overall_avg_power_w": 900.0,
        })
        collector = make_collector(power=700.0)
        collector.get_current_state = MagicMock(return_value={"power": 700.0})
        sensor = SavingsAccumulatedSensor(agent, collector, storage)
        sensor.hass = make_hass({"input_number.electricity_price": "not-a-number"})

        await sensor._async_update_state()
        assert sensor._attr_native_value == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_savings_provisional_with_nonpositive_baseline(self):
        active_since = datetime(2026, 2, 1)
        agent = make_agent(phase="active", active_since=active_since, baseline_consumption=0.0)
        storage = AsyncMock()
        storage.get_active_phase_savings = AsyncMock(return_value={
            "days_with_data": 0,
            "total_savings_kwh": 0.0,
            "overall_avg_power_w": 0.0,
        })
        collector = make_collector(power=700.0)
        collector.get_current_state = MagicMock(return_value={"power": 700.0})
        sensor = SavingsAccumulatedSensor(agent, collector, storage)
        sensor.hass = make_hass({})

        await sensor._async_update_state()
        assert sensor._attr_extra_state_attributes["saving_watts"] == 0

    @pytest.mark.asyncio
    async def test_co2_saved_provisional_with_nonpositive_baseline(self):
        active_since = datetime(2026, 2, 1)
        agent = make_agent(phase="active", active_since=active_since, baseline_consumption=0.0)
        storage = AsyncMock()
        storage.get_active_phase_savings = AsyncMock(return_value={
            "days_with_data": 0,
            "total_savings_kwh": 0.0,
            "overall_avg_power_w": 0.0,
        })
        collector = make_collector(power=700.0)
        collector.get_current_state = MagicMock(return_value={"power": 700.0})
        sensor = CO2SavedSensor(agent, collector, storage)
        sensor.hass = make_hass({})

        await sensor._async_update_state()
        assert sensor._attr_native_value == 0
        assert sensor._attr_extra_state_attributes["trees"] == 0

    @pytest.mark.asyncio
    async def test_daily_tasks_async_update_callback_and_load(self):
        storage = AsyncMock()
        storage.get_today_tasks = AsyncMock(return_value=[make_task("t1")])
        sensor = DailyTasksSensor(storage)
        sensor.hass = MagicMock()
        sensor.async_write_ha_state = MagicMock()

        await sensor.async_added_to_hass()
        assert sensor.state == 1

        sensor.hass.async_create_task.reset_mock()

        sensor._update_callback()
        sensor.hass.async_create_task.assert_called_once()

        await sensor._async_update_and_write()
        sensor.async_write_ha_state.assert_called_once()

    def test_active_notifications_invalid_timestamp_sets_unknown(self):
        agent = make_agent(notification_history=[{
            "notification_id": "n1",
            "action_type": "specific",
            "timestamp": "not-an-iso-date",
            "responded": False,
        }])
        sensor = ActiveNotificationsSensor(agent)
        attrs = sensor.extra_state_attributes
        assert attrs["notifications"][0]["time_ago"] == "Unknown"

    def test_get_time_ago_handles_naive_timestamp_when_now_is_tz_aware(self):
        sensor = ActiveNotificationsSensor(make_agent())
        with patch.object(sensor_mod, "datetime") as mock_dt:
            aware_now = datetime.now(timezone.utc)
            mock_dt.now.return_value = aware_now
            naive_ts = datetime.now() - timedelta(minutes=5)
            result = sensor._get_time_ago(naive_ts)
        assert result.endswith("ago")
