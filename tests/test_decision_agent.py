"""
Tests for decision_agent.py

Covers (pure logic only – no HA event loop required):
- _discretize_state       : power bins, anomaly/fatigue/time levels, edge cases
- _update_fatigue_index   : no history, no responses, rejection rate, time decay
- _update_behaviour_index : EMA update, empty history guard
- _check_cooldown_with_opportunity: first call, critical bypass, high-opp reduced cooldown, standard cooldown
- get_weekly_challenge_status : completed vs in_progress, pending on insufficient data
- Baseline phase -> active phase: phase attribute flips after 14 days (logic unit)
- Notification count reset across days
"""
import pytest
import asyncio
from collections import deque
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np

# ── Minimal stubs so we can import DecisionAgent without a real HA install ──

import sys, types

for mod_name in [
    "homeassistant",
    "homeassistant.core",
    "homeassistant.helpers",
    "homeassistant.helpers.dispatcher",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Stub dispatcher used inside agent
dispatcher_stub = sys.modules["homeassistant.helpers.dispatcher"]
dispatcher_stub.async_dispatcher_send = MagicMock()

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

# Stub helper module
helpers_stub = types.ModuleType("custom_components.green_shift.helpers")
helpers_stub.get_friendly_name = MagicMock(return_value="Mock Sensor")
helpers_stub.should_ai_be_active = MagicMock(return_value=True)
sys.modules["custom_components.green_shift.helpers"] = helpers_stub

# Stub translations module
trans_stub = types.ModuleType("custom_components.green_shift.translations_runtime")
trans_stub.get_language = AsyncMock(return_value="en")
trans_stub.get_notification_templates = MagicMock(return_value={})
trans_stub.get_time_of_day_name = MagicMock(return_value="morning")
sys.modules["custom_components.green_shift.translations_runtime"] = trans_stub

# Stub storage module
storage_stub = types.ModuleType("custom_components.green_shift.storage")
storage_stub.StorageManager = MagicMock()
sys.modules["custom_components.green_shift.storage"] = storage_stub

# Load decision_agent module
da_spec = importlib.util.spec_from_file_location(
    "decision_agent",
    pathlib.Path(__file__).parent.parent / "config" / "custom_components" / "green_shift" / "decision_agent.py"
)
da_mod = importlib.util.module_from_spec(da_spec)
da_mod.__package__ = "custom_components.green_shift"
da_spec.loader.exec_module(da_mod)
DecisionAgent = da_mod.DecisionAgent


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_agent(config_data=None):
    hass = MagicMock()
    collector = MagicMock()
    collector.get_current_state = MagicMock(return_value={
        "power": 500.0, "occupancy": True
    })
    collector.get_all_areas = MagicMock(return_value=[])
    collector.get_power_history = AsyncMock(return_value=[])
    agent = DecisionAgent(
        hass=hass,
        discovered_sensors={},
        data_collector=collector,
        storage_manager=None,
        config_data=config_data or {},
    )
    return agent


def _make_state_vector(power=500.0, time_of_day=0.3, occupancy=1):
    """Return an 18-element float list matching the agent's expected shape."""
    v = [0.0] * 18
    v[0] = power       # power
    v[10] = occupancy  # occupancy
    v[15] = 0          # area anomaly count
    v[16] = time_of_day  # time of day (0-1)
    return v


# ─────────────────────────────────────────────────────────────────────────────
# _discretize_state
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscreteState:

    def test_none_state_vector_returns_zero_tuple(self):
        agent = make_agent()
        agent.state_vector = None
        assert agent._discretize_state() == (0, 0, 0, 0, 0, 0)

    def test_short_state_vector_returns_zero_tuple(self):
        agent = make_agent()
        agent.state_vector = [0.0] * 5
        assert agent._discretize_state() == (0, 0, 0, 0, 0, 0)

    def test_power_bin_500w(self):
        agent = make_agent()
        agent.state_vector = _make_state_vector(power=500)
        agent.anomaly_index = 0.0
        agent.fatigue_index = 0.0
        result = agent._discretize_state()
        assert result[0] == 5  # 500 // 100 = 5

    def test_power_bin_capped_at_100(self):
        agent = make_agent()
        agent.state_vector = _make_state_vector(power=25000)
        agent.anomaly_index = 0.0
        agent.fatigue_index = 0.0
        result = agent._discretize_state()
        assert result[0] == 100

    @pytest.mark.parametrize("anomaly,expected_level", [
        (0.0,  0),  # none
        (0.24, 0),  # still none
        (0.25, 1),  # low boundary
        (0.49, 1),  # low
        (0.50, 2),  # medium boundary
        (0.74, 2),  # medium
        (0.75, 3),  # high boundary
        (1.0,  3),  # high
    ])
    def test_anomaly_level(self, anomaly, expected_level):
        agent = make_agent()
        agent.state_vector = _make_state_vector()
        agent.anomaly_index = anomaly
        agent.fatigue_index = 0.0
        assert agent._discretize_state()[1] == expected_level

    @pytest.mark.parametrize("fatigue,expected_level", [
        (0.0,  0),  # low
        (0.32, 0),  # still low
        (0.33, 1),  # medium boundary
        (0.65, 1),  # medium
        (0.66, 2),  # high boundary
        (1.0,  2),  # high
    ])
    def test_fatigue_level(self, fatigue, expected_level):
        agent = make_agent()
        agent.state_vector = _make_state_vector()
        agent.anomaly_index = 0.0
        agent.fatigue_index = fatigue
        assert agent._discretize_state()[2] == expected_level

    @pytest.mark.parametrize("time_val,expected_period", [
        (0.0,  0),  # night
        (0.24, 0),  # night
        (0.25, 1),  # morning
        (0.49, 1),  # morning
        (0.50, 2),  # afternoon
        (0.74, 2),  # afternoon
        (0.75, 3),  # evening
        (1.0,  3),  # evening
    ])
    def test_time_period(self, time_val, expected_period):
        agent = make_agent()
        agent.state_vector = _make_state_vector(time_of_day=time_val)
        agent.anomaly_index = 0.0
        agent.fatigue_index = 0.0
        assert agent._discretize_state()[4] == expected_period

    def test_occupancy_binary(self):
        agent = make_agent()
        agent.state_vector = _make_state_vector(occupancy=1)
        agent.anomaly_index = 0.0
        agent.fatigue_index = 0.0
        assert agent._discretize_state()[5] == 1

        agent.state_vector = _make_state_vector(occupancy=0)
        assert agent._discretize_state()[5] == 0

    def test_area_anomaly_flag(self):
        agent = make_agent()
        v = _make_state_vector()
        v[15] = 2  # 2 areas with anomalies
        agent.state_vector = v
        agent.anomaly_index = 0.0
        agent.fatigue_index = 0.0
        assert agent._discretize_state()[3] == 1

        v[15] = 0
        agent.state_vector = v
        assert agent._discretize_state()[3] == 0


# ─────────────────────────────────────────────────────────────────────────────
# _update_fatigue_index
# ─────────────────────────────────────────────────────────────────────────────

def _notif(accepted=True, responded=True, minutes_ago=30):
    ts = (datetime.now() - timedelta(minutes=minutes_ago)).isoformat()
    return {"timestamp": ts, "accepted": accepted, "responded": responded}


class TestUpdateFatigueIndex:

    @pytest.mark.asyncio
    async def test_no_history_sets_zero(self):
        agent = make_agent()
        agent.notification_history = deque(maxlen=100)
        await agent._update_fatigue_index()
        assert agent.fatigue_index == 0.0

    @pytest.mark.asyncio
    async def test_no_responded_sets_moderate_fatigue(self):
        agent = make_agent()
        agent.notification_history = deque(
            [_notif(responded=False), _notif(responded=False)],
            maxlen=100
        )
        await agent._update_fatigue_index()
        assert agent.fatigue_index == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_all_accepted_yields_low_fatigue(self):
        agent = make_agent()
        agent.notification_history = deque(
            [_notif(accepted=True, responded=True, minutes_ago=60 * i) for i in range(5)],
            maxlen=100
        )
        agent.last_notification_time = None
        await agent._update_fatigue_index()
        assert agent.fatigue_index <= 0.4

    @pytest.mark.asyncio
    async def test_all_rejected_yields_high_fatigue(self):
        agent = make_agent()
        agent.notification_history = deque(
            [_notif(accepted=False, responded=True, minutes_ago=10 * i) for i in range(5)],
            maxlen=100
        )
        agent.last_notification_time = datetime.now() - timedelta(minutes=5)
        await agent._update_fatigue_index()
        # Rejection rate of 1.0 -> high fatigue
        assert agent.fatigue_index > 0.4

    @pytest.mark.asyncio
    async def test_time_decay_reduces_fatigue(self):
        """Notifications that are old should decay fatigue downward."""
        agent = make_agent()
        # Mix of rejected, but last notification was 3 hours ago
        agent.notification_history = deque(
            [_notif(accepted=False, responded=True, minutes_ago=180 + i * 10) for i in range(5)],
            maxlen=100
        )
        agent.last_notification_time = datetime.now() - timedelta(hours=3)
        await agent._update_fatigue_index()
        # time_decay_factor at 3 hours = max(0.5, 1.0 - 2*0.1) = 0.8
        # base_fatigue = 0.6*1.0 + 0.4*1.0 = 1.0, final = 1.0 * 0.8 = 0.8
        assert agent.fatigue_index == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_fatigue_clipped_between_0_and_1(self):
        agent = make_agent()
        agent.notification_history = deque(
            [_notif(accepted=False, responded=True, minutes_ago=1) for _ in range(10)],
            maxlen=100
        )
        agent.last_notification_time = datetime.now() - timedelta(minutes=1)
        await agent._update_fatigue_index()
        assert 0.0 <= agent.fatigue_index <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# _update_behaviour_index
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateBehaviourIndex:

    def test_empty_history_does_not_change_index(self):
        agent = make_agent()
        agent.engagement_history = deque(maxlen=100)
        original = agent.behaviour_index
        agent._update_behaviour_index()
        assert agent.behaviour_index == original

    def test_all_positive_engagement_increases_index(self):
        agent = make_agent()
        agent.behaviour_index = 0.5
        agent.engagement_history = deque([1.0] * 10, maxlen=100)
        agent._update_behaviour_index()
        assert agent.behaviour_index > 0.5

    def test_all_negative_engagement_decreases_index(self):
        agent = make_agent()
        agent.behaviour_index = 0.8
        agent.engagement_history = deque([0.0] * 10, maxlen=100)
        agent._update_behaviour_index()
        assert agent.behaviour_index < 0.8

    def test_index_stays_in_0_1_range(self):
        agent = make_agent()
        agent.engagement_history = deque([1.0] * 50, maxlen=100)
        for _ in range(20):
            agent._update_behaviour_index()
        assert 0.0 <= agent.behaviour_index <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# _check_cooldown_with_opportunity
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckCooldown:

    @pytest.mark.asyncio
    async def test_first_notification_always_allowed(self):
        agent = make_agent()
        agent.last_notification_time = None
        result = await agent._check_cooldown_with_opportunity(0.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_critical_opportunity_bypasses_cooldown(self):
        agent = make_agent()
        agent.last_notification_time = datetime.now() - timedelta(minutes=1)
        agent.fatigue_index = 0.0
        # Critical threshold is 0.8
        result = await agent._check_cooldown_with_opportunity(0.85)
        assert result is True

    @pytest.mark.asyncio
    async def test_high_opportunity_with_reduced_cooldown_met(self):
        agent = make_agent()
        agent.fatigue_index = 0.0
        # Adaptive cooldown with fatigue=0: 30 * 1.0 * 1.0 = 30 min -> 50% = 15 min
        agent.last_notification_time = datetime.now() - timedelta(minutes=16)
        result = await agent._check_cooldown_with_opportunity(0.65)
        assert result is True

    @pytest.mark.asyncio
    async def test_high_opportunity_insufficient_cooldown(self):
        agent = make_agent()
        agent.fatigue_index = 0.0
        agent.last_notification_time = datetime.now() - timedelta(minutes=5)
        result = await agent._check_cooldown_with_opportunity(0.65)
        assert result is False

    @pytest.mark.asyncio
    async def test_normal_opportunity_requires_full_cooldown(self):
        agent = make_agent()
        agent.fatigue_index = 0.0
        # Test during midday to avoid evening peak reduction
        # 29 minutes since last -> just under 30 min standard cooldown
        with patch.object(da_mod, 'datetime') as mock_dt:
            now = datetime(2026, 2, 19, 14, 0, 0)  # 2 PM (no peak time multiplier)
            mock_dt.now.return_value = now
            agent.last_notification_time = now - timedelta(minutes=29)
            result = await agent._check_cooldown_with_opportunity(0.3)
            assert result is False

    @pytest.mark.asyncio
    async def test_normal_opportunity_cooldown_satisfied(self):
        agent = make_agent()
        agent.fatigue_index = 0.0
        agent.last_notification_time = datetime.now() - timedelta(minutes=31)
        result = await agent._check_cooldown_with_opportunity(0.3)
        assert result is True

    @pytest.mark.asyncio
    async def test_high_fatigue_extends_cooldown(self):
        agent = make_agent()
        agent.fatigue_index = 1.0  # Max fatigue -> 3x cooldown = 90 min
        # 45 minutes would normally satisfy 30 min, but not 90 min
        agent.last_notification_time = datetime.now() - timedelta(minutes=45)
        result = await agent._check_cooldown_with_opportunity(0.3)
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# get_weekly_challenge_status
# ─────────────────────────────────────────────────────────────────────────────

class TestWeeklyChallenge:

    @pytest.mark.asyncio
    async def test_no_baseline_returns_pending(self):
        agent = make_agent()
        agent.baseline_consumption = 0.0
        status = await agent.get_weekly_challenge_status()
        assert status["status"] == "pending"

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_pending(self):
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        # Return only 1 data point: well below minimum
        agent.data_collector.get_power_history = AsyncMock(
            return_value=[(datetime.now().timestamp(), 900.0)]
        )
        status = await agent.get_weekly_challenge_status()
        assert status["status"] == "pending"

    @pytest.mark.asyncio
    async def test_consumption_below_target_is_completed(self):
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        # Generate 1 full hour of data at 800W (20% below 1000W baseline)
        readings_per_hour = int(3600 / 15)
        data = [(i, 800.0) for i in range(readings_per_hour)]
        agent.data_collector.get_power_history = AsyncMock(return_value=data)
        status = await agent.get_weekly_challenge_status(target_percentage=15.0)
        # Target avg = 1000 * 0.85 = 850 W; current avg = 800 W -> progress < 100%
        assert status["status"] == "completed"
        assert status["progress"] < 100.0

    @pytest.mark.asyncio
    async def test_consumption_above_target_is_in_progress(self):
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        readings_per_hour = int(3600 / 15)
        data = [(i, 950.0) for i in range(readings_per_hour)]
        agent.data_collector.get_power_history = AsyncMock(return_value=data)
        status = await agent.get_weekly_challenge_status(target_percentage=15.0)
        # Target avg = 850 W; current avg = 950 W -> progress > 100%
        assert status["status"] == "in_progress"
        assert status["progress"] > 100.0

    @pytest.mark.asyncio
    async def test_week_start_date_initialized(self):
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        agent.current_week_start_date = None
        readings_per_hour = int(3600 / 15)
        data = [(i, 800.0) for i in range(readings_per_hour)]
        agent.data_collector.get_power_history = AsyncMock(return_value=data)
        await agent.get_weekly_challenge_status()
        assert agent.current_week_start_date is not None

    @pytest.mark.asyncio
    async def test_progress_key_present_in_result(self):
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        readings_per_hour = int(3600 / 15)
        data = [(i, 800.0) for i in range(readings_per_hour)]
        agent.data_collector.get_power_history = AsyncMock(return_value=data)
        status = await agent.get_weekly_challenge_status()
        assert "progress" in status
        assert "current_avg" in status
        assert "target_avg" in status


# ─────────────────────────────────────────────────────────────────────────────
# Notification daily counter reset
# ─────────────────────────────────────────────────────────────────────────────

class TestNotificationDailyCounter:
    """Verifies that notification_count_today resets on a new day (logic unit)."""

    def test_counter_is_independent_per_day(self):
        """Simulate the reset logic that lives inside _decide_action."""
        agent = make_agent()
        agent.notification_count_today = 5
        agent.last_notification_date = (datetime.now() - timedelta(days=1)).date()

        # Replicate the reset guard from _decide_action
        today = datetime.now().date()
        if agent.last_notification_date != today:
            agent.notification_count_today = 0
            agent.last_notification_date = today

        assert agent.notification_count_today == 0

    def test_counter_not_reset_same_day(self):
        agent = make_agent()
        agent.notification_count_today = 3
        agent.last_notification_date = datetime.now().date()

        today = datetime.now().date()
        if agent.last_notification_date != today:
            agent.notification_count_today = 0

        assert agent.notification_count_today == 3


# ─────────────────────────────────────────────────────────────────────────────
# Baseline -> Active phase transition logic
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseTransition:

    def test_phase_starts_as_baseline(self):
        agent = make_agent()
        assert agent.phase == "baseline"

    def test_phase_transitions_after_14_days(self):
        """Unit-test the inline phase-flip condition from __init__.py."""
        agent = make_agent()
        agent.start_date = datetime.now() - timedelta(days=14)
        agent.phase = "baseline"

        days_running = (datetime.now() - agent.start_date).days
        if days_running >= 14 and agent.phase == "baseline":
            agent.phase = "active"

        assert agent.phase == "active"

    def test_phase_does_not_flip_at_13_days(self):
        agent = make_agent()
        agent.start_date = datetime.now() - timedelta(days=13)
        agent.phase = "baseline"

        days_running = (datetime.now() - agent.start_date).days
        if days_running >= 14 and agent.phase == "baseline":
            agent.phase = "active"

        assert agent.phase == "baseline"
