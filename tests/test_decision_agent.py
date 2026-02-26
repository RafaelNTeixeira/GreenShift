"""
Tests for decision_agent.py

Covers (pure logic only – no HA event loop required):
- _discretize_state       : power bins, anomaly/fatigue/time levels, edge cases
- _update_fatigue_index   : count-based last-10 window, rejection rate, silence_penalty,
                            frequency_factor, time decay, edge cases (no streak penalty)
- _update_behaviour_index : EMA update, empty history guard
- _check_cooldown_with_opportunity: first call, critical bypass, high-opp reduced cooldown, standard cooldown
- get_weekly_challenge_status : on_track vs off_track, pending on insufficient data
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
helpers_stub.get_normalized_value = MagicMock(return_value=(100.0, "W"))
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
    """
    Tests for the count-based (last-10) fatigue window with components:
      - interaction_factor: unified score - explicit rejection (+1.0), passive silence (+0.4),
                            accepted (+0.0); normalised by window size
      - frequency_factor  : how quickly the last 3 notifications arrived
      - time_decay_factor : smoothly reduces fatigue when user has been left alone
    """

    @pytest.mark.asyncio
    async def test_no_history_sets_zero(self):
        agent = make_agent()
        agent.notification_history = deque(maxlen=100)
        await agent._update_fatigue_index()
        assert agent.fatigue_index == 0.0

    @pytest.mark.asyncio
    async def test_unanswered_notifications_raise_fatigue(self):
        """Unanswered notifications now contribute +0.4 each via interaction_factor."""
        agent = make_agent()
        # 2 unanswered (< 3 -> freq=0): interaction_score=0.8, factor=0.4
        # base = 0.6*0.4 + 0.4*0.0 = 0.24; no last_notification_time -> no decay
        agent.notification_history = deque(
            [_notif(responded=False), _notif(responded=False)],
            maxlen=100
        )
        await agent._update_fatigue_index()
        assert agent.fatigue_index == pytest.approx(0.24)

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
        assert agent.fatigue_index > 0.4

    @pytest.mark.asyncio
    async def test_time_decay_reduces_fatigue(self):
        """Time decay reduces fatigue when the user has been left alone for several hours."""
        agent = make_agent()
        agent.notification_history = deque(
            [_notif(accepted=False, responded=True, minutes_ago=180 + i * 10) for i in range(5)],
            maxlen=100
        )
        agent.last_notification_time = datetime.now() - timedelta(hours=3)
        await agent._update_fatigue_index()
        # All rejected -> interaction_factor=1.0, timestamps span negative -> freq=1.0
        # base_fatigue = 0.6*1.0 + 0.4*1.0 = 1.0
        # time_decay = max(0.2, 1.0 - (3-1)*0.1) = 0.8 -> fatigue = 1.0 * 0.8 = 0.8
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

    @pytest.mark.asyncio
    async def test_time_decay_reduces_old_rejection_fatigue(self):
        """Count-based window includes old notifications, but time_decay still reduces their impact."""
        agent = make_agent()
        agent.notification_history = deque(
            [_notif(accepted=False, responded=True, minutes_ago=300 + i * 10) for i in range(5)],
            maxlen=100
        )
        agent.last_notification_time = datetime.now() - timedelta(hours=5)
        await agent._update_fatigue_index()
        # All rejected -> interaction_factor=1.0, freq=1.0 -> base = 0.6*1.0 + 0.4*1.0 = 1.0
        # time_decay = max(0.2, 1-(5-1)*0.1) = 0.6 -> fatigue = 1.0 * 0.6 = 0.6
        assert agent.fatigue_index > 0.0              # still penalised (old rejects are in window)
        assert agent.fatigue_index == pytest.approx(0.6)  # time_decay lands exactly at 0.6

    @pytest.mark.asyncio
    async def test_recent_rejections_raise_fatigue_despite_old_accepts(self):
        """Recent rejections drive up fatigue even when many older responses were acceptances."""
        agent = make_agent()
        old_accepts = [_notif(accepted=True, responded=True, minutes_ago=300 + i * 10) for i in range(5)]
        recent_rejects = [_notif(accepted=False, responded=True, minutes_ago=10 + i * 5) for i in range(3)]
        agent.notification_history = deque(old_accepts + recent_rejects, maxlen=100)
        agent.last_notification_time = datetime.now() - timedelta(minutes=10)
        await agent._update_fatigue_index()
        # Both rejection_rate (3/8=0.375) and streak_penalty (3->0.6) push fatigue up
        assert agent.fatigue_index > 0.3

    @pytest.mark.asyncio
    async def test_all_unanswered_raises_fatigue_via_silence(self):
        """All-unanswered history: silence penalty (+0.4 each) raises fatigue, no early return."""
        agent = make_agent()
        # 10 unanswered at 0..90 min ago -> interaction_factor = (10*0.4)/10 = 0.4
        # last 3 timestamps (70,80,90 min ago) give negative span -> freq = 1.0
        # base = 0.6*0.4 + 0.4*1.0 = 0.64; last_notif=5min ago -> no decay
        agent.notification_history = deque(
            [_notif(responded=False, minutes_ago=10 * i) for i in range(10)],
            maxlen=100
        )
        agent.last_notification_time = datetime.now() - timedelta(minutes=5)
        await agent._update_fatigue_index()
        assert agent.fatigue_index == pytest.approx(0.64)

    @pytest.mark.asyncio
    async def test_silence_contributes_via_interaction_factor(self):
        """Unanswered notifications raise fatigue via the interaction_factor (0.4 each)."""
        agent = make_agent()
        # 2 responded (1 rejected, 1 accepted), 3 unanswered
        # interaction_score = 1.0 + 0.4 + 0.4 + 0.0 + 0.4 = 2.2 -> factor = 2.2/5 = 0.44
        agent.notification_history = deque([
            _notif(accepted=False, responded=True,  minutes_ago=25),
            _notif(responded=False,                 minutes_ago=20),
            _notif(responded=False,                 minutes_ago=15),
            _notif(accepted=True,  responded=True,  minutes_ago=10),
            _notif(responded=False,                 minutes_ago=5),
        ], maxlen=100)
        agent.last_notification_time = datetime.now() - timedelta(minutes=5)
        await agent._update_fatigue_index()
        assert agent.fatigue_index > 0.0

    @pytest.mark.asyncio
    async def test_rejection_rate_drives_fatigue(self):
        """Higher rejection rate yields higher fatigue than lower rejection rate (no streak effect)."""
        agent = make_agent()

        # Scenario A: 1 out of 4 rejected (rejection_rate = 0.25)
        agent.notification_history = deque([
            _notif(accepted=True,  responded=True, minutes_ago=20),
            _notif(accepted=True,  responded=True, minutes_ago=15),
            _notif(accepted=True,  responded=True, minutes_ago=10),
            _notif(accepted=False, responded=True, minutes_ago=5),
        ], maxlen=100)
        agent.last_notification_time = datetime.now() - timedelta(minutes=5)
        await agent._update_fatigue_index()
        fatigue_low_rejection = agent.fatigue_index

        # Scenario B: 4 out of 4 rejected (rejection_rate = 1.0), same pattern
        agent.notification_history = deque([
            _notif(accepted=False, responded=True, minutes_ago=20),
            _notif(accepted=False, responded=True, minutes_ago=15),
            _notif(accepted=False, responded=True, minutes_ago=10),
            _notif(accepted=False, responded=True, minutes_ago=5),
        ], maxlen=100)
        agent.last_notification_time = datetime.now() - timedelta(minutes=5)
        await agent._update_fatigue_index()
        fatigue_high_rejection = agent.fatigue_index

        assert fatigue_high_rejection > fatigue_low_rejection


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

    def test_rejection_scores_normalized_not_clipped(self):
        """Engagement scores are +1.0 (accept) or -0.5 (reject)."""
        agent = make_agent()
        agent.behaviour_index = 0.8
        agent.engagement_history = deque([-0.5] * 20, maxlen=100)
        for _ in range(10):
            agent._update_behaviour_index()
        # Should converge toward 0.0, definitely well below the initial 0.8
        assert agent.behaviour_index < 0.4

    def test_50_50_engagement_converges_to_midpoint(self):
        """Equal accepts (+1.0) and rejects (-0.5) should produce a weighted average
        near 0.25 raw, which normalises to 0.5 — so the index should hover near 0.5."""
        agent = make_agent()
        agent.behaviour_index = 0.5
        agent.engagement_history = deque([1.0, -0.5] * 10, maxlen=100)
        for _ in range(15):
            agent._update_behaviour_index()
        # After enough updates with 50/50 signal the index should be close to 0.5
        assert agent.behaviour_index == pytest.approx(0.5, abs=0.1)

    def test_all_rejections_drive_index_toward_zero(self):
        """History full of -0.5 scores should drive the index toward 0, not to 0.25."""
        agent = make_agent()
        agent.behaviour_index = 1.0
        agent.engagement_history = deque([-0.5] * 50, maxlen=100)
        for _ in range(30):
            agent._update_behaviour_index()
        assert agent.behaviour_index < 0.15


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
    async def test_consumption_below_target_is_on_track(self):
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        # Generate 1 full hour of data at 800W (20% below 1000W baseline)
        readings_per_hour = int(3600 / 15)
        data = [(i, 800.0) for i in range(readings_per_hour)]
        agent.data_collector.get_power_history = AsyncMock(return_value=data)
        status = await agent.get_weekly_challenge_status(target_percentage=15.0)
        # Target avg = 1000 * 0.85 = 850 W; current avg = 800 W -> progress < 100%
        assert status["status"] == "on_track"
        assert status["progress"] < 100.0

    @pytest.mark.asyncio
    async def test_consumption_above_target_is_off_track(self):
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        readings_per_hour = int(3600 / 15)
        data = [(i, 950.0) for i in range(readings_per_hour)]
        agent.data_collector.get_power_history = AsyncMock(return_value=data)
        status = await agent.get_weekly_challenge_status(target_percentage=15.0)
        # Target avg = 850 W; current avg = 950 W -> progress > 100%
        assert status["status"] == "off_track"
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


# ─────────────────────────────────────────────────────────────────────────────
# _update_anomaly_index — baseline-consumption signal
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateAnomalyIndex:
    """
    Verifies that _update_anomaly_index uses *both* signals:
      1. Local z-score  — detects sudden spikes within the 1-hour window
      2. Baseline deviation — detects sustained high consumption that the
                              local z-score misses (window mean rises with load)
    """

    @pytest.mark.asyncio
    async def test_insufficient_data_sets_zero(self):
        agent = make_agent()
        # Fewer readings than 80% of the expected hourly count -> index stays 0
        agent.data_collector.get_power_history = AsyncMock(return_value=[(i, 500.0) for i in range(5)])
        await agent._update_anomaly_index()
        assert agent.anomaly_index == 0.0

    @pytest.mark.asyncio
    async def test_no_variance_and_no_baseline_sets_zero(self):
        """Constant consumption with no historical baseline -> zero anomaly."""
        agent = make_agent()
        agent.baseline_consumption = 0.0
        # Generate enough readings (≥80% of hourly) all equal -> std=0
        readings_per_hour = int(3600 / 15)
        data = [(i, 500.0) for i in range(readings_per_hour)]
        agent.data_collector.get_power_history = AsyncMock(return_value=data)
        await agent._update_anomaly_index()
        assert agent.anomaly_index == 0.0

    @pytest.mark.asyncio
    async def test_sustained_high_consumption_detected_via_baseline(self):
        """Consumption consistently 50% above baseline should raise anomaly_index
        even when there is no variance in the 1-hour window (z-score would be 0)."""
        agent = make_agent()
        agent.baseline_consumption = 500.0  # Historical baseline
        readings_per_hour = int(3600 / 15)
        # All readings at 750 W (50% above 500 W baseline): no variance, z-score = 0
        data = [(i, 750.0) for i in range(readings_per_hour)]
        agent.data_collector.get_power_history = AsyncMock(return_value=data)
        await agent._update_anomaly_index()
        # baseline_deviation = (750-500)/500 = 0.5  ->  anomaly_index = 0.5
        assert agent.anomaly_index == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_baseline_below_threshold_not_flagged(self):
        """Consumption 20% above baseline is below the 30% threshold -> no anomaly."""
        agent = make_agent()
        agent.baseline_consumption = 500.0
        readings_per_hour = int(3600 / 15)
        data = [(i, 600.0) for i in range(readings_per_hour)]  # 20% above baseline
        agent.data_collector.get_power_history = AsyncMock(return_value=data)
        await agent._update_anomaly_index()
        assert agent.anomaly_index == 0.0

    @pytest.mark.asyncio
    async def test_sudden_spike_detected_by_zscore_even_without_baseline(self):
        """A spike within the window should still be caught by the local z-score
        when baseline_consumption is 0 (baseline phase)."""
        agent = make_agent()
        agent.baseline_consumption = 0.0
        readings_per_hour = int(3600 / 15)
        # Mostly 500 W, last reading is a large spike at 1500 W
        data = [(i, 500.0) for i in range(readings_per_hour - 1)] + [(readings_per_hour, 1500.0)]
        agent.data_collector.get_power_history = AsyncMock(return_value=data)
        await agent._update_anomaly_index()
        # z-score of spike should be large enough to produce non-zero anomaly
        assert agent.anomaly_index > 0.0

    @pytest.mark.asyncio
    async def test_anomaly_index_capped_at_one(self):
        """anomaly_index must never exceed 1.0 regardless of how extreme the deviation is."""
        agent = make_agent()
        agent.baseline_consumption = 100.0
        readings_per_hour = int(3600 / 15)
        # 10× the baseline — deviation = 9.0 -> clamped to 1.0
        data = [(i, 1000.0) for i in range(readings_per_hour)]
        agent.data_collector.get_power_history = AsyncMock(return_value=data)
        await agent._update_anomaly_index()
        assert agent.anomaly_index <= 1.0
        assert agent.anomaly_index > 0.5


# ─────────────────────────────────────────────────────────────────────────────
# _calculate_reward_with_feedback
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateRewardWithFeedback:
    """Verify that _calculate_reward_with_feedback uses initial_power (notification-
    time snapshot) in addition to baseline_consumption, so a real power drop is
    credited even when current power is still above the historical baseline."""

    @pytest.mark.asyncio
    async def test_large_drop_from_initial_power_credited(self):
        """2500 W -> 1200 W drop gives positive energy_saving even when 1200 > baseline."""
        agent = make_agent()
        agent.baseline_consumption = 1000.0  # baseline below current (old code would give 0)
        agent.fatigue_index = 0.0
        agent.data_collector.get_current_state = MagicMock(return_value={"power": 1200.0})

        # direct_impact  = (2500 - 1200) / 2500 = 0.52
        # baseline_comp  = (1000 - 1200) / 1000 = -0.20
        # energy_saving  = max(0, 0.5*0.52 + 0.5*(-0.20)) = max(0, 0.16) = 0.16
        # reward         = 1.0*0.16 + 0.5*1.0 - 0.3*0 = 0.66
        reward = await agent._calculate_reward_with_feedback(accepted=True, initial_power=2500.0)

        assert reward == pytest.approx(0.66, abs=0.01)

    @pytest.mark.asyncio
    async def test_no_change_from_initial_power_uses_baseline_signal(self):
        """When power doesn't change from initial, only baseline signal contributes."""
        agent = make_agent()
        agent.baseline_consumption = 2000.0
        agent.fatigue_index = 0.0
        agent.data_collector.get_current_state = MagicMock(return_value={"power": 1500.0})

        # direct_impact  = (1500 - 1500) / 1500 = 0.0
        # baseline_comp  = (2000 - 1500) / 2000 = 0.25
        # energy_saving  = max(0, 0.5*0 + 0.5*0.25) = 0.125
        # reward         = 1.0*0.125 + 0.5*1.0 = 0.625
        reward = await agent._calculate_reward_with_feedback(accepted=True, initial_power=1500.0)

        assert reward == pytest.approx(0.625, abs=0.01)

    @pytest.mark.asyncio
    async def test_zero_initial_power_falls_back_to_baseline_only(self):
        """When initial_power=0 (data unavailable), falls back to baseline comparison."""
        agent = make_agent()
        agent.baseline_consumption = 800.0
        agent.fatigue_index = 0.0
        agent.data_collector.get_current_state = MagicMock(return_value={"power": 600.0})

        # initial_power = 0 -> falls back to baseline-only branch
        # energy_saving = max(0, (800 - 600) / 800) = 0.25
        # reward        = 1.0*0.25 + 0.5*1.0 = 0.75
        reward = await agent._calculate_reward_with_feedback(accepted=True, initial_power=0.0)

        assert reward == pytest.approx(0.75, abs=0.01)

    @pytest.mark.asyncio
    async def test_accepted_yields_higher_reward_than_rejected(self):
        """Same power context: accepting a notification must reward more than rejecting."""
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        agent.fatigue_index = 0.0

        agent.data_collector.get_current_state = MagicMock(return_value={"power": 700.0})
        reward_accept = await agent._calculate_reward_with_feedback(accepted=True, initial_power=1000.0)

        agent.data_collector.get_current_state = MagicMock(return_value={"power": 700.0})
        reward_reject = await agent._calculate_reward_with_feedback(accepted=False, initial_power=1000.0)

        assert reward_accept > reward_reject

    @pytest.mark.asyncio
    async def test_rejection_with_energy_increase_gives_negative_reward(self):
        """Rejected notification + power went up -> reward must be negative."""
        agent = make_agent()
        agent.baseline_consumption = 500.0
        agent.fatigue_index = 0.0
        # Power went UP from initial: direct_impact < 0; baseline also below
        agent.data_collector.get_current_state = MagicMock(return_value={"power": 800.0})

        # direct_impact  = (500 - 800) / 500 = -0.6
        # baseline_comp  = (500 - 800) / 500 = -0.6
        # energy_saving  = max(0, ...) = 0.0
        # feedback       = -0.5
        # reward         = 0 + 0.5*(-0.5) - 0 = -0.25
        reward = await agent._calculate_reward_with_feedback(accepted=False, initial_power=500.0)

        assert reward < 0
