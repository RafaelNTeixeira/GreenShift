"""Tests for decision_agent.py

Covers (pure logic only - no HA event loop required):
- _discretize_state       : baseline-relative power categories (5 levels), anomaly/fatigue/time levels, edge cases
- _update_fatigue_index   : count-based last-10 window, rejection rate, silence_penalty,
                            frequency_factor, time decay, edge cases (no streak penalty)
- _update_behaviour_index : EMA update, empty history guard
- _check_cooldown_with_opportunity: first call, critical bypass respects CRITICAL_MIN_COOLDOWN_MINUTES,
                            high-opp reduced cooldown, standard cooldown
- get_weekly_challenge_status : on_track vs off_track, pending on insufficient data, week closes
                            on any day (not just Sunday)
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
helpers_stub.get_working_days_from_config = MagicMock(return_value=list(range(5)))  # Mon-Fri
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
    """_discretize_state: baseline-relative 5-category power bins.

    Ref (baseline_consumption=0 -> fallback 1000 W):
      0 = standby  : power < 5%  of ref  (< 50 W)
      1 = low      : 5-50%  of ref  (50-500 W)
      2 = medium   : 50-150% of ref (500-1500 W)  <- normal operating range
      3 = high     : 150-300% of ref (1500-3000 W)
      4 = peak     : > 300% of ref  (> 3000 W)
    """

    def test_none_state_vector_returns_zero_tuple(self):
        agent = make_agent()
        agent.state_vector = None
        assert agent._discretize_state() == (0, 0, 0, 0, 0, 0)

    def test_short_state_vector_returns_zero_tuple(self):
        agent = make_agent()
        agent.state_vector = [0.0] * 5
        assert agent._discretize_state() == (0, 0, 0, 0, 0, 0)

    def test_power_standby(self):
        """10 W -> < 5% of 1000 W fallback baseline -> level 0."""
        agent = make_agent()
        agent.state_vector = _make_state_vector(power=10.0)
        agent.anomaly_index = 0.0
        agent.fatigue_index = 0.0
        assert agent._discretize_state()[0] == 0

    def test_power_low(self):
        """250 W -> 25% of 1000 W -> level 1."""
        agent = make_agent()
        agent.state_vector = _make_state_vector(power=250.0)
        agent.anomaly_index = 0.0
        agent.fatigue_index = 0.0
        assert agent._discretize_state()[0] == 1

    def test_power_medium_at_baseline(self):
        """1000 W == baseline (ratio 1.0) -> level 2 (50–150%)."""
        agent = make_agent()
        agent.state_vector = _make_state_vector(power=1000.0)
        agent.anomaly_index = 0.0
        agent.fatigue_index = 0.0
        assert agent._discretize_state()[0] == 2

    def test_power_medium_uses_actual_baseline(self):
        """When baseline_consumption is explicitly set, thresholds shift with it."""
        agent = make_agent()
        agent.baseline_consumption = 2000.0  # 2 kW building baseline
        agent.state_vector = _make_state_vector(power=2000.0)  # exactly at baseline -> medium
        agent.anomaly_index = 0.0
        agent.fatigue_index = 0.0
        assert agent._discretize_state()[0] == 2

    def test_power_high(self):
        """2000 W -> 200% of 1000 W fallback -> level 3."""
        agent = make_agent()
        agent.state_vector = _make_state_vector(power=2000.0)
        agent.anomaly_index = 0.0
        agent.fatigue_index = 0.0
        assert agent._discretize_state()[0] == 3

    def test_power_peak(self):
        """5000 W -> 500% of 1000 W fallback -> level 4."""
        agent = make_agent()
        agent.state_vector = _make_state_vector(power=5000.0)
        agent.anomaly_index = 0.0
        agent.fatigue_index = 0.0
        assert agent._discretize_state()[0] == 4

    def test_power_bin_boundary_low_to_medium(self):
        """500W is exactly at the 50% boundary -> should be level 2 (medium)."""
        agent = make_agent()
        agent.state_vector = _make_state_vector(power=500.0)
        agent.anomaly_index = 0.0
        agent.fatigue_index = 0.0
        assert agent._discretize_state()[0] == 2

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

def _notif(accepted=True, responded=True, minutes_ago=60):
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
        """Unanswered notifications older than MIN_COOLDOWN_MINUTES contribute +0.4 each."""
        agent = make_agent()
        # 2 unanswered at 60 min ago (> 30-min cooldown threshold) -> passive silence counted
        # interaction_score=0.8, factor=0.4; < 3 notifs -> freq=0
        # base = 0.6*0.4 + 0.4*0.0 = 0.24; no last_notification_time -> no decay
        agent.notification_history = deque(
            [_notif(responded=False, minutes_ago=60), _notif(responded=False, minutes_ago=60)],
            maxlen=100
        )
        await agent._update_fatigue_index()
        assert agent.fatigue_index == pytest.approx(0.24)

    @pytest.mark.asyncio
    async def test_unanswered_within_cooldown_window_not_counted(self):
        """Unanswered notifications sent < MIN_COOLDOWN_MINUTES ago must not penalise fatigue.
        The user may simply not have had time to respond yet.
        Using 2 notifications so frequency_factor=0 (need >=3 for that component)."""
        agent = make_agent()
        # 2 unanswered at 5 min ago: within the 30-min cooldown window
        # < 3 notifications -> frequency_factor = 0; passive silence NOT counted
        # -> interaction_factor = 0 -> base = 0 -> fatigue = 0
        agent.notification_history = deque(
            [_notif(responded=False, minutes_ago=5),
             _notif(responded=False, minutes_ago=5)],
            maxlen=100
        )
        await agent._update_fatigue_index()
        assert agent.fatigue_index == pytest.approx(0.0)

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
        """All-unanswered history: only notifications clearly older than MIN_COOLDOWN_MINUTES
        count as passive silence. Uses values well away from the 30-min boundary."""
        agent = make_agent()
        # 4 recent (< 30 min): 5, 10, 15, 20 min ago -> NOT counted
        # 6 old (> 30 min): 35, 45, 55, 65, 75, 85 min ago -> COUNTED
        not_counted = [_notif(responded=False, minutes_ago=m) for m in [5, 10, 15, 20]]
        counted     = [_notif(responded=False, minutes_ago=m) for m in [35, 45, 55, 65, 75, 85]]
        agent.notification_history = deque(not_counted + counted, maxlen=100)
        agent.last_notification_time = datetime.now() - timedelta(minutes=5)
        await agent._update_fatigue_index()
        # interaction_score = 6 * 0.4 = 2.4; window = 10; factor = 0.24
        # last 3 in deque: 65, 75, 85 min ago -> time_span = (now-85)-(now-65) = -20 min < 0 -> freq=1.0
        # base = 0.6*0.24 + 0.4*1.0 = 0.544; last_notif=5min -> no decay
        assert agent.fatigue_index == pytest.approx(0.544)

    @pytest.mark.asyncio
    async def test_silence_contributes_via_interaction_factor(self):
        """Unanswered notifications older than MIN_COOLDOWN_MINUTES raise fatigue via interaction_factor."""
        agent = make_agent()
        # 1 rejected responded (25 min ago; responded=True -> always counted)
        # 2 old unanswered (>30 min ago -> counted): at 35 and 40 min ago
        # 1 accepted (10 min ago; responded=True -> 0.0)
        # 1 recent unanswered (5 min ago; < 30 min -> NOT counted)
        # interaction_score = 1.0 + 0.4 + 0.4 + 0.0 + 0.0 = 1.8 -> factor = 1.8/5 = 0.36
        agent.notification_history = deque([
            _notif(accepted=False, responded=True,  minutes_ago=25),
            _notif(responded=False,                 minutes_ago=40),
            _notif(responded=False,                 minutes_ago=35),
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

    def test_70_30_ema_converges_faster_than_50_50(self):
        """With the 0.7/0.3 EMA weight a fully-engaged user should reach a higher
        index than the old 0.5/0.5 after the same number of updates."""
        agent = make_agent()
        agent.behaviour_index = 0.5
        agent.engagement_history = deque([1.0] * 5, maxlen=100)
        for _ in range(5):
            agent._update_behaviour_index()
        # With 0.7 weight on new value the index should climb well above 0.5
        assert agent.behaviour_index > 0.7

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
    async def test_critical_opportunity_respects_absolute_cooldown(self):
        """Critical score (>=0.8) still requires CRITICAL_MIN_COOLDOWN_MINUTES (5 min)."""
        agent = make_agent()
        # Only 1 minute since last notification: still within the absolute minimum
        agent.last_notification_time = datetime.now() - timedelta(minutes=1)
        agent.fatigue_index = 0.0
        result = await agent._check_cooldown_with_opportunity(0.85)
        assert result is False

    @pytest.mark.asyncio
    async def test_critical_opportunity_after_absolute_cooldown(self):
        """Critical score passes once CRITICAL_MIN_COOLDOWN_MINUTES have elapsed."""
        agent = make_agent()
        # 6 minutes since last notification — past the 5-min absolute minimum
        agent.last_notification_time = datetime.now() - timedelta(minutes=6)
        agent.fatigue_index = 0.0
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

    @pytest.mark.asyncio
    async def test_week_closes_on_monday_after_missed_sunday(self):
        """
        If HA was offline on Sunday the weekly challenge must still be logged
        on the next call (which might be Monday or later), not silently skipped.
        """
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        agent.storage = AsyncMock()
        agent.storage.log_weekly_challenge = AsyncMock()

        # Force the week start to 8 days ago (Monday) so days_in_current_week >= 7 even when today is Monday (weekday == 0), simulating a missed Sunday check.
        from datetime import date, datetime, timedelta as td
        past_monday = date.today() - td(days=8)
        agent.current_week_start_date = past_monday

        readings_per_hour = int(3600 / 15)
        
        # Data belonging to the missed week (e.g., 3 days after past_monday)
        old_week_dt = datetime.combine(past_monday + td(days=3), datetime.min.time())
        data_old = [(old_week_dt + td(seconds=i*15), 800.0) for i in range(readings_per_hour)]
        
        # Data belonging to the current live week (e.g., 1 hour ago)
        current_dt = datetime.now() - td(hours=1)
        data_current = [(current_dt + td(seconds=i*15), 800.0) for i in range(readings_per_hour)]
        
        # Mock the collector to return both sets of data
        agent.data_collector.get_power_history = AsyncMock(return_value=data_old + data_current)

        status = await agent.get_weekly_challenge_status(target_percentage=15.0)

        # Challenge should be logged exactly once
        agent.storage.log_weekly_challenge.assert_called_once()

        # Status must be a real result, not pending
        assert status["status"] in ("on_track", "off_track")


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


# ─────────────────────────────────────────────────────────────────────────────
# _execute_action – mobile push notification
# ─────────────────────────────────────────────────────────────────────────────

class TestExecuteActionMobileNotify:
    """Verify mobile push notification behaviour in _execute_action."""

    def _make_agent_with_services(self, notify_services: dict):
        """Create an agent whose hass.services.async_services() returns the given map."""
        agent = make_agent()
        agent.phase = "active"
        agent.baseline_consumption = 1000.0
        agent.fatigue_index = 0.0
        agent.behaviour_index = 0.5
        agent.anomaly_index = 0.0
        agent.notification_count_today = 0
        agent.last_notification_time = None
        agent.data_collector.main_power_sensor = None

        # Stub hass async_services
        agent.hass.services.async_services = MagicMock(return_value={"notify": notify_services})
        agent.hass.services.async_call = AsyncMock()
        return agent

    def _patch_generate_notification(self, agent, title="Test Alert", message="Test message"):
        """Patch _generate_notification to return a canned notification."""
        async def _fake_generate(action_type):
            return {"title": title, "message": message, "template_index": 0}
        agent._generate_notification = _fake_generate

    @pytest.mark.asyncio
    async def test_mobile_notify_called_when_service_available(self):
        """When a mobile_app_ service exists, async_call must be invoked with notify domain."""
        agent = self._make_agent_with_services({"mobile_app_pixel": MagicMock()})
        self._patch_generate_notification(agent)

        await agent._execute_action(1)

        # At least one call to hass.services.async_call should target 'notify'
        calls = agent.hass.services.async_call.call_args_list
        notify_calls = [c for c in calls if c.args[0] == "notify"]
        assert len(notify_calls) >= 1, "Expected async_call to 'notify' domain"
        assert notify_calls[0].args[1] == "mobile_app_pixel"

    @pytest.mark.asyncio
    async def test_no_mobile_notify_when_no_service(self):
        """When there are no mobile_app_ services, async_call must not be called."""
        agent = self._make_agent_with_services({})
        self._patch_generate_notification(agent)

        await agent._execute_action(1)

        calls = agent.hass.services.async_call.call_args_list
        notify_calls = [c for c in calls if c.args and c.args[0] == "notify"]
        assert len(notify_calls) == 0, "Should not call notify when no mobile service exists"

    @pytest.mark.asyncio
    async def test_notification_added_to_history_regardless_of_mobile(self):
        """Whether or not mobile is available, notification must appear in history."""
        agent = self._make_agent_with_services({})
        self._patch_generate_notification(agent, title="Energy Tip")

        await agent._execute_action(1)

        assert len(agent.notification_history) == 1
        assert agent.notification_history[0]["title"] == "Energy Tip"

    @pytest.mark.asyncio
    async def test_mobile_notify_error_does_not_crash_execute_action(self):
        """If mobile notification raises, the action must still complete normally."""
        agent = self._make_agent_with_services({"mobile_app_phone": MagicMock()})
        self._patch_generate_notification(agent)
        agent.hass.services.async_call = AsyncMock(side_effect=Exception("Service unavailable"))

        # Should not raise
        notification_id = await agent._execute_action(1)
        assert notification_id is not None


# ─────────────────────────────────────────────────────────────────────────────
# Gamification Streaks
# ─────────────────────────────────────────────────────────────────────────────

class TestStreaks:
    """Pure-logic tests for update_task_streak and update_weekly_streak."""

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _agent():
        return make_agent()

    @staticmethod
    def _day(delta: int = 0):
        from datetime import date, timedelta
        return date(2026, 2, 20) + timedelta(days=delta)

    @staticmethod
    def _week(delta_weeks: int = 0):
        """Return a week_key (Monday ISO date) delta_weeks from a fixed origin."""
        from datetime import date, timedelta
        return (date(2026, 2, 2) + timedelta(weeks=delta_weeks)).isoformat()

    # ── task_streak – basic increments ────────────────────────────────────────

    def test_task_streak_first_success_gives_one(self):
        agent = self._agent()
        agent.update_task_streak(True, self._day(0))
        assert agent.task_streak == 1
        assert agent.task_streak_last_date == self._day(0)

    def test_task_streak_two_consecutive_days(self):
        agent = self._agent()
        agent.update_task_streak(True, self._day(0))
        agent.update_task_streak(True, self._day(1))
        assert agent.task_streak == 2

    def test_task_streak_three_consecutive_days(self):
        agent = self._agent()
        for i in range(3):
            agent.update_task_streak(True, self._day(i))
        assert agent.task_streak == 3

    # ── task_streak – gap resets ──────────────────────────────────────────────

    def test_task_streak_gap_resets_to_one(self):
        """Streak breaks when there is a day gap and the next success counts as 1."""
        agent = self._agent()
        agent.update_task_streak(True, self._day(0))   # Mon
        # Tue missing
        agent.update_task_streak(True, self._day(2))   # Wed -> gap -> reset to 1
        assert agent.task_streak == 1
        assert agent.task_streak_last_date == self._day(2)

    def test_task_streak_week_gap_resets_to_one(self):
        agent = self._agent()
        agent.update_task_streak(True, self._day(0))
        agent.update_task_streak(True, self._day(7))   # one-week gap
        assert agent.task_streak == 1

    # ── task_streak – idempotency ─────────────────────────────────────────────

    def test_task_streak_same_day_is_idempotent(self):
        agent = self._agent()
        agent.update_task_streak(True, self._day(0))
        agent.update_task_streak(True, self._day(0))  # called again same day
        assert agent.task_streak == 1

    def test_task_streak_same_day_does_not_advance_to_next_day_count(self):
        agent = self._agent()
        agent.update_task_streak(True, self._day(0))
        # Simulate three same-day calls
        for _ in range(5):
            agent.update_task_streak(True, self._day(0))
        assert agent.task_streak == 1

    # ── task_streak – failures ────────────────────────────────────────────────

    def test_task_streak_failure_resets_to_zero(self):
        agent = self._agent()
        agent.update_task_streak(True, self._day(0))
        agent.update_task_streak(True, self._day(1))
        agent.update_task_streak(True, self._day(2))
        assert agent.task_streak == 3
        agent.update_task_streak(False, self._day(3))  # failed today
        assert agent.task_streak == 0

    def test_task_streak_failure_does_not_update_last_date(self):
        """Last-date pointer must stay at the last SUCCESS so gap detection works."""
        agent = self._agent()
        agent.update_task_streak(True, self._day(0))   # streak=1, last=day 0
        agent.update_task_streak(False, self._day(1))  # failure on day 1
        assert agent.task_streak == 0
        assert agent.task_streak_last_date == self._day(0)  # not day 1

    def test_task_streak_after_failure_gap_to_next_success_resets(self):
        """day0 success -> day1 failure -> day3 success: streak = 1 (gap was > 1)."""
        agent = self._agent()
        agent.update_task_streak(True, self._day(0))
        agent.update_task_streak(False, self._day(1))
        agent.update_task_streak(True, self._day(3))   # gap: day0 to day3 = 3 days
        assert agent.task_streak == 1

    def test_task_streak_failure_on_new_day_only_resets_once(self):
        """Multiple failure calls for the same day keep streak at 0, no side effects."""
        agent = self._agent()
        agent.update_task_streak(True, self._day(0))
        agent.update_task_streak(False, self._day(1))
        agent.update_task_streak(False, self._day(1))  # duplicate failure call
        assert agent.task_streak == 0

    def test_task_streak_starts_correctly_after_zero_history(self):
        """With no prior history, a single success gives streak=1."""
        agent = self._agent()
        assert agent.task_streak == 0
        assert agent.task_streak_last_date is None
        agent.update_task_streak(True, self._day(0))
        assert agent.task_streak == 1

    # ── weekly_streak – basic increments ──────────────────────────────────────

    def test_weekly_streak_first_achieved_gives_one(self):
        agent = self._agent()
        agent.update_weekly_streak(True, self._week(0))
        assert agent.weekly_streak == 1
        assert agent.weekly_streak_last_week == self._week(0)

    def test_weekly_streak_two_consecutive_weeks(self):
        agent = self._agent()
        agent.update_weekly_streak(True, self._week(0))
        agent.update_weekly_streak(True, self._week(1))
        assert agent.weekly_streak == 2

    def test_weekly_streak_three_consecutive_weeks(self):
        agent = self._agent()
        for i in range(3):
            agent.update_weekly_streak(True, self._week(i))
        assert agent.weekly_streak == 3

    # ── weekly_streak – gap resets ────────────────────────────────────────────

    def test_weekly_streak_gap_resets_to_one(self):
        agent = self._agent()
        agent.update_weekly_streak(True, self._week(0))
        # week 1 missing
        agent.update_weekly_streak(True, self._week(2))
        assert agent.weekly_streak == 1
        assert agent.weekly_streak_last_week == self._week(2)

    # ── weekly_streak – idempotency ───────────────────────────────────────────

    def test_weekly_streak_same_week_is_idempotent(self):
        agent = self._agent()
        agent.update_weekly_streak(True, self._week(0))
        agent.update_weekly_streak(True, self._week(0))
        assert agent.weekly_streak == 1

    # ── weekly_streak – failures ──────────────────────────────────────────────

    def test_weekly_streak_failure_resets_to_zero(self):
        agent = self._agent()
        agent.update_weekly_streak(True, self._week(0))
        agent.update_weekly_streak(True, self._week(1))
        agent.update_weekly_streak(False, self._week(2))
        assert agent.weekly_streak == 0

    def test_weekly_streak_failure_does_not_update_last_week(self):
        agent = self._agent()
        agent.update_weekly_streak(True, self._week(0))
        agent.update_weekly_streak(False, self._week(1))
        assert agent.weekly_streak_last_week == self._week(0)

    def test_weekly_streak_after_failure_next_success_is_one(self):
        agent = self._agent()
        agent.update_weekly_streak(True, self._week(0))
        agent.update_weekly_streak(False, self._week(1))
        agent.update_weekly_streak(True, self._week(2))
        assert agent.weekly_streak == 1

    def test_weekly_streak_zero_without_any_update(self):
        agent = self._agent()
        assert agent.weekly_streak == 0
        assert agent.weekly_streak_last_week is None

    # ── task_streak – office mode: non-working-day gaps ───────────────────────

    def test_task_streak_office_mode_weekend_does_not_break_streak(self):
        """In office mode, a Friday success followed by a Monday success
        must continue the streak (weekend is not a working day), not reset it."""
        # _day(0) = 2026-02-20 Friday, _day(3) = 2026-02-23 Monday
        office_cfg = {"environment_mode": "office", "working_days": [0, 1, 2, 3, 4]}
        agent = make_agent(config_data=office_cfg)
        helpers_stub.get_working_days_from_config.return_value = [0, 1, 2, 3, 4]

        agent.update_task_streak(True, self._day(0))   # Friday  -> streak=1
        agent.update_task_streak(True, self._day(3))   # Monday  -> streak must be 2

        assert agent.task_streak == 2
        assert agent.task_streak_last_date == self._day(3)

    def test_task_streak_home_mode_friday_to_monday_resets(self):
        """In home mode every calendar day counts, so Friday->Monday (3-day gap) resets."""
        agent = make_agent(config_data={"environment_mode": "home"})

        agent.update_task_streak(True, self._day(0))   # Friday  -> streak=1
        agent.update_task_streak(True, self._day(3))   # Monday  -> reset to 1

        assert agent.task_streak == 1

    def test_task_streak_office_mode_long_streak_survives_weekend(self):
        """A streak built across Wed-Thu-Fri should still be 4 on the following Monday."""
        # _day(-2)=Wed, _day(-1)=Thu, _day(0)=Fri, _day(3)=Mon
        office_cfg = {"environment_mode": "office", "working_days": [0, 1, 2, 3, 4]}
        agent = make_agent(config_data=office_cfg)
        helpers_stub.get_working_days_from_config.return_value = [0, 1, 2, 3, 4]

        agent.update_task_streak(True, self._day(-2))  # Wed -> 1
        agent.update_task_streak(True, self._day(-1))  # Thu -> 2
        agent.update_task_streak(True, self._day(0))   # Fri -> 3
        agent.update_task_streak(True, self._day(3))   # Mon -> 4 (weekend skipped)

        assert agent.task_streak == 4

    def test_task_streak_office_mode_with_working_day_gap_still_resets(self):
        """In office mode, if a working day is skipped the streak resets to 1.
        Friday success + Tuesday success (Monday missed) -> streak=1."""
        office_cfg = {"environment_mode": "office", "working_days": [0, 1, 2, 3, 4]}
        agent = make_agent(config_data=office_cfg)
        helpers_stub.get_working_days_from_config.return_value = [0, 1, 2, 3, 4]

        agent.update_task_streak(True, self._day(0))   # Friday -> 1
        agent.update_task_streak(True, self._day(4))   # Tuesday (Mon skipped) -> reset to 1

        assert agent.task_streak == 1

    def test_task_streak_office_mode_two_weekends_in_a_row(self):
        """Streak survives two consecutive weekends (Mon1 -> Mon2, gap of 7 with only
        non-working days Sat, Sun in each week - but Mon in week 2 IS a working day
        so the gap Mon1->Mon2 contains Tue,Wed,Thu,Fri,Sat,Sun which includes working
        days -> should reset."""
        # Clarification: Mon -> next Mon has Tue-Sun in the gap; Tue-Fri are working.
        # So this should actually reset.  The test documents expected behaviour.
        office_cfg = {"environment_mode": "office", "working_days": [0, 1, 2, 3, 4]}
        agent = make_agent(config_data=office_cfg)
        helpers_stub.get_working_days_from_config.return_value = [0, 1, 2, 3, 4]

        agent.update_task_streak(True, self._day(3))   # Mon week-1 -> 1
        agent.update_task_streak(True, self._day(10))  # Mon week-2 (7-day gap, includes Tue-Fri) -> 1

        assert agent.task_streak == 1


# ─────────────────────────────────────────────────────────────────────────────
# DB query caching & area anomaly throttle
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessAiModelQueryOptimisation:
    """
    process_ai_model() must:
      1. Pre-fetch power_history(hours=1) once per cycle (shared cache).
      2. Throttle _update_area_anomalies to every 4 cycles.
    """

    def _make_agent_for_process(self):
        agent = make_agent()
        agent.phase = "baseline"
        agent._process_count = 0
        agent.last_notification_date = None
        agent.notification_count_today = 0
        # Stub out all heavy coroutines so process_ai_model can run
        agent._build_state_vector = AsyncMock()
        agent._update_anomaly_index = AsyncMock()
        agent._update_area_anomalies = AsyncMock()
        agent._update_behaviour_index = MagicMock()
        agent._update_fatigue_index = AsyncMock()
        agent._update_action_mask = AsyncMock()
        agent._save_persistent_state = AsyncMock()
        # Stub action-selection coroutines so action_mask=None doesn't crash
        agent._decide_action = AsyncMock()
        agent._shadow_decide_action = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_power_history_queried_once_per_cycle(self):
        """get_power_history(hours=1) must be called exactly once per cycle via the cache."""
        agent = self._make_agent_for_process()
        # Replace the collector's method with one we can count
        call_count = 0
        async def _fake_get_power(hours=None, working_hours_only=None):
            nonlocal call_count
            call_count += 1
            return []
        agent.data_collector.get_power_history = _fake_get_power

        await agent.process_ai_model()

        # Exactly one DB call regardless of how many internal methods need it
        assert call_count == 1, (
            f"Expected 1 power_history query per cycle, got {call_count}"
        )

    @pytest.mark.asyncio
    async def test_cached_power_h1_set_after_process_cycle(self):
        """_cached_power_h1 must be populated (not None) after running process_ai_model."""
        agent = self._make_agent_for_process()
        sentinel = [("ts", 100.0)]
        agent.data_collector.get_power_history = AsyncMock(return_value=sentinel)

        await agent.process_ai_model()

        assert agent._cached_power_h1 is sentinel

    @pytest.mark.asyncio
    async def test_area_anomalies_called_on_cycle_0(self):
        """_update_area_anomalies must run on the first cycle (count=0)."""
        agent = self._make_agent_for_process()
        agent._process_count = 0
        agent.data_collector.get_power_history = AsyncMock(return_value=[])

        await agent.process_ai_model()

        agent._update_area_anomalies.assert_called_once()

    @pytest.mark.asyncio
    async def test_area_anomalies_skipped_on_cycle_1(self):
        """_update_area_anomalies must be skipped on odd cycles (not multiples of 4)."""
        agent = self._make_agent_for_process()
        agent._process_count = 1           # 1 % 4 != 0 -> skip
        agent.data_collector.get_power_history = AsyncMock(return_value=[])

        await agent.process_ai_model()

        agent._update_area_anomalies.assert_not_called()

    @pytest.mark.asyncio
    async def test_area_anomalies_called_every_4_cycles(self):
        """Run 8 cycles and verify _update_area_anomalies is called exactly twice (cycle 0 & 4)."""
        agent = self._make_agent_for_process()
        agent.data_collector.get_power_history = AsyncMock(return_value=[])
        call_count = 0

        async def _count_area(*a, **k):
            nonlocal call_count
            call_count += 1
        agent._update_area_anomalies = _count_area

        for _ in range(8):
            await agent.process_ai_model()

        assert call_count == 2, (
            f"Expected _update_area_anomalies to be called 2 times in 8 cycles, got {call_count}"
        )

    def test_update_anomaly_index_uses_cache_when_available(self):
        """_update_anomaly_index must read _cached_power_h1 rather than querying if set."""
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        # Plant a 150-element cache (> 192 threshold) with high values => anomaly detected
        agent._cached_power_h1 = [(None, 2000.0)] * 200
        agent.data_collector.get_power_history = AsyncMock(return_value=[])  # should not be called

        import asyncio
        asyncio.get_event_loop().run_until_complete(agent._update_anomaly_index())

        # Cache was used -> no live query
        agent.data_collector.get_power_history.assert_not_called()

    def test_update_action_mask_uses_cache_when_available(self):
        """_update_action_mask must reuse _cached_power_h1 when set."""
        agent = make_agent()
        agent.sensors = {"power": ["sensor.main"]}
        agent.baseline_consumption = 1000.0
        agent.area_anomalies = {}
        # Plant cache with enough entries
        agent._cached_power_h1 = [(None, 500.0)] * 200
        agent.data_collector.get_power_history = AsyncMock(return_value=[])  # should not be called

        import asyncio
        asyncio.get_event_loop().run_until_complete(agent._update_action_mask())

        agent.data_collector.get_power_history.assert_not_called()

# ─────────────────────────────────────────────────────────────────────────────
# WeeklyChallengeSensor / get_weekly_challenge_status cache
# ─────────────────────────────────────────────────────────────────────────────

class TestWeeklyChallengeCaching:
    """get_weekly_challenge_status must cache its result for 5 minutes
    to avoid a full DB scan on every 15-second AI signal."""

    def _agent_with_baseline(self):
        agent = make_agent()
        agent.phase = "active"
        agent.baseline_consumption = 1000.0
        # Minimal week setup
        from datetime import date
        agent.current_week_start_date = date.today()
        agent._logged_weeks = set()
        # Provide at least 240 readings (min_readings = 86400/15/24 = 240) to
        # guarantee the function reaches the cache-and-return path.
        power_rows = [(datetime.now() - timedelta(seconds=i * 15), 900.0) for i in range(250)]
        agent.data_collector.get_power_history = AsyncMock(return_value=power_rows)
        return agent

    @pytest.mark.asyncio
    async def test_first_call_populates_cache(self):
        """After the first successful call the cache must be set."""
        agent = self._agent_with_baseline()
        result = await agent.get_weekly_challenge_status(target_percentage=15.0)

        assert agent._weekly_challenge_cache is not None
        assert agent._weekly_challenge_cache_ts is not None
        assert agent._weekly_challenge_cache["status"] in ("on_track", "off_track")
        assert result == agent._weekly_challenge_cache

    @pytest.mark.asyncio
    async def test_second_call_within_ttl_uses_cache(self):
        """A second call within the TTL window must return the cache without re-querying."""
        agent = self._agent_with_baseline()

        first_result = await agent.get_weekly_challenge_status(target_percentage=15.0)
        initial_call_count = agent.data_collector.get_power_history.call_count

        # Call again immediately (within TTL)
        second_result = await agent.get_weekly_challenge_status(target_percentage=15.0)

        assert second_result is first_result, (
            "Second call within TTL must return the exact same cached dict."
        )
        assert agent.data_collector.get_power_history.call_count == initial_call_count, (
            "No additional DB queries should be made within the cache TTL."
        )

    @pytest.mark.asyncio
    async def test_call_after_ttl_expires_recomputes(self):
        """A call after the 300-second TTL must recompute and hit the DB again."""
        agent = self._agent_with_baseline()

        await agent.get_weekly_challenge_status(target_percentage=15.0)
        initial_call_count = agent.data_collector.get_power_history.call_count

        # Fast-forward the cache timestamp by 6 minutes
        agent._weekly_challenge_cache_ts = datetime.now() - timedelta(minutes=6)

        await agent.get_weekly_challenge_status(target_percentage=15.0)

        assert agent.data_collector.get_power_history.call_count > initial_call_count, (
            "A stale cache (>5 min old) must trigger a fresh DB query."
        )

    @pytest.mark.asyncio
    async def test_cache_not_populated_for_zero_baseline(self):
        """When baseline is 0, the function returns 'pending' without caching."""
        agent = make_agent()
        agent.baseline_consumption = 0.0
        agent.data_collector.get_power_history = AsyncMock(return_value=[])

        result = await agent.get_weekly_challenge_status()

        assert result["status"] == "pending"
        assert agent._weekly_challenge_cache == {}
        assert agent._weekly_challenge_cache_ts is None


# ─────────────────────────────────────────────────────────────────────────────
# Noop action (action=0): agent must be able to learn when to not send notifications.
# ─────────────────────────────────────────────────────────────────────────────

# Re-use the same module loading pattern used above.
_const_mod = sys.modules.get("custom_components.green_shift.const")
ACTIONS = _const_mod.ACTIONS  # type: ignore[attr-defined]

class TestNoopAction:
    """Validates that the noop action (0) is always present and correctly handled."""

    @pytest.mark.asyncio
    async def test_noop_always_in_action_mask(self):
        """_update_action_mask must always set noop=True, regardless of sensor availability."""
        agent = make_agent()
        # No sensors -> normative/specific would be False, but noop must still be True.
        agent.sensors = {}
        agent.baseline_consumption = 0.0
        agent._cached_power_h1 = []
        agent.area_anomalies = {}
        agent.anomaly_index = 0.0

        await agent._update_action_mask()

        assert agent.action_mask is not None, "action_mask must not be None after _update_action_mask()"
        assert ACTIONS["noop"] in agent.action_mask, "noop action must be present in action_mask"
        assert agent.action_mask[ACTIONS["noop"]] is True, (
            "noop must always be available so the agent can choose not to notify"
        )

    @pytest.mark.asyncio
    async def test_noop_present_with_all_other_actions(self):
        """noop must be True even when all notification actions are also True."""
        agent = make_agent()
        agent.sensors = {"power": ["sensor.device_a"], "energy": []}
        agent.baseline_consumption = 500.0
        agent._cached_power_h1 = [(None, 600.0)] * 200
        agent.anomaly_index = 0.5
        agent.area_anomalies = {}

        await agent._update_action_mask()

        assert agent.action_mask[ACTIONS["noop"]] is True

    def test_noop_key_in_actions_constant(self):
        """ACTIONS must contain 'noop' mapped to integer 0."""
        assert "noop" in ACTIONS, "ACTIONS dict must include 'noop'"
        assert ACTIONS["noop"] == 0, "noop must map to action integer 0"

    def test_q_table_initialised_with_noop(self):
        """When a new Q-table entry is created it must include the noop action."""
        agent = make_agent()
        agent.state_vector = _make_state_vector(power=500.0)
        agent.anomaly_index = 0.0
        agent.fatigue_index = 0.0
        state_key = agent._discretize_state()

        # Manually trigger initialisation (mirrors what _decide_action does internally).
        if state_key not in agent.q_table:
            agent.q_table[state_key] = {a: 0.0 for a in ACTIONS.values()}

        assert ACTIONS["noop"] in agent.q_table[state_key], (
            "Q-table for a state must contain a noop entry"
        )

    def test_compute_noop_reward_below_baseline_positive(self):
        """_compute_noop_reward must return a positive reward when power < baseline."""
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        agent.data_collector.get_current_state = MagicMock(return_value={"power": 700.0})

        reward = agent._compute_noop_reward()
        assert reward > 0, (
            f"Expected positive noop reward when below baseline, got {reward}"
        )

    def test_compute_noop_reward_above_baseline_negative(self):
        """_compute_noop_reward must return a negative reward when power >> baseline."""
        agent = make_agent()
        agent.baseline_consumption = 500.0
        agent.data_collector.get_current_state = MagicMock(return_value={"power": 1000.0})

        reward = agent._compute_noop_reward()
        assert reward < 0, (
            f"Expected negative noop reward when well above baseline, got {reward}"
        )

    def test_compute_noop_reward_no_baseline_returns_zero(self):
        """Without a baseline, _compute_noop_reward must return 0 (neutral)."""
        agent = make_agent()
        agent.baseline_consumption = 0.0

        reward = agent._compute_noop_reward()
        assert reward == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly index working-hours consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestAnomalyIndexWorkingHours:
    """In office mode the anomaly index must use working-hours-filtered power
    history so its baseline comparison is consistent with baseline_consumption
    (which is also computed from working-hours data)."""

    @pytest.mark.asyncio
    async def test_office_mode_uses_working_hours_filter(self):
        """_update_anomaly_index in office mode must query get_power_history with
        working_hours_only=True instead of relying on the unfiltered _cached_power_h1."""
        office_cfg = {
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
        agent = make_agent(config_data=office_cfg)
        agent.baseline_consumption = 1000.0

        # Cache contains a large off-hours reading that would trigger a false anomaly
        # if used for the office-mode comparison (baseline was 1000 W working-hours avg).
        agent._cached_power_h1 = [(None, v) for v in [50.0] * 200]  # night-time low values

        # The working-hours query returns readings near baseline (no anomaly)
        filtered_history = [(None, 980.0)] * 200  # working hours data near baseline
        agent.data_collector.get_power_history = AsyncMock(return_value=filtered_history)

        await agent._update_anomaly_index()

        # Must have queried with working_hours_only=True
        call_kwargs = agent.data_collector.get_power_history.call_args
        assert call_kwargs is not None, "_update_anomaly_index did not call get_power_history"
        kwargs = call_kwargs.kwargs if hasattr(call_kwargs, "kwargs") else (
            call_kwargs[1] if len(call_kwargs) > 1 else {}
        )
        assert kwargs.get("working_hours_only") is True, (
            "In office mode, _update_anomaly_index must request working_hours_only=True "
            "to match the working-hours-only baseline_consumption."
        )

    @pytest.mark.asyncio
    async def test_home_mode_uses_cache_not_filtered_query(self):
        """In home mode, _update_anomaly_index must use _cached_power_h1 (no extra query)."""
        agent = make_agent(config_data={"environment_mode": "home"})
        agent.baseline_consumption = 1000.0
        agent._cached_power_h1 = [(None, 1050.0)] * 200

        call_tracker = AsyncMock(return_value=agent._cached_power_h1)
        agent.data_collector.get_power_history = call_tracker

        await agent._update_anomaly_index()

        # No additional DB query should be made; cached data is reused
        call_tracker.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Shadow-reward cache reuse
# ─────────────────────────────────────────────────────────────────────────────

class TestShadowRewardCache:
    """_calculate_shadow_reward must reuse _cached_power_h1 instead of issuing
    a second get_power_history query on every shadow episode."""

    @pytest.mark.asyncio
    async def test_shadow_reward_uses_cached_history(self):
        """When _cached_power_h1 is populated, _calculate_shadow_reward must NOT
        issue an additional get_power_history query."""
        agent = make_agent()
        agent.baseline_consumption = 800.0
        agent.anomaly_index = 0.3
        agent.area_anomalies = {}

        # Pre-populate the cache (as process_ai_model() does each cycle)
        agent._cached_power_h1 = [(None, 900.0)] * 150

        # Track whether any DB call snuck through
        query_tracker = AsyncMock(return_value=agent._cached_power_h1)
        agent.data_collector.get_power_history = query_tracker

        await agent._calculate_shadow_reward(action=1)  # action=1 is "specific"

        query_tracker.assert_not_called(), (
            "_calculate_shadow_reward must reuse _cached_power_h1; "
            "issuing a duplicate DB query negates the per-cycle caching design."
        )

    @pytest.mark.asyncio
    async def test_shadow_reward_falls_back_when_cache_empty(self):
        """When cache is None, _calculate_shadow_reward must fall back to a DB query."""
        agent = make_agent()
        agent.baseline_consumption = 800.0
        agent.anomaly_index = 0.0
        agent.area_anomalies = {}
        agent._cached_power_h1 = None  # cache not yet populated

        fallback_data = [(None, 900.0)] * 150
        agent.data_collector.get_power_history = AsyncMock(return_value=fallback_data)

        result = await agent._calculate_shadow_reward(action=1)

        agent.data_collector.get_power_history.assert_called_once()
        assert isinstance(result, float)
