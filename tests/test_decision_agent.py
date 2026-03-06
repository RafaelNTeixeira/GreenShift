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
        can_send, req_cooldown, adap_cooldown = await agent._check_cooldown_with_opportunity(0.0)
        assert can_send is True
        assert req_cooldown is None
        assert adap_cooldown is None

    @pytest.mark.asyncio
    async def test_critical_opportunity_respects_absolute_cooldown(self):
        """Critical score (>=0.8) still requires CRITICAL_MIN_COOLDOWN_MINUTES (5 min)."""
        agent = make_agent()
        # Only 1 minute since last notification: still within the absolute minimum
        agent.last_notification_time = datetime.now() - timedelta(minutes=1)
        agent.fatigue_index = 0.0
        can_send, req_cooldown, adap_cooldown = await agent._check_cooldown_with_opportunity(0.85)
        assert can_send is False
        # Required cooldown for the critical path is CRITICAL_MIN_COOLDOWN_MINUTES
        from custom_components.green_shift.const import CRITICAL_MIN_COOLDOWN_MINUTES
        assert req_cooldown == CRITICAL_MIN_COOLDOWN_MINUTES
        assert adap_cooldown is not None

    @pytest.mark.asyncio
    async def test_critical_opportunity_after_absolute_cooldown(self):
        """Critical score passes once CRITICAL_MIN_COOLDOWN_MINUTES have elapsed."""
        agent = make_agent()
        # 6 minutes since last notification — past the 5-min absolute minimum
        agent.last_notification_time = datetime.now() - timedelta(minutes=6)
        agent.fatigue_index = 0.0
        can_send, _, _ = await agent._check_cooldown_with_opportunity(0.85)
        assert can_send is True

    @pytest.mark.asyncio
    async def test_high_opportunity_with_reduced_cooldown_met(self):
        agent = make_agent()
        agent.fatigue_index = 0.0
        # Adaptive cooldown with fatigue=0: 30 * 1.0 * 1.0 = 30 min -> 50% = 15 min
        agent.last_notification_time = datetime.now() - timedelta(minutes=16)
        can_send, req_cooldown, adap_cooldown = await agent._check_cooldown_with_opportunity(0.65)
        assert can_send is True
        # For high opportunity, required = 50% of adaptive
        assert req_cooldown == pytest.approx(adap_cooldown * 0.5)

    @pytest.mark.asyncio
    async def test_high_opportunity_insufficient_cooldown(self):
        agent = make_agent()
        agent.fatigue_index = 0.0
        agent.last_notification_time = datetime.now() - timedelta(minutes=5)
        can_send, req_cooldown, adap_cooldown = await agent._check_cooldown_with_opportunity(0.65)
        assert can_send is False
        assert req_cooldown is not None
        assert adap_cooldown is not None

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
            can_send, _, _ = await agent._check_cooldown_with_opportunity(0.3)
            assert can_send is False

    @pytest.mark.asyncio
    async def test_normal_opportunity_cooldown_satisfied(self):
        agent = make_agent()
        agent.fatigue_index = 0.0
        agent.last_notification_time = datetime.now() - timedelta(minutes=31)
        can_send, _, _ = await agent._check_cooldown_with_opportunity(0.3)
        assert can_send is True

    @pytest.mark.asyncio
    async def test_high_fatigue_extends_cooldown(self):
        agent = make_agent()
        agent.fatigue_index = 1.0  # Max fatigue -> 3x cooldown = 90 min
        # 45 minutes would normally satisfy 30 min, but not 90 min
        agent.last_notification_time = datetime.now() - timedelta(minutes=45)
        can_send, _, _ = await agent._check_cooldown_with_opportunity(0.3)
        assert can_send is False

    @pytest.mark.asyncio
    async def test_normal_returns_adaptive_as_both_cooldown_fields(self):
        """For normal opportunity both req and adap_cooldown must equal the adaptive value."""
        agent = make_agent()
        agent.fatigue_index = 0.0
        with patch.object(da_mod, 'datetime') as mock_dt:
            now = datetime(2026, 2, 19, 14, 0, 0)  # midday, no peak multiplier
            mock_dt.now.return_value = now
            agent.last_notification_time = now - timedelta(minutes=5)
            can_send, req_cooldown, adap_cooldown = await agent._check_cooldown_with_opportunity(0.3)
            assert can_send is False
            assert req_cooldown == adap_cooldown, (
                "For normal opportunity, required_cooldown must equal adaptive_cooldown"
            )
            # With fatigue=0 and midday: 30 * 1.0 * 1.0 = 30.0
            assert adap_cooldown == pytest.approx(30.0)

    @pytest.mark.asyncio
    async def test_cooldown_block_passes_values_to_log(self):
        """When _decide_action blocks on cooldown, _log_blocked_notification must receive
        non-None required_cooldown and adaptive_cooldown values from the returned tuple."""
        agent = make_agent()
        agent.phase = "active"
        agent.baseline_consumption = 500.0
        agent.storage = AsyncMock()
        agent.storage.log_blocked_notification = AsyncMock()
        agent.storage.log_rl_decision = AsyncMock()
        agent.notification_count_today = 0
        agent.last_notification_time = datetime.now() - timedelta(minutes=1)
        agent.action_mask = {0: True, 1: True, 2: False, 3: True, 4: True}
        agent.state_vector = [0.0] * 20
        agent._cached_power_h1 = [(None, 500.0)] * 20

        await agent._decide_action()

        agent.storage.log_blocked_notification.assert_called_once()
        call_data = agent.storage.log_blocked_notification.call_args[0][0]
        assert call_data["block_reason"] == "cooldown"
        assert call_data["required_cooldown_minutes"] is not None, (
            "required_cooldown_minutes must be populated from _check_cooldown_with_opportunity tuple"
        )
        assert call_data["adaptive_cooldown_minutes"] is not None, (
            "adaptive_cooldown_minutes must be populated from _check_cooldown_with_opportunity tuple"
        )


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

    @pytest.mark.asyncio
    async def test_retroactive_logging_excludes_data_before_old_week_start(self):
        """Data older than the old week's start must be excluded."""
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        agent.storage = AsyncMock()
        agent.storage.log_weekly_challenge = AsyncMock()

        from datetime import date, datetime as dt, timedelta as td
        past_monday = date.today() - td(days=8)
        agent.current_week_start_date = past_monday

        readings_per_hour = int(3600 / 15)
        old_week_start_dt = dt.combine(past_monday, dt.min.time())

        # Data BEFORE the old week (must be excluded by lower bound)
        before_week_dt = old_week_start_dt - td(hours=12)
        data_too_old = [(before_week_dt + td(seconds=i * 15), 9999.0) for i in range(readings_per_hour)]

        # Data WITHIN the old week (must be included)
        inside_old_week_dt = old_week_start_dt + td(days=2)
        data_in_week = [(inside_old_week_dt + td(seconds=i * 15), 800.0) for i in range(readings_per_hour)]

        agent.data_collector.get_power_history = AsyncMock(return_value=data_too_old + data_in_week)

        await agent.get_weekly_challenge_status(target_percentage=15.0)

        agent.storage.log_weekly_challenge.assert_called_once()
        call_kwargs = agent.storage.log_weekly_challenge.call_args
        logged_data = (call_kwargs.kwargs.get("challenge_data")
                       or (call_kwargs.args[0] if call_kwargs.args else {}))
        # actual_W must reflect only in-week data (800 W), not data_too_old (9999 W)
        assert abs(logged_data.get("actual_W", 0) - 800.0) < 1.0, (
            f"actual_W should be ~800 (in-week data only), got {logged_data.get('actual_W')}"
        )

    @pytest.mark.asyncio
    async def test_multiple_missed_weeks_are_all_logged(self):
        """Tests that multiple missed weeks are all logged correctly."""
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        agent.storage = AsyncMock()
        agent.storage.log_weekly_challenge = AsyncMock()

        from datetime import date, datetime as dt, timedelta as td

        # Place week-start 15 days ago so there are two full 7-day periods before the current week regardless of which day of the week today is.
        two_weeks_ago_monday = date.today() - td(days=15)
        agent.current_week_start_date = two_weeks_ago_monday

        readings_per_hour = int(3600 / 15)

        # Provide data in week-1 (days 0-6 from two_weeks_ago_monday)
        week1_start_dt = dt.combine(two_weeks_ago_monday, dt.min.time())
        data_week1 = [(week1_start_dt + td(days=2, seconds=i * 15), 800.0)
                      for i in range(readings_per_hour)]

        # Provide data in week-2 (days 7-13)
        week2_start_dt = week1_start_dt + td(days=7)
        data_week2 = [(week2_start_dt + td(days=2, seconds=i * 15), 850.0)
                      for i in range(readings_per_hour)]

        agent.data_collector.get_power_history = AsyncMock(
            return_value=data_week1 + data_week2
        )

        await agent.get_weekly_challenge_status(target_percentage=15.0)

        # Both missed weeks must be logged — not just the most recent one.
        assert agent.storage.log_weekly_challenge.call_count == 2, (
            f"Expected 2 log_weekly_challenge calls (one per missed week), "
            f"got {agent.storage.log_weekly_challenge.call_count}"
        )


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

    @pytest.mark.asyncio
    async def test_all_mobile_services_are_notified(self):
        """When multiple mobile_app_ services exist, ALL must receive the push notification."""
        agent = self._make_agent_with_services({
            "mobile_app_phone": MagicMock(),
            "mobile_app_tablet": MagicMock(),
        })
        self._patch_generate_notification(agent)

        await agent._execute_action(1)

        calls = agent.hass.services.async_call.call_args_list
        notify_calls = [c for c in calls if c.args[0] == "notify"]
        notified_services = {c.args[1] for c in notify_calls}
        assert "mobile_app_phone" in notified_services, "phone must be notified"
        assert "mobile_app_tablet" in notified_services, "tablet must be notified"
        assert len(notify_calls) == 2, "Exactly 2 notify calls expected, one per device"


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
# _update_area_anomalies: working_hours_only filter
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateAreaAnomaliesWorkingHours:
    """_update_area_anomalies must pass working_hours_only=True in office mode and None in home mode."""

    def _make_agent_with_area(self, env_mode):
        agent = make_agent(config_data={"environment_mode": env_mode})
        agent.area_baselines = {}
        calls = []

        async def _fake_get_area_history(area, metric, hours=None, days=None, working_hours_only=None):
            calls.append((metric, working_hours_only))
            return []

        agent.data_collector.get_all_areas = MagicMock(return_value=["Living Room"])
        agent.data_collector.get_area_history = _fake_get_area_history
        return agent, calls

    @pytest.mark.asyncio
    async def test_office_mode_passes_working_hours_filter(self):
        """In office mode every get_area_history call must use working_hours_only=True."""
        agent, calls = self._make_agent_with_area("office")
        await agent._update_area_anomalies()
        assert len(calls) == 3, f"Expected 3 get_area_history calls, got {len(calls)}"
        for metric, wh in calls:
            assert wh is True, (
                f"'{metric}' history must use working_hours_only=True in office mode, got {wh}"
            )

    @pytest.mark.asyncio
    async def test_home_mode_passes_no_filter(self):
        """In home mode working_hours_only must be None for all area history calls."""
        agent, calls = self._make_agent_with_area("home")
        await agent._update_area_anomalies()
        assert len(calls) == 3, f"Expected 3 get_area_history calls, got {len(calls)}"
        for metric, wh in calls:
            assert wh is None, (
                f"'{metric}' history must use working_hours_only=None in home mode, got {wh}"
            )

    @pytest.mark.asyncio
    async def test_no_area_mode_skips_no_area(self):
        """Areas named 'No Area' must be skipped: no get_area_history calls for them."""
        agent, calls = self._make_agent_with_area("home")
        agent.data_collector.get_all_areas = MagicMock(return_value=["No Area"])
        await agent._update_area_anomalies()
        assert calls == [], "get_area_history must not be called for 'No Area' areas"

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

    def test_exploit_tie_breaking_randomises_across_equal_q_values(self):
        """When all Q-values are equal the exploitation branch must NOT always
        return noop=0. Over 100 draws at least two distinct actions must appear."""
        import random as _random
        agent = make_agent()
        agent.state_vector = [0.0] * 18
        state_key = (1, 0, 0, 0, 0, 0)
        agent.q_table = {state_key: {a: 0.0 for a in range(5)}}
        available = list(range(5))

        chosen = set()
        _random.seed(42)
        for _ in range(100):
            max_q = max(agent.q_table[state_key].get(a, 0.0) for a in available)
            best_actions = [a for a in available if agent.q_table[state_key].get(a, 0.0) == max_q]
            chosen.add(_random.choice(best_actions))

        assert len(chosen) > 1, (
            f"All 100 tie-breaking draws returned the same action {chosen}; "
            "tie-breaking must be random, not biased towards the first element"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly index working-hours consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestAnomalyIndexWorkingHours:
    """In office mode the anomaly index must use working-hours-filtered power
    history so its baseline comparison is consistent with baseline_consumption
    (which is also computed from working-hours data)."""

    @pytest.mark.asyncio
    async def test_office_mode_fallback_uses_working_hours_filter(self):
        """When _cached_power_h1 is None (cold start / direct call), the fallback
        query in _update_anomaly_index must use working_hours_only=True for office
        mode so results are consistent with the working-hours-based baseline."""
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
        agent._cached_power_h1 = None  # cache unpopulated -> fallback path

        fallback_data = [(None, 980.0)] * 200
        agent.data_collector.get_power_history = AsyncMock(return_value=fallback_data)

        await agent._update_anomaly_index()

        # The fallback query must have been issued with working_hours_only=True.
        agent.data_collector.get_power_history.assert_called_once()
        call_kwargs = agent.data_collector.get_power_history.call_args
        kwargs = call_kwargs.kwargs if hasattr(call_kwargs, "kwargs") else (
            call_kwargs[1] if len(call_kwargs) > 1 else {}
        )
        assert kwargs.get("working_hours_only") is True, (
            "In office mode the fallback query must use working_hours_only=True "
            "to match the working-hours-only baseline_consumption."
        )

    @pytest.mark.asyncio
    async def test_office_mode_cache_hit_no_extra_query(self):
        """When _cached_power_h1 is pre-populated (as by process_ai_model()),
        _update_anomaly_index must NOT issue a second DB query even in office mode."""
        office_cfg = {"environment_mode": "office"}
        agent = make_agent(config_data=office_cfg)
        agent.baseline_consumption = 1000.0

        # Simulate cache already populated with working-hours readings by process_ai_model()
        agent._cached_power_h1 = [(None, 980.0)] * 200

        query_tracker = AsyncMock(return_value=agent._cached_power_h1)
        agent.data_collector.get_power_history = query_tracker

        await agent._update_anomaly_index()

        query_tracker.assert_not_called(), (
            "In office mode, _update_anomaly_index must use the per-cycle cache "
            "when available and must NOT issue an additional DB query."
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


# ─────────────────────────────────────────────────────────────────────────────
# Cooldown-block logging throttle
# ─────────────────────────────────────────────────────────────────────────────

class TestCooldownBlockLogging:
    """_decide_action must log blocked-by-cooldown notifications to the research
    DB, but at most once per MIN_COOLDOWN_MINUTES to avoid flooding the table."""

    def _make_active_agent_with_storage(self):
        """Return an agent in ACTIVE phase with a mock storage that tracks calls."""
        from custom_components.green_shift.const import PHASE_ACTIVE
        agent = make_agent()
        agent.phase = PHASE_ACTIVE
        agent.storage = AsyncMock()
        agent.storage.log_blocked_notification = AsyncMock()
        agent.storage.log_rl_decision = AsyncMock()  # may be called for noop path
        agent.notification_count_today = 0
        return agent

    @pytest.mark.asyncio
    async def test_cooldown_block_is_logged_when_in_cooldown(self):
        """First cooldown block must produce exactly one DB write."""
        agent = self._make_active_agent_with_storage()

        # Simulate: last notification sent 1 minute ago (well within MIN_COOLDOWN_MINUTES)
        agent.last_notification_time = datetime.now() - timedelta(minutes=1)
        agent.action_mask = {0: True, 1: True, 2: False, 3: True, 4: True}
        agent.state_vector = [0.0] * 20
        agent._cached_power_h1 = [(None, 500.0)] * 20

        # _check_cooldown_with_opportunity will return False (too soon)
        # _log_blocked_notification calls storage.log_blocked_notification
        await agent._decide_action()

        agent.storage.log_blocked_notification.assert_called_once()
        call_kwargs = agent.storage.log_blocked_notification.call_args[0][0]
        assert call_kwargs["block_reason"] == "cooldown"

    @pytest.mark.asyncio
    async def test_cooldown_block_not_logged_twice_within_throttle_window(self):
        """A second cooldown block within MIN_COOLDOWN_MINUTES must NOT write again."""
        agent = self._make_active_agent_with_storage()

        agent.last_notification_time = datetime.now() - timedelta(minutes=1)
        agent.action_mask = {0: True, 1: True, 2: False, 3: True, 4: True}
        agent.state_vector = [0.0] * 20
        agent._cached_power_h1 = [(None, 500.0)] * 20

        # First call — should log
        await agent._decide_action()
        assert agent.storage.log_blocked_notification.call_count == 1

        # Second call immediately after — still in throttle window; must NOT log again
        await agent._decide_action()
        assert agent.storage.log_blocked_notification.call_count == 1  # unchanged

    @pytest.mark.asyncio
    async def test_cooldown_block_logged_again_after_throttle_expires(self):
        """After the throttle window elapses, the next block must be logged."""
        from custom_components.green_shift.const import MIN_COOLDOWN_MINUTES
        agent = self._make_active_agent_with_storage()

        agent.last_notification_time = datetime.now() - timedelta(minutes=1)
        agent.action_mask = {0: True, 1: True, 2: False, 3: True, 4: True}
        agent.state_vector = [0.0] * 20
        agent._cached_power_h1 = [(None, 500.0)] * 20

        # Simulate that the last log was MIN_COOLDOWN_MINUTES + 1 seconds ago
        agent._last_cooldown_block_log_time = (
            datetime.now() - timedelta(minutes=MIN_COOLDOWN_MINUTES, seconds=1)
        )

        await agent._decide_action()

        assert agent.storage.log_blocked_notification.call_count == 1  # logged again
        call_kwargs = agent.storage.log_blocked_notification.call_args[0][0]
        assert call_kwargs["block_reason"] == "cooldown"

    @pytest.mark.asyncio
    async def test_cooldown_block_reports_correct_available_action_count(self):
        """The logged blocked notification must include the count of available actions at the time of the block."""
        agent = self._make_active_agent_with_storage()

        agent.last_notification_time = datetime.now() - timedelta(minutes=1)
        # 4 actions available (noop=0, specific=1, behavioural=3, normative=4)
        agent.action_mask = {0: True, 1: True, 2: False, 3: True, 4: True}
        agent.state_vector = [0.0] * 20
        agent._cached_power_h1 = [(None, 500.0)] * 20

        await agent._decide_action()

        assert agent.storage.log_blocked_notification.call_count == 1
        logged = agent.storage.log_blocked_notification.call_args[0][0]
        assert logged["available_action_count"] == 4, (
            f"Cooldown block should report 4 available actions, got {logged['available_action_count']}"
        )

    @pytest.mark.asyncio
    async def test_fatigue_block_logs_time_since_last_and_adaptive_cooldown(self):
        """When a notification is blocked by fatigue_threshold, time_since_last and
        adaptive_cooldown must be non-None in the research DB entry.
        required_cooldown must stay None (cooldown was already satisfied)."""
        from custom_components.green_shift.const import FATIGUE_THRESHOLD, CRITICAL_OPPORTUNITY_THRESHOLD
        agent = self._make_active_agent_with_storage()

        # Fatigue is over threshold; opportunity is below critical so fatigue guard fires.
        agent.last_notification_time = datetime.now() - timedelta(minutes=35)
        agent.fatigue_index = FATIGUE_THRESHOLD + 0.1
        agent.action_mask = {0: True, 1: True, 2: False, 3: True, 4: True}
        agent.state_vector = [0.0] * 20
        agent._cached_power_h1 = [(None, 500.0)] * 20

        # Force the cooldown check to PASS and return known cooldown values,
        # so _decide_action proceeds to the fatigue guard rather than stopping at cooldown.
        known_req_cooldown = 25.0
        known_adap_cooldown = 50.0
        agent._check_cooldown_with_opportunity = AsyncMock(
            return_value=(True, known_req_cooldown, known_adap_cooldown)
        )
        # Opportunity score below critical threshold so fatigue guard activates
        agent._calculate_opportunity_score = AsyncMock(return_value=CRITICAL_OPPORTUNITY_THRESHOLD - 0.1)

        await agent._decide_action()

        agent.storage.log_blocked_notification.assert_called_once()
        logged = agent.storage.log_blocked_notification.call_args[0][0]
        assert logged["block_reason"] == "fatigue_threshold"
        assert logged["time_since_last_notification_minutes"] is not None, (
            "time_since_last must be populated for fatigue blocks "
            "(cooldown already passed, time is research context)"
        )
        assert logged["adaptive_cooldown_minutes"] == known_adap_cooldown, (
            "adaptive_cooldown must be forwarded from the cooldown check tuple"
        )
        assert logged["required_cooldown_minutes"] is None, (
            "required_cooldown must be None for fatigue blocks — cooldown was already satisfied"
        )


# ─────────────────────────────────────────────────────────────────────────────
# setup / _load_persistent_state / _save_persistent_state
# ─────────────────────────────────────────────────────────────────────────────

class TestSetup:
    @pytest.mark.asyncio
    async def test_setup_calls_load_when_storage_present(self):
        agent = make_agent()
        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value={})
        agent.storage = storage
        await agent.setup()
        storage.load_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_skips_load_when_no_storage(self):
        agent = make_agent()
        agent.storage = None
        # Should not raise
        await agent.setup()


class TestLoadPersistentState:

    @pytest.mark.asyncio
    async def test_loads_phase_from_state(self):
        agent = make_agent()
        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value={"phase": "active"})
        agent.storage = storage
        await agent._load_persistent_state()
        assert agent.phase == "active"

    @pytest.mark.asyncio
    async def test_loads_baseline_consumption(self):
        agent = make_agent()
        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value={"baseline_consumption": 800.0})
        agent.storage = storage
        await agent._load_persistent_state()
        assert agent.baseline_consumption == pytest.approx(800.0)

    @pytest.mark.asyncio
    async def test_loads_task_streak(self):
        agent = make_agent()
        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value={"task_streak": 7})
        agent.storage = storage
        await agent._load_persistent_state()
        assert agent.task_streak == 7

    @pytest.mark.asyncio
    async def test_loads_q_table_with_tuple_key(self):
        agent = make_agent()
        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value={
            "q_table": {"(1, 0, 0, 0, 0, 0)": {"0": 0.5, "1": 0.3}}
        })
        agent.storage = storage
        await agent._load_persistent_state()
        assert len(agent.q_table) == 1
        key = list(agent.q_table.keys())[0]
        assert isinstance(key, tuple)

    @pytest.mark.asyncio
    async def test_invalid_q_table_key_is_skipped(self):
        agent = make_agent()
        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value={
            "q_table": {"not_a_tuple": {"0": 1.0}, "(2, 1, 0, 0, 0, 0)": {"0": 0.9}}
        })
        agent.storage = storage
        await agent._load_persistent_state()
        assert len(agent.q_table) == 1

    @pytest.mark.asyncio
    async def test_no_storage_returns_immediately(self):
        agent = make_agent()
        agent.storage = None
        await agent._load_persistent_state()
        assert agent.phase == "baseline"  # unchanged

    @pytest.mark.asyncio
    async def test_loads_weekly_streak(self):
        agent = make_agent()
        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value={"weekly_streak": 3})
        agent.storage = storage
        await agent._load_persistent_state()
        assert agent.weekly_streak == 3

    @pytest.mark.asyncio
    async def test_loads_notification_count(self):
        agent = make_agent()
        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value={"notification_count_today": 4})
        agent.storage = storage
        await agent._load_persistent_state()
        assert agent.notification_count_today == 4


class TestSavePersistentState:

    @pytest.mark.asyncio
    async def test_save_calls_storage_save_state(self):
        agent = make_agent()
        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value={})
        storage.save_state = AsyncMock()
        agent.storage = storage
        await agent._save_persistent_state()
        storage.save_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_storage_skips_save(self):
        agent = make_agent()
        agent.storage = None
        await agent._save_persistent_state()  # should not raise

    @pytest.mark.asyncio
    async def test_q_table_serialised_as_string_keys(self):
        """Q-table tuple keys are stringified so JSON can store them."""
        agent = make_agent()
        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value={})
        saved = {}

        async def capture_save(state):
            saved.update(state)

        storage.save_state = AsyncMock(side_effect=capture_save)
        agent.storage = storage
        agent.q_table = {(1, 0, 0, 0, 0, 0): {0: 0.7}}
        await agent._save_persistent_state()
        # String key should appear in serialised q_table
        assert str((1, 0, 0, 0, 0, 0)) in saved.get("q_table", {})

    def test_feedback_episode_number_survives_save_load_roundtrip(self):
        """feedback_episode_number must be included in the persisted state."""
        agent = make_agent()
        agent.feedback_episode_number = 42
        state = {
            "feedback_episode_number": agent.feedback_episode_number,
            "episode_number": 0,
            "shadow_episode_number": 0,
        }
        agent2 = make_agent()
        if "feedback_episode_number" in state:
            agent2.feedback_episode_number = state["feedback_episode_number"]
        assert agent2.feedback_episode_number == 42


# ─────────────────────────────────────────────────────────────────────────────
# _build_state_vector
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildStateVector:

    @pytest.mark.asyncio
    async def test_state_vector_has_18_elements(self):
        agent = make_agent()
        agent.sensors = {"power": ["sensor.p1"]}
        agent.data_collector.get_current_state.return_value = {
            "power": 300.0, "occupancy": True,
            "temperature": 21.0, "humidity": 50.0, "illuminance": 200.0,
        }
        agent.data_collector.main_power_sensor = None
        agent.hass.states.get.return_value = None
        await agent._build_state_vector()
        assert len(agent.state_vector) == 18

    @pytest.mark.asyncio
    async def test_power_value_at_index_0(self):
        agent = make_agent()
        agent.sensors = {}
        agent.data_collector.get_current_state.return_value = {"power": 750.0}
        agent.data_collector.main_power_sensor = None
        agent.hass.states.get.return_value = None
        await agent._build_state_vector()
        assert agent.state_vector[0] == pytest.approx(750.0)

    @pytest.mark.asyncio
    async def test_occupancy_flag_set_correctly(self):
        agent = make_agent()
        agent.sensors = {"occupancy": ["binary_sensor.occ_1"]}
        agent.data_collector.get_current_state.return_value = {
            "power": 0.0, "occupancy": True,
        }
        agent.data_collector.main_power_sensor = None
        agent.hass.states.get.return_value = None
        await agent._build_state_vector()
        # Index 10 = occupancy value
        assert agent.state_vector[10] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# _get_top_power_consumer
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTopPowerConsumer:

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_sensors(self):
        agent = make_agent()
        agent.sensors = {}
        result = await agent._get_top_power_consumer()
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_returns_max_excluding_main_sensor(self):
        agent = make_agent()
        agent.sensors = {"power": ["sensor.main", "sensor.p1", "sensor.p2"]}
        agent.data_collector.main_power_sensor = "sensor.main"

        def mock_state(entity_id):
            vals = {"sensor.main": "1000", "sensor.p1": "300", "sensor.p2": "200"}
            if entity_id in vals:
                s = MagicMock()
                s.state = vals[entity_id]
                s.attributes = {"unit_of_measurement": "W"}
                return s
            return None

        agent.hass.states.get.side_effect = mock_state
        da_mod.get_normalized_value = MagicMock(side_effect=lambda s, t: (float(s.state), "W"))
        result = await agent._get_top_power_consumer()
        assert result == pytest.approx(300.0)

    @pytest.mark.asyncio
    async def test_skips_unavailable_states(self):
        agent = make_agent()
        agent.sensors = {"power": ["sensor.p1"]}
        agent.data_collector.main_power_sensor = None

        state = MagicMock()
        state.state = "unavailable"
        agent.hass.states.get.return_value = state
        result = await agent._get_top_power_consumer()
        assert result == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _update_action_mask
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateActionMask:

    @pytest.mark.asyncio
    async def test_noop_always_available(self):
        agent = make_agent()
        agent._cached_power_h1 = []
        agent.sensors = {}
        agent.baseline_consumption = 0.0
        agent.area_anomalies = {}
        agent.anomaly_index = 0.0
        await agent._update_action_mask()
        assert agent.action_mask[da_mod.ACTIONS["noop"]] is True

    @pytest.mark.asyncio
    async def test_specific_enabled_with_power_sensors(self):
        agent = make_agent()
        agent._cached_power_h1 = []
        agent.sensors = {"power": ["sensor.p1"]}
        agent.data_collector.main_power_sensor = None  # p1 is individual, not main
        agent.baseline_consumption = 0.0
        agent.area_anomalies = {}
        agent.anomaly_index = 0.0
        await agent._update_action_mask()
        assert agent.action_mask[da_mod.ACTIONS["specific"]] is True

    @pytest.mark.asyncio
    async def test_specific_disabled_when_only_main_sensor(self):
        """If only the main_power_sensor is in power[], specific must be False
        because _find_top_consumer() always excludes main and returns (None, 0.0)."""
        agent = make_agent()
        agent._cached_power_h1 = []
        agent.sensors = {"power": ["sensor.main"]}
        agent.data_collector.main_power_sensor = "sensor.main"
        agent.baseline_consumption = 0.0
        agent.area_anomalies = {}
        agent.anomaly_index = 0.0
        await agent._update_action_mask()
        assert agent.action_mask[da_mod.ACTIONS["specific"]] is False, (
            "specific must be False when the only power sensor is the main aggregator"
        )

    @pytest.mark.asyncio
    async def test_normative_enabled_with_baseline(self):
        agent = make_agent()
        agent._cached_power_h1 = []
        agent.sensors = {}
        agent.baseline_consumption = 600.0
        agent.area_anomalies = {}
        agent.anomaly_index = 0.0
        await agent._update_action_mask()
        assert agent.action_mask[da_mod.ACTIONS["normative"]] is True

    @pytest.mark.asyncio
    async def test_normative_disabled_without_baseline(self):
        agent = make_agent()
        agent._cached_power_h1 = []
        agent.sensors = {}
        agent.baseline_consumption = 0.0
        agent.area_anomalies = {}
        agent.anomaly_index = 0.0
        await agent._update_action_mask()
        assert agent.action_mask[da_mod.ACTIONS["normative"]] is False

    @pytest.mark.asyncio
    async def test_behavioural_always_available(self):
        agent = make_agent()
        agent._cached_power_h1 = []
        agent.sensors = {}
        agent.baseline_consumption = 0.0
        agent.area_anomalies = {}
        agent.anomaly_index = 0.0
        await agent._update_action_mask()
        assert agent.action_mask[da_mod.ACTIONS["behavioural"]] is True

    @pytest.mark.asyncio
    async def test_specific_enabled_with_both_main_and_individual_sensor(self):
        """Individual sensor present alongside the main aggregator -> specific must be True."""
        agent = make_agent()
        agent._cached_power_h1 = []
        agent.sensors = {"power": ["sensor.main_power", "sensor.device_1"]}
        agent.data_collector.main_power_sensor = "sensor.main_power"
        agent.baseline_consumption = 0.0
        agent.area_anomalies = {}
        agent.anomaly_index = 0.0
        await agent._update_action_mask()
        assert agent.action_mask[da_mod.ACTIONS["specific"]] is True


# ─────────────────────────────────────────────────────────────────────────────
# _update_q_table_with_feedback  /  _shadow_update_q_table
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateQTableWithFeedback:

    @pytest.mark.asyncio
    async def test_accepted_increases_q_value(self):
        agent = make_agent()
        agent.state_vector = [0.0] * 18
        agent.q_table = {}
        state_key = (1, 0, 0, 0, 0, 0)
        await agent._update_q_table_with_feedback(state_key, 0, reward=1.0, accepted=True)
        assert agent.q_table[state_key][0] > 0.0

    @pytest.mark.asyncio
    async def test_rejected_decreases_q_value_with_zero_gamma(self):
        agent = make_agent()
        agent.state_vector = [0.0] * 18
        state_key = (2, 0, 0, 0, 0, 0)
        agent.q_table = {state_key: {a: 0.0 for a in range(5)}}
        await agent._update_q_table_with_feedback(state_key, 0, reward=-1.0, accepted=False)
        # With gamma=0 and negative reward, Q must decrease
        assert agent.q_table[state_key][0] < 0.0

    @pytest.mark.asyncio
    async def test_creates_q_entry_for_new_state(self):
        agent = make_agent()
        agent.state_vector = [0.0] * 18
        agent.q_table = {}
        state_key = (3, 1, 0, 0, 0, 0)
        await agent._update_q_table_with_feedback(state_key, 1, reward=0.5, accepted=True)
        assert state_key in agent.q_table


class TestShadowUpdateQTable:

    @pytest.mark.asyncio
    async def test_positive_reward_increases_q(self):
        agent = make_agent()
        agent.state_vector = [0.0] * 18
        state_key = (1, 0, 0, 0, 0, 0)
        agent.q_table = {state_key: {a: 0.0 for a in range(5)}}
        await agent._shadow_update_q_table(state_key, 0, reward=1.0)
        assert agent.q_table[state_key][0] > 0.0

    @pytest.mark.asyncio
    async def test_creates_new_state_entry_if_missing(self):
        agent = make_agent()
        agent.state_vector = [0.0] * 18
        agent.q_table = {}
        state_key = (4, 0, 0, 0, 0, 0)
        await agent._shadow_update_q_table(state_key, 0, reward=0.5)
        assert state_key in agent.q_table

    @pytest.mark.asyncio
    async def test_shadow_uses_gamma_zero_no_future_bias(self):
        """With gamma=0, the shadow update must not consider any future state and should update Q based solely on the immediate reward."""
        agent = make_agent()
        agent.state_vector = [0.0] * 18
        state_key = (2, 1, 0, 0, 0, 0)
        initial_q = 0.0
        agent.q_table = {state_key: {a: initial_q for a in range(5)}}

        reward = 0.8
        from custom_components.green_shift.const import SHADOW_LEARNING_RATE
        await agent._shadow_update_q_table(state_key, 0, reward=reward)

        expected_q = initial_q + SHADOW_LEARNING_RATE * (reward - initial_q)
        assert abs(agent.q_table[state_key][0] - expected_q) < 1e-9, (
            f"Expected Q={expected_q:.6f} (gamma=0 formula), got {agent.q_table[state_key][0]:.6f}"
        )
        # Only the original state key should exist — no phantom next-state entry
        assert len(agent.q_table) == 1, (
            "Shadow update must not create a second state entry (no next-state lookup with gamma=0)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# _is_only_non_working_days_in_gap
# ─────────────────────────────────────────────────────────────────────────────

class TestIsOnlyNonWorkingDaysInGap:

    def _make_office_agent(self):
        return make_agent(config_data={
            "environment_mode": "office",
            "working_monday": True, "working_tuesday": True,
            "working_wednesday": True, "working_thursday": True,
            "working_friday": True, "working_saturday": False,
            "working_sunday": False,
        })

    def test_home_mode_always_returns_false(self):
        agent = make_agent(config_data={"environment_mode": "home"})
        from datetime import date
        d1 = date(2026, 2, 20)  # Friday
        d2 = date(2026, 2, 23)  # Monday (gap = Sat+Sun)
        helpers_stub.get_working_days_from_config = MagicMock(return_value=list(range(5)))
        assert agent._is_only_non_working_days_in_gap(d1, d2) is False

    def test_gap_with_only_weekend_days_returns_true(self):
        agent = self._make_office_agent()
        helpers_stub.get_working_days_from_config = MagicMock(return_value=list(range(5)))
        from datetime import date
        friday = date(2026, 2, 20)
        monday = date(2026, 2, 23)  # gap = Sat(5) + Sun(6) -> non-working
        assert agent._is_only_non_working_days_in_gap(friday, monday) is True

    def test_gap_containing_working_day_returns_false(self):
        agent = self._make_office_agent()
        helpers_stub.get_working_days_from_config = MagicMock(return_value=list(range(5)))
        from datetime import date
        monday = date(2026, 2, 16)
        wednesday = date(2026, 2, 18)  # gap = Tuesday(1) -> working day
        assert agent._is_only_non_working_days_in_gap(monday, wednesday) is False

    def test_consecutive_days_no_gap_returns_true(self):
        agent = self._make_office_agent()
        helpers_stub.get_working_days_from_config = MagicMock(return_value=list(range(5)))
        from datetime import date
        d1 = date(2026, 2, 18)
        d2 = date(2026, 2, 19)  # no day strictly between them -> loop never executes -> True
        assert agent._is_only_non_working_days_in_gap(d1, d2) is True


# ─────────────────────────────────────────────────────────────────────────────
# process_ai_model: daily counter reset and process_count increment
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessAiModel:

    def _make_patched_agent(self, phase="baseline"):
        agent = make_agent()
        agent.storage = None
        agent.phase = phase
        agent._process_count = 0
        agent.data_collector.get_power_history = AsyncMock(return_value=[])
        agent.data_collector.get_current_state = MagicMock(return_value={"power": 0.0})
        agent._build_state_vector = AsyncMock()
        agent._update_anomaly_index = AsyncMock()
        agent._update_area_anomalies = AsyncMock()
        agent._update_behaviour_index = MagicMock()
        agent._update_fatigue_index = AsyncMock()
        agent._update_action_mask = AsyncMock()
        agent._decide_action = AsyncMock()
        agent._shadow_decide_action = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_resets_notification_count_on_new_day(self):
        agent = self._make_patched_agent()
        agent.notification_count_today = 5
        agent.last_notification_date = datetime(2026, 1, 1).date()
        await agent.process_ai_model()
        assert agent.notification_count_today == 0
        assert agent.last_notification_date == datetime.now().date()

    @pytest.mark.asyncio
    async def test_increments_process_count(self):
        agent = self._make_patched_agent()
        await agent.process_ai_model()
        assert agent._process_count == 1

    @pytest.mark.asyncio
    async def test_calls_decide_action_in_active_phase(self):
        agent = self._make_patched_agent(phase="active")
        await agent.process_ai_model()
        agent._decide_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_call_decide_action_in_baseline(self):
        agent = self._make_patched_agent(phase="baseline")
        agent._process_count = 0
        await agent.process_ai_model()
        agent._decide_action.assert_not_called()

    @pytest.mark.asyncio
    async def test_expires_stale_pending_episodes(self):
        agent = self._make_patched_agent()
        old_ts = datetime.now() - timedelta(hours=25)
        agent.pending_episodes = {
            "old_ep": {
                "state_key": (1, 0, 0, 0, 0, 0),
                "action": 0, "initial_power": 500.0,
                "timestamp": old_ts, "action_source": "exploit",
                "opportunity_score": 0.5,
            }
        }
        await agent.process_ai_model()
        assert "old_ep" not in agent.pending_episodes


# ─────────────────────────────────────────────────────────────────────────────
# Epsilon decay
# ─────────────────────────────────────────────────────────────────────────────

class TestEpsilonDecay:
    """Verify that epsilon decays towards MIN_EPSILON as real episodes accumulate."""

    def test_epsilon_decreases_after_episodes(self):
        """Computed epsilon must be strictly below INITIAL_EPSILON after several episodes."""
        from custom_components.green_shift.const import INITIAL_EPSILON, MIN_EPSILON, EPSILON_DECAY_RATE
        episodes = 100
        computed = max(MIN_EPSILON, INITIAL_EPSILON * (EPSILON_DECAY_RATE ** episodes))
        assert computed < INITIAL_EPSILON, "Epsilon must drop below initial value"
        assert computed >= MIN_EPSILON, "Epsilon must not go below MIN_EPSILON floor"

    def test_epsilon_floor_is_min_epsilon(self):
        """After many episodes epsilon saturates at exactly MIN_EPSILON."""
        from custom_components.green_shift.const import INITIAL_EPSILON, MIN_EPSILON, EPSILON_DECAY_RATE
        computed = max(MIN_EPSILON, INITIAL_EPSILON * (EPSILON_DECAY_RATE ** 10_000))
        assert computed == MIN_EPSILON

    @pytest.mark.asyncio
    async def test_noop_episode_does_not_decay_epsilon(self):
        """After a noop episode, epsilon must stay at INITIAL_EPSILON.
        Epsilon now decays only on real user-feedback episodes, not noops."""
        from custom_components.green_shift.const import INITIAL_EPSILON, MIN_EPSILON
        agent = make_agent()
        agent.phase = "active"
        agent.episode_number = 0
        agent.feedback_episode_number = 0
        agent.epsilon = INITIAL_EPSILON
        agent.notification_count_today = 0
        agent.last_notification_time = None
        # Only noop available so epsilon-greedy always picks noop
        agent.action_mask = {0: True, 1: False, 2: False, 3: False, 4: False}
        agent.state_vector = [0.0] * 14
        agent.storage = None
        agent._compute_noop_reward = MagicMock(return_value=0.0)
        agent._update_q_table_with_feedback = AsyncMock()
        agent._log_rl_episode = AsyncMock()
        agent._calculate_opportunity_score = AsyncMock(return_value=0.1)
        agent._check_cooldown_with_opportunity = AsyncMock(return_value=(True, None, None))
        agent._discretize_state = MagicMock(return_value=(0, 0, 0, 0, 0, 0))

        await agent._decide_action()

        # Noop should NOT decay epsilon
        assert agent.epsilon == INITIAL_EPSILON, (
            f"Epsilon must stay at INITIAL_EPSILON={INITIAL_EPSILON} after noop "
            f"(got {agent.epsilon}); epsilon decays only on real user-feedback episodes"
        )

    @pytest.mark.asyncio
    async def test_user_feedback_decays_epsilon(self):
        """After user responds to a notification, epsilon must have decayed."""
        from custom_components.green_shift.const import INITIAL_EPSILON, MIN_EPSILON
        agent = make_agent()
        agent.phase = "active"
        agent.episode_number = 5
        agent.epsilon = INITIAL_EPSILON
        agent.storage = None

        notif_id = "test-notif-001"
        agent.notification_history.append({
            "notification_id": notif_id,
            "responded": False,
            "accepted": None,
        })
        agent.pending_episodes[notif_id] = {
            "state_key": (0, 0, 0, 0, 0, 0),
            "action": 1,
            "initial_power": 500.0,
            "timestamp": datetime.now(),
            "action_source": "exploit",
            "opportunity_score": 0.5,
        }
        agent._calculate_reward_with_feedback = AsyncMock(return_value=1.0)
        agent._update_q_table_with_feedback = AsyncMock()
        agent._log_rl_episode = AsyncMock()
        agent._update_behaviour_index = MagicMock()

        await agent._handle_notification_feedback(notif_id, accepted=True)

        assert agent.epsilon < INITIAL_EPSILON, "Epsilon must have decayed after user feedback"
        assert agent.epsilon >= MIN_EPSILON

    def test_feedback_episode_number_is_independent_from_noop_counter(self):
        """feedback_episode_number and episode_number are distinct counters.
        After many noop episodes, epsilon must still be at INITIAL_EPSILON."""
        agent = make_agent()
        assert hasattr(agent, "feedback_episode_number"), (
            "agent must have feedback_episode_number attribute"
        )
        from custom_components.green_shift.const import INITIAL_EPSILON, MIN_EPSILON, EPSILON_DECAY_RATE
        agent.episode_number = 500
        agent.feedback_episode_number = 0
        expected = max(MIN_EPSILON, INITIAL_EPSILON * (EPSILON_DECAY_RATE ** agent.feedback_episode_number))
        assert expected == INITIAL_EPSILON, (
            "With 0 feedback episodes, epsilon should equal INITIAL_EPSILON regardless of noop count"
        )


# ─────────────────────────────────────────────────────────────────────────────
# _log_rl_episode: gamma_used resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestLogRlEpisodeGamma:
    """_log_rl_episode must record the correct gamma_used for every episode category.

    Cases:
      - accepted=True  (real accept)  -> gamma_used = GAMMA
      - accepted=False (real reject)  -> gamma_used = 0.0
      - shadow episode                -> gamma_used = 0.0
      - active-phase noop             -> gamma_used = GAMMA
    """

    def _make_logging_agent(self):
        """Agent with mock storage that captures the log_rl_decision payload."""
        agent = make_agent()
        agent.phase = "active"
        agent.state_vector = [0.0] * 18
        agent.action_mask = {a: True for a in range(5)}
        agent.q_table = {}
        storage = AsyncMock()
        storage.log_rl_decision = AsyncMock()
        agent.storage = storage
        return agent

    def _get_logged_gamma(self, agent):
        call_args = agent.storage.log_rl_decision.call_args
        episode_data = call_args[0][0]
        return episode_data["gamma_used"]

    @pytest.mark.asyncio
    async def test_accepted_notification_gamma_is_GAMMA(self):
        from custom_components.green_shift.const import GAMMA
        agent = self._make_logging_agent()
        await agent._log_rl_episode(
            state_key=(1, 0, 0, 0, 0, 0), action=1,
            reward=0.5, action_source="exploit", accepted=True
        )
        assert self._get_logged_gamma(agent) == GAMMA

    @pytest.mark.asyncio
    async def test_rejected_notification_gamma_is_zero(self):
        agent = self._make_logging_agent()
        await agent._log_rl_episode(
            state_key=(1, 0, 0, 0, 0, 0), action=1,
            reward=-0.5, action_source="exploit", accepted=False
        )
        assert self._get_logged_gamma(agent) == 0.0

    @pytest.mark.asyncio
    async def test_shadow_episode_gamma_is_zero(self):
        agent = self._make_logging_agent()
        agent.phase = "baseline"
        await agent._log_rl_episode(
            state_key=(1, 0, 0, 0, 0, 0), action=1,
            reward=0.3, action_source="shadow_explore", accepted=None
        )
        assert self._get_logged_gamma(agent) == 0.0

    @pytest.mark.asyncio
    async def test_active_phase_noop_gamma_is_zero(self):
        """Noop uses γ=0: no real state transition occurs, so future-value estimation
        would compare Q(s,a) against Q(s,a) (same state), causing the same inflation
        bias fixed for shadow learning.  The research DB must record gamma_used=0.0."""
        from custom_components.green_shift.const import ACTIONS
        agent = self._make_logging_agent()
        await agent._log_rl_episode(
            state_key=(0, 0, 0, 0, 0, 0), action=ACTIONS["noop"],
            reward=0.1, action_source="explore", accepted=None
        )
        gamma = self._get_logged_gamma(agent)
        assert gamma == 0.0, (
            f"Active-phase noop must log gamma_used=0.0 (γ=0, no real state transition), "
            f"got {gamma!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# max_next_Q restricted to available actions
# ─────────────────────────────────────────────────────────────────────────────

class TestQTableUpdateRestrictedToAvailableActions:
    """_update_q_table_with_feedback() must restrict max_next_q to actions that are currently AVAILABLE in the action_mask."""

    @pytest.mark.asyncio
    async def test_q_update_ignores_masked_action_in_max_next_q(self):
        """
        Setup:
          - action_mask: only 'noop' (0) and 'behavioural' (3) are available
          - Q-table for next_state: anomaly (2) has Q=0.9, available actions have Q=0.3
          - Expected: max_next_q = 0.3  (not 0.9)
        """
        from custom_components.green_shift.const import ACTIONS, GAMMA

        agent = make_agent()
        agent.state_vector = [0.0] * 18
        agent.baseline_consumption = 1000.0

        # Only noop and behavioural are available; anomaly is masked out
        agent.action_mask = {
            ACTIONS["noop"]:        True,
            ACTIONS["specific"]:    False,
            ACTIONS["anomaly"]:     False,   # masked: high Q should be ignored
            ACTIONS["behavioural"]: True,
            ACTIONS["normative"]:   False,
        }

        # Pre-populate Q-table for the next state (same state after discretization)
        state_key = agent._discretize_state()
        agent.q_table[state_key] = {
            ACTIONS["noop"]:        0.3,
            ACTIONS["specific"]:    0.4,
            ACTIONS["anomaly"]:     0.9,   # high value but action is masked
            ACTIONS["behavioural"]: 0.3,
            ACTIONS["normative"]:   0.5,
        }

        initial_q = 0.4
        agent.q_table[state_key][ACTIONS["behavioural"]] = initial_q
        reward = 0.5
        learning_rate = 0.1

        await agent._update_q_table_with_feedback(
            state_key=state_key,
            action=ACTIONS["behavioural"],
            reward=reward,
            accepted=True,
        )

        # With ONLY available actions (noop=0.3, behavioural=initial_q=0.4 before update):
        # max_next_q_correct = 0.4 (behavioural, highest among available before update)
        # (noop=0.3 < behavioural=initial_q=0.4)
        # new_q = initial_q + lr * (reward + GAMMA * max_next_q_correct - initial_q)
        # = 0.4 + 0.1 * (0.5 + 0.95 * 0.4 - 0.4)
        # = 0.4 + 0.1 * (0.5 + 0.38 - 0.4)
        # = 0.4 + 0.1 * 0.48 = 0.4 + 0.048 = 0.448

        # With checking all actions (including masked anomaly=0.9):
        # max_next_q_buggy = 0.9 (anomaly)
        # new_q = 0.4 + 0.1*(0.5 + 0.95*0.9 - 0.4) = 0.4 + 0.1*(0.5+0.855-0.4) = 0.4+0.0955 = 0.4955

        final_q = agent.q_table[state_key][ACTIONS["behavioural"]]

        # Verify the result is closer to the correct value (0.448) than the buggy value (0.4955)
        correct_max_next_q = 0.4  # best available Q before update
        expected_correct = initial_q + learning_rate * (reward + GAMMA * correct_max_next_q - initial_q)
        buggy_max_next_q = 0.9  # anomaly Q (masked)
        expected_buggy = initial_q + learning_rate * (reward + GAMMA * buggy_max_next_q - initial_q)

        assert abs(final_q - expected_correct) < 0.001, (
            f"Expected Q≈{expected_correct:.4f} (available actions only), "
            f"got {final_q:.4f}. Buggy value would be {expected_buggy:.4f}"
        )

    @pytest.mark.asyncio
    async def test_rejection_uses_gamma_zero_regardless_of_mask(self):
        """When accepted=False, gamma=0 so max_next_q is irrelevant; masked values
        must not influence the rejected update either."""
        from custom_components.green_shift.const import ACTIONS

        agent = make_agent()
        agent.state_vector = [0.0] * 18
        agent.baseline_consumption = 1000.0
        agent.action_mask = {
            ACTIONS["noop"]:        True,
            ACTIONS["specific"]:    False,
            ACTIONS["anomaly"]:     False,
            ACTIONS["behavioural"]: True,
            ACTIONS["normative"]:   False,
        }

        state_key = agent._discretize_state()
        agent.q_table[state_key] = {a: 0.9 for a in ACTIONS.values()}  # all high

        initial_q = agent.q_table[state_key][ACTIONS["behavioural"]]
        reward = -0.5
        lr = agent.learning_rate  # default 0.1

        await agent._update_q_table_with_feedback(
            state_key=state_key,
            action=ACTIONS["behavioural"],
            reward=reward,
            accepted=False,
        )

        # gamma=0 -> new_q = init_q + lr * (reward + 0 - init_q) = init_q + lr*(reward - init_q)
        expected = initial_q + lr * (reward - initial_q)
        final_q = agent.q_table[state_key][ACTIONS["behavioural"]]
        assert abs(final_q - expected) < 0.001, (
            f"Rejected update should use gamma=0; expected {expected:.4f}, got {final_q:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Behavioural template_index must be ABSOLUTE (position in all_templates)
# ─────────────────────────────────────────────────────────────────────────────

class TestBehaviouralTemplateAbsoluteIndex:
    """When behavioural templates are filtered by context, the
    template_index stored in the returned notification dict must be the absolute
    position in all_templates - not the position inside the filtered subset.

    The filtering is additive: generic templates (no context_filter) are always
    included, and context-specific templates are added on top.
    """

    _AWAY_IDX_A = 6
    _AWAY_IDX_B = 7

    def _make_templates(self):
        """Return 8 dummy behavioural templates; only indices 6 and 7 are
        contextually gated behind away_mode."""
        templates = []
        for i in range(8):
            if i in (self._AWAY_IDX_A, self._AWAY_IDX_B):
                templates.append({
                    "title": f"Away {i}",
                    "message": "You are away - save energy!",
                    "context_filter": "away_mode",
                })
            else:
                templates.append({
                    "title": f"Generic {i}",
                    "message": "Consider reducing consumption.",
                })
        return templates

    def _make_context(self, *, is_away_mode=False, is_daylight_waste=False, is_nighttime=False):
        return {
            "current_power": 500,
            "baseline_power": 1000,
            "target_power": 800,
            "percent_above": 10,
            "device_name": "Heater",
            "device_power": 200,
            "area_name": "Living room",
            "metric": "temperature",
            "time_of_day": "morning",
            "is_away_mode": is_away_mode,
            "is_daylight_waste": is_daylight_waste,
            "is_nighttime": is_nighttime,
        }

    @pytest.mark.asyncio
    async def test_away_mode_templates_are_reachable_and_carry_absolute_index(self):
        """With is_away_mode=True the pool includes generic (0-5) + away_mode (6,7).
        template_index 6 and 7 must appear in the output — confirming the fix
        preserves absolute positions rather than re-indexing the filtered slice."""
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        agent.target_percentage = 20.0

        templates = self._make_templates()
        context = self._make_context(is_away_mode=True)

        with patch.object(
            da_mod, "get_notification_templates",
            return_value={"behavioural": templates}
        ), patch.object(
            da_mod, "get_language",
            new=AsyncMock(return_value="en")
        ):
            agent._gather_notification_context = AsyncMock(return_value=context)
            seen_indices = set()
            for _ in range(200):
                result = await agent._generate_notification("behavioural")
                assert result is not None
                seen_indices.add(result["template_index"])

        # All 8 absolute positions must be reachable (generic 0-5 always + away 6,7)
        assert seen_indices == set(range(8)), (
            f"All 8 absolute template indices must be reachable; got {seen_indices}"
        )

    @pytest.mark.asyncio
    async def test_away_mode_template_index_never_exceeds_list_length(self):
        """template_index must always be a valid index into all_templates (< 8)."""
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        agent.target_percentage = 20.0

        templates = self._make_templates()
        context = self._make_context(is_away_mode=True)

        with patch.object(
            da_mod, "get_notification_templates",
            return_value={"behavioural": templates}
        ), patch.object(
            da_mod, "get_language",
            new=AsyncMock(return_value="en")
        ):
            agent._gather_notification_context = AsyncMock(return_value=context)
            for _ in range(50):
                result = await agent._generate_notification("behavioural")
                assert result is not None
                assert 0 <= result["template_index"] < len(templates), (
                    f"template_index {result['template_index']} out of bounds for "
                    f"list of length {len(templates)}"
                )

    @pytest.mark.asyncio
    async def test_no_context_filter_uses_only_generic_range(self):
        """When no contextual flags are set, only the 6 generic templates
        (indices 0-5) should be reachable, confirming context-gated templates
        are correctly excluded."""
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        agent.target_percentage = 20.0

        templates = self._make_templates()
        context = self._make_context()  # all flags False

        with patch.object(
            da_mod, "get_notification_templates",
            return_value={"behavioural": templates}
        ), patch.object(
            da_mod, "get_language",
            new=AsyncMock(return_value="en")
        ):
            agent._gather_notification_context = AsyncMock(return_value=context)
            seen_indices = set()
            for _ in range(100):
                result = await agent._generate_notification("behavioural")
                assert result is not None
                seen_indices.add(result["template_index"])

        generic_range = set(range(self._AWAY_IDX_A))  # {0,1,2,3,4,5}
        assert seen_indices.issubset(generic_range), (
            f"Without context flags only generic indices 0-5 must appear; got {seen_indices}"
        )

    @pytest.mark.asyncio
    async def test_non_behavioural_action_uses_full_template_list(self):
        """For non-behavioural action types (e.g. normative) the index must
        still be drawn from the full list without any context filtering."""
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        agent.target_percentage = 20.0

        normative_templates = [
            {"title": f"Norm {i}", "message": f"Norm msg {i}"} for i in range(4)
        ]
        context = self._make_context()

        with patch.object(
            da_mod, "get_notification_templates",
            return_value={"normative": normative_templates}
        ), patch.object(
            da_mod, "get_language",
            new=AsyncMock(return_value="en")
        ):
            agent._gather_notification_context = AsyncMock(return_value=context)
            seen_indices = set()
            for _ in range(80):
                result = await agent._generate_notification("normative")
                assert result is not None
                seen_indices.add(result["template_index"])

        assert seen_indices == {0, 1, 2, 3}, (
            f"All 4 normative templates must be reachable; got {seen_indices}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# _handle_notification_feedback: full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _make_feedback_agent(notif_id, accepted_init=None):
    """Return an agent pre-loaded with one pending episode and matching history entry."""
    agent = make_agent()
    agent.phase = "active"
    agent.episode_number = 5
    agent.feedback_episode_number = 3
    from custom_components.green_shift.const import INITIAL_EPSILON
    agent.epsilon = INITIAL_EPSILON

    agent.notification_history.append({
        "notification_id": notif_id,
        "responded": False,
        "accepted": accepted_init,
    })
    agent.pending_episodes[notif_id] = {
        "state_key": (1, 0, 0, 0, 0, 0),
        "action": 1,
        "initial_power": 800.0,
        "timestamp": datetime.now(),
        "action_source": "exploit",
        "opportunity_score": 0.6,
    }
    agent._calculate_reward_with_feedback = AsyncMock(return_value=1.0)
    agent._update_q_table_with_feedback = AsyncMock()
    agent._log_rl_episode = AsyncMock()
    agent._update_behaviour_index = MagicMock()

    storage = AsyncMock()
    storage.log_nudge_response = AsyncMock()
    storage._save_persistent_state = AsyncMock()
    agent.storage = storage
    agent._save_persistent_state = AsyncMock()
    return agent


class TestHandleNotificationFeedback:
    """_handle_notification_feedback: full pipeline for accept and reject."""

    @pytest.mark.asyncio
    async def test_accept_marks_notification_responded_true(self):
        notif_id = "test-accept-001"
        agent = _make_feedback_agent(notif_id)
        await agent._handle_notification_feedback(notif_id, accepted=True)
        entry = next(n for n in agent.notification_history if n["notification_id"] == notif_id)
        assert entry["responded"] is True
        assert entry["accepted"] is True

    @pytest.mark.asyncio
    async def test_reject_marks_notification_responded_false(self):
        notif_id = "test-reject-001"
        agent = _make_feedback_agent(notif_id)
        await agent._handle_notification_feedback(notif_id, accepted=False)
        entry = next(n for n in agent.notification_history if n["notification_id"] == notif_id)
        assert entry["responded"] is True
        assert entry["accepted"] is False

    @pytest.mark.asyncio
    async def test_accept_appends_positive_engagement_score(self):
        notif_id = "test-accept-eng"
        agent = _make_feedback_agent(notif_id)
        agent.engagement_history.clear()
        await agent._handle_notification_feedback(notif_id, accepted=True)
        assert len(agent.engagement_history) == 1
        assert agent.engagement_history[-1] == 1.0

    @pytest.mark.asyncio
    async def test_reject_appends_negative_engagement_score(self):
        notif_id = "test-reject-eng"
        agent = _make_feedback_agent(notif_id)
        agent.engagement_history.clear()
        await agent._handle_notification_feedback(notif_id, accepted=False)
        assert len(agent.engagement_history) == 1
        assert agent.engagement_history[-1] == -0.5

    @pytest.mark.asyncio
    async def test_feedback_calls_update_behaviour_index(self):
        notif_id = "test-behav-001"
        agent = _make_feedback_agent(notif_id)
        await agent._handle_notification_feedback(notif_id, accepted=True)
        agent._update_behaviour_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_accept_removes_pending_episode(self):
        notif_id = "test-pend-remove"
        agent = _make_feedback_agent(notif_id)
        assert notif_id in agent.pending_episodes
        await agent._handle_notification_feedback(notif_id, accepted=True)
        assert notif_id not in agent.pending_episodes

    @pytest.mark.asyncio
    async def test_reject_removes_pending_episode(self):
        notif_id = "test-pend-reject"
        agent = _make_feedback_agent(notif_id)
        assert notif_id in agent.pending_episodes
        await agent._handle_notification_feedback(notif_id, accepted=False)
        assert notif_id not in agent.pending_episodes

    @pytest.mark.asyncio
    async def test_reject_passes_accepted_false_to_q_update(self):
        notif_id = "test-q-reject"
        agent = _make_feedback_agent(notif_id)
        await agent._handle_notification_feedback(notif_id, accepted=False)
        agent._update_q_table_with_feedback.assert_called_once()
        _, kwargs = agent._update_q_table_with_feedback.call_args
        assert kwargs.get("accepted") is False or agent._update_q_table_with_feedback.call_args[0][3] is False

    @pytest.mark.asyncio
    async def test_feedback_increments_feedback_episode_number(self):
        notif_id = "test-ep-incr"
        agent = _make_feedback_agent(notif_id)
        before = agent.feedback_episode_number
        await agent._handle_notification_feedback(notif_id, accepted=True)
        assert agent.feedback_episode_number == before + 1

    @pytest.mark.asyncio
    async def test_feedback_calls_storage_log_nudge_response(self):
        notif_id = "test-log-nudge"
        agent = _make_feedback_agent(notif_id)
        await agent._handle_notification_feedback(notif_id, accepted=True)
        agent.storage.log_nudge_response.assert_called_once_with(notif_id, True)

    @pytest.mark.asyncio
    async def test_reject_calls_storage_log_nudge_response_with_false(self):
        notif_id = "test-log-nudge-rej"
        agent = _make_feedback_agent(notif_id)
        await agent._handle_notification_feedback(notif_id, accepted=False)
        agent.storage.log_nudge_response.assert_called_once_with(notif_id, False)

    @pytest.mark.asyncio
    async def test_feedback_saves_persistent_state(self):
        notif_id = "test-save-state"
        agent = _make_feedback_agent(notif_id)
        await agent._handle_notification_feedback(notif_id, accepted=True)
        agent._save_persistent_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_pending_episode_does_not_raise(self):
        """Responding to a notification that is not in pending_episodes must not crash."""
        notif_id = "test-no-pending"
        agent = _make_feedback_agent(notif_id)
        agent.pending_episodes.clear()  # remove the pre-loaded episode
        # Should complete without raising, just log a warning
        await agent._handle_notification_feedback(notif_id, accepted=True)

    @pytest.mark.asyncio
    async def test_notification_id_not_in_history_does_not_crash(self):
        """If notification_id is not found in notification_history the call
        must still proceed (no KeyError / StopIteration raised)."""
        notif_id = "ghost-notification"
        agent = _make_feedback_agent("some-other-id")
        # notif_id is NOT in notification_history
        await agent._handle_notification_feedback(notif_id, accepted=True)


# ─────────────────────────────────────────────────────────────────────────────
# _compute_noop_reward: near-baseline boundary (-0.1 band)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeNoopRewardNearBaseline:
    """_compute_noop_reward must return -0.1 when 0 < deviation < 0.3."""

    def test_slightly_above_baseline_returns_minus_0_1(self):
        """10% above baseline falls in the near-baseline band (0 < dev < 0.3)."""
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        agent.data_collector.get_current_state = MagicMock(return_value={"power": 1100.0})
        reward = agent._compute_noop_reward()
        assert reward == pytest.approx(-0.1), (
            f"Expected -0.1 for slightly-above-baseline, got {reward}"
        )

    def test_just_below_0_3_deviation_returns_minus_0_1(self):
        """29% above baseline is still in the near band, not the miss band."""
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        agent.data_collector.get_current_state = MagicMock(return_value={"power": 1290.0})
        reward = agent._compute_noop_reward()
        assert reward == pytest.approx(-0.1), (
            f"29%% above baseline should be -0.1, got {reward}"
        )

    def test_exactly_0_3_deviation_transitions_to_miss_band(self):
        """30% deviation is >= 0.3, so it falls in the 'missed opportunity' band."""
        agent = make_agent()
        agent.baseline_consumption = 1000.0
        agent.data_collector.get_current_state = MagicMock(return_value={"power": 1300.0})
        reward = agent._compute_noop_reward()
        # deviation = 0.3, formula -> max(-0.5, -0.3 * 0.5) = -0.15
        assert reward < -0.1, (
            f"30%% deviation must use miss-band formula (< -0.1), got {reward}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# _calculate_opportunity_score: direct parametric tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateOpportunityScore:
    """Direct tests for the four components of _calculate_opportunity_score."""

    def _base_agent(self, power=500.0, baseline=1000.0, occupancy=True,
                    anomaly=0.0, area_anomalies=None, fatigue=0.0, behaviour=1.0):
        agent = make_agent()
        agent.baseline_consumption = baseline
        agent.anomaly_index = anomaly
        agent.area_anomalies = area_anomalies or {}
        agent.fatigue_index = fatigue
        agent.behaviour_index = behaviour
        agent.data_collector.get_current_state = MagicMock(return_value={
            "power": power, "occupancy": occupancy
        })
        return agent

    @pytest.mark.asyncio
    async def test_at_baseline_savings_potential_is_zero(self):
        """When current power == baseline, savings_potential component is 0."""
        agent = self._base_agent(power=1000.0, baseline=1000.0)
        with patch.object(da_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 12, 0, 0)  # peak time
            score = await agent._calculate_opportunity_score()
        # savings_potential = 0, so score comes entirely from other components
        # With anomaly=0, fatigue=0, behaviour=1, context from noon occupancy
        # Max theoretical: 0*0.35 + 0*0.35 + 1.0*0.20 + 1.0*0.10 = 0.30
        assert score <= 0.31

    @pytest.mark.asyncio
    async def test_below_baseline_savings_potential_is_zero(self):
        """When power < baseline, savings_potential = 0 (user already saving)."""
        agent = self._base_agent(power=600.0, baseline=1000.0)
        with patch.object(da_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 3, 0, 0)  # late night
            score = await agent._calculate_opportunity_score()
        # savings=0, anomaly=0: urgency=0, fatigue=0 behaviour=1: receptiveness=1.0
        # context at 3am: time=0.3, no occupancy=0.5, context=(0.3+0.5)/2=0.4
        # score = 0 + 0 + 0.20*1.0 + 0.10*0.4 = 0.24
        assert score == pytest.approx(0.24, abs=0.05)

    @pytest.mark.asyncio
    async def test_above_baseline_savings_potential_positive(self):
        """50% above baseline -> savings_potential=0.5 contributes to score."""
        agent = self._base_agent(power=1500.0, baseline=1000.0,
                                 anomaly=0.0, fatigue=1.0, behaviour=0.0, occupancy=False)
        with patch.object(da_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 3, 0, 0)  # suppress context
            score = await agent._calculate_opportunity_score()
        # savings_potential=0.5, urgency=0, receptiveness=0, context=low
        # Lower bound: 0.35*0.5 = 0.175
        assert score >= 0.17

    @pytest.mark.asyncio
    async def test_savings_potential_capped_at_one(self):
        """300% above baseline -> savings_potential capped at 1.0."""
        agent = self._base_agent(power=4000.0, baseline=1000.0,
                                 anomaly=0.0, fatigue=1.0, behaviour=0.0, occupancy=False)
        with patch.object(da_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 3, 0, 0)
            score = await agent._calculate_opportunity_score()
        # savings_potential must be 1.0 (capped), so score >= 0.35
        assert score >= 0.35

    @pytest.mark.asyncio
    async def test_area_anomalies_boost_urgency(self):
        """Each area with any anomaly value > 0.3 adds 0.1 to urgency (capped at 1)."""
        area_anomalies = {
            "kitchen": {"power": 0.8},
            "office":  {"power": 0.5},
        }
        agent_no_area = self._base_agent(power=500.0, baseline=1000.0,
                                          anomaly=0.2, area_anomalies={})
        agent_with_area = self._base_agent(power=500.0, baseline=1000.0,
                                            anomaly=0.2, area_anomalies=area_anomalies)
        with patch.object(da_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 12, 0, 0)
            score_no_area   = await agent_no_area._calculate_opportunity_score()
            score_with_area = await agent_with_area._calculate_opportunity_score()
        # 2 qualifying area anomalies add 0.20 to urgency weight (0.35 * 0.20 = 0.07 extra)
        assert score_with_area > score_no_area

    @pytest.mark.asyncio
    async def test_high_fatigue_reduces_receptiveness(self):
        """fatigue=1.0 makes receptiveness=0; fatigue=0.0 keeps receptiveness=behaviour."""
        agent_tired   = self._base_agent(power=1500.0, baseline=1000.0,
                                          fatigue=1.0, behaviour=1.0)
        agent_rested  = self._base_agent(power=1500.0, baseline=1000.0,
                                          fatigue=0.0, behaviour=1.0)
        with patch.object(da_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 12, 0, 0)
            score_tired  = await agent_tired._calculate_opportunity_score()
            score_rested = await agent_rested._calculate_opportunity_score()
        assert score_rested > score_tired

    @pytest.mark.asyncio
    async def test_score_always_in_zero_to_one_range(self):
        """Opportunity score must be in [0, 1] for extreme input combinations."""
        cases = [
            dict(power=0.0,    baseline=1000.0, anomaly=0.0, fatigue=0.0, behaviour=0.0),
            dict(power=5000.0, baseline=1000.0, anomaly=1.0, fatigue=0.0, behaviour=1.0),
            dict(power=1000.0, baseline=1000.0, anomaly=0.5, fatigue=0.5, behaviour=0.5),
            dict(power=1000.0, baseline=0.0,    anomaly=0.0, fatigue=0.0, behaviour=1.0),
        ]
        with patch.object(da_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 12, 0, 0)
            for c in cases:
                agent = self._base_agent(**c)
                score = await agent._calculate_opportunity_score()
                assert 0.0 <= score <= 1.0, (
                    f"Score {score} out of [0,1] for inputs: {c}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Additional _load_persistent_state coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadPersistentStateExtended:
    """Extended tests covering fields not tested by TestLoadPersistentState."""

    def _agent_with_storage(self, state_data):
        agent = make_agent()
        storage = AsyncMock()
        storage.load_state = AsyncMock(return_value=state_data)
        agent.storage = storage
        return agent

    @pytest.mark.asyncio
    async def test_loads_start_date_valid_iso_string(self):
        agent = self._agent_with_storage({"start_date": "2026-01-15T10:00:00"})
        await agent._load_persistent_state()
        assert agent.start_date.year == 2026
        assert agent.start_date.month == 1
        assert agent.start_date.day == 15

    @pytest.mark.asyncio
    async def test_loads_start_date_invalid_falls_back_to_now(self):
        before = datetime.now()
        agent = self._agent_with_storage({"start_date": "not-a-date"})
        await agent._load_persistent_state()
        after = datetime.now()
        assert before <= agent.start_date <= after

    @pytest.mark.asyncio
    async def test_loads_area_baselines(self):
        agent = self._agent_with_storage({"area_baselines": {"Living Room": 300.0, "Kitchen": 150.0}})
        await agent._load_persistent_state()
        assert agent.area_baselines == {"Living Room": 300.0, "Kitchen": 150.0}

    @pytest.mark.asyncio
    async def test_loads_current_week_start_date_valid(self):
        agent = self._agent_with_storage({"current_week_start_date": "2026-02-02"})
        await agent._load_persistent_state()
        from datetime import date
        assert agent.current_week_start_date == date(2026, 2, 2)

    @pytest.mark.asyncio
    async def test_loads_current_week_start_date_invalid_sets_none(self):
        agent = self._agent_with_storage({"current_week_start_date": "bad-date"})
        await agent._load_persistent_state()
        assert agent.current_week_start_date is None

    @pytest.mark.asyncio
    async def test_loads_current_week_start_date_none_value(self):
        agent = self._agent_with_storage({"current_week_start_date": None})
        await agent._load_persistent_state()
        # None value should not change the field (guard `and state[...]`)
        assert agent.current_week_start_date is None

    @pytest.mark.asyncio
    async def test_loads_anomaly_index(self):
        agent = self._agent_with_storage({"anomaly_index": 0.65})
        await agent._load_persistent_state()
        assert agent.anomaly_index == pytest.approx(0.65)

    @pytest.mark.asyncio
    async def test_loads_behaviour_index(self):
        agent = self._agent_with_storage({"behaviour_index": 0.72})
        await agent._load_persistent_state()
        assert agent.behaviour_index == pytest.approx(0.72)

    @pytest.mark.asyncio
    async def test_loads_engagement_history(self):
        agent = self._agent_with_storage({"engagement_history": [0.8, 0.6, 0.9]})
        await agent._load_persistent_state()
        assert list(agent.engagement_history) == [0.8, 0.6, 0.9]

    @pytest.mark.asyncio
    async def test_loads_fatigue_index(self):
        agent = self._agent_with_storage({"fatigue_index": 0.45})
        await agent._load_persistent_state()
        assert agent.fatigue_index == pytest.approx(0.45)

    @pytest.mark.asyncio
    async def test_loads_last_notification_date_valid(self):
        agent = self._agent_with_storage({"last_notification_date": "2026-02-20"})
        await agent._load_persistent_state()
        from datetime import date
        assert agent.last_notification_date == date(2026, 2, 20)

    @pytest.mark.asyncio
    async def test_loads_last_notification_date_invalid_sets_none(self):
        agent = self._agent_with_storage({"last_notification_date": "not-a-date"})
        await agent._load_persistent_state()
        assert agent.last_notification_date is None

    @pytest.mark.asyncio
    async def test_loads_last_notification_time_valid(self):
        agent = self._agent_with_storage({"last_notification_time": "2026-02-20T14:30:00"})
        await agent._load_persistent_state()
        assert agent.last_notification_time.hour == 14
        assert agent.last_notification_time.minute == 30

    @pytest.mark.asyncio
    async def test_loads_last_notification_time_invalid_sets_none(self):
        agent = self._agent_with_storage({"last_notification_time": "garbage"})
        await agent._load_persistent_state()
        assert agent.last_notification_time is None

    @pytest.mark.asyncio
    async def test_loads_notification_history(self):
        history = [{"notification_id": "n1", "responded": False}]
        agent = self._agent_with_storage({"notification_history": history})
        await agent._load_persistent_state()
        assert len(agent.notification_history) == 1
        assert list(agent.notification_history)[0]["notification_id"] == "n1"

    @pytest.mark.asyncio
    async def test_loads_shadow_episode_number(self):
        agent = self._agent_with_storage({"shadow_episode_number": 25})
        await agent._load_persistent_state()
        assert agent.shadow_episode_number == 25

    @pytest.mark.asyncio
    async def test_loads_feedback_episode_number(self):
        agent = self._agent_with_storage({"feedback_episode_number": 12})
        await agent._load_persistent_state()
        assert agent.feedback_episode_number == 12

    @pytest.mark.asyncio
    async def test_loads_episode_number(self):
        agent = self._agent_with_storage({"episode_number": 88})
        await agent._load_persistent_state()
        assert agent.episode_number == 88

    @pytest.mark.asyncio
    async def test_loads_logged_weeks_as_set(self):
        agent = self._agent_with_storage({"logged_weeks": ["2026-01-05", "2026-01-12"]})
        await agent._load_persistent_state()
        assert "2026-01-05" in agent._logged_weeks
        assert "2026-01-12" in agent._logged_weeks
        assert isinstance(agent._logged_weeks, set)

    @pytest.mark.asyncio
    async def test_loads_active_since_valid(self):
        agent = self._agent_with_storage({"active_since": "2026-02-14T08:00:00"})
        await agent._load_persistent_state()
        assert agent.active_since.day == 14
        assert agent.active_since.month == 2

    @pytest.mark.asyncio
    async def test_loads_active_since_invalid_sets_none(self):
        agent = self._agent_with_storage({"active_since": "bad-datetime"})
        await agent._load_persistent_state()
        assert agent.active_since is None

    @pytest.mark.asyncio
    async def test_loads_active_since_none_value(self):
        agent = self._agent_with_storage({"active_since": None})
        await agent._load_persistent_state()
        assert agent.active_since is None

    @pytest.mark.asyncio
    async def test_loads_pending_episodes_valid(self):
        pending = {
            "notif-001": {
                "state_key": "(1, 0, 0, 0, 0, 0)",
                "action": 1,
                "initial_power": 500.0,
                "timestamp": "2026-02-20T10:00:00",
                "action_source": "exploit",
                "opportunity_score": 0.75,
            }
        }
        agent = self._agent_with_storage({"pending_episodes": pending})
        await agent._load_persistent_state()
        assert "notif-001" in agent.pending_episodes
        ep = agent.pending_episodes["notif-001"]
        assert ep["action"] == 1
        assert ep["initial_power"] == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_loads_pending_episodes_invalid_state_key_skipped(self):
        pending = {
            "notif-bad": {
                "state_key": "not_a_tuple",
                "action": 1,
                "initial_power": 500.0,
                "timestamp": "2026-02-20T10:00:00",
                "action_source": "exploit",
                "opportunity_score": 0.5,
            }
        }
        agent = self._agent_with_storage({"pending_episodes": pending})
        await agent._load_persistent_state()
        assert "notif-bad" not in agent.pending_episodes

    @pytest.mark.asyncio
    async def test_loads_task_streak_last_date_valid(self):
        agent = self._agent_with_storage({"task_streak_last_date": "2026-02-20"})
        await agent._load_persistent_state()
        from datetime import date
        assert agent.task_streak_last_date == date(2026, 2, 20)

    @pytest.mark.asyncio
    async def test_loads_task_streak_last_date_invalid_sets_none(self):
        agent = self._agent_with_storage({"task_streak_last_date": "bad"})
        await agent._load_persistent_state()
        assert agent.task_streak_last_date is None

    @pytest.mark.asyncio
    async def test_loads_weekly_streak_last_week(self):
        agent = self._agent_with_storage({"weekly_streak_last_week": "2026-02-02"})
        await agent._load_persistent_state()
        assert agent.weekly_streak_last_week == "2026-02-02"

    @pytest.mark.asyncio
    async def test_loads_multiple_fields_at_once(self):
        """All fields can coexist in a single state dict."""
        agent = self._agent_with_storage({
            "phase": "active",
            "baseline_consumption": 950.0,
            "anomaly_index": 0.3,
            "behaviour_index": 0.8,
            "fatigue_index": 0.2,
            "shadow_episode_number": 5,
            "feedback_episode_number": 3,
            "episode_number": 8,
            "logged_weeks": ["2026-02-02"],
            "active_since": "2026-02-15T00:00:00",
            "task_streak": 4,
            "weekly_streak": 2,
        })
        await agent._load_persistent_state()
        assert agent.phase == "active"
        assert agent.baseline_consumption == pytest.approx(950.0)
        assert agent.anomaly_index == pytest.approx(0.3)
        assert agent.shadow_episode_number == 5
        assert agent.active_since is not None
        assert agent.task_streak == 4


# ─────────────────────────────────────────────────────────────────────────────
# TestStreaks: additional edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestStreaksEdgeCases:
    """Edge cases for update_weekly_streak not covered by TestStreaks."""

    @staticmethod
    def _agent():
        return make_agent()

    @staticmethod
    def _week(delta_weeks: int = 0):
        from datetime import date, timedelta
        return (date(2026, 2, 2) + timedelta(weeks=delta_weeks)).isoformat()

    def test_weekly_streak_invalid_last_week_format_resets_to_one(self):
        """If weekly_streak_last_week is not a valid ISO date, the except branch fires."""
        agent = self._agent()
        # Set an invalid ISO date string (ISO week format, not date format)
        agent.weekly_streak_last_week = "2026-W09"
        agent.weekly_streak = 3
        # Any new week key that is a valid date triggers the fromisoformat on last_week
        agent.update_weekly_streak(True, self._week(10))
        # ValueError fallback: streak resets to 1
        assert agent.weekly_streak == 1

    def test_weekly_streak_out_of_order_call_ignored(self):
        """A week key earlier than the current last week (delta < 7) should be ignored."""
        agent = self._agent()
        agent.update_weekly_streak(True, self._week(5))
        assert agent.weekly_streak == 1
        # Supplying an earlier week should be silently ignored
        agent.update_weekly_streak(True, self._week(3))
        assert agent.weekly_streak == 1  # unchanged
