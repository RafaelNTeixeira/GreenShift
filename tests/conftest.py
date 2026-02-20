"""
Shared fixtures for Green Shift unit tests.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime
from collections import deque


# ---------------------------------------------------------------------------
# Minimal Home Assistant mock (avoids importing the full HA stack)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.states = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    return hass


# ---------------------------------------------------------------------------
# Storage mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.load_state = AsyncMock(return_value={})
    storage.save_state = AsyncMock()
    storage.get_today_tasks = AsyncMock(return_value=[])
    storage.save_daily_tasks = AsyncMock()
    storage.log_task_generation = AsyncMock()
    return storage


# ---------------------------------------------------------------------------
# Data collector mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_collector():
    collector = MagicMock()
    collector.get_current_state = MagicMock(return_value={
        "power": 500.0,
        "temperature": 21.0,
        "humidity": 50.0,
        "occupancy": True,
    })
    collector.get_power_history = AsyncMock(return_value=[])
    collector.get_temperature_history = AsyncMock(return_value=[])
    collector.get_area_history = AsyncMock(return_value=[])
    collector.get_all_areas = MagicMock(return_value=["Living Room", "Kitchen"])
    return collector


# ---------------------------------------------------------------------------
# Config data helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def home_config():
    """Standard home-mode configuration."""
    return {"environment_mode": "home", "electricity_price": 0.25, "currency": "EUR"}


@pytest.fixture
def office_config():
    """Office-mode configuration with Mon-Fri 08:00-18:00."""
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


# ---------------------------------------------------------------------------
# Sensor state mock helper
# ---------------------------------------------------------------------------

def make_sensor_state(value, unit):
    """Return a minimal sensor state object."""
    state = MagicMock()
    state.state = str(value)
    state.attributes = {"unit_of_measurement": unit}
    return state