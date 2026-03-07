# Testing Documentation

Green Shift includes a comprehensive test suite covering core functionality and AI logic.

## Coverage Summary

**930 tests** across 11 modules - total measured coverage **75.6%**.

> **Note on `__init__.py`:** The main integration setup file requires a live Home Assistant runtime for its service registration, event bus subscriptions and platform setup. Its measured coverage is 12%: this is expected and cannot be meaningfully increased with unit tests alone. Excluding it, the **adjusted coverage across all other modules is ≈ 89%**.

| Module | Tests | Coverage | Notes |
|--------|-------|----------|-------|
| `test_backup_manager.py` | 31 | **99%** | Backup creation, cleanup, restoration |
| `test_config_flow.py` | 51 | **97%** | Config flow, sensor discovery, area assignment |
| `test_data_collector.py` | 73 | **84%** | Real-time monitoring, energy tracking |
| `test_decision_agent.py` | 268 | **77%** | AI model, Q-learning, fatigue tracking, engagement persistence, state persistence, action masking |
| `test_helpers.py` | 46 | **99%** | Utility functions, conversions, entity/area resolution, working hours |
| `test_init.py` | 33 | - | Tests service registration and restore-backup reload behaviour |
| `test_select.py` | 36 | **94%** | |
| `test_sensor.py` | 119 | **95%** | |
| `test_storage.py` | 107 | **90%** | Deeply nested SQL error handlers not exercised |
| `test_task_manager.py` | 105 | **93%** | Database operations, data persistence, task interactions, research data |
| `test_translations_runtime.py` | 61 | **91%** | Multilingual support, templates |

## Running Tests

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run All Tests

Run from the **workspace root** (not from `tests/`):

```bash
python3 -m pytest tests/
```

Coverage is automatically generated in `tests/htmlcov/`.

### Run Specific Module

```bash
python3 -m pytest tests/test_decision_agent.py
python3 -m pytest tests/test_backup_manager.py -v  # verbose output
```

### Run without Coverage (faster)

```bash
python3 -m pytest tests/ --no-cov
```

### View Coverage Report

After running tests, open `tests/htmlcov/index.html` in a browser to see detailed per-line coverage.

---

## Test Structure

### `test_backup_manager.py` - 31 tests, **99%** coverage
- Directory structure creation
- Automatic backup generation (startup, shutdown, daily triggers)
- Cleanup policies (old backup removal by count/age)
- Backup restoration (full state recovery)
- Edge cases (missing backups, corrupted files, concurrent access)

### `test_config_flow.py` - 51 tests, **97%** coverage
- Multi-step flow navigation (5 steps)
- Environment mode branching (home vs office)
- Sensor discovery and value-based sorting
- Main sensor selection and injection
- Area assignment to entity registry
- Complete integration flows (home & office modes)
- Input validation and edge cases

### `test_data_collector.py` - 73 tests, **84%** coverage
- Real-time sensor monitoring
- Energy midnight point tracking
- Daily kWh calculations
- Area-based data aggregation
- Working hours filtering
- Sensor cache management
- *Note*: setup-phase callbacks that fire on HA startup cannot be exercised without the HA event loop.

### `test_decision_agent.py` - 268 tests, **77%** coverage
- State discretization (power bins, indices)
- Fatigue index calculation (rejection rate, time decay)
- Behavior index updates (EMA)
- Cooldown mechanisms (adaptive, opportunity-based)
- Q-learning updates (feedback acceptance, rejection, shadow updates)
- Phase transitions (baseline -> active)
- Notification limits and daily/weekly counters
- Weekly challenge target calculation and streak tracking
- State persistence (load/save from storage, Q-table serialisation, all field edge cases)
- State vector construction (18-element feature vector)
- Top power consumer detection
- Action mask updates (noop, specific, normative, behavioural modes)
- Non-working day gap detection
- Full AI model cycle (`process_ai_model`): daily counter reset, episode expiry
- *Note*: The full `_decide_action` orchestration, shadow-learning inner loops, and RL episode logging paths require deeply coupled dependencies and remain partially untested (77% -> bounded by integration complexity).

### `test_helpers.py` - 46 tests, **99%** coverage
- Unit conversions (kW->W, Wh->kWh)
- Environmental impact calculations (CO2, metaphors)
- Working hours validation (including timezone-aware datetimes)
- Configuration parsing
- Area assignment utilities
- Entity area resolution via entity registry, device registry and area registry
- Sensor grouping by area (including unknown-entity fallback)
- Friendly name resolution (attribute, registry, entity_id fallback)
- Daily working hours calculation (standard, custom, midnight-crossing, malformed config)

### `test_init.py` - 33 tests
- Service registration mocks
- Entry setup/teardown
- Event bus subscription hooks
- `restore_backup` service: in-memory reload after restore
- *Note*: coverage not reported separately (exercises `__init__.py` which is 12% due to HA runtime requirements)

### `test_select.py` - 36 tests, **94%** coverage
- `AreaViewSelect`: option population, current_option, async callbacks, `async_on_remove` unsubscription
- `NotificationSelect`: pending notification rendering, selection acknowledgment, option refresh
- `_update_callback` write-state path (sync and async)
- *Note*: `async_setup_entry` (lines 34-40) requires `hass.data[DOMAIN]` from HA platform loader and cannot be unit-tested.

### `test_sensor.py` - 119 tests, **95%** coverage
- `HardwareSensors`: normalized_value sanitization, occupancy forwarding, `None`-value skip
- `DailyTasksSensor`: task list state, attribute structure (completion values, `checked_at`, pending result)
- `EnergyUsageSensor` / `DailyCO2EstimateSensor`: state extraction, name/unique_id, unit
- `SavingsAccumulatedSensor`: baseline guard, no-active-since guard, provisional estimate, full kwh -> coins calculation, days_tracked attribute, currency units
- `CO2SavedSensor`: baseline guard, no-active-since guard, provisional estimate, CO2 kg calculation, total_kwh attribute
- `TasksCompletedSensor`: fetch from task_manager, zero count, state property
- `WeeklyChallengeSensor`: `_get_target_percentage` (all difficulty levels), baseline early return, active phase progress, active attrs
- `GreenShiftBaseSensor` / `GreenShiftAISensor` callbacks: `async_added_to_hass`, `async_on_remove`, `_update_callback` create_task dispatch
- *Note*: Lines 104/111-112 in sensor.py are unreachable dead code (see coverage note above).

### `test_storage.py` - 107 tests, **90%** coverage
- Database initialization (SQLite schema creation)
- Sensor snapshot storage and retrieval
- Area-based data tracking
- Historical data queries (rolling windows, daily aggregates)
- Task persistence (create, complete, fetch pending)
- Data retention and cleanup (age-based pruning)
- State file operations (JSON read/write, corrupt-file recovery)
- *Note*: Deeply nested SQL exception handlers (e.g. multi-level rollback paths) are not exercised.

### `test_task_manager.py` - 105 tests, **93%** coverage
- Task generation with sensor constraints
- Phase guards (baseline vs active)
- Working hours enforcement
- Difficulty multipliers
- Idempotence (no duplicate tasks in same window)
- Dynamic difficulty calculation with historical stats (clamping, adjustment, fallback)
- Task generator edge cases (no history, no daytime readings, no occupied areas)
- Task verification logic (temperature below/above target, no history, pre-verified tasks)
- Exception handling during verification

### `test_translations_runtime.py` - 61 tests, **91%** coverage
- Language detection and selection
- Notification template rendering (English and Portuguese)
- Task template localization
- Time period translations
- Difficulty level labels
- Phase transition messages

---

**Contributing?** Keep per-module coverage ≥ 85% (excluding `__init__.py`) and add tests for new features before submitting PRs. Run `python3 -m pytest tests/` from the workspace root to verify.
