# Testing Documentation

Green Shift includes a comprehensive test suite covering core functionality and AI logic.

## Coverage Summary

**1154 tests** across 11 modules - total measured coverage **100%**.

| Module | Tests | Coverage | Notes |
|--------|-------|----------|-------|
| `test_backup_manager.py` | 46 | **100%** | Backup creation, cleanup, restoration |
| `test_config_flow.py` | 58 | **100%** | Config flow, sensor discovery, area assignment |
| `test_data_collector.py` | 97 | **100%** | Real-time monitoring, energy tracking |
| `test_decision_agent.py` | 349 | **100%** | AI model, Q-learning, fatigue tracking, engagement persistence, state persistence, action masking |
| `test_helpers.py` | 58 | **100%** | Utility functions, conversions, entity/area resolution, working hours |
| `test_init.py` | 55 | **100%** | Integration setup/services/unload/discovery and callback/runtime edge paths |
| `test_select.py` | 37 | **100%** | |
| `test_sensor.py` | 129 | **100%** | |
| `test_storage.py` | 131 | **100%** | Database operations, data persistence, retention |
| `test_task_manager.py` | 128 | **100%** | Task generation/verification and persistence |
| `test_translations_runtime.py` | 66 | **100%** | Multilingual support, templates |

## Running Tests

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run All Tests

Run from the **workspace root** (not from `tests/`):

```bash
python3 -m pytest -n auto tests/
```

Coverage is automatically generated in `tests/htmlcov/`.

### Run Specific Module

```bash
python3 -m pytest -n auto tests/test_decision_agent.py
python3 -m pytest -n auto tests/test_backup_manager.py -v  # verbose output
```

### Run without Coverage (faster)

```bash
python3 -m pytest -n auto tests/ --no-cov
```

### View Coverage Report

After running tests, open `tests/htmlcov/index.html` in a browser to see detailed per-line coverage.

---

## Test Structure

### `test_backup_manager.py` - 46 tests, **100%** coverage
- Directory structure creation
- Automatic backup generation (startup, shutdown, daily triggers)
- Cleanup policies (old backup removal by count/age)
- Backup restoration (full state recovery)
- Edge cases (missing backups, corrupted files, concurrent access)

### `test_config_flow.py` - 58 tests, **100%** coverage
- Multi-step flow navigation (5 steps)
- Environment mode branching (home vs office)
- Sensor discovery and value-based sorting
- Main sensor selection and injection
- Area assignment to entity registry
- Complete integration flows (home & office modes)
- Input validation and edge cases

### `test_data_collector.py` - 97 tests, **100%** coverage
- Real-time sensor monitoring
- Energy midnight point tracking
- Daily kWh calculations
- Area-based data aggregation
- Working hours filtering
- Sensor cache management

### `test_decision_agent.py` - 349 tests, **100%** coverage
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
- Full decision-agent unit branch coverage achieved in pure logic tests.

### `test_helpers.py` - 58 tests, **100%** coverage
- Unit conversions (kW->W, Wh->kWh)
- Environmental impact calculations (CO2, metaphors)
- Working hours validation (including timezone-aware datetimes)
- Configuration parsing
- Area assignment utilities
- Entity area resolution via entity registry, device registry and area registry
- Sensor grouping by area (including unknown-entity fallback)
- Friendly name resolution (attribute, registry, entity_id fallback)
- Daily working hours calculation (standard, custom, midnight-crossing, malformed config)

### `test_init.py` - 55 tests, **100%** coverage
- Service registration mocks
- Entry setup/teardown
- Event bus subscription hooks
- `restore_backup` service: in-memory reload after restore
- Setup callbacks and phase-transition runtime branches
- Backup/cleanup success, failure and exception branches
- Unload error handlers and shutdown backup/state persistence paths

### `test_select.py` - 37 tests, **100%** coverage
- `AreaViewSelect`: option population, current_option, async callbacks, `async_on_remove` unsubscription
- `NotificationSelect`: pending notification rendering, selection acknowledgment, option refresh
- `_update_callback` write-state path (sync and async)

### `test_sensor.py` - 129 tests, **100%** coverage
- `HardwareSensors`: normalized_value sanitization, occupancy forwarding, `None`-value skip
- `DailyTasksSensor`: task list state, attribute structure (completion values, `checked_at`, pending result)
- `EnergyUsageSensor` / `DailyCO2EstimateSensor`: state extraction, name/unique_id, unit
- `SavingsAccumulatedSensor`: baseline guard, no-active-since guard, provisional estimate, full kwh -> coins calculation, days_tracked attribute, currency units
- `CO2SavedSensor`: baseline guard, no-active-since guard, provisional estimate, CO2 kg calculation, total_kwh attribute
- `TasksCompletedSensor`: fetch from task_manager, zero count, state property
- `WeeklyChallengeSensor`: `_get_target_percentage` (all difficulty levels), baseline early return, active phase progress, active attrs
- `GreenShiftBaseSensor` / `GreenShiftAISensor` callbacks: `async_added_to_hass`, `async_on_remove`, `_update_callback` create_task dispatch

### `test_storage.py` - 131 tests, **100%** coverage
- Database initialization (SQLite schema creation)
- Sensor snapshot storage and retrieval
- Area-based data tracking
- Historical data queries (rolling windows, daily aggregates)
- Task persistence (create, complete, fetch pending)
- Data retention and cleanup (age-based pruning)
- State file operations (JSON read/write, corrupt-file recovery)

### `test_task_manager.py` - 128 tests, **100%** coverage
- Task generation with sensor constraints
- Phase guards (baseline vs active)
- Working hours enforcement
- Difficulty multipliers
- Idempotence (no duplicate tasks in same window)
- Dynamic difficulty calculation with historical stats (clamping, adjustment, fallback)
- Task generator edge cases (no history, no daytime readings, no occupied areas)
- Task verification logic (temperature below/above target, no history, pre-verified tasks)
- Exception handling during verification

### `test_translations_runtime.py` - 66 tests, **100%** coverage
- Language detection and selection
- Notification template rendering (English and Portuguese)
- Task template localization
- Time period translations
- Difficulty level labels
- Phase transition messages

---

**Contributing?** Keep per-module coverage ≥ 85% (excluding `__init__.py`) and add tests for new features before submitting PRs. Run `python3 -m pytest -n auto tests/` from the workspace root to verify.
