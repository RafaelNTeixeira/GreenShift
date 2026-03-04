# Testing Documentation

Green Shift includes a comprehensive test suite covering core functionality and AI logic.

## Test Coverage

**607 tests** across 8 modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_backup_manager.py` | 27 | Backup creation, cleanup, restoration |
| `test_config_flow.py` | 41 | Config flow, sensor discovery, area assignment |
| `test_data_collector.py` | 69 | Real-time monitoring, energy tracking |
| `test_decision_agent.py` | 201 | AI model, Q-learning, fatigue tracking, engagement persistence, state persistence, action masking |
| `test_helpers.py` | 46 | Utility functions, conversions, entity/area resolution, working hours |
| `test_storage.py` | 107 | Database operations, data persistence, task interactions, research data |
| `test_task_manager.py` | 61 | Task generation, difficulty adjustment, verification, edge cases |
| `test_translations_runtime.py` | 55 | Multilingual support, templates |

## Running Tests

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
cd tests/
pytest
```

Coverage is automatically generated in `htmlcov/` directory.

### Run Specific Module

```bash
cd tests/
pytest test_decision_agent.py
pytest test_backup_manager.py -v  # verbose output
```

### Run without Coverage (faster)

```bash
cd tests/
pytest --no-cov
```

### View Coverage Report

After running tests, open `tests/htmlcov/index.html` in your browser to see detailed coverage information.

## Test Structure

Tests are organized by component:

### `test_backup_manager.py`
- Directory structure creation
- Automatic backup generation
- Cleanup policies (old backup removal)
- Backup restoration
- Edge cases (missing backups, corrupted files)

### `test_config_flow.py`
- Multi-step flow navigation (5 steps)
- Environment mode branching (home vs office)
- Sensor discovery and value-based sorting
- Main sensor selection and injection
- Area assignment to entity registry
- Complete integration flows (home & office modes)
- Input validation and edge cases

### `test_data_collector.py`
- Real-time sensor monitoring
- Energy midnight point tracking
- Daily kWh calculations
- Area-based data aggregation
- Working hours filtering
- Sensor cache management

### `test_decision_agent.py`
- State discretization (power bins, indices)
- Fatigue index calculation (rejection rate, time decay)
- Behavior index updates (EMA)
- Cooldown mechanisms (adaptive, opportunity-based)
- Q-learning updates (feedback acceptance, rejection, shadow updates)
- Phase transitions (baseline -> active)
- Notification limits
- Daily notification counter
- Weekly challenge tracking
- State persistence (load/save from storage, Q-table serialisation)
- State vector construction (18-element feature vector)
- Top power consumer detection
- Action mask updates (noop, specific, normative, behavioural modes)
- Non-working day gap detection
- Full AI model cycle (`process_ai_model`): daily counter reset, episode expiry

### `test_helpers.py`
- Unit conversions (kW->W, Wh->kWh)
- Environmental impact calculations (CO2, metaphors)
- Working hours validation (including timezone-aware datetimes)
- Configuration parsing
- Area assignment utilities
- Entity area resolution via entity registry, device registry and area registry
- Sensor grouping by area (including unknown-entity fallback)
- Friendly name resolution (attribute, registry, entity_id fallback)
- Daily working hours calculation (standard, custom, midnight-crossing, malformed config)

### `test_storage.py`
- Database initialization (SQLite)
- Sensor snapshot storage
- Area-based data tracking
- Historical data queries
- Task persistence
- Data retention and cleanup
- State file operations (JSON)

### `test_task_manager.py`
- Task generation with sensor constraints
- Phase guards (baseline vs active)
- Working hours enforcement
- Difficulty multipliers
- Idempotence (no duplicate tasks)
- Dynamic difficulty calculation with historical stats (clamping, adjustment, fallback)
- Task generator edge cases (no history, no daytime readings, no occupied areas)
- Task verification logic (temperature below/above target, no history, pre-verified tasks)
- Exception handling during verification

### `test_translations_runtime.py`
- Language detection and selection
- Notification template rendering
- Task template localization
- Time period translations
- Difficulty level labels
- Phase transition messages
- English and Portuguese coverage

---

**Contributing?** Add tests for new features before submitting PRs!
