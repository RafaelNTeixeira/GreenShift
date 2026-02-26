# Testing Documentation

Green Shift includes a comprehensive test suite covering core functionality and AI logic.

## Test Coverage

**293 tests** across 8 modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_backup_manager.py` | 19 | Backup creation, cleanup, restoration |
| `test_config_flow.py` | 24 | Config flow, sensor discovery, area assignment |
| `test_data_collector.py` | 28 | Real-time monitoring, energy tracking |
| `test_decision_agent.py` | 75 | AI model, Q-learning, fatigue tracking, engagement persistence |
| `test_helpers.py` | 28 | Utility functions, conversions |
| `test_storage.py` | 37 | Database operations, data persistence, task interactions |
| `test_task_manager.py` | 28 | Task generation, difficulty adjustment, verification |
| `test_translations_runtime.py` | 54 | Multilingual support, templates |

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
- Q-learning updates
- Phase transitions (baseline -> active)
- Notification limits
- Daily notification counter
- Weekly challenge tracking

### `test_helpers.py`
- Unit conversions (kW->W, Wh->kWh)
- Environmental impact calculations (CO2, metaphors)
- Working hours validation
- Configuration parsing
- Area assignment utilities

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
