# Testing Documentation

Green Shift includes a comprehensive test suite covering core functionality and AI logic.

## Test Coverage

**117 tests** across 4 modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_backup_manager.py` | 19 | Backup creation, cleanup, restoration |
| `test_decision_agent.py` | 56 | AI model, Q-learning, fatigue tracking |
| `test_helpers.py` | 28 | Utility functions, conversions |
| `test_task_manager.py` | 14 | Task generation, difficulty adjustment |

## Running Tests

### Prerequisites

```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
cd tests/
pytest
```

### Run Specific Module

```bash
pytest test_decision_agent.py
pytest test_backup_manager.py -v  # verbose output
```

### Run with Coverage Report

```bash
pytest --cov=config/custom_components/green_shift --cov-report=html
```

## Test Structure

Tests are organized by component:

### `test_backup_manager.py`
- Directory structure creation
- Automatic backup generation
- Cleanup policies (old backup removal)
- Backup restoration
- Edge cases (missing backups, corrupted files)

### `test_decision_agent.py`
- State discretization (power bins, indices)
- Fatigue index calculation (rejection rate, time decay)
- Behavior index updates (EMA)
- Cooldown mechanisms (adaptive, opportunity-based)
- Q-learning updates
- Phase transitions (baseline -> active)
- Notification limits

### `test_helpers.py`
- Unit conversions (kW->W, Wh->kWh)
- Environmental impact calculations (CO2, metaphors)
- Working hours validation
- Configuration parsing

### `test_task_manager.py`
- Task generation with sensor constraints
- Phase guards (baseline vs active)
- Working hours enforcement
- Difficulty multipliers
- Idempotence (no duplicate tasks)

---

**Contributing?** Add tests for new features before submitting PRs!
