# Research Data Storage Plan for GreenShift Intervention Study

## Overview

GreenShift uses **two SQLite databases** and **one JSON file** for data persistence:

1. **`sensor_data.db`** - Temporal sensor data (14-day rolling window) + daily tasks
2. **`research_data.db`** - Permanent research data (never purged)
3. **`state.json`** - AI agent persistent state (Q-table, indices, phase info)

---

## Database Schema

### 1. Research Database (`research_data.db`)

#### Table: `research_phase_metadata`
Tracks study phase transitions (baseline → active).

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `phase` | TEXT | "baseline" or "active" |
| `start_timestamp` | REAL | Unix timestamp when phase started |
| `end_timestamp` | REAL | Unix timestamp when phase ended (NULL if current) |
| `baseline_consumption_kwh` | REAL | Learned baseline consumption (kWh) |
| `baseline_occupancy_avg` | REAL | Average occupancy count during baseline |
| `notes` | TEXT | Additional metadata |

#### Table: `research_daily_aggregates`
Daily rollup of all metrics for time-series analysis.

| Column | Type | Description |
|--------|------|-------------|
| `date` | TEXT | Primary key (YYYY-MM-DD) |
| `phase` | TEXT | "baseline" or "active" |
| `total_energy_kwh` | REAL | Total energy consumed that day |
| `avg_power_w` | REAL | Average power consumption (W) |
| `peak_power_w` | REAL | Peak power consumption (W) |
| `min_power_w` | REAL | Minimum power consumption (W) |
| `avg_occupancy_count` | REAL | Average number of occupied areas |
| `total_occupied_hours` | REAL | Total hours with occupancy |
| `avg_temperature` | REAL | Daily average temperature (°C) |
| `avg_humidity` | REAL | Daily average humidity (%) |
| `avg_illuminance` | REAL | Daily average light level (lx) |
| `tasks_generated` | INTEGER | Tasks generated that day |
| `tasks_completed` | INTEGER | Tasks marked as completed |
| `tasks_verified` | INTEGER | Tasks auto-verified by system |
| `nudges_sent` | INTEGER | Total notifications sent |
| `nudges_accepted` | INTEGER | Notifications accepted by user |
| `nudges_dismissed` | INTEGER | Notifications explicitly rejected |
| `nudges_ignored` | INTEGER | Notifications not responded to |
| `avg_anomaly_index` | REAL | Average anomaly index (0-1) |
| `avg_behaviour_index` | REAL | Average behaviour index (0-1) |
| `avg_fatigue_index` | REAL | Average user fatigue index (0-1) |
| `outdoor_temp_celsius` | REAL | External temperature (for normalization) |
| `hdd_base18` | REAL | Heating degree days (base 18°C) |

#### Table: `research_rl_episodes`
Every RL agent decision with full context.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `timestamp` | REAL | Unix timestamp of decision |
| `episode_number` | INTEGER | Sequential episode counter |
| `phase` | TEXT | "baseline" or "active" |
| `state_vector` | TEXT | JSON serialized state (all sensors) |
| `state_key` | TEXT | Discretized state for Q-table |
| `action` | INTEGER | Chosen action (0-4) |
| `action_name` | TEXT | "noop", "specific", "anomaly", "behavioural", "normative" |
| `action_source` | TEXT | "explore" or "exploit" |
| `reward` | REAL | Reward received after action |
| `q_values` | TEXT | JSON of Q-values for all actions |
| `max_q_value` | REAL | Maximum Q-value at decision time |
| `epsilon` | REAL | Exploration rate (0-1) |
| **`action_mask`** | **TEXT** | **JSON of which actions were available** ⭐ NEW |
| **`time_of_day_hour`** | **INTEGER** | **Hour of decision (0-23)** ⭐ NEW |
| **`baseline_power_reference`** | **REAL** | **Baseline consumption at decision time** ⭐ NEW |
| `current_power` | REAL | Current power consumption (W) |
| `anomaly_index` | REAL | Anomaly index at decision time |
| `behaviour_index` | REAL | Behaviour index at decision time |
| `fatigue_index` | REAL | User fatigue index at decision time |

**Enhanced Context Fields**:
- **`action_mask`**: Tracks which actions were blocked (e.g., no device to nudge → specific blocked)
  - Example: `{"noop": true, "specific": false, "anomaly": true, ...}`
- **`time_of_day_hour`**: Enables temporal pattern analysis (when are nudges most effective?)
- **`baseline_power_reference`**: Compares current vs baseline to measure intervention impact

#### Table: `research_nudge_log`
Individual notification tracking.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `timestamp` | REAL | When notification was sent |
| `notification_id` | TEXT | Unique ID for tracking |
| `action_type` | TEXT | Nudge type ("specific", "anomaly", etc.) |
| `phase` | TEXT | "baseline" or "active" |
| `message_title` | TEXT | Notification title shown to user |
| `message_body` | TEXT | Notification message content |
| `responded` | INTEGER | 1 if user responded, 0 if ignored |
| `accepted` | INTEGER | 1 if accepted, 0 if rejected |
| `response_time_seconds` | REAL | Time taken to respond |
| `current_power_w` | REAL | Power at notification time |
| `anomaly_index` | REAL | Anomaly index when sent |

**No "verified" field**: Notifications are simpler than tasks - user either responds or doesn't.

#### Table: `research_task_interactions`
Task generation, completion, and verification.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `date` | TEXT | Task date (YYYY-MM-DD) |
| `task_id` | TEXT | Unique task identifier |
| `task_type` | TEXT | Task category |
| `generation_timestamp` | REAL | When task was generated |
| `difficulty_level` | INTEGER | 1-5 (adaptive difficulty) |
| `target_value` | REAL | What user needs to achieve |
| `baseline_value` | REAL | Historical baseline for comparison |
| `completed` | INTEGER | 1 if user marked complete |
| `verified` | INTEGER | 1 if system auto-verified |
| `completion_timestamp` | REAL | When marked complete |
| `completion_value` | REAL | Actual measured value |
| `user_feedback` | TEXT | "too_easy", "just_right", "too_hard" |

**Why "verified" matters**: 
- `completed = 1` means user clicked "done"
- `verified = 1` means sensor data confirms they actually did it
- This prevents false positives in research data (research integrity)

#### Table: `research_weekly_challenges`
Weekly energy reduction challenge outcomes.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `week_start_date` | TEXT | Monday date (YYYY-MM-DD) |
| `target_percentage` | REAL | Reduction goal (e.g., 15%) |
| `baseline_avg_w` | REAL | Reference baseline consumption |
| `actual_avg_w` | REAL | Actual week average consumption |
| `success` | INTEGER | 1 if goal achieved, 0 otherwise |
| `logged_at` | REAL | When outcome was recorded |

Logged automatically on **Sunday night** if full week (Mon-Sun) is complete. Used for gamification analysis.

#### Table: `research_area_daily_stats`
Area-level daily statistics (e.g., per room/zone).

| Column | Type | Description |
|--------|------|-------------|
| `date` | TEXT | YYYY-MM-DD |
| `area_name` | TEXT | Area identifier |
| `phase` | TEXT | "baseline" or "active" |
| `avg_power_w` | REAL | Average power for that area |
| `max_power_w` | REAL | Peak power |
| `avg_temperature` | REAL | Average temperature |
| `avg_humidity` | REAL | Average humidity |
| `avg_illuminance` | REAL | Average light level |
| `total_occupied_hours` | REAL | Hours with occupancy |
| `occupancy_percentage` | REAL | % of day occupied |

---

## Key Research Metrics

### 1. Energy Efficiency

```sql
-- Normalized energy consumption per person
SELECT 
    date,
    phase,
    total_energy_kwh / NULLIF(avg_occupancy_count, 0) as energy_per_person,
    total_energy_kwh / NULLIF(hdd_base18, 0) as energy_normalized
FROM research_daily_aggregates
WHERE phase = 'baseline'
ORDER BY date;

-- Compare baseline vs intervention
SELECT 
    phase,
    AVG(total_energy_kwh / NULLIF(avg_occupancy_count, 0)) as avg_energy_per_person,
    AVG(avg_power_w) as avg_power
FROM research_daily_aggregates
GROUP BY phase;
```

### 2. Engagement Metrics

```sql
-- Task completion rate over time
SELECT 
    date,
    tasks_completed * 1.0 / NULLIF(tasks_generated, 0) as completion_rate,
    tasks_verified * 1.0 / NULLIF(tasks_completed, 0) as verification_rate
FROM research_daily_aggregates
WHERE phase = 'active'
ORDER BY date;

-- Engagement decay analysis
SELECT 
    date,
    completion_rate,
    ROW_NUMBER() OVER (ORDER BY date) as day_number
FROM (
    SELECT 
        date,
        tasks_completed * 1.0 / NULLIF(tasks_generated, 0) as completion_rate
    FROM research_daily_aggregates
    WHERE phase = 'active'
);
-- Then run linear regression on day_number vs completion_rate
```

### 3. Nudge Effectiveness

```sql
-- Overall acceptance rate
SELECT 
    SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END) * 1.0 / 
    NULLIF(SUM(CASE WHEN responded = 1 THEN 1 ELSE 0 END), 0) as acceptance_rate
FROM research_nudge_log
WHERE phase = 'active';

-- Acceptance by nudge type
SELECT 
    action_type,
    COUNT(*) as total_sent,
    SUM(responded) as total_responded,
    SUM(accepted) as total_accepted,
    ROUND(SUM(accepted) * 100.0 / NULLIF(SUM(responded), 0), 2) as acceptance_rate_pct
FROM research_nudge_log
WHERE phase = 'active'
GROUP BY action_type;

-- Time-of-day effectiveness (NEW)
SELECT 
    time_of_day_hour as hour,
    COUNT(*) as nudges_sent,
    AVG(reward) as avg_reward,
    SUM(CASE WHEN action_name != 'noop' THEN 1 ELSE 0 END) as actual_notifications
FROM research_rl_episodes
WHERE time_of_day_hour IS NOT NULL
GROUP BY time_of_day_hour
ORDER BY avg_reward DESC;
```

### 4. RL Agent Learning

```sql
-- Reward trajectory over episodes
SELECT 
    episode_number,
    AVG(reward) as avg_reward,
    MIN(reward) as min_reward,
    MAX(reward) as max_reward
FROM research_rl_episodes
WHERE episode_number IS NOT NULL
GROUP BY episode_number
ORDER BY episode_number;

-- Exploration vs Exploitation over time
SELECT 
    DATE(timestamp, 'unixepoch') as date,
    SUM(CASE WHEN action_source = 'explore' THEN 1 ELSE 0 END) as explorations,
    SUM(CASE WHEN action_source = 'exploit' THEN 1 ELSE 0 END) as exploitations
FROM research_rl_episodes
GROUP BY date
ORDER BY date;

-- Action distribution
SELECT 
    action_name,
    COUNT(*) as times_selected,
    AVG(reward) as avg_reward
FROM research_rl_episodes
WHERE action_name IS NOT NULL
GROUP BY action_name
ORDER BY times_selected DESC;

-- Action constraint analysis (NEW)
SELECT 
    DATE(timestamp, 'unixepoch') as date,
    COUNT(*) as total_decisions,
    SUM(CASE WHEN action_mask LIKE '%"specific": false%' THEN 1 ELSE 0 END) as specific_blocked,
    SUM(CASE WHEN action_mask LIKE '%"anomaly": false%' THEN 1 ELSE 0 END) as anomaly_blocked
FROM research_rl_episodes
WHERE action_mask IS NOT NULL
GROUP BY date;
```

### 5. Weekly Challenge Performance (NEW)

```sql
-- Weekly challenge success rate
SELECT 
    COUNT(*) as total_weeks,
    SUM(success) as successful_weeks,
    ROUND(SUM(success) * 100.0 / COUNT(*), 2) as success_rate_pct,
    AVG(target_percentage) as avg_target_pct,
    AVG((baseline_avg_w - actual_avg_w) / baseline_avg_w * 100) as avg_actual_reduction_pct
FROM research_weekly_challenges;

-- Weekly challenge trend
SELECT 
    week_start_date,
    target_percentage,
    ROUND((baseline_avg_w - actual_avg_w) / baseline_avg_w * 100, 1) as actual_reduction_pct,
    CASE WHEN success = 1 THEN 'SUCCESS' ELSE 'FAILED' END as outcome
FROM research_weekly_challenges
ORDER BY week_start_date;
```

### 6. Baseline Comparison (NEW)

```sql
-- How much above baseline when nudges are sent?
SELECT 
    action_name,
    COUNT(*) as times_used,
    ROUND(AVG(current_power), 2) as avg_power_at_nudge,
    ROUND(AVG(baseline_power_reference), 2) as avg_baseline_ref,
    ROUND(AVG((current_power - baseline_power_reference) / baseline_power_reference * 100), 1) as avg_pct_above_baseline
FROM research_rl_episodes
WHERE baseline_power_reference IS NOT NULL
  AND action_name != 'noop'
GROUP BY action_name;
```

---

## Data Export

All research data can be exported to CSV using the built-in `export_research_data()` method:

```python
await storage.export_research_data("./research_export/")
```

This creates 7 CSV files:
1. `daily_aggregates.csv`
2. `rl_episodes.csv`
3. `nudge_log.csv`
4. `task_interactions.csv`
5. `phase_metadata.csv`
6. `area_daily_stats.csv`
7. `weekly_challenges.csv` 
