# Research Data Storage Plan for GreenShift Intervention Study

## Metrics to Calculate

### Energy Efficiency Metrics

```sql
-- After intervention, with outdoor temperature data
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

### Engagement Metrics

```sql
-- Collective participation rate
SELECT 
    date,
    tasks_completed * 1.0 / NULLIF(tasks_generated, 0) as completion_rate
FROM research_daily_aggregates
WHERE phase = 'active'
ORDER BY date;

-- Engagement decay over time
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

### System Accuracy (Nudge Acceptance)

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
    SUM(accepted) * 1.0 / NULLIF(SUM(responded), 0) as acceptance_rate
FROM research_nudge_log
WHERE phase = 'active'
GROUP BY action_type;

-- Response time analysis
SELECT 
    action_type,
    AVG(response_time_seconds) as avg_response_time,
    MIN(response_time_seconds) as min_response_time,
    MAX(response_time_seconds) as max_response_time
FROM research_nudge_log
WHERE responded = 1
GROUP BY action_type;
```

### RL Agent Convergence

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
    SUM(CASE WHEN action_source = 'exploit' THEN 1 ELSE 0 END) as exploitations,
    COUNT(*) as total_decisions
FROM research_rl_episodes
GROUP BY date
ORDER BY date;

-- Action distribution over time
SELECT 
    action_name,
    COUNT(*) as times_selected,
    AVG(reward) as avg_reward
FROM research_rl_episodes
WHERE action_name IS NOT NULL
GROUP BY action_name
ORDER BY times_selected DESC;
```

### Engagement-Energy Correlation

```sql
-- Daily engagement vs energy savings
SELECT 
    d.date,
    d.tasks_completed * 1.0 / NULLIF(d.tasks_generated, 0) as participation_index,
    d.total_energy_kwh,
    d.total_energy_kwh - b.avg_baseline_energy as energy_savings
FROM research_daily_aggregates d
CROSS JOIN (
    SELECT AVG(total_energy_kwh) as avg_baseline_energy
    FROM research_daily_aggregates
    WHERE phase = 'baseline'
) b
WHERE d.phase = 'active'
ORDER BY d.date;
-- Then compute Spearman correlation in Python
```
