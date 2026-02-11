-- =============================================================================
-- RESEARCH DATABASE TEST QUERIES
-- Run these queries to verify data collection is working correctly
-- Database: config/green_shift_data/research_data.db
-- =============================================================================

-- =============================================================================
-- 1. TABLE EXISTENCE & ROW COUNTS
-- =============================================================================

-- Check all tables and their row counts
SELECT 'research_daily_aggregates' as table_name, COUNT(*) as row_count FROM research_daily_aggregates
UNION ALL
SELECT 'research_rl_episodes', COUNT(*) FROM research_rl_episodes
UNION ALL
SELECT 'research_nudge_log', COUNT(*) FROM research_nudge_log
UNION ALL
SELECT 'research_task_interactions', COUNT(*) FROM research_task_interactions
UNION ALL
SELECT 'research_phase_metadata', COUNT(*) FROM research_phase_metadata
UNION ALL
SELECT 'research_area_daily_stats', COUNT(*) FROM research_area_daily_stats
UNION ALL
SELECT 'research_weekly_challenges', COUNT(*) FROM research_weekly_challenges;

-- =============================================================================
-- 2. PHASE METADATA VERIFICATION
-- =============================================================================

-- Check phase transitions
SELECT 
    id,
    phase,
    datetime(start_timestamp, 'unixepoch') as start_date,
    datetime(end_timestamp, 'unixepoch') as end_date,
    baseline_consumption_kwh,
    baseline_occupancy_avg,
    notes
FROM research_phase_metadata
ORDER BY start_timestamp;

-- =============================================================================
-- 3. DAILY AGGREGATES VERIFICATION
-- =============================================================================

-- Check daily aggregates summary
SELECT 
    COUNT(*) as total_days,
    MIN(date) as first_date,
    MAX(date) as last_date,
    SUM(CASE WHEN phase = 'baseline' THEN 1 ELSE 0 END) as baseline_days,
    SUM(CASE WHEN phase = 'active' THEN 1 ELSE 0 END) as active_days
FROM research_daily_aggregates;

-- Check recent daily aggregates (last 7 days)
SELECT 
    date,
    phase,
    ROUND(avg_power_w, 2) as avg_power_w,
    ROUND(total_energy_kwh, 2) as total_energy_kwh,
    ROUND(avg_occupancy_count, 2) as avg_occupancy,
    tasks_generated,
    tasks_completed,
    nudges_sent,
    nudges_accepted
FROM research_daily_aggregates
ORDER BY date DESC
LIMIT 7;

-- Check for missing or NULL values in daily aggregates
SELECT 
    date,
    CASE WHEN avg_power_w IS NULL THEN 'avg_power_w' END as null_power,
    CASE WHEN avg_occupancy_count IS NULL THEN 'avg_occupancy' END as null_occupancy,
    CASE WHEN avg_temperature IS NULL THEN 'avg_temperature' END as null_temp
FROM research_daily_aggregates
WHERE avg_power_w IS NULL 
   OR avg_occupancy_count IS NULL 
   OR avg_temperature IS NULL
ORDER BY date DESC;

-- =============================================================================
-- 4. RL EPISODES VERIFICATION
-- =============================================================================

-- Check RL episode summary
SELECT 
    COUNT(*) as total_episodes,
    COUNT(DISTINCT episode_number) as unique_episodes,
    MIN(datetime(timestamp, 'unixepoch')) as first_episode,
    MAX(datetime(timestamp, 'unixepoch')) as last_episode,
    AVG(reward) as avg_reward,
    MIN(reward) as min_reward,
    MAX(reward) as max_reward
FROM research_rl_episodes;

-- Check action distribution
SELECT 
    action_name,
    COUNT(*) as times_used,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM research_rl_episodes), 2) as percentage,
    ROUND(AVG(reward), 4) as avg_reward
FROM research_rl_episodes
WHERE action_name IS NOT NULL
GROUP BY action_name
ORDER BY times_used DESC;

-- Check exploration vs exploitation
SELECT 
    action_source,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM research_rl_episodes WHERE action_source IS NOT NULL), 2) as percentage,
    ROUND(AVG(reward), 4) as avg_reward
FROM research_rl_episodes
WHERE action_source IS NOT NULL
GROUP BY action_source;

-- Check recent RL episodes (last 10) with new context fields
SELECT 
    datetime(timestamp, 'unixepoch') as time,
    episode_number,
    action_name,
    action_source,
    ROUND(reward, 4) as reward,
    ROUND(epsilon, 2) as epsilon,
    ROUND(max_q_value, 4) as max_q,
    ROUND(current_power, 2) as power_w,
    time_of_day_hour,
    ROUND(baseline_power_reference, 2) as baseline_ref,
    action_mask
FROM research_rl_episodes
ORDER BY timestamp DESC
LIMIT 10;

-- Check reward trend over time
SELECT 
    DATE(timestamp, 'unixepoch') as date,
    COUNT(*) as episodes,
    ROUND(AVG(reward), 4) as avg_reward,
    ROUND(MIN(reward), 4) as min_reward,
    ROUND(MAX(reward), 4) as max_reward
FROM research_rl_episodes
GROUP BY date
ORDER BY date DESC
LIMIT 7;

-- =============================================================================
-- 5. NUDGE LOG VERIFICATION
-- =============================================================================

-- Check nudge summary
SELECT 
    COUNT(*) as total_nudges,
    SUM(responded) as total_responded,
    SUM(accepted) as total_accepted,
    ROUND(SUM(accepted) * 100.0 / NULLIF(SUM(responded), 0), 2) as acceptance_rate_pct,
    ROUND(AVG(response_time_seconds), 2) as avg_response_time_sec
FROM research_nudge_log;

-- Check nudge type distribution
SELECT 
    action_type,
    COUNT(*) as total_sent,
    SUM(responded) as responded,
    SUM(accepted) as accepted,
    ROUND(SUM(accepted) * 100.0 / NULLIF(SUM(responded), 0), 2) as acceptance_rate_pct,
    ROUND(AVG(CASE WHEN responded = 1 THEN response_time_seconds END), 2) as avg_response_time_sec
FROM research_nudge_log
GROUP BY action_type
ORDER BY total_sent DESC;

-- Check recent nudges (last 10)
SELECT 
    datetime(timestamp, 'unixepoch') as sent_time,
    action_type,
    title,
    CASE WHEN responded = 1 THEN 'Yes' ELSE 'No' END as responded,
    CASE WHEN accepted = 1 THEN 'Yes' ELSE 'No' END as accepted,
    ROUND(response_time_seconds, 2) as response_time_sec
FROM research_nudge_log
ORDER BY timestamp DESC
LIMIT 10;

-- Check nudges with no response (might indicate issues)
SELECT 
    datetime(timestamp, 'unixepoch') as sent_time,
    action_type,
    title,
    notification_id
FROM research_nudge_log
WHERE responded = 0
ORDER BY timestamp DESC
LIMIT 10;

-- =============================================================================
-- 6. TASK INTERACTIONS VERIFICATION
-- =============================================================================

-- Check task summary
SELECT 
    COUNT(*) as total_tasks,
    SUM(completed) as completed,
    SUM(verified) as verified,
    ROUND(SUM(completed) * 100.0 / COUNT(*), 2) as completion_rate_pct,
    ROUND(SUM(verified) * 100.0 / COUNT(*), 2) as verification_rate_pct,
    ROUND(AVG(time_to_complete_seconds) / 3600.0, 2) as avg_time_to_complete_hours
FROM research_task_interactions;

-- Check task type distribution
SELECT 
    task_type,
    COUNT(*) as total,
    SUM(completed) as completed,
    SUM(verified) as verified,
    ROUND(SUM(completed) * 100.0 / COUNT(*), 2) as completion_rate_pct,
    ROUND(AVG(difficulty_level), 2) as avg_difficulty
FROM research_task_interactions
GROUP BY task_type
ORDER BY total DESC;

-- Check task difficulty feedback
SELECT 
    difficulty_level,
    COUNT(*) as total_tasks,
    SUM(CASE WHEN user_feedback = 'too_easy' THEN 1 ELSE 0 END) as too_easy,
    SUM(CASE WHEN user_feedback = 'just_right' THEN 1 ELSE 0 END) as just_right,
    SUM(CASE WHEN user_feedback = 'too_hard' THEN 1 ELSE 0 END) as too_hard,
    SUM(CASE WHEN user_feedback IS NULL THEN 1 ELSE 0 END) as no_feedback
FROM research_task_interactions
GROUP BY difficulty_level
ORDER BY difficulty_level;

-- Check recent tasks (last 10)
SELECT 
    date,
    task_type,
    ROUND(target_value, 2) as target,
    target_unit,
    difficulty_level,
    CASE WHEN completed = 1 THEN 'Yes' ELSE 'No' END as completed,
    CASE WHEN verified = 1 THEN 'Yes' ELSE 'No' END as verified,
    user_feedback
FROM research_task_interactions
ORDER BY date DESC, generation_timestamp DESC
LIMIT 10;

-- Check tasks by date
SELECT 
    date,
    COUNT(*) as tasks,
    SUM(completed) as completed,
    SUM(verified) as verified
FROM research_task_interactions
GROUP BY date
ORDER BY date DESC
LIMIT 7;

-- =============================================================================
-- 7. AREA DAILY STATS VERIFICATION
-- =============================================================================

-- Check if area stats are being collected
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT area_name) as unique_areas,
    COUNT(DISTINCT date) as unique_dates,
    MIN(date) as first_date,
    MAX(date) as last_date
FROM research_area_daily_stats;

-- Check recent area stats
SELECT 
    date,
    area_name,
    ROUND(avg_power_w, 2) as avg_power,
    ROUND(avg_temperature, 2) as avg_temp,
    ROUND(avg_humidity, 2) as avg_humidity,
    ROUND(occupancy_percentage, 2) as occupancy_pct
FROM research_area_daily_stats
ORDER BY date DESC, area_name
LIMIT 20;

-- =============================================================================
-- 8. DATA QUALITY CHECKS
-- =============================================================================

-- Check for duplicate dates in daily aggregates
SELECT 
    date,
    COUNT(*) as duplicate_count
FROM research_daily_aggregates
GROUP BY date
HAVING COUNT(*) > 1;

-- Check for gaps in daily aggregates
SELECT 
    date,
    DATE(date, '+1 day') as next_expected,
    (SELECT MIN(date) FROM research_daily_aggregates WHERE date > d.date) as next_actual
FROM research_daily_aggregates d
WHERE DATE(date, '+1 day') != (SELECT MIN(date) FROM research_daily_aggregates WHERE date > d.date)
ORDER BY date DESC;

-- Check for orphaned task interactions (no matching daily aggregate)
SELECT 
    t.date,
    COUNT(*) as orphaned_tasks
FROM research_task_interactions t
LEFT JOIN research_daily_aggregates d ON t.date = d.date
WHERE d.date IS NULL
GROUP BY t.date;

-- =============================================================================
-- 9. RESEARCH METRICS PREVIEW
-- =============================================================================

-- Energy Efficiency by Phase
SELECT 
    phase,
    COUNT(*) as days,
    ROUND(AVG(avg_power_w), 2) as avg_power,
    ROUND(AVG(total_energy_kwh), 2) as avg_energy,
    ROUND(AVG(total_energy_kwh / NULLIF(avg_occupancy_count, 0)), 2) as energy_per_person
FROM research_daily_aggregates
WHERE avg_occupancy_count > 0
GROUP BY phase;

-- Engagement Rate (Active Phase Only)
SELECT 
    ROUND(SUM(tasks_completed) * 100.0 / NULLIF(SUM(tasks_generated), 0), 2) as overall_completion_rate_pct,
    ROUND(SUM(nudges_accepted) * 100.0 / NULLIF(SUM(nudges_sent), 0), 2) as overall_nudge_acceptance_pct,
    COUNT(*) as active_days
FROM research_daily_aggregates
WHERE phase = 'active';

-- RL Learning Progress
SELECT 
    CASE 
        WHEN episode_number <= 100 THEN '1-100'
        WHEN episode_number <= 500 THEN '101-500'
        WHEN episode_number <= 1000 THEN '501-1000'
        ELSE '1000+'
    END as episode_range,
    COUNT(*) as episodes,
    ROUND(AVG(reward), 4) as avg_reward,
    ROUND(AVG(max_q_value), 4) as avg_max_q,
    ROUND(SUM(CASE WHEN action_source = 'explore' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as exploration_pct
FROM research_rl_episodes
WHERE episode_number IS NOT NULL
GROUP BY episode_range
ORDER BY MIN(episode_number);

-- =============================================================================
-- 10. SYSTEM HEALTH CHECK
-- =============================================================================

-- Check last data collection time for each table
SELECT 
    'daily_aggregates' as table_name,
    MAX(date) as last_date,
    datetime(MAX(created_at)) as last_created
FROM research_daily_aggregates

UNION ALL

SELECT 
    'rl_episodes',
    DATE(MAX(timestamp), 'unixepoch'),
    datetime(MAX(created_at))
FROM research_rl_episodes

UNION ALL

SELECT 
    'nudge_log',
    DATE(MAX(timestamp), 'unixepoch'),
    datetime(MAX(created_at))
FROM research_nudge_log

UNION ALL

SELECT 
    'task_interactions',
    MAX(date),
    datetime(MAX(created_at))
FROM research_task_interactions

UNION ALL

SELECT 
    'phase_metadata',
    DATE(MAX(start_timestamp), 'unixepoch'),
    datetime(MAX(created_at))
FROM research_phase_metadata;

-- Check current phase
SELECT 
    phase,
    datetime(start_timestamp, 'unixepoch') as started,
    ROUND((julianday('now') - julianday(start_timestamp, 'unixepoch')), 1) as days_in_phase,
    notes
FROM research_phase_metadata
WHERE end_timestamp IS NULL
ORDER BY start_timestamp DESC
LIMIT 1;

-- =============================================================================
-- 11. EXPORT-READY QUERIES
-- =============================================================================

-- Daily time series (for graphing)
SELECT 
    date,
    phase,
    avg_power_w,
    total_energy_kwh,
    avg_occupancy_count,
    tasks_generated,
    tasks_completed,
    nudges_sent,
    nudges_accepted,
    avg_anomaly_index,
    avg_behaviour_index,
    avg_fatigue_index
FROM research_daily_aggregates
ORDER BY date;

-- Nudge acceptance by type and phase
SELECT 
    phase,
    action_type,
    COUNT(*) as sent,
    SUM(responded) as responded,
    SUM(accepted) as accepted,
    ROUND(SUM(accepted) * 100.0 / NULLIF(SUM(responded), 0), 2) as acceptance_rate_pct
FROM research_nudge_log
GROUP BY phase, action_type
ORDER BY phase, sent DESC;

-- Task completion trends
SELECT 
    date,
    task_type,
    difficulty_level,
    completed,
    verified,
    user_feedback
FROM research_task_interactions
ORDER BY date, generation_timestamp;

-- RL episode trajectory
SELECT 
    episode_number,
    AVG(reward) as avg_reward,
    AVG(max_q_value) as avg_max_q,
    SUM(CASE WHEN action_source = 'explore' THEN 1 ELSE 0 END) as explorations,
    SUM(CASE WHEN action_source = 'exploit' THEN 1 ELSE 0 END) as exploitations
FROM research_rl_episodes
WHERE episode_number IS NOT NULL
GROUP BY episode_number
ORDER BY episode_number;

-- Weekly challenge tracking
SELECT 
    week_start_date,
    target_percentage,
    ROUND(baseline_avg_w, 2) as baseline_w,
    ROUND(actual_avg_w, 2) as actual_w,
    ROUND((actual_avg_w / baseline_avg_w - 1) * 100, 1) as change_pct,
    CASE WHEN success = 1 THEN 'SUCCESS' ELSE 'FAILED' END as outcome
FROM research_weekly_challenges
ORDER BY week_start_date DESC;

-- Action constraint analysis (action_mask usage)
SELECT 
    DATE(timestamp, 'unixepoch') as date,
    COUNT(*) as total_decisions,
    SUM(CASE WHEN action_mask LIKE '%"noop": false%' THEN 1 ELSE 0 END) as noop_blocked,
    SUM(CASE WHEN action_mask LIKE '%"specific": false%' THEN 1 ELSE 0 END) as specific_blocked,
    SUM(CASE WHEN action_mask LIKE '%"anomaly": false%' THEN 1 ELSE 0 END) as anomaly_blocked
FROM research_rl_episodes
WHERE action_mask IS NOT NULL
GROUP BY date
ORDER BY date DESC
LIMIT 7;

-- Time-of-day decision patterns
SELECT 
    time_of_day_hour as hour,
    COUNT(*) as decisions,
    SUM(CASE WHEN action_name != 'noop' THEN 1 ELSE 0 END) as notifications_sent,
    ROUND(AVG(CASE WHEN action_name != 'noop' THEN reward END), 4) as avg_reward_when_notified
FROM research_rl_episodes
WHERE time_of_day_hour IS NOT NULL
GROUP BY time_of_day_hour
ORDER BY time_of_day_hour;

-- Baseline comparison effectiveness
SELECT 
    action_name,
    COUNT(*) as times_used,
    ROUND(AVG(current_power), 2) as avg_power_at_decision,
    ROUND(AVG(baseline_power_reference), 2) as avg_baseline_ref,
    ROUND(AVG(current_power - baseline_power_reference), 2) as avg_power_diff
FROM research_rl_episodes
WHERE baseline_power_reference IS NOT NULL
  AND action_name IS NOT NULL
GROUP BY action_name
ORDER BY times_used DESC;

-- =============================================================================
-- END OF TEST QUERIES
-- =============================================================================
