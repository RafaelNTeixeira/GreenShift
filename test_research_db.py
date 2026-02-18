"""
Research Database Test Script
Run this to verify data collection in research_data.db
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import sys

def connect_db(db_path):
    """Connect to research database."""
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print(f"   Expected location: {db_path.absolute()}")
        return None
    
    print(f"✅ Database found: {db_path}")
    print(f"   Size: {db_path.stat().st_size / 1024:.2f} KB\n")
    return sqlite3.connect(db_path)

def run_query(conn, query, description):
    """Run a query and display results."""
    print(f"{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        if not rows:
            print("⚠️  No data returned\n")
            return
        
        # Calculate column widths
        col_widths = [max(len(str(col)), 15) for col in columns]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(val)))
        
        # Print header
        header = " | ".join(str(col).ljust(col_widths[i]) for i, col in enumerate(columns))
        print(header)
        print("-" * len(header))
        
        # Print rows
        for row in rows:
            print(" | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)))
        
        print(f"\nRows returned: {len(rows)}\n")
        
    except sqlite3.Error as e:
        print(f"❌ Error: {e}\n")

def main():
    """Run all test queries."""
    # Determine database path
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    else:
        db_path = Path("config/green_shift_data/research_data.db")
    
    # Connect to database
    conn = connect_db(db_path)
    if not conn:
        return
    
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Table existence and row counts
    run_query(conn, """
        SELECT 'research_daily_aggregates' as table_name, COUNT(*) as row_count FROM research_daily_aggregates
        UNION ALL
        SELECT 'research_rl_episodes', COUNT(*) FROM research_rl_episodes
        UNION ALL
        SELECT 'research_nudge_log', COUNT(*) FROM research_nudge_log
        UNION ALL
        SELECT 'research_blocked_notifications', COUNT(*) FROM research_blocked_notifications
        UNION ALL
        SELECT 'research_task_interactions', COUNT(*) FROM research_task_interactions
        UNION ALL
        SELECT 'research_phase_metadata', COUNT(*) FROM research_phase_metadata
        UNION ALL
        SELECT 'research_area_daily_stats', COUNT(*) FROM research_area_daily_stats
        UNION ALL
        SELECT 'research_weekly_challenges', COUNT(*) FROM research_weekly_challenges;
    """, "1. TABLE ROW COUNTS")
    
    # 2. Phase metadata
    run_query(conn, """
        SELECT 
            phase,
            datetime(start_timestamp, 'unixepoch') as start_date,
            datetime(end_timestamp, 'unixepoch') as end_date,
            ROUND(baseline_consumption_W, 2) as baseline_W,
            notes
        FROM research_phase_metadata
        ORDER BY start_timestamp;
    """, "2. PHASE METADATA")
    
    # 3. Current phase
    run_query(conn, """
        SELECT 
            phase,
            datetime(start_timestamp, 'unixepoch') as started,
            ROUND((julianday('now') - julianday(start_timestamp, 'unixepoch')), 1) as days_in_phase
        FROM research_phase_metadata
        WHERE end_timestamp IS NULL
        ORDER BY start_timestamp DESC
        LIMIT 1;
    """, "3. CURRENT PHASE")
    
    # 4. Recent daily aggregates
    run_query(conn, """
        SELECT 
            date,
            phase,
            ROUND(avg_power_w, 2) as power_w,
            ROUND(total_energy_kwh, 2) as energy_kwh,
            ROUND(avg_occupancy_count, 2) as occupancy,
            tasks_generated as tasks_gen,
            tasks_completed as tasks_done,
            nudges_sent,
            nudges_accepted,
            nudges_blocked
        FROM research_daily_aggregates
        ORDER BY date DESC
        LIMIT 5;
    """, "4. RECENT DAILY AGGREGATES (Last 5 Days)")
    
    # 5. Area daily stats summary
    run_query(conn, """
        SELECT 
            COUNT(DISTINCT area_name) as total_areas,
            COUNT(DISTINCT date) as days_tracked,
            COUNT(*) as total_records
        FROM research_area_daily_stats;
    """, "5. AREA DAILY STATS SUMMARY")
    
    # 6. Recent area daily stats
    run_query(conn, """
        SELECT 
            date,
            area_name,
            phase,
            ROUND(avg_power_w, 2) as avg_power_w,
            ROUND(avg_temperature, 1) as temp_c,
            ROUND(avg_humidity, 1) as humidity_pct,
            ROUND(total_occupied_hours, 1) as occupied_hrs,
            ROUND(occupancy_percentage, 1) as occupancy_pct
        FROM research_area_daily_stats
        ORDER BY date DESC, area_name
        LIMIT 10;
    """, "6. RECENT AREA DAILY STATS (Last 10 Records)")
    
    # 7. Area power consumption comparison (latest day)
    run_query(conn, """
        SELECT 
            area_name,
            ROUND(avg_power_w, 2) as avg_power_w,
            ROUND(max_power_w, 2) as max_power_w,
            ROUND(occupancy_percentage, 1) as occupancy_pct,
            ROUND(avg_power_w / NULLIF(occupancy_percentage, 0), 2) as power_per_occupancy_pct
        FROM research_area_daily_stats
        WHERE date = (SELECT MAX(date) FROM research_area_daily_stats)
        ORDER BY avg_power_w DESC;
    """, "7. AREA POWER COMPARISON (Latest Day)")
    
    # 8. RL episodes summary
    run_query(conn, """
        SELECT 
            COUNT(*) as total_episodes,
            COUNT(DISTINCT episode_number) as unique_episodes,
            MIN(datetime(timestamp, 'unixepoch')) as first_episode,
            MAX(datetime(timestamp, 'unixepoch')) as last_episode,
            ROUND(AVG(reward), 4) as avg_reward
        FROM research_rl_episodes;
    """, "8. RL EPISODES SUMMARY")
    
    # 9. Action distribution
    run_query(conn, """
        SELECT 
            action_name,
            COUNT(*) as times_used,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM research_rl_episodes), 2) as percentage,
            ROUND(AVG(reward), 4) as avg_reward
        FROM research_rl_episodes
        WHERE action_name IS NOT NULL
        GROUP BY action_name
        ORDER BY times_used DESC;
    """, "9. ACTION DISTRIBUTION")
    
    # 10. Exploration vs Exploitation
    run_query(conn, """
        SELECT 
            action_source,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM research_rl_episodes WHERE action_source IS NOT NULL), 2) as percentage,
            ROUND(AVG(reward), 4) as avg_reward
        FROM research_rl_episodes
        WHERE action_source IS NOT NULL
        GROUP BY action_source;
    """, "10. EXPLORATION VS EXPLOITATION")
    
    # 11. Weekly Challenge Summary
    run_query(conn, """
        SELECT 
            week_start_date,
            target_percentage,
            ROUND(baseline_W, 2) as baseline_W,
            ROUND(actual_W, 2) as actual_W,
            ROUND(savings_W, 2) as savings_W,
            ROUND(savings_percentage, 1) as savings_pct,
            CASE WHEN achieved = 1 THEN 'SUCCESS' ELSE 'FAILED' END as outcome
        FROM research_weekly_challenges
        ORDER BY week_start_date DESC
        LIMIT 5;
    """, "11. WEEKLY CHALLENGE SUMMARY (Last 5)")
    
    # 12. Time-of-Day Notification Patterns (Active Phase)
    run_query(conn, """
        SELECT 
            time_of_day_hour as hour,
            COUNT(*) as total_decisions,
            COUNT(CASE WHEN action_name IS NOT NULL THEN 1 END) as notifications_attempted,
            ROUND(AVG(reward), 4) as avg_reward,
            ROUND(AVG(opportunity_score), 3) as avg_opportunity
        FROM research_rl_episodes
        WHERE time_of_day_hour IS NOT NULL
          AND phase = 'active'
        GROUP BY time_of_day_hour
        ORDER BY hour;
    """, "12. TIME-OF-DAY NOTIFICATION PATTERNS - Active Phase")
    
    # 13. Action Constraint Analysis (ACTIVE PHASE ONLY - Real RL Decisions)
    run_query(conn, """
        SELECT 
            DATE(timestamp, 'unixepoch') as date,
            COUNT(*) as total_decisions,
            SUM(CASE WHEN action_mask LIKE '%"1": false%' THEN 1 ELSE 0 END) as specific_masked,
            SUM(CASE WHEN action_mask LIKE '%"2": false%' THEN 1 ELSE 0 END) as anomaly_masked,
            SUM(CASE WHEN action_mask LIKE '%"3": false%' THEN 1 ELSE 0 END) as behavioural_masked,
            SUM(CASE WHEN action_mask LIKE '%"4": false%' THEN 1 ELSE 0 END) as normative_masked,
            ROUND(AVG(
                CAST((action_mask LIKE '%"1": true%') AS INTEGER) +
                CAST((action_mask LIKE '%"2": true%') AS INTEGER) +
                CAST((action_mask LIKE '%"3": true%') AS INTEGER) +
                CAST((action_mask LIKE '%"4": true%') AS INTEGER)
            ), 1) as avg_available_actions
        FROM research_rl_episodes
        WHERE action_mask IS NOT NULL
          AND phase = 'active'
        GROUP BY date
        ORDER BY date DESC
        LIMIT 5;
    """, "13. ACTION CONSTRAINT ANALYSIS - Active Phase Only (Last 5 Days)")
    
    # 13b. Shadow Phase Learning Summary (Baseline Phase - No Real Notifications)
    run_query(conn, """
        SELECT 
            DATE(timestamp, 'unixepoch') as date,
            COUNT(*) as shadow_episodes,
            SUM(CASE WHEN action_source LIKE 'shadow%' THEN 1 ELSE 0 END) as shadow_decisions,
            ROUND(AVG(reward), 3) as avg_shadow_reward,
            ROUND(AVG(max_q_value), 3) as avg_max_q
        FROM research_rl_episodes
        WHERE phase = 'baseline'
        GROUP BY date
        ORDER BY date DESC
        LIMIT 5;
    """, "13b. SHADOW LEARNING SUMMARY - Baseline Phase (Last 5 Days)")
    
    # 14. Baseline Comparison Effectiveness
    run_query(conn, """
        SELECT 
            action_name,
            COUNT(*) as times_used,
            ROUND(AVG(current_power), 2) as avg_power,
            ROUND(AVG(baseline_power_reference), 2) as avg_baseline,
            ROUND(AVG(current_power - baseline_power_reference), 2) as avg_diff
        FROM research_rl_episodes
        WHERE baseline_power_reference IS NOT NULL
          AND action_name IS NOT NULL
        GROUP BY action_name
        ORDER BY times_used DESC;
    """, "14. BASELINE COMPARISON EFFECTIVENESS")
    
    # 15. Nudge summary
    run_query(conn, """
        SELECT 
            COUNT(*) as total_nudges,
            SUM(responded) as responded,
            SUM(accepted) as accepted,
            ROUND(SUM(accepted) * 100.0 / NULLIF(SUM(responded), 0), 2) as acceptance_rate_pct,
            ROUND(AVG(response_time_seconds), 2) as avg_response_sec
        FROM research_nudge_log;
    """, "15. NUDGE SUMMARY")
    
    # 16. Nudge type distribution
    run_query(conn, """
        SELECT 
            action_type,
            COUNT(*) as sent,
            SUM(responded) as responded,
            SUM(accepted) as accepted,
            ROUND(SUM(accepted) * 100.0 / NULLIF(SUM(responded), 0), 2) as accept_rate_pct
        FROM research_nudge_log
        GROUP BY action_type
        ORDER BY sent DESC;
    """, "16. NUDGE TYPE DISTRIBUTION")
    
    # 16b. Blocked Notifications Summary (Active Phase Only)
    run_query(conn, """
        SELECT 
            COUNT(*) as total_blocked,
            SUM(CASE WHEN block_reason = 'max_daily_limit' THEN 1 ELSE 0 END) as blocked_max_daily,
            SUM(CASE WHEN block_reason = 'cooldown' THEN 1 ELSE 0 END) as blocked_cooldown,
            SUM(CASE WHEN block_reason = 'fatigue_threshold' THEN 1 ELSE 0 END) as blocked_fatigue,
            SUM(CASE WHEN block_reason = 'no_available_actions' THEN 1 ELSE 0 END) as blocked_no_actions,
            ROUND(AVG(opportunity_score), 3) as avg_opportunity,
            ROUND(AVG(fatigue_index), 3) as avg_fatigue_at_block
        FROM research_blocked_notifications
        WHERE phase = 'active';
    """, "16b. BLOCKED NOTIFICATIONS SUMMARY - Active Phase Only")
    
    # 16c. Blocked Notifications by Reason (Daily Breakdown)
    run_query(conn, """
        SELECT 
            DATE(timestamp, 'unixepoch') as date,
            block_reason,
            COUNT(*) as count,
            ROUND(AVG(opportunity_score), 3) as avg_opportunity,
            ROUND(AVG(fatigue_index), 3) as avg_fatigue
        FROM research_blocked_notifications
        WHERE phase = 'active'
        GROUP BY date, block_reason
        ORDER BY date DESC, count DESC
        LIMIT 15;
    """, "16c. BLOCKED NOTIFICATIONS BY REASON (Recent Days)")
    
    # 16d. Notification Success vs Block Rate
    run_query(conn, """
        SELECT 
            DATE(n.timestamp, 'unixepoch') as date,
            COUNT(n.notification_id) as sent,
            (SELECT COUNT(*) FROM research_blocked_notifications 
             WHERE DATE(timestamp, 'unixepoch') = DATE(n.timestamp, 'unixepoch') 
             AND phase = 'active') as blocked,
            ROUND(COUNT(n.notification_id) * 100.0 / 
                  NULLIF(COUNT(n.notification_id) + 
                         (SELECT COUNT(*) FROM research_blocked_notifications 
                          WHERE DATE(timestamp, 'unixepoch') = DATE(n.timestamp, 'unixepoch') 
                          AND phase = 'active'), 0), 1) as send_rate_pct
        FROM research_nudge_log n
        GROUP BY date
        ORDER BY date DESC
        LIMIT 5;
    """, "16d. NOTIFICATION SUCCESS VS BLOCK RATE (Last 5 Days)")
    
    # 17. Task summary
    run_query(conn, """
        SELECT 
            COUNT(*) as total_tasks,
            SUM(completed) as completed,
            SUM(verified) as verified,
            ROUND(SUM(completed) * 100.0 / COUNT(*), 2) as completion_rate_pct
        FROM research_task_interactions;
    """, "17. TASK SUMMARY")
    
    # 18. Task type distribution
    run_query(conn, """
        SELECT 
            task_type,
            COUNT(*) as total,
            SUM(completed) as completed,
            ROUND(SUM(completed) * 100.0 / COUNT(*), 2) as completion_pct
        FROM research_task_interactions
        GROUP BY task_type
        ORDER BY total DESC;
    """, "18. TASK TYPE DISTRIBUTION")
    
    # 19. Energy Consumption by Phase (Intervention Impact)
    run_query(conn, """
        SELECT 
            phase,
            COUNT(*) as days,
            ROUND(AVG(avg_power_w), 2) as avg_power_w,
            ROUND(AVG(peak_power_w), 2) as avg_peak_w,
            ROUND(AVG(total_energy_kwh), 2) as avg_daily_kwh,
            ROUND(AVG(total_energy_kwh / NULLIF(total_occupied_hours, 0)), 3) as kwh_per_occupied_hour
        FROM research_daily_aggregates
        WHERE total_occupied_hours > 0
        GROUP BY phase;
    """, "19. ENERGY CONSUMPTION BY PHASE (Intervention Impact)")
    
    # 20. Power Consumption Trend Over Time
    run_query(conn, """
        SELECT 
            date,
            phase,
            ROUND(avg_power_w, 2) as avg_power_w,
            ROUND(total_energy_kwh, 2) as energy_kwh,
            ROUND(avg_occupancy_count, 1) as occupancy
        FROM research_daily_aggregates
        ORDER BY date DESC
        LIMIT 14;
    """, "20. POWER CONSUMPTION TREND (Last 14 Days)")
    
    # 21. User Engagement Trends Over Time
    run_query(conn, """
        SELECT 
            date,
            phase,
            tasks_generated,
            tasks_completed,
            ROUND(tasks_completed * 100.0 / NULLIF(tasks_generated, 0), 1) as completion_rate_pct,
            nudges_sent,
            nudges_accepted,
            ROUND(nudges_accepted * 100.0 / NULLIF(nudges_sent, 0), 1) as acceptance_rate_pct
        FROM research_daily_aggregates
        WHERE tasks_generated > 0 OR nudges_sent > 0
        ORDER BY date DESC
        LIMIT 14;
    """, "21. USER ENGAGEMENT TRENDS (Last 14 Days)")
    
    # 22. AI Indices Progression Over Time
    run_query(conn, """
        SELECT 
            date,
            phase,
            ROUND(avg_anomaly_index, 3) as anomaly,
            ROUND(avg_behaviour_index, 3) as behaviour,
            ROUND(avg_fatigue_index, 3) as fatigue
        FROM research_daily_aggregates
        WHERE avg_anomaly_index IS NOT NULL
        ORDER BY date DESC
        LIMIT 14;
    """, "22. AI INDICES PROGRESSION (Last 14 Days)")
    
    # 23. Task Difficulty Feedback Analysis
    run_query(conn, """
        SELECT 
            task_type,
            difficulty_level,
            COUNT(*) as total,
            SUM(completed) as completed,
            ROUND(AVG(CASE WHEN user_feedback = 'too_easy' THEN 1 ELSE 0 END) * 100, 1) as too_easy_pct,
            ROUND(AVG(CASE WHEN user_feedback = 'just_right' THEN 1 ELSE 0 END) * 100, 1) as just_right_pct,
            ROUND(AVG(CASE WHEN user_feedback = 'too_hard' THEN 1 ELSE 0 END) * 100, 1) as too_hard_pct
        FROM research_task_interactions
        WHERE user_feedback IS NOT NULL
        GROUP BY task_type, difficulty_level
        ORDER BY task_type, difficulty_level;
    """, "23. TASK DIFFICULTY FEEDBACK ANALYSIS")
    
    # 24. Notification Response Time Analysis
    run_query(conn, """
        SELECT 
            action_type,
            COUNT(*) as total_sent,
            SUM(responded) as responded,
            SUM(accepted) as accepted,
            ROUND(AVG(response_time_seconds), 1) as avg_response_sec,
            ROUND(MIN(response_time_seconds), 1) as min_response_sec,
            ROUND(MAX(response_time_seconds), 1) as max_response_sec
        FROM research_nudge_log
        WHERE responded = 1
        GROUP BY action_type;
    """, "24. NOTIFICATION RESPONSE TIME ANALYSIS")
    
    # 25. Phase Transition Impact (Before/After Comparison)
    run_query(conn, """
        SELECT 
            phase,
            COUNT(*) as days,
            ROUND(AVG(avg_power_w), 2) as avg_power,
            ROUND(AVG(tasks_completed), 1) as avg_tasks_completed,
            ROUND(AVG(nudges_accepted * 100.0 / NULLIF(nudges_sent, 0)), 1) as avg_acceptance_rate,
            ROUND(AVG(avg_behaviour_index), 3) as avg_behaviour,
            ROUND(AVG(avg_fatigue_index), 3) as avg_fatigue
        FROM research_daily_aggregates
        GROUP BY phase;
    """, "25. PHASE TRANSITION IMPACT (Baseline vs Active Comparison)")
    
    # 26. Q-Learning Convergence Analysis
    run_query(conn, """
        SELECT 
            DATE(timestamp, 'unixepoch') as date,
            phase,
            COUNT(*) as episodes,
            ROUND(AVG(max_q_value), 4) as avg_max_q,
            ROUND(AVG(reward), 4) as avg_reward,
            ROUND(AVG(epsilon), 3) as avg_epsilon
        FROM research_rl_episodes
        GROUP BY date, phase
        ORDER BY date DESC
        LIMIT 14;
    """, "26. Q-LEARNING CONVERGENCE (Last 14 Days)")
    
    # 27. Correlation: Indices vs User Behavior
    run_query(conn, """
        SELECT 
            CASE 
                WHEN avg_fatigue_index < 0.3 THEN 'Low Fatigue'
                WHEN avg_fatigue_index < 0.7 THEN 'Medium Fatigue'
                ELSE 'High Fatigue'
            END as fatigue_level,
            COUNT(*) as days,
            ROUND(AVG(nudges_accepted * 100.0 / NULLIF(nudges_sent, 0)), 1) as avg_acceptance_rate,
            ROUND(AVG(tasks_completed * 100.0 / NULLIF(tasks_generated, 0)), 1) as avg_task_completion
        FROM research_daily_aggregates
        WHERE nudges_sent > 0 AND avg_fatigue_index IS NOT NULL
        GROUP BY fatigue_level;
    """, "27. FATIGUE INDEX vs USER ENGAGEMENT")
    
    # 28. Area-Based Intervention Impact
    run_query(conn, """
        SELECT 
            area_name,
            phase,
            COUNT(*) as days,
            ROUND(AVG(avg_power_w), 2) as avg_power_w,
            ROUND(AVG(occupancy_percentage), 1) as avg_occupancy_pct
        FROM research_area_daily_stats
        GROUP BY area_name, phase
        ORDER BY area_name, phase;
    """, "28. AREA-BASED INTERVENTION IMPACT")
    
    # 29. Last Data Collection Time
    run_query(conn, """
        SELECT 
            'daily_aggregates' as table_name,
            MAX(date) as last_date
        FROM research_daily_aggregates
        UNION ALL
        SELECT 
            'area_daily_stats',
            MAX(date)
        FROM research_area_daily_stats
        UNION ALL
        SELECT 
            'rl_episodes',
            DATE(MAX(timestamp), 'unixepoch')
        FROM research_rl_episodes
        UNION ALL
        SELECT 
            'nudge_log',
            DATE(MAX(timestamp), 'unixepoch')
        FROM research_nudge_log
        UNION ALL
        SELECT 
            'task_interactions',
            MAX(date)
        FROM research_task_interactions;
    """, "29. LAST DATA COLLECTION TIME")
    
    # 30. Gamification Effectiveness: Task Completion Times
    run_query(conn, """
        SELECT 
            task_type,
            COUNT(*) as total_completed,
            ROUND(AVG(time_to_complete_seconds / 3600.0), 2) as avg_hours_to_complete,
            ROUND(MIN(time_to_complete_seconds / 3600.0), 2) as min_hours,
            ROUND(MAX(time_to_complete_seconds / 3600.0), 2) as max_hours
        FROM research_task_interactions
        WHERE completed = 1 AND time_to_complete_seconds IS NOT NULL
        GROUP BY task_type
        ORDER BY total_completed DESC;
    """, "30. TASK COMPLETION TIME ANALYSIS")
    
    # Close connection
    conn.close()
    
    print("="*80)
    print("✅ Test completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
