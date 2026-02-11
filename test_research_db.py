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
            ROUND(baseline_consumption_kwh, 2) as baseline_kwh,
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
            nudges_accepted
        FROM research_daily_aggregates
        ORDER BY date DESC
        LIMIT 5;
    """, "4. RECENT DAILY AGGREGATES (Last 5 Days)")
    
    # 5. RL episodes summary
    run_query(conn, """
        SELECT 
            COUNT(*) as total_episodes,
            COUNT(DISTINCT episode_number) as unique_episodes,
            MIN(datetime(timestamp, 'unixepoch')) as first_episode,
            MAX(datetime(timestamp, 'unixepoch')) as last_episode,
            ROUND(AVG(reward), 4) as avg_reward
        FROM research_rl_episodes;
    """, "5. RL EPISODES SUMMARY")
    
    # 6. Action distribution
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
    """, "6. ACTION DISTRIBUTION")
    
    # 7. Exploration vs Exploitation
    run_query(conn, """
        SELECT 
            action_source,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM research_rl_episodes WHERE action_source IS NOT NULL), 2) as percentage,
            ROUND(AVG(reward), 4) as avg_reward
        FROM research_rl_episodes
        WHERE action_source IS NOT NULL
        GROUP BY action_source;
    """, "7. EXPLORATION VS EXPLOITATION")
    
    # 8. Weekly Challenge Summary
    run_query(conn, """
        SELECT 
            week_start_date,
            target_percentage,
            ROUND(baseline_kwh, 2) as baseline_kwh,
            ROUND(actual_kwh, 2) as actual_kwh,
            ROUND(savings_kwh, 2) as savings_kwh,
            ROUND(savings_percentage, 1) as savings_pct,
            CASE WHEN achieved = 1 THEN 'SUCCESS' ELSE 'FAILED' END as outcome
        FROM research_weekly_challenges
        ORDER BY week_start_date DESC
        LIMIT 5;
    """, "8. WEEKLY CHALLENGE SUMMARY (Last 5)")
    
    # 9. Time-of-Day Decision Patterns
    run_query(conn, """
        SELECT 
            time_of_day_hour as hour,
            COUNT(*) as decisions,
            SUM(CASE WHEN action_name != 'noop' THEN 1 ELSE 0 END) as notifications,
            ROUND(AVG(CASE WHEN action_name != 'noop' THEN reward END), 4) as avg_reward
        FROM research_rl_episodes
        WHERE time_of_day_hour IS NOT NULL
        GROUP BY time_of_day_hour
        ORDER BY decisions DESC
        LIMIT 10;
    """, "9. TIME-OF-DAY DECISION PATTERNS (Top 10 Hours)")
    
    # 10. Action Constraint Analysis
    run_query(conn, """
        SELECT 
            DATE(timestamp, 'unixepoch') as date,
            COUNT(*) as total_decisions,
            SUM(CASE WHEN action_mask LIKE '%\"noop\": false%' THEN 1 ELSE 0 END) as noop_blocked,
            SUM(CASE WHEN action_mask LIKE '%\"specific\": false%' THEN 1 ELSE 0 END) as specific_blocked
        FROM research_rl_episodes
        WHERE action_mask IS NOT NULL
        GROUP BY date
        ORDER BY date DESC
        LIMIT 5;
    """, "10. ACTION CONSTRAINT ANALYSIS (Last 5 Days)")
    
    # 11. Baseline Comparison Effectiveness
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
    """, "11. BASELINE COMPARISON EFFECTIVENESS")
    
    # 12. Nudge summary
    run_query(conn, """
        SELECT 
            COUNT(*) as total_nudges,
            SUM(responded) as responded,
            SUM(accepted) as accepted,
            ROUND(SUM(accepted) * 100.0 / NULLIF(SUM(responded), 0), 2) as acceptance_rate_pct,
            ROUND(AVG(response_time_seconds), 2) as avg_response_sec
        FROM research_nudge_log;
    """, "12. NUDGE SUMMARY")
    
    # 13. Nudge type distribution
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
    """, "13. NUDGE TYPE DISTRIBUTION")
    
    # 14. Task summary
    run_query(conn, """
        SELECT 
            COUNT(*) as total_tasks,
            SUM(completed) as completed,
            SUM(verified) as verified,
            ROUND(SUM(completed) * 100.0 / COUNT(*), 2) as completion_rate_pct
        FROM research_task_interactions;
    """, "14. TASK SUMMARY")
    
    # 15. Task type distribution
    run_query(conn, """
        SELECT 
            task_type,
            COUNT(*) as total,
            SUM(completed) as completed,
            ROUND(SUM(completed) * 100.0 / COUNT(*), 2) as completion_pct
        FROM research_task_interactions
        GROUP BY task_type
        ORDER BY total DESC;
    """, "15. TASK TYPE DISTRIBUTION")
    
    # 16. Energy efficiency by phase
    run_query(conn, """
        SELECT 
            phase,
            COUNT(*) as days,
            ROUND(AVG(avg_power_w), 2) as avg_power_w,
            ROUND(AVG(total_energy_kwh / NULLIF(avg_occupancy_count, 0)), 2) as energy_per_person
        FROM research_daily_aggregates
        WHERE avg_occupancy_count > 0
        GROUP BY phase;
    """, "16. ENERGY EFFICIENCY BY PHASE")
    
    # 17. Last data collection time
    run_query(conn, """
        SELECT 
            'daily_aggregates' as table_name,
            MAX(date) as last_date
        FROM research_daily_aggregates
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
    """, "17. LAST DATA COLLECTION TIME")
    
    # Close connection
    conn.close()
    
    print("="*80)
    print("✅ Test completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
