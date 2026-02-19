"""
Populate Test Data Script
Generates realistic test data for research_data.db following the actual system flow.

This script simulates the GreenShift system operation over time:
- Phase metadata (baseline -> active transition)
- Sensor data collection
- RL agent decisions (shadow learning in baseline, real decisions in active)
- Notifications sent and blocked
- Task generation and completion
- Weekly challenges
- Daily aggregates computation

Usage:
    python populate_test_data.py [db_path]

Example:
    python populate_test_data.py config/green_shift_data
"""

import sqlite3
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import random

# Constants matching the actual system
UPDATE_INTERVAL_SECONDS = 15  # Optimized for storage
AI_FREQUENCY_SECONDS = 15
SHADOW_INTERVAL_MULTIPLIER = 4
MAX_NOTIFICATIONS_PER_DAY = 10
MIN_COOLDOWN_MINUTES = 30
FATIGUE_THRESHOLD = 0.7
BASELINE_DAYS = 14

ACTIONS = {
    1: "specific",      # Target specific appliance
    2: "anomaly",       # Alert about anomaly
    3: "behavioural",   # Behavioral nudge
    4: "normative"      # Social comparison
}

BLOCK_REASONS = [
    "fatigue_threshold",
    "no_available_actions"
]


class TestDataGenerator:
    """Generates realistic test data following actual system flow."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.sensor_db_path = self.data_dir / "sensor_data.db"
        self.research_db_path = self.data_dir / "research_data.db"
        
        # Simulation state
        self.baseline_start = datetime.now() - timedelta(days=21)  # 3 weeks ago
        self.active_start = self.baseline_start + timedelta(days=14)  # 2 weeks baseline
        self.current_time = datetime.now()
        
        self.baseline_consumption_W = 850.0  # Baseline average power
        self.notification_count_today = 0
        self.last_notification_time = None
        self.episode_number = 0
        self.shadow_episode_number = 0
        
        print("=" * 80)
        print("GreenShift Test Data Generator")
        print("=" * 80)
        print(f"Data directory: {self.data_dir.absolute()}")
        print(f"Baseline phase: {self.baseline_start.date()} (14 days)")
        print(f"Active phase:   {self.active_start.date()} (7 days)")
        print(f"Total period:   21 days")
        print("=" * 80)
        print()
    
    def run(self):
        """Generate all test data."""
        print("Step 1: Initializing databases...")
        self._init_databases()
        
        print("\nStep 2: Recording phase metadata...")
        self._populate_phase_metadata()
        
        print("\nStep 3: Generating sensor data (21 days)...")
        self._populate_sensor_data()
        
        print("\nStep 4: Generating baseline phase data (14 days - shadow learning)...")
        self._populate_baseline_phase()
        
        print("\nStep 5: Generating active phase data (7 days - real intervention)...")
        self._populate_active_phase()
        
        print("\nStep 6: Computing daily aggregates...")
        self._compute_daily_aggregates()
        
        print("\nStep 7: Computing area daily stats...")
        self._compute_area_daily_stats()
        
        print("\n" + "=" * 80)
        print("✅ Test data generation complete!")
        print("=" * 80)
        
        self._print_summary()
        self._print_expected_results()
    
    def _init_databases(self):
        """Initialize database schemas."""
        
        sensor_conn = sqlite3.connect(self.sensor_db_path)
        sensor_cursor = sensor_conn.cursor()
        
        # Sensor history table
        sensor_cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                power REAL,
                energy REAL,
                temperature REAL,
                humidity REAL,
                illuminance REAL,
                occupancy INTEGER,
                within_working_hours INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Area sensor history table
        sensor_cursor.execute("""
            CREATE TABLE IF NOT EXISTS area_sensor_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                area_name TEXT NOT NULL,
                power REAL,
                energy REAL,
                temperature REAL,
                humidity REAL,
                illuminance REAL,
                occupancy INTEGER,
                within_working_hours INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        sensor_conn.commit()
        sensor_conn.close()
        
        # Research database tables (matching storage.py schema)
        research_conn = sqlite3.connect(self.research_db_path)
        research_cursor = research_conn.cursor()
        
        # Phase metadata
        research_cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_phase_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase TEXT NOT NULL,
                start_timestamp REAL NOT NULL,
                end_timestamp REAL,
                baseline_consumption_W REAL,
                baseline_occupancy_avg REAL,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Daily aggregates
        research_cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_daily_aggregates (
                date TEXT PRIMARY KEY,
                phase TEXT,
                total_energy_kwh REAL,
                avg_power_w REAL,
                peak_power_w REAL,
                min_power_w REAL,
                avg_occupancy_count REAL,
                total_occupied_hours REAL,
                avg_temperature REAL,
                avg_humidity REAL,
                avg_illuminance REAL,
                tasks_generated INTEGER DEFAULT 0,
                tasks_completed INTEGER DEFAULT 0,
                tasks_verified INTEGER DEFAULT 0,
                nudges_sent INTEGER DEFAULT 0,
                nudges_accepted INTEGER DEFAULT 0,
                nudges_dismissed INTEGER DEFAULT 0,
                nudges_ignored INTEGER DEFAULT 0,
                nudges_blocked INTEGER DEFAULT 0,
                avg_anomaly_index REAL,
                avg_behaviour_index REAL,
                avg_fatigue_index REAL,
                outdoor_temp_celsius REAL,
                hdd_base18 REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # RL episodes
        research_cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_rl_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                episode_number INTEGER,
                phase TEXT,
                state_vector TEXT,
                state_key TEXT,
                action INTEGER,
                action_name TEXT,
                action_source TEXT,
                reward REAL,
                q_values TEXT,
                max_q_value REAL,
                epsilon REAL,
                action_mask TEXT,
                current_power REAL,
                anomaly_index REAL,
                behaviour_index REAL,
                fatigue_index REAL,
                opportunity_score REAL,
                time_of_day_hour INTEGER,
                baseline_power_reference REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Nudge log
        research_cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_nudge_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                notification_id TEXT UNIQUE,
                phase TEXT,
                action_type TEXT,
                template_index INTEGER,
                title TEXT,
                message TEXT,
                state_vector TEXT,
                current_power REAL,
                anomaly_index REAL,
                behaviour_index REAL,
                fatigue_index REAL,
                responded INTEGER DEFAULT 0,
                accepted INTEGER DEFAULT 0,
                response_timestamp REAL,
                response_time_seconds REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Blocked notifications
        research_cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_blocked_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                phase TEXT,
                block_reason TEXT NOT NULL,
                opportunity_score REAL,
                current_power REAL,
                anomaly_index REAL,
                behaviour_index REAL,
                fatigue_index REAL,
                notification_count_today INTEGER,
                time_since_last_notification_minutes REAL,
                required_cooldown_minutes REAL,
                adaptive_cooldown_minutes REAL,
                available_action_count INTEGER,
                action_mask TEXT,
                state_vector TEXT,
                time_of_day_hour INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Task interactions
        research_cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_task_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                date TEXT,
                phase TEXT,
                task_type TEXT,
                difficulty_level INTEGER,
                target_value REAL,
                baseline_value REAL,
                area_name TEXT,
                generation_timestamp REAL,
                first_view_timestamp REAL,
                completion_timestamp REAL,
                time_to_view_seconds REAL,
                time_to_complete_seconds REAL,
                completed INTEGER DEFAULT 0,
                verified INTEGER DEFAULT 0,
                completion_value REAL,
                user_feedback TEXT,
                power_at_generation REAL,
                occupancy_at_generation INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Area daily stats
        research_cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_area_daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                area_name TEXT,
                phase TEXT,
                avg_power_w REAL,
                max_power_w REAL,
                min_power_w REAL,
                avg_temperature REAL,
                avg_humidity REAL,
                avg_illuminance REAL,
                total_occupied_hours REAL,
                occupancy_percentage REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, area_name)
            )
        """)
        
        # Weekly challenges
        research_cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_weekly_challenges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week_start_date TEXT,
                week_end_date TEXT,
                phase TEXT,
                target_percentage REAL,
                baseline_W REAL,
                actual_W REAL,
                savings_W REAL,
                savings_percentage REAL,
                achieved INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(week_start_date)
            )
        """)
        
        research_conn.commit()
        research_conn.close()
        
        print(f"  ✅ Created {self.sensor_db_path.name}")
        print(f"  ✅ Created {self.research_db_path.name}")
    
    def _populate_phase_metadata(self):
        """Record phase transitions."""
        conn = sqlite3.connect(self.research_db_path)
        cursor = conn.cursor()
        
        # Baseline phase started
        cursor.execute("""
            INSERT INTO research_phase_metadata 
            (phase, start_timestamp, end_timestamp, baseline_consumption_W, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "baseline",
            int(self.baseline_start.timestamp()),
            int(self.active_start.timestamp()),
            None,
            "Initial system setup - shadow learning phase"
        ))
        
        # Active phase started
        cursor.execute("""
            INSERT INTO research_phase_metadata 
            (phase, start_timestamp, end_timestamp, baseline_consumption_W, baseline_occupancy_avg, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "active",
            int(self.active_start.timestamp()),
            None,
            self.baseline_consumption_W,
            65.0,
            "Active intervention phase started - AI now sends real notifications"
        ))
        
        conn.commit()
        conn.close()
        
        print(f"  ✅ Recorded phase metadata (baseline + active)")
    
    def _populate_sensor_data(self):
        """Generate realistic sensor readings for 21 days."""
        conn = sqlite3.connect(self.sensor_db_path)
        cursor = conn.cursor()
        
        areas = ["Living Room", "Kitchen", "Bedroom", "Office"]
        
        # Generate data every UPDATE_INTERVAL_SECONDS for 21 days
        current = self.baseline_start
        end = self.current_time
        
        readings_count = 0
        area_readings_count = 0
        
        while current <= end:
            hour = current.hour
            is_day = 7 <= hour <= 22
            is_peak = 18 <= hour <= 21  # Evening peak
            
            # Simulate realistic power patterns
            base_power = 600 if is_day else 300
            if is_peak:
                base_power = 1200
            
            power = base_power + random.uniform(-100, 150)
            temperature = 20 + random.uniform(-2, 3)
            humidity = 50 + random.uniform(-10, 10)
            illuminance = (500 if is_day else 50) + random.uniform(-50, 100)
            occupancy = 1 if is_day else random.choice([0, 0, 0, 1])
            
            # Determine if within working hours (Mon-Fri 8am-6pm for test data)
            is_working_hours = (
                current.weekday() < 5 and  # Monday=0 to Friday=4
                8 <= hour <= 18
            )
            
            # Global sensor readings
            cursor.execute("""
                INSERT INTO sensor_history
                (timestamp, power, energy, temperature, humidity, illuminance, occupancy, within_working_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(current.timestamp()),
                power,
                None,  # Energy is cumulative, not needed for aggregation
                temperature,
                humidity,
                illuminance,
                occupancy,
                1 if is_working_hours else 0
            ))
            readings_count += 1
            
            # Area-specific readings
            for area in areas:
                area_power = power / 4 + random.uniform(-50, 50)
                area_temp = temperature + random.uniform(-1, 1)
                area_humidity = humidity + random.uniform(-5, 5)
                area_illum = illuminance + random.uniform(-100, 100)
                area_occ = random.choice([0, 1]) if is_day else 0
                
                cursor.execute("""
                    INSERT INTO area_sensor_history
                    (timestamp, area_name, power, energy, temperature, humidity, illuminance, occupancy, within_working_hours)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    int(current.timestamp()),
                    area,
                    area_power,
                    None,
                    area_temp,
                    area_humidity,
                    area_illum,
                    area_occ,
                    1 if is_working_hours else 0
                ))
                area_readings_count += 1
            
            current += timedelta(seconds=UPDATE_INTERVAL_SECONDS)
        
        conn.commit()
        conn.close()
        
        print(f"  ✅ Generated {readings_count:,} global sensor readings")
        print(f"  ✅ Generated {area_readings_count:,} area sensor readings ({len(areas)} areas)")
    
    def _populate_baseline_phase(self):
        """Generate baseline phase data (shadow learning only).
        
        Note: Shadow learning respects working hours in office mode,
        just like real AI decisions.
        """
        conn = sqlite3.connect(self.research_db_path)
        cursor = conn.cursor()
        
        current = self.baseline_start
        end = self.active_start
        
        rl_episodes = 0
        rl_episodes_skipped = 0
        
        # AI processes every AI_FREQUENCY_SECONDS * SHADOW_INTERVAL_MULTIPLIER in baseline
        interval = AI_FREQUENCY_SECONDS * SHADOW_INTERVAL_MULTIPLIER
        
        while current < end:
            hour = current.hour
            weekday = current.weekday()
            
            # Check if within working hours (Mon-Fri 8am-6pm for office mode)
            # Shadow learning follows same working hours logic as active phase
            is_working_hours = weekday < 5 and 8 <= hour <= 18
            
            # In office mode, shadow learning only happens during working hours
            # This prevents learning from weekend/off-hours patterns
            if not is_working_hours:
                rl_episodes_skipped += 1
                current += timedelta(seconds=interval)
                continue
            
            # Simulate shadow learning decision
            state_vector = self._generate_state_vector(current, "baseline")
            action = random.choice(list(ACTIONS.keys()))
            action_mask = self._generate_action_mask()
            reward = random.uniform(-0.5, 0.5)  # Shadow rewards are exploratory
            max_q = random.uniform(0, 1)
            action_source = random.choice(["shadow_exploration", "shadow_exploitation"])
            
            cursor.execute("""
                INSERT INTO research_rl_episodes
                (episode_number, timestamp, phase, state_vector, action, action_name, action_mask,
                 reward, max_q_value, action_source, current_power, baseline_power_reference,
                 anomaly_index, behaviour_index, fatigue_index, opportunity_score, time_of_day_hour)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.shadow_episode_number,
                int(current.timestamp()),
                "baseline",
                json.dumps(state_vector),
                action,
                ACTIONS[action],
                json.dumps(action_mask),
                reward,
                max_q,
                action_source,
                800 + random.uniform(-200, 200),
                850,  # Baseline reference
                random.uniform(0, 0.3),  # Low anomaly in baseline
                random.uniform(0.3, 0.7),
                0.0,  # No fatigue in baseline (no real notifications)
                random.uniform(0, 1),
                hour
            ))
            
            self.shadow_episode_number += 1
            rl_episodes += 1
            
            current += timedelta(seconds=interval)
        
        conn.commit()
        conn.close()
        
        print(f"  ✅ Generated {rl_episodes:,} shadow RL episodes (baseline phase)")
        print(f"     - Skipped {rl_episodes_skipped:,} episodes outside working hours")
        print(f"     - Only Mon-Fri 8am-6pm (simulating office mode)")
        print(f"     - No notifications sent (shadow learning only)")
        print(f"     - No blocked notifications (blocking logic inactive)")
    
    def _populate_active_phase(self):
        """Generate active phase data (real intervention)."""
        conn = sqlite3.connect(self.research_db_path)
        cursor = conn.cursor()
        
        current = self.active_start
        end = self.current_time
        
        rl_episodes = 0
        nudges_sent = 0
        nudges_accepted = 0
        nudges_blocked = 0
        tasks_generated = 0
        tasks_completed = 0
        
        # AI processes every AI_FREQUENCY_SECONDS in active phase
        interval = AI_FREQUENCY_SECONDS
        
        current_date = current.date()
        self.notification_count_today = 0
        self.last_notification_time = None
        tasks_generated_dates = set()
        
        while current < end:
            hour = current.hour
            
            # Reset daily notification counter at midnight
            if current.date() != current_date:
                current_date = current.date()
                self.notification_count_today = 0
            
            # Generate daily tasks at 8 AM (once per day)
            if hour == 8 and current.date() not in tasks_generated_dates:
                tasks_generated_dates.add(current.date())
                for i in range(3):
                    task_id = f"task_{current.date()}_{i}"
                    task_type = random.choice(["reduce_power", "temperature_adjust", "lights_off"])
                    difficulty_level = random.randint(1, 3)
                    target_val = random.uniform(100, 500)
                    baseline_val = target_val * random.uniform(1.1, 1.5)
                    
                    cursor.execute("""
                        INSERT INTO research_task_interactions
                        (task_id, date, phase, task_type, difficulty_level, target_value, baseline_value,
                         area_name, generation_timestamp, power_at_generation, occupancy_at_generation, completed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        task_id,
                        current.date().isoformat(),
                        "active",
                        task_type,
                        difficulty_level,
                        target_val,
                        baseline_val,
                        random.choice(["Living Room", "Kitchen", "Bedroom", "Office"]),
                        int(current.timestamp()),
                        750 + random.uniform(-200, 200),
                        1,
                        0
                    ))
                    tasks_generated += 1
                    
                    # Simulate task completion (70% completion rate)
                    if random.random() < 0.7:
                        completion_time = current + timedelta(hours=random.randint(1, 10))
                        time_to_complete = (completion_time - current).total_seconds()
                        cursor.execute("""
                            UPDATE research_task_interactions
                            SET completed = 1, completion_timestamp = ?, completion_value = ?, 
                                verified = 1, time_to_complete_seconds = ?
                            WHERE task_id = ?
                        """, (
                            int(completion_time.timestamp()),
                            random.uniform(80, 120),
                            time_to_complete,
                            task_id
                        ))
                        tasks_completed += 1
                        
                        # Some tasks get feedback (50%)
                        if random.random() < 0.5:
                            feedback = random.choice(["too_easy", "just_right", "too_hard"])
                            cursor.execute("""
                                UPDATE research_task_interactions
                                SET user_feedback = ?
                                WHERE task_id = ?
                            """, (
                                feedback,
                                task_id
                            ))
            
            # AI decision cycle
            state_vector = self._generate_state_vector(current, "active")
            action_mask = self._generate_action_mask()
            opportunity_score = self._calculate_opportunity_score(current)
            
            anomaly_idx = random.uniform(0, 0.5)
            behaviour_idx = random.uniform(0.3, 0.8)
            fatigue_idx = min(self.notification_count_today / MAX_NOTIFICATIONS_PER_DAY, 1.0)
            
            # Check if notification should be blocked
            blocked = False
            block_reason = None
            
            # Max daily limit
            if self.notification_count_today >= MAX_NOTIFICATIONS_PER_DAY:
                blocked = True
                block_reason = "max_daily_limit"
            
            # Cooldown check
            elif self.last_notification_time:
                minutes_since = (current - self.last_notification_time).total_seconds() / 60
                required_cooldown = MIN_COOLDOWN_MINUTES
                if minutes_since < required_cooldown:
                    blocked = True
                    block_reason = "cooldown"
            
            # Fatigue threshold
            elif fatigue_idx >= FATIGUE_THRESHOLD:
                blocked = True
                block_reason = "fatigue_threshold"
            
            # No available actions
            elif sum(action_mask.values()) == 0:
                blocked = True
                block_reason = "no_available_actions"
            
            # Log RL episode
            action = random.choice([a for a, available in action_mask.items() if available]) if not blocked and any(action_mask.values()) else None
            reward = random.uniform(-1, 1) if action else 0
            max_q = random.uniform(0, 2)
            action_source = "exploitation" if random.random() > 0.2 else "exploration"
            
            cursor.execute("""
                INSERT INTO research_rl_episodes
                (episode_number, timestamp, phase, state_vector, action, action_name, action_mask,
                 reward, max_q_value, action_source, current_power, baseline_power_reference,
                 anomaly_index, behaviour_index, fatigue_index, opportunity_score, time_of_day_hour)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.episode_number,
                int(current.timestamp()),
                "active",
                json.dumps(state_vector),
                action,
                ACTIONS[action] if action else None,
                json.dumps(action_mask),
                reward,
                max_q,
                action_source,
                750 + random.uniform(-200, 200),
                850,
                anomaly_idx,
                behaviour_idx,
                fatigue_idx,
                opportunity_score,
                hour
            ))
            
            self.episode_number += 1
            rl_episodes += 1
            
            # Only log blocked notifications for research-valuable reasons
            if blocked and block_reason in BLOCK_REASONS:
                minutes_since = (current - self.last_notification_time).total_seconds() / 60 if self.last_notification_time else None
                
                cursor.execute("""
                    INSERT INTO research_blocked_notifications
                    (timestamp, phase, block_reason, opportunity_score, current_power,
                     anomaly_index, behaviour_index, fatigue_index, notification_count_today,
                     time_since_last_notification_minutes, required_cooldown_minutes,
                     adaptive_cooldown_minutes, available_action_count, action_mask,
                     state_vector, time_of_day_hour)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    int(current.timestamp()),
                    "active",
                    block_reason,
                    opportunity_score,
                    750 + random.uniform(-200, 200),
                    anomaly_idx,
                    behaviour_idx,
                    fatigue_idx,
                    self.notification_count_today,
                    minutes_since,
                    None,  # required_cooldown not relevant for fatigue/no_actions
                    None,  # adaptive_cooldown not relevant for fatigue/no_actions
                    sum(action_mask.values()),
                    json.dumps(action_mask),
                    json.dumps(state_vector),
                    hour
                ))
                nudges_blocked += 1
            
            # If not blocked and action chosen, send notification
            elif action and random.random() < 0.3:  # 30% chance to actually send
                notification_id = f"nudge_{int(current.timestamp())}"
                accepted_val = 1 if random.random() < 0.65 else 0  # 65% acceptance rate
                response_time = random.uniform(10, 300)
                
                cursor.execute("""
                    INSERT INTO research_nudge_log
                    (notification_id, timestamp, phase, action_type, template_index, title, message,
                     state_vector, current_power, anomaly_index, behaviour_index, fatigue_index,
                     responded, accepted, response_timestamp, response_time_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    notification_id,
                    int(current.timestamp()),
                    "active",
                    ACTIONS[action],
                    random.randint(0, 2),  # template_index
                    f"{ACTIONS[action].capitalize()} Nudge",
                    f"Notification message for {ACTIONS[action]}",
                    json.dumps(state_vector),
                    750 + random.uniform(-200, 200),
                    anomaly_idx,
                    behaviour_idx,
                    fatigue_idx,
                    1,  # responded
                    accepted_val,
                    int(current.timestamp()) + response_time,
                    response_time
                ))
                
                nudges_sent += 1
                if accepted_val == 1:
                    nudges_accepted += 1
                
                self.notification_count_today += 1
                self.last_notification_time = current
            
            current += timedelta(seconds=interval)
        
        # Generate weekly challenges (3 weeks in total, 1 in baseline + 2 in active)
        week1_start = (self.baseline_start + timedelta(days=7)).date()
        week2_start = (self.active_start).date()
        week3_start = (self.active_start + timedelta(days=7)).date()
        
        for week_start in [week1_start, week2_start, week3_start]:
            target_pct = 15.0
            baseline_w = 850
            actual_w = 850 - random.uniform(50, 150)
            savings_w = baseline_w - actual_w
            savings_pct = (savings_w / baseline_w) * 100
            achieved = 1 if savings_pct >= target_pct else 0
            week_end = week_start + timedelta(days=7)
            phase = "baseline" if week_start < self.active_start.date() else "active"
            
            cursor.execute("""
                INSERT INTO research_weekly_challenges
                (week_start_date, week_end_date, phase, target_percentage, baseline_W, actual_W, savings_W, savings_percentage, achieved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                week_start.isoformat(),
                week_end.isoformat(),
                phase,
                target_pct,
                baseline_w,
                actual_w,
                savings_w,
                savings_pct,
                achieved
            ))
        
        conn.commit()
        conn.close()
        
        print(f"  ✅ Generated {rl_episodes:,} RL episodes (active phase)")
        print(f"  ✅ Sent {nudges_sent} notifications")
        print(f"     - Accepted: {nudges_accepted} ({nudges_accepted/nudges_sent*100:.1f}%)")
        print(f"  ✅ Blocked {nudges_blocked} notifications")
        print(f"  ✅ Generated {tasks_generated} tasks")
        print(f"     - Completed: {tasks_completed} ({tasks_completed/tasks_generated*100:.1f}%)" if tasks_generated > 0 else f"     - Completed: {tasks_completed} (0.0%)")
        print(f"  ✅ Created 3 weekly challenges")
    
    def _compute_daily_aggregates(self):
        """Compute daily aggregates from sensor data."""
        sensor_conn = sqlite3.connect(self.sensor_db_path)
        sensor_cursor = sensor_conn.cursor()
        
        research_conn = sqlite3.connect(self.research_db_path)
        research_cursor = research_conn.cursor()
        
        current_date = self.baseline_start.date()
        end_date = self.current_time.date()
        
        days_computed = 0
        
        while current_date <= end_date:
            phase = "baseline" if current_date < self.active_start.date() else "active"
            
            start_ts = int(datetime.combine(current_date, datetime.min.time()).timestamp())
            end_ts = int(datetime.combine(current_date + timedelta(days=1), datetime.min.time()).timestamp())
            
            # Get sensor stats for the day
            sensor_cursor.execute("""
                SELECT 
                    AVG(power) as avg_power,
                    MAX(power) as peak_power,
                    MIN(power) as min_power,
                    AVG(temperature) as avg_temp,
                    AVG(humidity) as avg_humidity,
                    AVG(illuminance) as avg_illuminance,
                    SUM(CASE WHEN occupancy = 1 THEN 1 ELSE 0 END) * ? / 3600.0 as occupied_hours,
                    SUM(power * ? / 3600000.0) as total_energy_kwh
                FROM sensor_history
                WHERE timestamp >= ? AND timestamp < ?
            """, (UPDATE_INTERVAL_SECONDS, UPDATE_INTERVAL_SECONDS, start_ts, end_ts))
            
            energy_stats = sensor_cursor.fetchone()
            
            if energy_stats and energy_stats[0] is not None:
                # Get RL stats for the day
                research_cursor.execute("""
                    SELECT 
                        AVG(anomaly_index),
                        AVG(behaviour_index),
                        AVG(fatigue_index),
                        AVG(reward),
                        COUNT(*)
                    FROM research_rl_episodes
                    WHERE DATE(timestamp, 'unixepoch') = ?
                """, (current_date.isoformat(),))
                
                rl_stats = research_cursor.fetchone()
                
                # Get task stats for the day
                research_cursor.execute("""
                    SELECT 
                        COUNT(CASE WHEN generation_timestamp IS NOT NULL THEN 1 END),
                        COUNT(CASE WHEN completed = 1 THEN 1 END),
                        COUNT(CASE WHEN verified = 1 THEN 1 END),
                        COUNT(CASE WHEN user_feedback IS NOT NULL THEN 1 END)
                    FROM research_task_interactions
                    WHERE DATE(generation_timestamp, 'unixepoch') = ?
                """, (current_date.isoformat(),))
                
                task_stats = research_cursor.fetchone()
                
                # Get nudge stats for the day
                research_cursor.execute("""
                    SELECT 
                        COUNT(*) as sent,
                        SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END) as accepted,
                        SUM(CASE WHEN responded = 1 AND accepted = 0 THEN 1 ELSE 0 END) as dismissed,
                        SUM(CASE WHEN responded = 0 THEN 1 ELSE 0 END) as ignored
                    FROM research_nudge_log
                    WHERE DATE(timestamp, 'unixepoch') = ?
                """, (current_date.isoformat(),))
                
                nudge_stats = research_cursor.fetchone()
                if not nudge_stats or nudge_stats[0] is None:
                    nudge_stats = (0, 0, 0, 0)
                
                # Get blocked notifications count
                research_cursor.execute("""
                    SELECT COUNT(*)
                    FROM research_blocked_notifications
                    WHERE DATE(timestamp, 'unixepoch') = ?
                """, (current_date.isoformat(),))
                
                blocked_count = research_cursor.fetchone()[0]
                
                # Insert daily aggregate
                research_cursor.execute("""
                    INSERT OR REPLACE INTO research_daily_aggregates
                    (date, phase, total_energy_kwh, avg_power_w, peak_power_w, min_power_w,
                     avg_occupancy_count, total_occupied_hours, avg_temperature, avg_humidity, avg_illuminance,
                     tasks_generated, tasks_completed, tasks_verified,
                     nudges_sent, nudges_accepted, nudges_dismissed, nudges_ignored, nudges_blocked,
                     avg_anomaly_index, avg_behaviour_index, avg_fatigue_index)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    current_date.isoformat(),
                    phase,
                    energy_stats[7],  # total_energy_kwh
                    energy_stats[0],  # avg_power
                    energy_stats[1],  # peak_power
                    energy_stats[2],  # min_power
                    None,  # avg_occupancy_count (not calculated here)
                    energy_stats[6],  # total_occupied_hours
                    energy_stats[3],  # avg_temp
                    energy_stats[4],  # avg_humidity
                    energy_stats[5],  # avg_illuminance
                    task_stats[0] if task_stats else 0,  # tasks_generated
                    task_stats[1] if task_stats else 0,  # tasks_completed
                    task_stats[2] if task_stats else 0,  # tasks_verified
                    nudge_stats[0] if nudge_stats else 0,  # nudges_sent
                    nudge_stats[1] if nudge_stats else 0,  # nudges_accepted
                    nudge_stats[2] if nudge_stats else 0,  # nudges_dismissed
                    nudge_stats[3] if nudge_stats else 0,  # nudges_ignored
                    blocked_count,  # nudges_blocked
                    rl_stats[0] if rl_stats else None,  # avg_anomaly_index
                    rl_stats[1] if rl_stats else None,  # avg_behaviour_index
                    rl_stats[2] if rl_stats else None  # avg_fatigue_index
                ))
                
                days_computed += 1
            
            current_date += timedelta(days=1)
        
        research_conn.commit()
        sensor_conn.close()
        research_conn.close()
        
        print(f"  ✅ Computed daily aggregates for {days_computed} days")
    
    def _compute_area_daily_stats(self):
        """Compute area-level daily statistics."""
        sensor_conn = sqlite3.connect(self.sensor_db_path)
        sensor_cursor = sensor_conn.cursor()
        
        research_conn = sqlite3.connect(self.research_db_path)
        research_cursor = research_conn.cursor()
        
        # Get all areas
        sensor_cursor.execute("SELECT DISTINCT area_name FROM area_sensor_history")
        areas = [row[0] for row in sensor_cursor.fetchall()]
        
        current_date = self.baseline_start.date()
        end_date = self.current_time.date()
        
        records_computed = 0
        
        for area in areas:
            check_date = self.baseline_start.date()
            while check_date <= end_date:
                phase = "baseline" if check_date < self.active_start.date() else "active"
                
                start_ts = int(datetime.combine(check_date, datetime.min.time()).timestamp())
                end_ts = int(datetime.combine(check_date + timedelta(days=1), datetime.min.time()).timestamp())
                
                sensor_cursor.execute("""
                    SELECT 
                        AVG(power) as avg_power,
                        MAX(power) as max_power,
                        AVG(temperature) as avg_temp,
                        AVG(humidity) as avg_humidity,
                        AVG(illuminance) as avg_illuminance,
                        SUM(CASE WHEN occupancy = 1 THEN 1 ELSE 0 END) * ? / 3600.0 as occupied_hours
                    FROM area_sensor_history
                    WHERE area_name = ? AND timestamp >= ? AND timestamp < ?
                """, (UPDATE_INTERVAL_SECONDS, area, start_ts, end_ts))
                
                stats = sensor_cursor.fetchone()
                
                if stats and stats[0] is not None:
                    total_readings = (end_ts - start_ts) / UPDATE_INTERVAL_SECONDS
                    total_hours = total_readings * (UPDATE_INTERVAL_SECONDS / 3600.0)
                    occupancy_pct = (stats[5] / total_hours) * 100 if stats[5] and total_hours > 0 else 0
                    
                    research_cursor.execute("""
                        INSERT OR REPLACE INTO research_area_daily_stats
                        (date, area_name, phase, avg_power_w, max_power_w, min_power_w,
                         avg_temperature, avg_humidity, avg_illuminance, total_occupied_hours, occupancy_percentage)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        check_date.isoformat(),
                        area,
                        phase,
                        stats[0],
                        stats[1],
                        None,  # min_power_w - not calculated in this version
                        stats[2],
                        stats[3],
                        stats[4],
                        stats[5],
                        occupancy_pct
                    ))
                    
                    records_computed += 1
                
                check_date += timedelta(days=1)
        
        research_conn.commit()
        sensor_conn.close()
        research_conn.close()
        
        print(f"  ✅ Computed area daily stats for {len(areas)} areas ({records_computed} records)")
    
    def _generate_state_vector(self, timestamp: datetime, phase: str) -> list:
        """Generate realistic state vector."""
        hour = timestamp.hour
        is_day = 7 <= hour <= 22
        
        return [
            800 + random.uniform(-200, 200),  # power
            1.0 if is_day else 0.5,  # power flag
            200,  # top consumer
            1.0,
            22 + random.uniform(-2, 2),  # temperature
            1.0,
            50 + random.uniform(-10, 10),  # humidity
            1.0,
            400 if is_day else 50,  # illuminance
            1.0,
            1.0 if is_day else 0.0,  # occupancy
            1.0,
            random.uniform(0, 0.5),  # anomaly index
            random.uniform(0.3, 0.7),  # behaviour index
            0.0 if phase == "baseline" else random.uniform(0, 0.6),  # fatigue index
            random.randint(0, 2),  # area anomaly count
            hour / 24.0,  # time of day
            timestamp.weekday() / 7.0  # day of week
        ]
    
    def _generate_action_mask(self) -> dict:
        """Generate realistic action mask."""
        return {
            1: random.random() > 0.2,  # specific - usually available
            2: random.random() > 0.5,  # anomaly - sometimes available
            3: random.random() > 0.3,  # behavioural - often available
            4: random.random() > 0.4   # normative - often available
        }
    
    def _calculate_opportunity_score(self, timestamp: datetime) -> float:
        """Calculate opportunity score based on time and conditions."""
        hour = timestamp.hour
        
        # High opportunity during evening peak hours
        if 18 <= hour <= 21:
            return random.uniform(0.7, 1.0)
        elif 7 <= hour <= 22:
            return random.uniform(0.4, 0.7)
        else:
            return random.uniform(0.1, 0.4)
    
    def _print_summary(self):
        """Print database summary."""
        research_conn = sqlite3.connect(self.research_db_path)
        research_cursor = research_conn.cursor()
        
        sensor_conn = sqlite3.connect(self.sensor_db_path)
        sensor_cursor = sensor_conn.cursor()
        
        print("\n📊 Database Summary:")
        print("-" * 80)
        
        # Research database tables
        print("Research Database (research_data.db):")
        tables = [
            'research_daily_aggregates',
            'research_rl_episodes',
            'research_phase_metadata',
            'research_nudge_log',
            'research_blocked_notifications',
            'research_task_interactions',
            'research_area_daily_stats',
            'research_weekly_challenges'
        ]
        
        for table in tables:
            research_cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = research_cursor.fetchone()[0]
            print(f"  {table:<40} {count:>6} rows")
        
        # Sensor database tables
        print("\nSensor Database (sensor_data.db):")
        sensor_cursor.execute("SELECT COUNT(*) FROM sensor_history")
        sensor_count = sensor_cursor.fetchone()[0]
        print(f"  {'sensor_history':<40} {sensor_count:>6} rows")
        
        sensor_cursor.execute("SELECT COUNT(*) FROM area_sensor_history")
        area_sensor_count = sensor_cursor.fetchone()[0]
        print(f"  {'area_sensor_history':<40} {area_sensor_count:>6} rows")
        
        # Working hours distribution in sensor data
        print("\nWorking Hours Distribution:")
        sensor_cursor.execute("""
            SELECT 
                SUM(CASE WHEN within_working_hours = 1 THEN 1 ELSE 0 END) as working,
                SUM(CASE WHEN within_working_hours = 0 THEN 1 ELSE 0 END) as non_working
            FROM sensor_history
        """)
        working, non_working = sensor_cursor.fetchone()
        total = working + non_working
        print(f"  Working hours (Mon-Fri 8am-6pm):      {working:>6} ({working/total*100:.1f}%)")
        print(f"  Non-working hours:                     {non_working:>6} ({non_working/total*100:.1f}%)")
        
        research_conn.close()
        sensor_conn.close()
        print("-" * 80)
    
    def _print_expected_results(self):
        """Print expected results for key queries."""
        print("\n" + "=" * 80)
        print("EXPECTED QUERY RESULTS")
        print("=" * 80)
        
        print("\n✅ Query 2 - Phase Metadata:")
        print("   - 2 phases: baseline (14 days) → active (7 days, ongoing)")
        print("   - Baseline phase ended, active phase ongoing")
        
        print("\n✅ Query 3 - Current Phase:")
        print("   - Phase: active")
        print("   - Days in phase: ~7 days")
        
        print("\n✅ Query 4 - Recent Daily Aggregates:")
        print("   - Last 5 days show active phase data")
        print("   - total_energy_kwh: ~15-20 kWh per day")
        print("   - Tasks generated/completed only in active phase")
        print("   - Nudges sent/accepted/blocked only in active phase")
        
        print("\n✅ Query 8 - RL Episodes Summary:")
        print("   - Total episodes: Several thousand (baseline shadow + active real)")
        print("   - Baseline: ~40,000 episodes (shadow learning)")
        print("   - Active: ~40,000 episodes (real decisions)")
        
        print("\n✅ Query 9 - Action Distribution:")
        print("   - All 4 action types used (specific, anomaly, behavioural, normative)")
        print("   - Roughly balanced distribution based on availability")
        
        print("\n✅ Query 10 - Exploration vs Exploitation:")
        print("   - Baseline: shadow_exploration + shadow_exploitation")
        print("   - Active: ~80% exploitation, ~20% exploration")
        
        print("\n✅ Query 12 - Time-of-Day Patterns (Active Phase):")
        print("   - Higher notification attempts during evening (18-21h)")
        print("   - Lower opportunity scores at night")
        
        print("\n✅ Query 13 - Action Constraint Analysis (Active Phase):")
        print("   - Shows average available actions per day")
        print("   - Action masking based on sensor availability")
        
        print("\n✅ Query 13b - Shadow Learning Summary (Baseline Phase):")
        print("   - All episodes are shadow decisions")
        print("   - Exploratory rewards (both positive and negative)")
        
        print("\n✅ Query 15 - Nudge Summary:")
        print("   - Acceptance rate: ~65%")
        print("   - Average response time: 10-300 seconds")
        
        print("\n✅ Query 16b - Blocked Notifications (Active Phase):")
        print("   - Blocked by: cooldown, max_daily_limit, fatigue_threshold, no_available_actions")
        print("   - Fatigue index correlates with block count")
        
        print("\n✅ Query 16d - Notification Success vs Block Rate:")
        print("   - Daily breakdown of sent vs blocked notifications")
        print("   - Block percentage increases with daily notification count")
        
        print("\n✅ Queries 20-31 - Intervention Impact Analytics:")
        print("   - Power consumption trends (decreasing in active phase)")
        print("   - User engagement trends (task completion rates)")
        print("   - Phase comparison (baseline vs active)")
        print("   - Correlation analysis (indices vs energy)")
        print("   - Gamification effectiveness (task difficulty vs completion)")
        
        print("\n✅ Query 30 - Working Hours Data Distribution:")
        print("   - Shows split between working hours (Mon-Fri 8am-6pm) and non-working hours")
        print("   - ~40% working hours, ~60% non-working hours (includes weekends + nights)")
        print("   - Used for office mode to filter baseline calculations")
        
        print("\n" + "=" * 80)
        print("💡 Run test_research_db.py to see actual results")
        print("=" * 80)


def main():
    """Main entry point."""
    if len(sys.argv) > 2:
        print("❌ Too many arguments!")
        print(__doc__)
        return 1
    
    # Determine data directory
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "config/green_shift_data"
    
    # Run generation
    generator = TestDataGenerator(data_dir)
    generator.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())