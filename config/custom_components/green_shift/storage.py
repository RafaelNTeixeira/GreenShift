"""
Storage management for Green Shift integration.
- SQLite sensor_data.db: Temporal sensor data (14 days rolling window) with area-based tracking + Daily tasks
- SQLite research_data.db: Permanent research data (never purged) for post-intervention analysis
- JSON: Persistent state (AI configuration, indices, Q-table)
"""
import logging
import sqlite3
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import asyncio
from homeassistant.core import HomeAssistant
from .const import UPDATE_INTERVAL_SECONDS, RL_EPISODE_RETENTION_DAYS

_LOGGER = logging.getLogger(__name__)


class StorageManager:
    """Manages SQLite and JSON storage for Green Shift."""

    def __init__(self, hass: HomeAssistant, config_dir: str = None):
        """Initialize storage manager."""
        self.hass = hass

        # Use Home Assistant's configuration directory
        if config_dir is None:
            config_dir = hass.config.path("green_shift_data")

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.db_path = self.config_dir / "sensor_data.db"  # Rolling 14-day data
        self.research_db_path = self.config_dir / "research_data.db"  # Permanent research data
        self.state_file = self.config_dir / "state.json"

        self._conn = None
        self._lock = asyncio.Lock()

        _LOGGER.info("Storage initialized at: %s", self.config_dir)
        _LOGGER.info("Research database: %s", self.research_db_path)

    async def setup(self):
        """Setup database schema and load state."""
        await self._init_database()
        await self._init_research_database()
        await self._cleanup_old_data()

    async def _init_database(self):
        """Initialize SQLite database with schema."""
        def _create_tables():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Enable WAL mode for better crash recovery and concurrent access
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and performance

            _LOGGER.info("SQLite WAL mode enabled for crash protection")

            # Main sensor data table with all metrics
            cursor.execute("""
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

            # Area-specific sensor data table
            cursor.execute("""
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

            # Daily tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL UNIQUE,
                    date TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    target_value REAL,
                    target_unit TEXT,
                    baseline_value REAL,
                    area_name TEXT,
                    difficulty_level INTEGER DEFAULT 1,
                    completed INTEGER DEFAULT 0,
                    verified INTEGER DEFAULT 0,
                    completion_value REAL,
                    completion_timestamp REAL,
                    user_feedback TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Task difficulty history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS task_difficulty_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    difficulty_level INTEGER NOT NULL,
                    feedback TEXT NOT NULL,
                    date TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Index for faster queries by timestamp
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON sensor_history(timestamp)
            """)

            # Index for faster area-based queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_area_timestamp
                ON area_sensor_history(area_name, timestamp)
            """)

            # Index for tasks by date
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_date
                ON daily_tasks(date)
            """)

            conn.commit()
            conn.close()
            _LOGGER.info("Database schema initialized")

        await self.hass.async_add_executor_job(_create_tables)

    async def _init_research_database(self):
        """Initialize permanent research database (never purged)."""
        def _create_research_tables():
            conn = sqlite3.connect(str(self.research_db_path))
            cursor = conn.cursor()

            # Enable WAL mode for better crash recovery and concurrent access
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")

            _LOGGER.info("Research database WAL mode enabled for crash protection")

            # Phase metadata table
            cursor.execute("""
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

            # Daily aggregates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_daily_aggregates (
                    date TEXT PRIMARY KEY,
                    phase TEXT NOT NULL,

                    -- Energy metrics
                    total_energy_kwh REAL,
                    avg_power_w REAL,
                    peak_power_w REAL,
                    min_power_w REAL,

                    -- Occupancy metrics (for normalization)
                    avg_occupancy_count REAL,
                    total_occupied_hours REAL,

                    -- Environmental context
                    avg_temperature REAL,
                    avg_humidity REAL,
                    avg_illuminance REAL,

                    -- Engagement metrics
                    tasks_generated INTEGER DEFAULT 0,
                    tasks_completed INTEGER DEFAULT 0,
                    tasks_verified INTEGER DEFAULT 0,

                    -- Nudge metrics
                    nudges_sent INTEGER DEFAULT 0,
                    nudges_accepted INTEGER DEFAULT 0,
                    nudges_dismissed INTEGER DEFAULT 0,
                    nudges_ignored INTEGER DEFAULT 0,
                    nudges_blocked INTEGER DEFAULT 0,

                    -- Indices
                    avg_anomaly_index REAL,
                    avg_behaviour_index REAL,
                    avg_fatigue_index REAL,

                    -- For weather normalization (to be filled later)
                    outdoor_temp_celsius REAL,
                    hdd_base18 REAL,

                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # RL episodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_rl_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    episode_number INTEGER,
                    phase TEXT,

                    -- RL components
                    state_vector TEXT,
                    state_key TEXT,
                    action INTEGER,
                    action_name TEXT,
                    action_source TEXT,
                    reward REAL,

                    -- Q-values at decision time
                    q_values TEXT,
                    max_q_value REAL,
                    epsilon REAL,

                    -- Action availability
                    action_mask TEXT,

                    -- Context
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

            # Nudge log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_nudge_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    notification_id TEXT UNIQUE,
                    phase TEXT,

                    -- Nudge details
                    action_type TEXT,
                    template_index INTEGER,
                    title TEXT,
                    message TEXT,

                    -- Context at nudge time
                    state_vector TEXT,
                    current_power REAL,
                    anomaly_index REAL,
                    behaviour_index REAL,
                    fatigue_index REAL,

                    -- User response
                    responded INTEGER DEFAULT 0,
                    accepted INTEGER DEFAULT 0,
                    response_timestamp REAL,
                    response_time_seconds REAL,

                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Blocked notifications table (active phase only)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_blocked_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    phase TEXT,

                    -- Block reason
                    block_reason TEXT NOT NULL,

                    -- Context at block time
                    opportunity_score REAL,
                    current_power REAL,
                    anomaly_index REAL,
                    behaviour_index REAL,
                    fatigue_index REAL,
                    notification_count_today INTEGER,

                    -- Cooldown details (if applicable)
                    time_since_last_notification_minutes REAL,
                    required_cooldown_minutes REAL,
                    adaptive_cooldown_minutes REAL,

                    -- Action availability
                    available_action_count INTEGER,
                    action_mask TEXT,

                    -- State
                    state_vector TEXT,
                    time_of_day_hour INTEGER,

                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Task interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_task_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    phase TEXT,

                    -- Task details
                    task_type TEXT,
                    difficulty_level INTEGER,
                    target_value REAL,
                    baseline_value REAL,
                    area_name TEXT,

                    -- Engagement tracking
                    generation_timestamp REAL,
                    first_view_timestamp REAL,
                    completion_timestamp REAL,
                    time_to_view_seconds REAL,
                    time_to_complete_seconds REAL,

                    -- Outcomes
                    completed INTEGER DEFAULT 0,
                    verified INTEGER DEFAULT 0,
                    completion_value REAL,
                    user_feedback TEXT,

                    -- Context at generation
                    power_at_generation REAL,
                    occupancy_at_generation INTEGER,

                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Area daily stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_area_daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    area_name TEXT NOT NULL,
                    phase TEXT,

                    -- Energy by area
                    avg_power_w REAL,
                    max_power_w REAL,
                    min_power_w REAL,

                    -- Environment by area
                    avg_temperature REAL,
                    avg_humidity REAL,
                    avg_illuminance REAL,

                    -- Occupancy by area
                    total_occupied_hours REAL,
                    occupancy_percentage REAL,

                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE(date, area_name)
                )
            """)

            # Indices for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_episodes_timestamp
                ON research_rl_episodes(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_nudge_log_timestamp
                ON research_nudge_log(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_blocked_notifications_timestamp
                ON research_blocked_notifications(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_interactions_date
                ON research_task_interactions(date)
            """)

            # Weekly challenges tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_weekly_challenges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    week_start_date TEXT NOT NULL,
                    week_end_date TEXT NOT NULL,
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

            conn.commit()
            conn.close()
            _LOGGER.info("Research database schema initialized (permanent storage)")

        await self.hass.async_add_executor_job(_create_research_tables)

    async def _cleanup_old_data(self):
        """Remove data older than 14 days from sensor_data.db (research_data.db is never purged)."""
        def _cleanup():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Calculate cutoff (14 days ago)
            cutoff = (datetime.now() - timedelta(days=14)).timestamp()

            # Clean global history
            cursor.execute(
                "DELETE FROM sensor_history WHERE timestamp < ?",
                (cutoff,)
            )
            deleted_global = cursor.rowcount

            # Clean area history
            cursor.execute(
                "DELETE FROM area_sensor_history WHERE timestamp < ?",
                (cutoff,)
            )
            deleted_area = cursor.rowcount

            # Clean old tasks (keep 30 days for historical analysis)
            cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            cursor.execute(
                "DELETE FROM daily_tasks WHERE date < ?",
                (cutoff_date,)
            )
            deleted_tasks = cursor.rowcount

            conn.commit()
            conn.close()

            if deleted_global > 0 or deleted_area > 0 or deleted_tasks > 0:
                _LOGGER.info("Cleaned up %d global, %d area, and %d task records", deleted_global, deleted_area, deleted_tasks)

        await self.hass.async_add_executor_job(_cleanup)

    async def _cleanup_old_rl_episodes(self):
        """Remove RL episodes older than RL_EPISODE_RETENTION_DAYS from research_data.db."""
        def _cleanup():
            conn = sqlite3.connect(str(self.research_db_path))
            cursor = conn.cursor()

            # Calculate cutoff (RL_EPISODE_RETENTION_DAYS ago)
            cutoff = (datetime.now() - timedelta(days=RL_EPISODE_RETENTION_DAYS)).timestamp()

            # Clean old RL episodes
            cursor.execute(
                "DELETE FROM research_rl_episodes WHERE timestamp < ?",
                (cutoff,)
            )
            deleted_episodes = cursor.rowcount

            conn.commit()
            conn.close()

            if deleted_episodes > 0:
                _LOGGER.info("Cleaned up %d old RL episodes (older than %d days)", deleted_episodes, RL_EPISODE_RETENTION_DAYS)

        await self.hass.async_add_executor_job(_cleanup)

    # ==================== TEMPORAL DATA (SQLite) ====================

    async def store_sensor_snapshot(
        self,
        timestamp: datetime,
        power: float = None,
        energy: float = None,
        temperature: float = None,
        humidity: float = None,
        illuminance: float = None,
        occupancy: bool = None,
        within_working_hours: bool = True
    ):
        """Store a single sensor snapshot."""
        def _insert():
            conn = None
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO sensor_history
                    (timestamp, power, energy, temperature, humidity, illuminance, occupancy, within_working_hours)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp.timestamp(),
                    power,
                    energy,
                    temperature,
                    humidity,
                    illuminance,
                    1 if occupancy else 0 if occupancy is not None else None,
                    1 if within_working_hours else 0
                ))

                conn.commit()

            except Exception as e:
                _LOGGER.error("Failed to store sensor snapshot: %s", e)
                if conn:
                    conn.rollback()
            finally:
                if conn:
                    conn.close()

        await self.hass.async_add_executor_job(_insert)

    async def store_area_snapshot(
        self,
        timestamp: datetime,
        area_name: str,
        power: float = None,
        energy: float = None,
        temperature: float = None,
        humidity: float = None,
        illuminance: float = None,
        occupancy: bool = None,
        within_working_hours: bool = True
    ):
        """Store a sensor snapshot for a specific area."""
        def _insert():
            conn = None
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO area_sensor_history
                    (timestamp, area_name, power, energy, temperature, humidity, illuminance, occupancy, within_working_hours)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp.timestamp(),
                    area_name,
                    power,
                    energy,
                    temperature,
                    humidity,
                    illuminance,
                    1 if occupancy else 0 if occupancy is not None else None,
                    1 if within_working_hours else 0
                ))

                conn.commit()

            except Exception as e:
                _LOGGER.error("Failed to store area snapshot for %s: %s", area_name, e)
                if conn:
                    conn.rollback()
            finally:
                if conn:
                    conn.close()

        await self.hass.async_add_executor_job(_insert)

    async def get_history(
        self,
        metric: str,
        hours: int = None,
        days: int = None,
        working_hours_only: bool = None
    ) -> List[Tuple[datetime, float]]:
        """
        Get historical data for a specific metric.

        Args:
            metric: One of 'power', 'energy', 'temperature', 'humidity', 'illuminance', 'occupancy'
            hours: Number of hours to retrieve (optional)
            days: Number of days to retrieve (optional)
            working_hours_only: Filter to only working hours (True), non-working hours (False), or all (None)

        Returns:
            List of (timestamp, value) tuples
        """
        def _query():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            query = f"SELECT timestamp, {metric} FROM sensor_history WHERE {metric} IS NOT NULL"

            if hours:
                cutoff = (datetime.now() - timedelta(hours=hours)).timestamp()
                query += f" AND timestamp >= {cutoff}"
            elif days:
                cutoff = (datetime.now() - timedelta(days=days)).timestamp()
                query += f" AND timestamp >= {cutoff}"

            # Filter by working hours if specified
            if working_hours_only is True:
                query += " AND within_working_hours = 1"
            elif working_hours_only is False:
                query += " AND within_working_hours = 0"

            query += " ORDER BY timestamp ASC"

            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()

            # Convert to (datetime, value) tuples
            return [(datetime.fromtimestamp(ts), val) for ts, val in rows]

        return await self.hass.async_add_executor_job(_query)

    async def get_area_history(
        self,
        area_name: str,
        metric: str,
        hours: int = None,
        days: int = None,
        working_hours_only: bool = None
    ) -> List[Tuple[datetime, float]]:
        """
        Get historical data for a specific metric in a specific area.

        Args:
            area_name: Name of the area
            metric: One of 'temperature', 'humidity', 'illuminance', 'occupancy'
            hours: Number of hours to retrieve (optional)
            days: Number of days to retrieve (optional)
            working_hours_only: Filter to only working hours (True), non-working hours (False), or all (None)

        Returns:
            List of (timestamp, value) tuples
        """
        def _query():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            query = f"""
                SELECT timestamp, {metric}
                FROM area_sensor_history
                WHERE area_name = ? AND {metric} IS NOT NULL
            """

            params = [area_name]

            if hours:
                cutoff = (datetime.now() - timedelta(hours=hours)).timestamp()
                query += " AND timestamp >= ?"
                params.append(cutoff)
            elif days:
                cutoff = (datetime.now() - timedelta(days=days)).timestamp()
                query += " AND timestamp >= ?"
                params.append(cutoff)

            # Filter by working hours if specified
            if working_hours_only is True:
                query += " AND within_working_hours = 1"
            elif working_hours_only is False:
                query += " AND within_working_hours = 0"

            query += " ORDER BY timestamp ASC"

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            # Convert to (datetime, value) tuples
            return [(datetime.fromtimestamp(ts), val) for ts, val in rows]

        return await self.hass.async_add_executor_job(_query)

    async def get_all_areas(self) -> List[str]:
        """Get list of all areas that have data."""
        def _query():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DISTINCT area_name
                FROM area_sensor_history
                ORDER BY area_name
            """)

            rows = cursor.fetchall()
            conn.close()

            return [row[0] for row in rows]

        return await self.hass.async_add_executor_job(_query)

    async def get_area_stats(
        self,
        area_name: str,
        metric: str,
        hours: int = None,
        days: int = None,
        working_hours_only: bool = None
    ) -> Dict[str, float]:
        """
        Get statistical summary for an area's metric.

        Args:
            area_name: Name of the area
            metric: Metric to analyze
            hours: Number of hours to retrieve
            days: Number of days to retrieve
            working_hours_only: Filter to only working hours (True), non-working hours (False), or all (None)

        Returns:
            Dictionary with 'mean', 'min', 'max', 'std' keys
        """
        history = await self.get_area_history(area_name, metric, hours, days, working_hours_only)

        if not history:
            return {"mean": 0, "min": 0, "max": 0, "std": 0}

        values = [val for _, val in history]

        return {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values))
        }

    async def get_recent_values(
        self,
        metric: str,
        count: int = 100
    ) -> List[float]:
        """Get the N most recent values for a metric (values only, no timestamps)."""
        def _query():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT {metric} FROM sensor_history
                WHERE {metric} IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT ?
            """, (count,))

            rows = cursor.fetchall()
            conn.close()

            # Return in chronological order (oldest first)
            return [row[0] for row in reversed(rows)]

        return await self.hass.async_add_executor_job(_query)

    # ==================== DAILY TASKS (SQLite) ====================

    async def save_daily_tasks(self, tasks: List[Dict]) -> bool:
        """Save daily tasks to database."""
        def _insert():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            for task in tasks:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO daily_tasks
                        (task_id, date, task_type, title, description, target_value,
                         target_unit, baseline_value, area_name, difficulty_level)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        task['task_id'],
                        task['date'],
                        task['task_type'],
                        task['title'],
                        task['description'],
                        task.get('target_value'),
                        task.get('target_unit'),
                        task.get('baseline_value'),
                        task.get('area_name'),
                        task.get('difficulty_level', 1)
                    ))
                except Exception as e:
                    _LOGGER.error("Error saving task %s: %s", task['task_id'], e)
                    conn.close()
                    return False

            conn.commit()
            conn.close()
            return True

        return await self.hass.async_add_executor_job(_insert)

    async def get_today_tasks(self) -> List[Dict]:
        """Get today's tasks."""
        today = datetime.now().strftime("%Y-%m-%d")

        def _query():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT task_id, date, task_type, title, description,
                       target_value, target_unit, baseline_value, area_name,
                       difficulty_level, completed, verified, completion_value,
                       completion_timestamp, user_feedback
                FROM daily_tasks
                WHERE date = ?
                ORDER BY id ASC
            """, (today,))

            rows = cursor.fetchall()
            conn.close()

            tasks = []
            for row in rows:
                tasks.append({
                    'task_id': row[0],
                    'date': row[1],
                    'task_type': row[2],
                    'title': row[3],
                    'description': row[4],
                    'target_value': row[5],
                    'target_unit': row[6],
                    'baseline_value': row[7],
                    'area_name': row[8],
                    'difficulty_level': row[9],
                    'completed': bool(row[10]),
                    'verified': bool(row[11]),
                    'completion_value': row[12],
                    'completion_timestamp': row[13],
                    'user_feedback': row[14]
                })

            return tasks

        return await self.hass.async_add_executor_job(_query)

    async def get_total_completed_tasks_count(self) -> int:
        """Get the total count of completed tasks in the last 30 days (rolling window)."""
        def _query():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*)
                FROM daily_tasks
                WHERE completed = 1 OR verified = 1
            """)

            count = cursor.fetchone()[0]
            conn.close()

            return count

        return await self.hass.async_add_executor_job(_query)

    async def get_total_completed_tasks_count_alltime(self) -> int:
        """Get the total count of all completed tasks across all time from research database."""
        def _query():
            conn = sqlite3.connect(str(self.research_db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*)
                FROM research_task_interactions
                WHERE completed = 1
            """)

            count = cursor.fetchone()[0]
            conn.close()

            return count

        return await self.hass.async_add_executor_job(_query)

    async def mark_task_completed(self, task_id: str, completion_value: float = None) -> bool:
        """Mark a task as completed."""
        def _update():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE daily_tasks
                SET completed = 1,
                    completion_value = ?,
                    completion_timestamp = ?
                WHERE task_id = ?
            """, (completion_value, datetime.now().timestamp(), task_id))

            _LOGGER.debug("Marking task %s as completed with value %s", task_id, completion_value)

            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return success

        return await self.hass.async_add_executor_job(_update)

    async def mark_task_verified(self, task_id: str, verified: bool = True) -> bool:
        """Mark a task as verified by the system."""
        def _update():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE daily_tasks
                SET verified = ?
                WHERE task_id = ?
            """, (1 if verified else 0, task_id))

            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return success

        return await self.hass.async_add_executor_job(_update)

    async def save_task_feedback(self, task_id: str, feedback: str) -> bool:
        """Save user feedback for a task (too_easy, just_right, too_hard)."""
        def _update():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            _LOGGER.debug("Saving feedback for task %s: %s", task_id, feedback)

            # First, get task info
            cursor.execute("""
                SELECT task_type, difficulty_level, date
                FROM daily_tasks
                WHERE task_id = ?
            """, (task_id,))

            row = cursor.fetchone()
            if not row:
                conn.close()
                return False

            task_type, difficulty_level, date = row

            # Update the task feedback
            cursor.execute("""
                UPDATE daily_tasks
                SET user_feedback = ?
                WHERE task_id = ?
            """, (feedback, task_id))

            # Save to difficulty history
            cursor.execute("""
                INSERT INTO task_difficulty_history
                (task_type, difficulty_level, feedback, date)
                VALUES (?, ?, ?, ?)
            """, (task_type, difficulty_level, feedback, date))

            conn.commit()
            conn.close()
            return True

        return await self.hass.async_add_executor_job(_update)

    async def delete_today_tasks(self) -> bool:
        """Delete today's tasks from both sensor and research databases."""
        today = datetime.now().strftime("%Y-%m-%d")

        def _delete():
            # Delete from sensor database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM daily_tasks WHERE date = ?
            """, (today,))

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            # Delete from research database
            res_conn = sqlite3.connect(str(self.research_db_path))
            res_cursor = res_conn.cursor()

            res_cursor.execute("""
                DELETE FROM research_task_interactions WHERE date = ?
            """, (today,))

            res_deleted_count = res_cursor.rowcount
            res_conn.commit()
            res_conn.close()

            _LOGGER.info("Deleted %d tasks from sensor DB and %d from research DB for %s",
                        deleted_count, res_deleted_count, today)

            return True

        return await self.hass.async_add_executor_job(_delete)

    async def get_task_difficulty_stats(self, task_type: str) -> Dict:
        """Get difficulty statistics for a task type to adjust future difficulty."""
        def _query():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Get recent feedback (last 30 days)
            cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            cursor.execute("""
                SELECT difficulty_level, feedback, COUNT(*) as count
                FROM task_difficulty_history
                WHERE task_type = ? AND date >= ?
                GROUP BY difficulty_level, feedback
                ORDER BY difficulty_level, feedback
            """, (task_type, cutoff_date))

            rows = cursor.fetchall()
            conn.close()

            # Process statistics
            stats = {
                'too_easy_count': 0,
                'just_right_count': 0,
                'too_hard_count': 0,
                'avg_difficulty': 1,
                'suggested_adjustment': 0
            }

            total = 0
            difficulty_sum = 0

            for difficulty, feedback, count in rows:
                total += count
                difficulty_sum += difficulty * count

                if feedback == 'too_easy':
                    stats['too_easy_count'] += count
                elif feedback == 'just_right':
                    stats['just_right_count'] += count
                elif feedback == 'too_hard':
                    stats['too_hard_count'] += count

            if total > 0:
                stats['avg_difficulty'] = difficulty_sum / total

                # Suggest adjustment based on feedback
                if stats['too_easy_count'] > stats['too_hard_count'] * 2:
                    stats['suggested_adjustment'] = 1  # Increase difficulty
                elif stats['too_hard_count'] > stats['too_easy_count'] * 2:
                    stats['suggested_adjustment'] = -1  # Decrease difficulty

            _LOGGER.debug("Difficulty stats for task type '%s': %s", task_type, stats)

            return stats

        return await self.hass.async_add_executor_job(_query)

    # ==================== PERSISTENT STATE (JSON) ====================

    async def save_state(self, state_data: Dict[str, Any]):
        """
        Save persistent state to JSON using atomic write.

        Expected keys:
        - phase: Current system phase
        - baseline_consumption: Baseline consumption value
        - anomaly_index: Current anomaly index
        - behaviour_index: Current behaviour index
        - fatigue_index: Current fatigue index
        - q_table: Q-learning table
        - energy_midnight_points: Midnight energy readings
        """
        def _write():
            try:
                # Convert datetime objects to ISO strings
                serializable_state = {}
                for key, value in state_data.items():
                    if isinstance(value, datetime):
                        serializable_state[key] = value.isoformat()
                    else:
                        serializable_state[key] = value

                # Atomic write: write to temporary file first, then rename
                temp_file = self.state_file.with_suffix('.tmp')

                with open(temp_file, 'w') as f:
                    json.dump(serializable_state, f, indent=2)

                # Atomic rename (replaces old file only after new one is complete)
                temp_file.replace(self.state_file)

                _LOGGER.debug("State saved atomically to %s", self.state_file)

            except Exception as e:
                _LOGGER.error("Failed to save state: %s", e)
                # Clean up temp file if it exists
                temp_file = self.state_file.with_suffix('.tmp')
                if temp_file.exists():
                    temp_file.unlink()
                raise

        async with self._lock:
            await self.hass.async_add_executor_job(_write)

    async def load_state(self) -> Dict[str, Any]:
        """
        Load persistent state from JSON.

        Returns empty dict if file doesn't exist.
        """
        def _read():
            if not self.state_file.exists():
                _LOGGER.info("No existing state file found")
                return {}

            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                _LOGGER.info("State loaded from %s", self.state_file)
                return state
            except Exception as e:
                _LOGGER.error("Failed to load state: %s", e)
                return {}

        async with self._lock:
            return await self.hass.async_add_executor_job(_read)

    async def update_state_field(self, key: str, value: Any):
        """Update a single field in the state file."""
        state = await self.load_state()
        state[key] = value
        await self.save_state(state)


    # ==================== RESEARCH DATA (Permanent SQLite) ====================

    async def record_phase_change(self, phase: str, baseline_consumption: float = None, baseline_occupancy: float = None, notes: str = None):
        """Record when system phase changes."""
        def _insert():
            conn = sqlite3.connect(str(self.research_db_path))
            cursor = conn.cursor()

            # End current phase
            cursor.execute("""
                UPDATE research_phase_metadata
                SET end_timestamp = ?
                WHERE end_timestamp IS NULL
            """, (datetime.now().timestamp(),))

            # Start new phase
            cursor.execute("""
                INSERT INTO research_phase_metadata
                (phase, start_timestamp, baseline_consumption_W, baseline_occupancy_avg, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (phase, datetime.now().timestamp(), baseline_consumption, baseline_occupancy, notes))

            conn.commit()
            conn.close()
            _LOGGER.info("Phase changed to: %s", phase)

        await self.hass.async_add_executor_job(_insert)

    async def log_rl_decision(self, episode_data: dict):
        """Log each RL agent decision for convergence analysis."""
        def _insert():
            conn = sqlite3.connect(str(self.research_db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO research_rl_episodes
                (timestamp, episode_number, phase, state_vector, state_key,
                 action, action_name, action_source, reward, q_values,
                 max_q_value, epsilon, action_mask, current_power, anomaly_index,
                 behaviour_index, fatigue_index, opportunity_score, time_of_day_hour, baseline_power_reference)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().timestamp(),
                episode_data.get('episode'),
                episode_data.get('phase'),
                json.dumps(episode_data.get('state_vector', [])),
                str(episode_data.get('state_key', '')),
                episode_data.get('action'),
                episode_data.get('action_name'),
                episode_data.get('action_source'),
                episode_data.get('reward'),
                json.dumps(episode_data.get('q_values', {})),
                episode_data.get('max_q'),
                episode_data.get('epsilon'),
                json.dumps(episode_data.get('action_mask', {})),
                episode_data.get('power'),
                episode_data.get('anomaly_index'),
                episode_data.get('behaviour_index'),
                episode_data.get('fatigue_index'),
                episode_data.get('opportunity_score'),
                episode_data.get('time_of_day_hour'),
                episode_data.get('baseline_power_reference')
            ))

            conn.commit()
            conn.close()

        await self.hass.async_add_executor_job(_insert)

    async def log_nudge_sent(self, nudge_data: dict):
        """Log comprehensive nudge information for acceptance rate analysis."""
        def _insert():
            conn = sqlite3.connect(str(self.research_db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO research_nudge_log
                (timestamp, notification_id, phase, action_type, template_index,
                 title, message, state_vector, current_power, anomaly_index,
                 behaviour_index, fatigue_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().timestamp(),
                nudge_data.get('notification_id'),
                nudge_data.get('phase'),
                nudge_data.get('action_type'),
                nudge_data.get('template_index'),
                nudge_data.get('title'),
                nudge_data.get('message'),
                json.dumps(nudge_data.get('state_vector', [])),
                nudge_data.get('current_power'),
                nudge_data.get('anomaly_index'),
                nudge_data.get('behaviour_index'),
                nudge_data.get('fatigue_index')
            ))

            conn.commit()
            conn.close()

        await self.hass.async_add_executor_job(_insert)

    async def log_nudge_response(self, notification_id: str, accepted: bool):
        """Log user response to nudge."""
        def _update():
            conn = sqlite3.connect(str(self.research_db_path))
            cursor = conn.cursor()

            # Get when nudge was sent
            cursor.execute("""
                SELECT timestamp FROM research_nudge_log
                WHERE notification_id = ?
            """, (notification_id,))
            result = cursor.fetchone()

            response_time = None
            if result:
                sent_time = result[0]
                response_time = datetime.now().timestamp() - sent_time

            cursor.execute("""
                UPDATE research_nudge_log
                SET responded = 1,
                    accepted = ?,
                    response_timestamp = ?,
                    response_time_seconds = ?
                WHERE notification_id = ?
            """, (1 if accepted else 0, datetime.now().timestamp(),
                  response_time, notification_id))

            conn.commit()
            conn.close()

        await self.hass.async_add_executor_job(_update)

    async def log_blocked_notification(self, block_data: dict):
        """Log when a notification attempt was blocked in active phase."""
        def _insert():
            conn = sqlite3.connect(str(self.research_db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO research_blocked_notifications
                (timestamp, phase, block_reason, opportunity_score,
                 current_power, anomaly_index, behaviour_index, fatigue_index,
                 notification_count_today, time_since_last_notification_minutes,
                 required_cooldown_minutes, adaptive_cooldown_minutes,
                 available_action_count, action_mask, state_vector, time_of_day_hour)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().timestamp(),
                block_data.get('phase'),
                block_data.get('block_reason'),
                block_data.get('opportunity_score'),
                block_data.get('current_power'),
                block_data.get('anomaly_index'),
                block_data.get('behaviour_index'),
                block_data.get('fatigue_index'),
                block_data.get('notification_count_today'),
                block_data.get('time_since_last_notification_minutes'),
                block_data.get('required_cooldown_minutes'),
                block_data.get('adaptive_cooldown_minutes'),
                block_data.get('available_action_count'),
                json.dumps(block_data.get('action_mask', {})),
                json.dumps(block_data.get('state_vector', [])),
                block_data.get('time_of_day_hour')
            ))

            conn.commit()
            conn.close()

        await self.hass.async_add_executor_job(_insert)

    async def log_task_generation(self, task_data: dict):
        """Log task generation with full context."""
        def _insert():
            conn = sqlite3.connect(str(self.research_db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO research_task_interactions
                (task_id, date, phase, task_type, difficulty_level,
                 target_value, baseline_value, area_name, generation_timestamp,
                 power_at_generation, occupancy_at_generation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_data.get('task_id'),
                task_data.get('date'),
                task_data.get('phase'),
                task_data.get('task_type'),
                task_data.get('difficulty_level'),
                task_data.get('target_value'),
                task_data.get('baseline_value'),
                task_data.get('area_name'),
                datetime.now().timestamp(),
                task_data.get('power_at_generation'),
                task_data.get('occupancy_at_generation')
            ))

            conn.commit()
            conn.close()

        await self.hass.async_add_executor_job(_insert)

    async def log_task_completion(self, task_id: str, completion_value: float = None):
        """Log task completion."""
        def _update():
            conn = sqlite3.connect(str(self.research_db_path))
            cursor = conn.cursor()

            # Get generation timestamp
            cursor.execute("""
                SELECT generation_timestamp FROM research_task_interactions
                WHERE task_id = ?
            """, (task_id,))
            result = cursor.fetchone()

            time_to_complete = None
            if result and result[0]:
                time_to_complete = datetime.now().timestamp() - result[0]

            cursor.execute("""
                UPDATE research_task_interactions
                SET completed = 1,
                    completion_timestamp = ?,
                    time_to_complete_seconds = ?,
                    completion_value = ?
                WHERE task_id = ?
            """, (datetime.now().timestamp(), time_to_complete, completion_value, task_id))

            conn.commit()
            conn.close()

        await self.hass.async_add_executor_job(_update)

    async def log_task_feedback(self, task_id: str, feedback: str):
        """Log task difficulty feedback."""
        def _update():
            conn = sqlite3.connect(str(self.research_db_path))
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE research_task_interactions
                SET user_feedback = ?
                WHERE task_id = ?
            """, (feedback, task_id))

            conn.commit()
            conn.close()

        await self.hass.async_add_executor_job(_update)

    async def log_weekly_challenge(self, challenge_data: dict):
        """Log weekly challenge progress for gamification analysis."""
        def _insert():
            conn = sqlite3.connect(str(self.research_db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO research_weekly_challenges
                (week_start_date, week_end_date, phase, target_percentage,
                 baseline_W, actual_W, savings_W, savings_percentage, achieved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                challenge_data.get('week_start_date'),
                challenge_data.get('week_end_date'),
                challenge_data.get('phase'),
                challenge_data.get('target_percentage'),
                challenge_data.get('baseline_W'),
                challenge_data.get('actual_W'),
                challenge_data.get('savings_W'),
                challenge_data.get('savings_percentage'),
                1 if challenge_data.get('achieved', False) else 0
            ))

            conn.commit()
            conn.close()

        await self.hass.async_add_executor_job(_insert)

    async def compute_daily_aggregates(self, date: str = None, phase: str = None):
        """
        Compute and store daily aggregates for research analysis.
        Continuously updates today's aggregate (INSERT OR REPLACE).
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Get all data for the day
        start_ts = datetime.strptime(date, "%Y-%m-%d").timestamp()
        end_ts = start_ts + 86400

        def _compute():
            # Connect to both databases
            sensor_conn = sqlite3.connect(str(self.db_path))
            research_conn = sqlite3.connect(str(self.research_db_path))

            sensor_cursor = sensor_conn.cursor()
            research_cursor = research_conn.cursor()

            # Get energy metrics from sensor database
            sensor_cursor.execute("""
                SELECT
                    energy as total_energy_kwh,
                    AVG(power) as avg_power,
                    MAX(power) as peak_power,
                    MIN(power) as min_power,
                    AVG(temperature) as avg_temp,
                    AVG(humidity) as avg_humidity,
                    AVG(illuminance) as avg_illuminance,
                    SUM(CASE WHEN occupancy = 1 THEN 1 ELSE 0 END) * ? / 3600.0 as occupied_hours
                FROM sensor_history
                WHERE timestamp >= ? AND timestamp < ?
            """, (UPDATE_INTERVAL_SECONDS, start_ts, end_ts))  # UPDATE_INTERVAL_SECONDS per reading
            energy_stats = sensor_cursor.fetchone()

            # Count occupancy per reading (could be multiple areas)
            sensor_cursor.execute("""
                SELECT AVG(occupied_count) FROM (
                    SELECT timestamp,
                           SUM(CASE WHEN occupancy = 1 THEN 1 ELSE 0 END) as occupied_count
                    FROM area_sensor_history
                    WHERE timestamp >= ? AND timestamp < ?
                    GROUP BY timestamp
                )
            """, (start_ts, end_ts))
            occupancy_result = sensor_cursor.fetchone()
            avg_occupancy = occupancy_result[0] if occupancy_result[0] else 0

            # Get tasks for the day
            sensor_cursor.execute("""
                SELECT
                    COUNT(*) as generated,
                    SUM(CASE WHEN completed = 1 THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN verified = 1 THEN 1 ELSE 0 END) as verified
                FROM daily_tasks
                WHERE date = ?
            """, (date,))
            task_stats = sensor_cursor.fetchone()

            # Get nudge stats from research database
            research_cursor.execute("""
                SELECT
                    COUNT(*) as sent,
                    SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END) as accepted,
                    SUM(CASE WHEN responded = 1 AND accepted = 0 THEN 1 ELSE 0 END) as dismissed,
                    SUM(CASE WHEN responded = 0 THEN 1 ELSE 0 END) as ignored
                FROM research_nudge_log
                WHERE DATE(timestamp, 'unixepoch') = ?
            """, (date,))
            nudge_stats = research_cursor.fetchone()
            if not nudge_stats or nudge_stats[0] is None:
                nudge_stats = (0, 0, 0, 0)

            # Get blocked notification count (active phase only)
            research_cursor.execute("""
                SELECT COUNT(*) FROM research_blocked_notifications
                WHERE DATE(timestamp, 'unixepoch') = ?
                  AND phase = 'active'
            """, (date,))
            blocked_result = research_cursor.fetchone()
            blocked_count = blocked_result[0] if blocked_result and blocked_result[0] else 0

            # Get average indices from research RL episodes
            research_cursor.execute("""
                SELECT
                    AVG(anomaly_index) as avg_anomaly,
                    AVG(behaviour_index) as avg_behaviour,
                    AVG(fatigue_index) as avg_fatigue
                FROM research_rl_episodes
                WHERE DATE(timestamp, 'unixepoch') = ?
            """, (date,))
            indices_stats = research_cursor.fetchone()
            if not indices_stats or indices_stats[0] is None:
                indices_stats = (0, 0, 0)

            # Insert aggregate
            research_cursor.execute("""
                INSERT OR REPLACE INTO research_daily_aggregates
                (date, phase, total_energy_kwh, avg_power_w, peak_power_w, min_power_w,
                 avg_temperature, avg_humidity, avg_illuminance,
                 avg_occupancy_count, total_occupied_hours,
                 tasks_generated, tasks_completed, tasks_verified,
                 nudges_sent, nudges_accepted, nudges_dismissed, nudges_ignored, nudges_blocked,
                 avg_anomaly_index, avg_behaviour_index, avg_fatigue_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (date, phase, energy_stats[0], energy_stats[1], energy_stats[2], energy_stats[3],
                   energy_stats[4], energy_stats[5], energy_stats[6], avg_occupancy, energy_stats[7],
                   *task_stats, *nudge_stats, blocked_count, *indices_stats))

            research_conn.commit()
            sensor_conn.close()
            research_conn.close()

            _LOGGER.info("Daily aggregates computed for %s", date)

        await self.hass.async_add_executor_job(_compute)

    async def compute_area_daily_aggregates(self, date: str = None, phase: str = None):
        """
        Compute and store area-specific daily aggregates for research analysis.
        Continuously updates today's aggregate for each area (INSERT OR REPLACE).
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Get all data for the day
        start_ts = datetime.strptime(date, "%Y-%m-%d").timestamp()
        end_ts = start_ts + 86400

        def _compute():
            # Connect to both databases
            sensor_conn = sqlite3.connect(str(self.db_path))
            research_conn = sqlite3.connect(str(self.research_db_path))

            sensor_cursor = sensor_conn.cursor()
            research_cursor = research_conn.cursor()

            # Get all areas from the area_sensor_history table
            sensor_cursor.execute("""
                SELECT DISTINCT area_name
                FROM area_sensor_history
                WHERE timestamp >= ? AND timestamp < ?
                AND area_name IS NOT NULL
            """, (start_ts, end_ts))

            areas = [row[0] for row in sensor_cursor.fetchall()]

            if not areas:
                _LOGGER.debug("No areas found for date %s", date)
                sensor_conn.close()
                research_conn.close()
                return

            # For each area, compute daily statistics
            for area_name in areas:
                # Get environmental and energy metrics for this area
                sensor_cursor.execute("""
                    SELECT
                        AVG(power) as avg_power,
                        MAX(power) as max_power,
                        MIN(power) as min_power,
                        AVG(temperature) as avg_temp,
                        AVG(humidity) as avg_humidity,
                        AVG(illuminance) as avg_illuminance,
                        SUM(CASE WHEN occupancy = 1 THEN 1 ELSE 0 END) * ? / 3600.0 as occupied_hours,
                        COUNT(*) as total_readings
                    FROM area_sensor_history
                    WHERE area_name = ? AND timestamp >= ? AND timestamp < ?
                """, (UPDATE_INTERVAL_SECONDS, area_name, start_ts, end_ts))

                area_stats = sensor_cursor.fetchone()

                if not area_stats or area_stats[7] == 0:  # total_readings
                    continue

                occupied_hours = area_stats[6] if area_stats[6] else 0
                total_readings = area_stats[7]
                total_hours = total_readings * (UPDATE_INTERVAL_SECONDS / 3600.0)
                occupancy_percentage = (occupied_hours / total_hours) * 100.0 if total_hours > 0 else 0

                # Insert or replace area aggregate
                research_cursor.execute("""
                    INSERT OR REPLACE INTO research_area_daily_stats
                    (date, area_name, phase,
                     avg_power_w, max_power_w, min_power_w,
                     avg_temperature, avg_humidity, avg_illuminance,
                     total_occupied_hours, occupancy_percentage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date, area_name, phase,
                    area_stats[0], area_stats[1], area_stats[2],  # power metrics
                    area_stats[3], area_stats[4], area_stats[5],  # environmental metrics
                    occupied_hours, occupancy_percentage
                ))

            research_conn.commit()
            sensor_conn.close()
            research_conn.close()

            _LOGGER.info("Area daily aggregates computed for %s (%d areas)", date, len(areas))

        await self.hass.async_add_executor_job(_compute)


    # ==================== CLEANUP ====================

    async def close(self):
        """Close database connection and cleanup."""
        if self._conn:
            def _close():
                self._conn.close()
            await self.hass.async_add_executor_job(_close)
            _LOGGER.info("Database connection closed")

    async def reset_all_data(self):
        """Reset all data (for testing/debugging)."""
        def _reset():
            # Clear sensor database
            if self.db_path.exists():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sensor_history")
                cursor.execute("DELETE FROM area_sensor_history")
                cursor.execute("DELETE FROM daily_tasks")
                cursor.execute("DELETE FROM task_difficulty_history")
                conn.commit()
                conn.close()

            # Clear research database
            if self.research_db_path.exists():
                conn = sqlite3.connect(str(self.research_db_path))
                cursor = conn.cursor()
                cursor.execute("DELETE FROM research_phase_metadata")
                cursor.execute("DELETE FROM research_daily_aggregates")
                cursor.execute("DELETE FROM research_rl_episodes")
                cursor.execute("DELETE FROM research_nudge_log")
                cursor.execute("DELETE FROM research_blocked_notifications")
                cursor.execute("DELETE FROM research_task_interactions")
                cursor.execute("DELETE FROM research_area_daily_stats")
                conn.commit()
                conn.close()

            # Clear state file
            if self.state_file.exists():
                self.state_file.unlink()

            _LOGGER.warning("All data reset (sensor DB + research DB + state)")

        await self.hass.async_add_executor_job(_reset)
