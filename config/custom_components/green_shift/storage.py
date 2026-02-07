"""
Storage management for Green Shift integration.
- SQLite: Temporal sensor data (14 days rolling window) with area-based tracking + Daily tasks
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
        
        self.db_path = self.config_dir / "sensor_data.db"
        self.state_file = self.config_dir / "state.json"
        
        self._conn = None
        self._lock = asyncio.Lock()
        
        _LOGGER.info("Storage initialized at: %s", self.config_dir)
    
    async def setup(self):
        """Setup database schema and load state."""
        await self._init_database()
        await self._cleanup_old_data()
    
    async def _init_database(self):
        """Initialize SQLite database with schema."""
        def _create_tables():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
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
    
    async def _cleanup_old_data(self):
        """Remove data older than 14 days."""
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
    
    # ==================== TEMPORAL DATA (SQLite) ====================
    
    async def store_sensor_snapshot(
        self,
        timestamp: datetime,
        power: float = None,
        energy: float = None,
        temperature: float = None,
        humidity: float = None,
        illuminance: float = None,
        occupancy: bool = None
    ):
        """Store a single sensor snapshot."""
        def _insert():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO sensor_history 
                (timestamp, power, energy, temperature, humidity, illuminance, occupancy)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp.timestamp(),
                power,
                energy,
                temperature,
                humidity,
                illuminance,
                1 if occupancy else 0 if occupancy is not None else None
            ))
            
            conn.commit()
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
        occupancy: bool = None
    ):
        """Store a sensor snapshot for a specific area."""
        def _insert():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO area_sensor_history 
                (timestamp, area_name, power, energy, temperature, humidity, illuminance, occupancy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp.timestamp(),
                area_name,
                power,
                energy,
                temperature,
                humidity,
                illuminance,
                1 if occupancy else 0 if occupancy is not None else None
            ))
            
            conn.commit()
            conn.close()
        
        await self.hass.async_add_executor_job(_insert)
    
    async def get_history(
        self,
        metric: str,
        hours: int = None,
        days: int = None
    ) -> List[Tuple[datetime, float]]:
        """
        Get historical data for a specific metric.
        
        Args:
            metric: One of 'power', 'energy', 'temperature', 'humidity', 'illuminance', 'occupancy'
            hours: Number of hours to retrieve (optional)
            days: Number of days to retrieve (optional)
        
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
        days: int = None
    ) -> List[Tuple[datetime, float]]:
        """
        Get historical data for a specific metric in a specific area.
        
        Args:
            area_name: Name of the area
            metric: One of 'temperature', 'humidity', 'illuminance', 'occupancy'
            hours: Number of hours to retrieve (optional)
            days: Number of days to retrieve (optional)
        
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
        days: int = None
    ) -> Dict[str, float]:
        """
        Get statistical summary for an area's metric.
        
        Returns:
            Dictionary with 'mean', 'min', 'max', 'std' keys
        """
        history = await self.get_area_history(area_name, metric, hours, days)
        
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
            
            return stats
        
        return await self.hass.async_add_executor_job(_query)
    
    # ==================== PERSISTENT STATE (JSON) ====================
    
    async def save_state(self, state_data: Dict[str, Any]):
        """
        Save persistent state to JSON.
        
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
            # Convert datetime objects to ISO strings
            serializable_state = {}
            for key, value in state_data.items():
                if isinstance(value, datetime):
                    serializable_state[key] = value.isoformat()
                else:
                    serializable_state[key] = value
            
            with open(self.state_file, 'w') as f:
                json.dump(serializable_state, f, indent=2)
            
            _LOGGER.debug("State saved to %s", self.state_file)
        
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
            # Clear database
            if self.db_path.exists():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()  
                cursor.execute("DELETE FROM sensor_history")
                cursor.execute("DELETE FROM area_sensor_history")
                cursor.execute("DELETE FROM daily_tasks")
                cursor.execute("DELETE FROM task_difficulty_history")
                conn.commit()
                conn.close()
            
            # Clear state file
            if self.state_file.exists():
                self.state_file.unlink()
            
            _LOGGER.warning("All data reset")
        
        await self.hass.async_add_executor_job(_reset)