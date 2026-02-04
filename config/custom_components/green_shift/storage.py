"""
Storage management for Green Shift integration.
- SQLite: Temporal sensor data (14 days rolling window)
- JSON: Persistent state (AI configuration, indices, Q-table)
"""
import logging
import sqlite3
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
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
            
            # Index for faster queries by timestamp
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON sensor_history(timestamp)
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
            
            cursor.execute(
                "DELETE FROM sensor_history WHERE timestamp < ?",
                (cutoff,)
            )
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted > 0:
                _LOGGER.info("Cleaned up %d old records", deleted)
        
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
                conn.commit()
                conn.close()
            
            # Clear state file
            if self.state_file.exists():
                self.state_file.unlink()
            
            _LOGGER.warning("All data reset")
        
        await self.hass.async_add_executor_job(_reset)