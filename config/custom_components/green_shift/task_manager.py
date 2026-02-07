"""
Task Management System for Green Shift
Handles generation, verification, and difficulty adjustment of daily tasks
"""
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from homeassistant.core import HomeAssistant

from .const import TASK_GENERATION_TIME

_LOGGER = logging.getLogger(__name__)


class TaskManager:
    """Manages daily energy-saving tasks with automatic verification and difficulty adjustment."""
    
    def __init__(self, hass: HomeAssistant, sensors: dict, data_collector, storage):
        self.hass = hass
        self.sensors = sensors
        self.data_collector = data_collector
        self.storage = storage
        
        # Task difficulty adjustments (multipliers for targets)
        self.difficulty_multipliers = {
            1: 0.5,   # Easy (50% of ideal target)
            2: 0.75,  # Medium-Easy (75%)
            3: 1.0,   # Normal (100%)
            4: 1.25,  # Medium-Hard (125%)
            5: 1.5,   # Hard (150%)
        }
    
    async def generate_daily_tasks(self) -> List[Dict]:
        """
        Generates 3 verifiable daily tasks based on available sensors and historical data.
        Tasks are measurable and can be automatically verified.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Check if tasks already exist for today
        existing_tasks = await self.storage.get_today_tasks()
        if existing_tasks:
            _LOGGER.debug("Tasks already exist for today")
            return existing_tasks
        
        available_task_generators = []
        
        # Temperature-based tasks (verifiable via temperature sensors)
        if self.sensors.get("temperature"):
            available_task_generators.append(self._generate_temperature_task)
        
        # Power-based tasks (verifiable via power sensors)
        if self.sensors.get("power"):
            available_task_generators.append(self._generate_power_reduction_task)
            available_task_generators.append(self._generate_standby_power_task)
            available_task_generators.append(self._generate_peak_avoidance_task)
        
        # Illuminance-based tasks (verifiable via light sensors)
        if self.sensors.get("illuminance") and self.sensors.get("power"):
            available_task_generators.append(self._generate_daylight_task)
        
        # Occupancy-based tasks (verifiable via motion sensors)
        if self.sensors.get("occupancy") and self.sensors.get("power"):
            available_task_generators.append(self._generate_unoccupied_power_task)
        
        # TODO: ADD TASK GENERATORS THAT USE ENERGY DATA
        # Energy-based tasks (always available if we have energy sensors)
        # if self.sensors.get("energy"):
        #     # ADD ENERGY-BASED TASK GENERATORS HERE
        
        # Select 3 random task generators
        num_tasks = min(3, len(available_task_generators))
        if num_tasks == 0:
            _LOGGER.warning("No sensors available for task generation")
            return []
        
        selected_generators = np.random.choice(
            available_task_generators, 
            size=num_tasks, 
            replace=False
        )
        
        # Generate tasks
        tasks = []
        for generator in selected_generators:
            try:
                task = await generator()
                if task:
                    task['date'] = today
                    tasks.append(task)
            except Exception as e:
                _LOGGER.error("Error generating task with %s: %s", generator.__name__, e)
        
        # Save to database
        if tasks:
            await self.storage.save_daily_tasks(tasks)
            _LOGGER.info("Generated %d tasks for %s", len(tasks), today)
        
        return tasks
    
    async def _generate_temperature_task(self) -> Optional[Dict]:
        """Generate a temperature reduction task."""
        # Get difficulty stats for this task type
        stats = await self.storage.get_task_difficulty_stats('temperature_reduction')
        difficulty = await self._calculate_task_difficulty(stats)
        
        # Get average temperature from last 7 days
        temp_history = await self.data_collector.get_temperature_history(days=7)
        if not temp_history:
            return None
        
        temps = [temp for _, temp in temp_history]
        baseline_temp = np.mean(temps)
        
        # Calculate target based on difficulty (reduce by 0.5°C to 2°C)
        base_reduction = 1.0  # Base reduction in °C
        reduction = base_reduction * self.difficulty_multipliers[difficulty]
        target_temp = baseline_temp - reduction
        
        return {
            'task_id': f"temp_{datetime.now().strftime('%Y%m%d')}",
            'task_type': 'temperature_reduction',
            'title': f'Reduce Temperature by {reduction:.1f}°C',
            'description': f'Keep average temperature below {target_temp:.1f}°C today (current avg: {baseline_temp:.1f}°C)',
            'target_value': target_temp,
            'target_unit': '°C',
            'baseline_value': baseline_temp,
            'difficulty_level': difficulty,
            'area_name': None,  # Global task
        }
    
    async def _generate_power_reduction_task(self) -> Optional[Dict]:
        """Generate a power consumption reduction task."""
        stats = await self.storage.get_task_difficulty_stats('power_reduction')
        difficulty = await self._calculate_task_difficulty(stats)
        
        # Get average power from last 7 days
        power_history = await self.data_collector.get_power_history(days=7)
        if not power_history:
            return None
        
        powers = [power for _, power in power_history]
        baseline_power = np.mean(powers)
        
        # Calculate target based on difficulty (reduce by 3% to 15%)
        base_reduction_pct = 8.0  # Base reduction percentage
        reduction_pct = base_reduction_pct * self.difficulty_multipliers[difficulty]
        target_power = baseline_power * (1 - reduction_pct / 100)
        
        return {
            'task_id': f"power_{datetime.now().strftime('%Y%m%d')}",
            'task_type': 'power_reduction',
            'title': f'Reduce Power by {reduction_pct:.1f}%',
            'description': f'Keep average power below {target_power:.0f}W today (7-day avg: {baseline_power:.0f}W)',
            'target_value': target_power,
            'target_unit': 'W',
            'baseline_value': baseline_power,
            'difficulty_level': difficulty,
            'area_name': None,
        }
    
    async def _generate_standby_power_task(self) -> Optional[Dict]:
        """Generate a task to reduce standby power during night hours."""
        stats = await self.storage.get_task_difficulty_stats('standby_reduction')
        difficulty = await self._calculate_task_difficulty(stats)
        
        # Get power usage during night hours (00:00-06:00) from last 7 days
        power_history = await self.data_collector.get_power_history(days=7)
        if not power_history:
            return None
        
        night_powers = [power for timestamp, power in power_history if 0 <= timestamp.hour < 6]
        if not night_powers:
            return None
        
        baseline_night_power = np.mean(night_powers)
        
        # Target: reduce night power by 5% to 25%
        base_reduction_pct = 12.0
        reduction_pct = base_reduction_pct * self.difficulty_multipliers[difficulty]
        target_power = baseline_night_power * (1 - reduction_pct / 100)
        
        return {
            'task_id': f"standby_{datetime.now().strftime('%Y%m%d')}",
            'task_type': 'standby_reduction',
            'title': f'Reduce Night Power by {reduction_pct:.1f}%',
            'description': f'Keep power below {target_power:.0f}W during 00:00-06:00 (avg: {baseline_night_power:.0f}W)',
            'target_value': target_power,
            'target_unit': 'W',
            'baseline_value': baseline_night_power,
            'difficulty_level': difficulty,
            'area_name': None,
        }
    
    async def _generate_daylight_task(self) -> Optional[Dict]:
        """Generate a task to maximize natural daylight usage."""
        stats = await self.storage.get_task_difficulty_stats('daylight_usage')
        difficulty = await self._calculate_task_difficulty(stats)
        
        # Get power usage during daylight hours (08:00-17:00)
        power_history = await self.data_collector.get_power_history(days=7)
        if not power_history:
            return None
        
        day_powers = [power for timestamp, power in power_history if 8 <= timestamp.hour < 17]
        if not day_powers:
            return None
        
        baseline_day_power = np.mean(day_powers)
        
        # Target: reduce daytime power by 4% to 20% (easier because natural light helps)
        base_reduction_pct = 10.0
        reduction_pct = base_reduction_pct * self.difficulty_multipliers[difficulty]
        target_power = baseline_day_power * (1 - reduction_pct / 100)
        
        return {
            'task_id': f"daylight_{datetime.now().strftime('%Y%m%d')}",
            'task_type': 'daylight_usage',
            'title': f'Use Natural Light ({reduction_pct:.1f}% less power)',
            'description': f'Keep daytime power (08:00-17:00) below {target_power:.0f}W by using natural light',
            'target_value': target_power,
            'target_unit': 'W',
            'baseline_value': baseline_day_power,
            'difficulty_level': difficulty,
            'area_name': None,
        }
    
    async def _generate_unoccupied_power_task(self) -> Optional[Dict]:
        """Generate a task to minimize power in unoccupied rooms."""
        stats = await self.storage.get_task_difficulty_stats('unoccupied_power')
        difficulty = await self._calculate_task_difficulty(stats)
        
        # Find the area with highest average power
        areas = self.data_collector.get_all_areas()
        if not areas or len(areas) == 0:
            return None
        
        max_power = 0
        target_area = None
        
        for area in areas:
            if area == "No Area":
                continue
            area_stats = await self.storage.get_area_stats(area, "power", days=7)
            if area_stats['mean'] > max_power:
                max_power = area_stats['mean']
                target_area = area
        
        if not target_area or max_power == 0:
            return None
        
        # Target: reduce power in that area by 10% to 40%
        base_reduction_pct = 20.0
        reduction_pct = base_reduction_pct * self.difficulty_multipliers[difficulty]
        target_power = max_power * (1 - reduction_pct / 100)
        
        return {
            'task_id': f"unoccupied_{datetime.now().strftime('%Y%m%d')}",
            'task_type': 'unoccupied_power',
            'title': f'Turn Off Devices in {target_area}',
            'description': f'Reduce power in {target_area} to below {target_power:.0f}W when unoccupied',
            'target_value': target_power,
            'target_unit': 'W',
            'baseline_value': max_power,
            'difficulty_level': difficulty,
            'area_name': target_area,
        }
    
    async def _generate_peak_avoidance_task(self) -> Optional[Dict]:
        """Generate a task to avoid peak consumption hours."""
        stats = await self.storage.get_task_difficulty_stats('peak_avoidance')
        difficulty = await self._calculate_task_difficulty(stats)
        
        # Identify peak hour from last 7 days
        power_history = await self.data_collector.get_power_history(days=7)
        if not power_history:
            return None
        
        hourly_avg = {}
        for timestamp, power in power_history:
            hour = timestamp.hour
            if hour not in hourly_avg:
                hourly_avg[hour] = []
            hourly_avg[hour].append(power)
        
        # Find peak hour
        peak_hour = max(hourly_avg.items(), key=lambda x: np.mean(x[1]))[0]
        peak_power = np.mean(hourly_avg[peak_hour])
        
        # Target: reduce peak hour power by 8% to 30%
        base_reduction_pct = 15.0
        reduction_pct = base_reduction_pct * self.difficulty_multipliers[difficulty]
        target_power = peak_power * (1 - reduction_pct / 100)
        
        return {
            'task_id': f"peak_{datetime.now().strftime('%Y%m%d')}",
            'task_type': 'peak_avoidance',
            'title': f'Reduce Peak Hour Usage ({peak_hour:02d}:00)',
            'description': f'Keep power below {target_power:.0f}W during {peak_hour:02d}:00-{(peak_hour+1) % 24:02d}:00',
            'target_value': target_power,
            'target_unit': 'W',
            'baseline_value': peak_power,
            'difficulty_level': difficulty,
            'area_name': None,
        }
    
    async def _calculate_task_difficulty(self, stats: Dict) -> int:
        """
        Calculate appropriate difficulty level (1-5) based on user feedback history.
        
        Returns:
            int: Difficulty level from 1 (easiest) to 5 (hardest)
        """
        if not stats or stats.get('too_easy_count', 0) + stats.get('just_right_count', 0) + stats.get('too_hard_count', 0) < 3:
            # Not enough data, start at normal difficulty
            return 3
        
        # Apply suggested adjustment
        current_difficulty = int(stats.get('avg_difficulty', 3))
        adjustment = stats.get('suggested_adjustment', 0)
        new_difficulty = current_difficulty + adjustment
        
        # Clamp to valid range
        return max(1, min(5, new_difficulty))
    
    async def verify_tasks(self) -> Dict[str, bool]:
        """
        Verify today's tasks against actual sensor data.
        Returns dict mapping task_id to verification status.
        """
        tasks = await self.storage.get_today_tasks()
        if not tasks:
            return {}
        
        results = {}
        
        for task in tasks:
            if task['verified']:
                # Already verified
                results[task['task_id']] = True
                continue
            
            verified = await self._verify_single_task(task)
            results[task['task_id']] = verified
            
            if verified:
                await self.storage.mark_task_verified(task['task_id'], True)
                _LOGGER.info("Task %s verified successfully", task['task_id'])
        
        return results
    
    async def _verify_single_task(self, task: Dict) -> bool:
        """Verify a single task based on its type and target."""
        task_type = task['task_type']
        target_value = task['target_value']
        area_name = task.get('area_name')
        
        # Get today's data
        today_start = datetime.now().replace(hour=TASK_GENERATION_TIME[0], minute=TASK_GENERATION_TIME[1], second=TASK_GENERATION_TIME[2], microsecond=0)
        hours_passed = (datetime.now() - today_start).total_seconds() / 3600
        
        if hours_passed < 1:
            # Too early to verify
            return False
        
        try:
            if task_type == 'temperature_reduction':
                # Check average temperature today
                temp_history = await self.data_collector.get_temperature_history(hours=int(hours_passed))
                if not temp_history:
                    return False
                avg_temp = np.mean([temp for _, temp in temp_history])
                return avg_temp <= target_value
            
            elif task_type in ['power_reduction', 'daylight_usage']:
                # Check average power today
                power_history = await self.data_collector.get_power_history(hours=int(hours_passed))
                if not power_history:
                    return False
                
                if task_type == 'daylight_usage':
                    # Filter for daytime hours only
                    power_history = [(ts, p) for ts, p in power_history if 9 <= ts.hour < 17]
                
                if not power_history:
                    return False
                
                avg_power = np.mean([power for _, power in power_history])
                return avg_power <= target_value
            
            elif task_type == 'standby_reduction':
                # Check night power
                power_history = await self.data_collector.get_power_history(hours=int(hours_passed))
                night_powers = [p for ts, p in power_history if 0 <= ts.hour < 6]
                
                if not night_powers:
                    return False  # Can only verify after 6am
                
                avg_night_power = np.mean(night_powers)
                return avg_night_power <= target_value
            
            elif task_type == 'unoccupied_power':
                # Check area-specific power
                if not area_name:
                    return False
                
                area_history = await self.data_collector.get_area_history(
                    area_name, 'power', hours=int(hours_passed)
                )
                if not area_history:
                    return False
                
                avg_area_power = np.mean([power for _, power in area_history])
                return avg_area_power <= target_value
            
            elif task_type == 'peak_avoidance':
                # Extract peak hour from description or use baseline
                power_history = await self.data_collector.get_power_history(hours=int(hours_passed))
                if not power_history:
                    return False
                
                # For now, check if any hour exceeded the target
                # Group by hour
                hourly_powers = {}
                for ts, power in power_history:
                    hour = ts.hour
                    if hour not in hourly_powers:
                        hourly_powers[hour] = []
                    hourly_powers[hour].append(power)
                
                # Check all hours
                for hour_powers in hourly_powers.values():
                    if np.mean(hour_powers) > target_value:
                        return False
                
                return True
            
        except Exception as e:
            _LOGGER.error("Error verifying task %s: %s", task['task_id'], e)
            return False
        
        return False