"""
File: task_manager.py
Description: This module defines the TaskManager class, which is responsible for managing daily energy-saving tasks in the Green Shift Home Assistant component.
The TaskManager generates 3 verifiable tasks each day based on available sensors and historical data, with difficulty levels adjusted according to user feedback.
Tasks are designed to be automatically verifiable using sensor data, and the TaskManager includes methods for verifying task completion and adjusting future task difficulty based on user feedback.
"""

import logging
import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from homeassistant.core import HomeAssistant

from .const import TASK_GENERATION_TIME, ENVIRONMENT_OFFICE
from .translations_runtime import get_language, get_task_templates, get_difficulty_display, get_verification_reason_templates
from .helpers import should_ai_be_active, get_working_days_from_config

_LOGGER = logging.getLogger(__name__)


class TaskManager:
    """Manages daily energy-saving tasks with automatic verification and difficulty adjustment."""

    def __init__(self, hass: HomeAssistant, sensors: dict, data_collector, storage, decision_agent=None, config_data: dict = None):
        self.hass = hass
        self.sensors = sensors
        self.data_collector = data_collector
        self.storage = storage
        self.decision_agent = decision_agent  # For accessing phase and other agent state
        self.config_data = config_data or {}

        # In-memory cache of the most recent verification run results.
        # Schema per task_id:
        #   {
        #     "verified": bool,          # target achieved
        #     "failed": bool,            # checked and missed target
        #     "pending": bool,           # evaluation deferred (peak hour not reached)
        #     "checked_at": datetime,    # timestamp of last check attempt
        #     "actual_value": float|None,
        #     "target_value": float|None,
        #     "reason": str              # human-readable outcome explanation
        #   }
        self._last_verification_results: dict = {}

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

        Returns:
            List[Dict]: A list of generated tasks with details such as task_id, type, title, description, target values, difficulty level and area (if applicable).
        """
        # In office mode: only generate tasks on working days.
        if self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE:
            working_days = get_working_days_from_config(self.config_data)
            today_weekday = datetime.now().weekday()
            if today_weekday not in working_days:
                _LOGGER.debug("Not a working day (weekday=%d, working_days=%s) - task generation skipped",
                    today_weekday, working_days)
                return []

        today = datetime.now().strftime("%Y-%m-%d")

        # Task-streak: check the last day that could have had tasks before generating today
        # In home mode that is simply yesterday
        # In office mode, yesterday may be a non-working day (e.g. Monday -> yesterday = Sunday)
        # In that case we must walk backwards to the most-recent working day (e.g. Friday)
        if self.decision_agent:
            is_office_mode = self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE
            check_date = datetime.now().date() - timedelta(days=1)

            if is_office_mode:
                working_days = get_working_days_from_config(self.config_data)
                for offset in range(1, 8):
                    candidate = datetime.now().date() - timedelta(days=offset)
                    if candidate.weekday() in working_days:
                        check_date = candidate
                        break

            streak_tasks = await self.storage.get_tasks_for_date(check_date)
            if streak_tasks:
                any_done = any(t['verified'] for t in streak_tasks)
                if not any_done:
                    self.decision_agent.update_task_streak(False, check_date)

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
            available_task_generators.append(self._generate_peak_avoidance_task)

        # Illuminance-based tasks (verifiable via light sensors)
        if self.sensors.get("illuminance") and self.sensors.get("power"):
            available_task_generators.append(self._generate_daylight_task)

        # Occupancy-based tasks (verifiable via motion sensors)
        if self.sensors.get("occupancy") and self.sensors.get("power"):
            available_task_generators.append(self._generate_unoccupied_power_task)

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

            # Log task generation to research database
            current_state = self.data_collector.get_current_state()
            phase = self.decision_agent.phase if self.decision_agent else "unknown"

            for task in tasks:
                await self.storage.log_task_generation({
                    "task_id": task["task_id"],
                    "date": task["date"],
                    "phase": phase,
                    "task_type": task["task_type"],
                    "difficulty_level": task["difficulty_level"],
                    "target_value": task.get("target_value"),
                    "baseline_value": task.get("baseline_value"),
                    "area_name": task.get("area_name"),
                    "power_at_generation": current_state.get("power", 0),
                    "occupancy_at_generation": 1 if current_state.get("occupancy") else 0
                })

            _LOGGER.info("Generated %d tasks for %s", len(tasks), today)

        return tasks

    async def _generate_temperature_task(self) -> Optional[Dict]:
        """
        Generate a temperature reduction task.
        
        Returns:
            dict: A task d: juneictionary with details for a temperature reduction task, or None if it cannot be generated.
        """
        # Get difficulty stats for this task type
        stats = await self.storage.get_task_difficulty_stats('temperature_reduction')
        difficulty = await self._calculate_task_difficulty(stats)

        # Get average temperature from last 7 days (working hours only in office mode)
        is_office_mode = self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE
        working_hours_filter = True if is_office_mode else None

        temp_history = await self.data_collector.get_temperature_history(days=7, working_hours_only=working_hours_filter)
        if not temp_history:
            return None

        temps = [temp for _, temp in temp_history]
        baseline_temp = np.mean(temps)

        # Calculate target based on difficulty (adjust by 0.5°C to 2°C)
        base_reduction = 1.0  # Base adjustment in °C
        reduction = base_reduction * self.difficulty_multipliers[difficulty]
        target_temp = round(baseline_temp - reduction, 1)  # lower setpoint to reduce heating

        # Get user's language and templates
        language = await get_language(self.hass)
        templates = get_task_templates(language)
        template = templates['temperature_reduction']

        return {
            'task_id': f"temp_{datetime.now().strftime('%Y%m%d')}",
            'task_type': 'temperature_reduction',
            'title': template['title'].format(reduction=reduction),
            'description': template['description'].format(
                target_temp=target_temp,
                baseline_temp=round(baseline_temp, 1)
            ),
            'target_value': target_temp,
            'target_unit': '°C',
            'baseline_value': round(baseline_temp, 1),
            'difficulty_level': difficulty,
            'difficulty_display': get_difficulty_display(difficulty, language),
            'area_name': None,  # Global task
        }

    async def _generate_power_reduction_task(self) -> Optional[Dict]:
        """
        Generate a power consumption reduction task.
        
        Returns:
            dict: A task dictionary with details for a power reduction task, or None if it cannot be generated.
        """
        stats = await self.storage.get_task_difficulty_stats('power_reduction')
        difficulty = await self._calculate_task_difficulty(stats)

        # Get average power from last 7 days (working hours only in office mode)
        is_office_mode = self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE
        working_hours_filter = True if is_office_mode else None

        power_history = await self.data_collector.get_power_history(days=7, working_hours_only=working_hours_filter)
        if not power_history:
            return None

        powers = [power for _, power in power_history]
        baseline_power = np.mean(powers)

        # Calculate target based on difficulty (reduce by 3% to 15%)
        base_reduction_pct = 8.0  # Base reduction percentage
        reduction_pct = base_reduction_pct * self.difficulty_multipliers[difficulty]
        target_power = round(baseline_power * (1 - reduction_pct / 100))

        # Get user's language and templates
        language = await get_language(self.hass)
        templates = get_task_templates(language)
        template = templates['power_reduction']

        return {
            'task_id': f"power_{datetime.now().strftime('%Y%m%d')}",
            'task_type': 'power_reduction',
            'title': template['title'].format(reduction_pct=reduction_pct),
            'description': template['description'].format(
                target_power=target_power,
                baseline_power=round(baseline_power)
            ),
            'target_value': target_power,
            'target_unit': 'W',
            'baseline_value': round(baseline_power),
            'difficulty_level': difficulty,
            'difficulty_display': get_difficulty_display(difficulty, language),
            'area_name': None,
        }

    async def _generate_daylight_task(self) -> Optional[Dict]:
        """
        Generate a task to maximize natural daylight usage.
        
        Returns:
            dict: A task dictionary with details for a daylight usage task, or None if it cannot be generated.
        """
        stats = await self.storage.get_task_difficulty_stats('daylight_usage')
        difficulty = await self._calculate_task_difficulty(stats)

        # Get power usage during daylight hours (08:00-17:00) (working hours only in office mode)
        is_office_mode = self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE
        working_hours_filter = True if is_office_mode else None

        power_history = await self.data_collector.get_power_history(days=7, working_hours_only=working_hours_filter)
        if not power_history:
            return None

        day_powers = [power for timestamp, power in power_history if 8 <= timestamp.hour < 17]
        if not day_powers:
            return None

        baseline_day_power = np.mean(day_powers)

        # Target: reduce daytime power by 4% to 20% (easier because natural light helps)
        base_reduction_pct = 10.0
        reduction_pct = base_reduction_pct * self.difficulty_multipliers[difficulty]
        target_power = round(baseline_day_power * (1 - reduction_pct / 100))

        # Get user's language and templates
        language = await get_language(self.hass)
        templates = get_task_templates(language)
        template = templates['daylight_usage']

        return {
            'task_id': f"daylight_{datetime.now().strftime('%Y%m%d')}",
            'task_type': 'daylight_usage',
            'title': template['title'].format(reduction_pct=reduction_pct),
            'description': template['description'].format(target_power=target_power),
            'target_value': target_power,
            'target_unit': 'W',
            'baseline_value': round(baseline_day_power),
            'difficulty_level': difficulty,
            'difficulty_display': get_difficulty_display(difficulty, language),
            'area_name': None,
        }

    async def _generate_unoccupied_power_task(self) -> Optional[Dict]:
        """
        Generate a task to minimize power in unoccupied rooms.
        
        Returns:
            dict: A task dictionary with details for an unoccupied power task, or None if it cannot be generated.
        """
        stats = await self.storage.get_task_difficulty_stats('unoccupied_power')
        difficulty = await self._calculate_task_difficulty(stats)

        # In office mode restrict history to working hours, consistent with other generators.
        is_office_mode = self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE
        working_hours_filter = True if is_office_mode else None

        # Find the area with highest average power during unoccupied periods
        areas = self.data_collector.get_all_areas()
        if not areas or len(areas) == 0:
            return None

        max_power = 0
        target_area = None

        for area in areas:
            if area == "No Area":
                continue

            power_history = await self.data_collector.get_area_history(area, 'power', days=7, working_hours_only=working_hours_filter)
            if not power_history:
                continue

            occ_history = await self.data_collector.get_area_history(area, 'occupancy', days=7, working_hours_only=working_hours_filter)
            if occ_history:
                # Build occupancy lookup keyed by minute (both streams share the same snapshot interval)
                occ_by_minute = {
                    ts.replace(second=0, microsecond=0): bool(occ)
                    for ts, occ in occ_history
                }
                unoccupied_powers = [
                    p for ts, p in power_history
                    if not occ_by_minute.get(ts.replace(second=0, microsecond=0), False)
                ]
            else:
                # No occupancy data: fall back to all power readings
                unoccupied_powers = [p for _, p in power_history]

            if not unoccupied_powers:
                continue

            avg_unoccupied = np.mean(unoccupied_powers)
            if avg_unoccupied > max_power:
                max_power = avg_unoccupied
                target_area = area

        if not target_area or max_power == 0:
            return None

        # Target: reduce power in that area by 10% to 40%
        base_reduction_pct = 20.0
        reduction_pct = base_reduction_pct * self.difficulty_multipliers[difficulty]
        target_power = round(max_power * (1 - reduction_pct / 100))

        # Get user's language and templates
        language = await get_language(self.hass)
        templates = get_task_templates(language)
        template = templates['unoccupied_power']

        return {
            'task_id': f"unoccupied_{datetime.now().strftime('%Y%m%d')}",
            'task_type': 'unoccupied_power',
            'title': template['title'].format(target_area=target_area),
            'description': template['description'].format(
                target_area=target_area,
                target_power=target_power
            ),
            'target_value': target_power,
            'target_unit': 'W',
            'baseline_value': round(max_power),
            'difficulty_level': difficulty,
            'difficulty_display': get_difficulty_display(difficulty, language),
            'area_name': target_area,
        }

    async def _generate_peak_avoidance_task(self) -> Optional[Dict]:
        """
        Generate a task to avoid peak consumption hours.
        
        Returns:
            dict: A task dictionary with details for a peak avoidance task, or None if it cannot be generated.
        """
        stats = await self.storage.get_task_difficulty_stats('peak_avoidance')
        difficulty = await self._calculate_task_difficulty(stats)

        # Identify peak hour from last 7 days (working hours only in office mode)
        is_office_mode = self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE
        working_hours_filter = True if is_office_mode else None

        power_history = await self.data_collector.get_power_history(days=7, working_hours_only=working_hours_filter)
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
        target_power = round(peak_power * (1 - reduction_pct / 100))

        # Get user's language and templates
        language = await get_language(self.hass)
        templates = get_task_templates(language)
        template = templates['peak_avoidance']

        return {
            'task_id': f"peak_{datetime.now().strftime('%Y%m%d')}",
            'task_type': 'peak_avoidance',
            'title': template['title'].format(peak_hour=peak_hour),
            'description': template['description'].format(
                target_power=target_power,
                peak_hour=peak_hour,
                next_hour=(peak_hour+1) % 24
            ),
            'target_value': target_power,
            'target_unit': 'W',
            'baseline_value': round(peak_power),
            'peak_hour': peak_hour,  # Stored so verification checks this hour exclusively
            'difficulty_level': difficulty,
            'difficulty_display': get_difficulty_display(difficulty, language),
            'area_name': None,
        }

    async def _calculate_task_difficulty(self, stats: Dict) -> int:
        """
        Calculate appropriate difficulty level (1-5) based on user feedback history.

        Args:
            stats (Dict): A dictionary containing counts of 'too_easy', 'just_right', and 'too_hard' feedback, as well as any suggested adjustments.

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
        
        Returns:
            Dict[str, bool]: A dictionary mapping task_id to verification result (True if verified, False if not verified).
        """
        tasks = await self.storage.get_today_tasks()
        if not tasks:
            return {}

        results = {}
        _now = datetime.now()
        _lang = await get_language(self.hass)
        _reasons = get_verification_reason_templates(_lang)

        for task in tasks:
            task_id = task['task_id']
            target_value = task.get('target_value')
            target_unit = task.get('target_unit', '')
            peak_hour = task.get('peak_hour')

            if task['verified']:
                # Already verified – preserve or set the success entry in the result cache
                results[task_id] = True
                if task_id not in self._last_verification_results or not self._last_verification_results[task_id].get('verified'):
                    self._last_verification_results[task_id] = {
                        "verified": True,
                        "failed": False,
                        "pending": False,
                        "checked_at": _now,
                        "actual_value": task.get('completion_value'),
                        "target_value": target_value,
                        "reason": _reasons["target_achieved"]
                    }
                continue

            task_verified, actual_value, pending = await self._verify_single_task(task)
            results[task_id] = task_verified

            # Determine human-readable reason for the UI
            if task_verified:
                _reason = _reasons["target_achieved"]
            elif pending:
                if peak_hour is not None:
                    _reason = _reasons["waiting_for_peak_hour"].format(peak_hour=peak_hour)
                else:
                    _reason = _reasons["evaluation_deferred"]
            elif actual_value is not None:
                _task_type_now = task.get('task_type', '')
                if _task_type_now == 'temperature_reduction':
                    _reason = _reasons.get("temp_reduction_failed", _reasons["avg_above_target"]).format(
                        actual=actual_value, target=target_value
                    )
                elif _task_type_now == 'power_reduction':
                    _reason = _reasons.get("power_above_target", _reasons["avg_above_target"]).format(
                        actual=actual_value, target=target_value
                    )
                elif _task_type_now == 'daylight_usage':
                    _reason = _reasons.get("daylight_above_target", _reasons["avg_above_target"]).format(
                        actual=actual_value, target=target_value
                    )
                elif _task_type_now == 'unoccupied_power':
                    _reason = _reasons.get("unoccupied_above_target", _reasons["avg_above_target"]).format(
                        actual=actual_value, target=target_value
                    )
                elif _task_type_now == 'peak_avoidance':
                    _reason = _reasons.get("peak_above_target", _reasons["avg_above_target"]).format(
                        actual=actual_value, target=target_value
                    )
                else:
                    _reason = _reasons["avg_above_target"].format(
                        actual=actual_value, unit=target_unit, target=target_value
                    )
            else:
                _reason = _reasons["insufficient_data"]

            self._last_verification_results[task_id] = {
                "verified": task_verified,
                "failed": not task_verified and not pending,
                "pending": pending,
                "checked_at": _now,
                "actual_value": actual_value,
                "target_value": target_value,
                "reason": _reason
            }

            if task_verified:
                await self.storage.mark_task_verified(task_id, True)

                # Also log completion to research database with the actual measured value
                await self.storage.log_task_completion(
                    task_id,
                    completion_value=actual_value
                )

                _LOGGER.info("Task %s verified successfully (measured: %s)", task_id, actual_value)

        # Task-streak: credit streak as soon as at least one task is verified today
        if self.decision_agent and tasks and any(results.values()):
            self.decision_agent.update_task_streak(True, datetime.now().date())

        return results

    async def _verify_single_task(self, task: Dict) -> tuple:
        """
        Verify a single task based on its type and target.
        
        Args:
            task (Dict): The task dictionary containing details for verification.

        Returns:
            tuple: (verified: bool, actual_value: Optional[float], pending: bool) where 'verified' indicates if the task target was achieved, 
            'actual_value' is the measured value used for verification (if available) and 'pending' indicates if evaluation is deferred (e.g., peak hour not reached).
        """
        task_type = task['task_type']
        target_value = task['target_value']
        area_name = task.get('area_name')

        # Get today's data. Use the task's created_at as the starting point to ensure we only evaluate data from the day the task was generated.
        created_at_str = task.get('created_at')
        if created_at_str:
            try:
                today_start = datetime.fromisoformat(str(created_at_str))
            except (ValueError, TypeError):
                today_start = datetime.now().replace(
                    hour=TASK_GENERATION_TIME[0], minute=TASK_GENERATION_TIME[1],
                    second=TASK_GENERATION_TIME[2], microsecond=0
                )
        else:
            today_start = datetime.now().replace(
                hour=TASK_GENERATION_TIME[0], minute=TASK_GENERATION_TIME[1],
                second=TASK_GENERATION_TIME[2], microsecond=0
            )
        hours_passed = (datetime.now() - today_start).total_seconds() / 3600

        if hours_passed < 1:
            # Too early to verify
            return False, None, False

        try:
            if task_type == 'temperature_reduction':
                # Check average temperature today (working hours only in office mode)
                is_office_mode = self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE
                working_hours_filter = True if is_office_mode else None

                temp_history = await self.data_collector.get_temperature_history(hours=math.ceil(hours_passed), working_hours_only=working_hours_filter)
                if not temp_history:
                    return False, None, False
                avg_temp = np.mean([temp for _, temp in temp_history])
                _LOGGER.debug("Average temperature for verification: %.1f°C, target: %.1f°C", avg_temp, target_value)
                verified = bool(avg_temp <= target_value)
                return verified, round(avg_temp, 2), False

            elif task_type in ['power_reduction', 'daylight_usage']:
                # Check average power today (working hours only in office mode)
                is_office_mode = self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE
                working_hours_filter = True if is_office_mode else None

                power_history = await self.data_collector.get_power_history(
                    hours=math.ceil(hours_passed), working_hours_only=working_hours_filter
                )
                if not power_history:
                    return False, None, False

                if task_type == 'daylight_usage':
                    # Filter for daytime hours only
                    power_history = [(ts, p) for ts, p in power_history if 8 <= ts.hour < 17]

                if not power_history:
                    return False, None, False

                avg_power = np.mean([power for _, power in power_history])
                _LOGGER.debug("Average power for verification: %.2fW, target: %.2fW", avg_power, target_value)
                return avg_power <= target_value, round(avg_power, 2), False

            elif task_type == 'unoccupied_power':
                # Check area-specific power during unoccupied periods only
                if not area_name:
                    return False, None, False

                # Apply working-hours filter in office mode for consistency with task generation.
                is_office_mode = self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE
                working_hours_filter = True if is_office_mode else None

                area_power_history = await self.data_collector.get_area_history(
                    area_name, 'power', hours=math.ceil(hours_passed), working_hours_only=working_hours_filter
                )
                if not area_power_history:
                    return False, None, False

                area_occ_history = await self.data_collector.get_area_history(
                    area_name, 'occupancy', hours=math.ceil(hours_passed), working_hours_only=working_hours_filter
                )
                if area_occ_history:
                    occ_by_minute = {
                        ts.replace(second=0, microsecond=0): bool(occ)
                        for ts, occ in area_occ_history
                    }
                    unoccupied_powers = [
                        p for ts, p in area_power_history
                        if not occ_by_minute.get(ts.replace(second=0, microsecond=0), False)
                    ]
                else:
                    unoccupied_powers = [p for _, p in area_power_history]

                if not unoccupied_powers:
                    return False, None, False

                avg_unoccupied_power = np.mean(unoccupied_powers)
                _LOGGER.debug(
                    "Average unoccupied power in area %s for verification: %.2fW, target: %.2fW",
                    area_name, avg_unoccupied_power, target_value
                )
                return avg_unoccupied_power <= target_value, round(avg_unoccupied_power, 2), False

            elif task_type == 'peak_avoidance':
                # Use the peak_hour stored at task generation: only that hour is evaluated
                peak_hour = task.get('peak_hour')
                if peak_hour is None:
                    _LOGGER.warning("peak_avoidance task %s missing peak_hour field; cannot verify", task.get('task_id'))
                    return False, None, False

                is_office_mode = self.config_data.get("environment_mode") == ENVIRONMENT_OFFICE
                working_hours_filter = True if is_office_mode else None
                power_history = await self.data_collector.get_power_history(hours=math.ceil(hours_passed), working_hours_only=working_hours_filter)
                if not power_history:
                    return False, None, False

                peak_hour_powers = [p for ts, p in power_history if ts.hour == peak_hour]
                if not peak_hour_powers:
                    # Peak hour has not been reached yet today – result is deferred, not failed
                    _LOGGER.debug("Peak avoidance task %s: peak hour %02d:00 not reached yet; deferring",
                                  task.get('task_id'), peak_hour)
                    return False, None, True  # pending=True

                avg_peak = np.mean(peak_hour_powers)
                _LOGGER.debug(
                    "Peak avoidance: hour %02d average %.2fW vs target %.2fW",
                    peak_hour, avg_peak, target_value
                )
                return avg_peak <= target_value, round(avg_peak, 2), False

        except Exception as e:
            _LOGGER.error("Error verifying task %s: %s", task['task_id'], e)
            return False, None, False

        return False, None, False
