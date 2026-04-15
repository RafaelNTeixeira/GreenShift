"""
clean_research_data.py
======================
Data cleaning script for the energy-saving research project (Green Shift / Gamification + AI).

Cleaned tables:
    - research_area_daily_stats
    - research_blocked_notifications
    - research_daily_aggregates
    - research_nudge_log
    - research_phase_metadata
    - research_rl_episodes
    - research_task_interactions
    - research_weekly_challenges

Usage:
    python clean_research_data.py [input_dir] [output_dir]

Examples:
    python clean_research_data.py                                    # uses ./ and ./cleaned_data/
    python clean_research_data.py ./research_export                  # uses custom input dir
    python clean_research_data.py ./research_export/ ./cleaned_data  # custom input and output dirs
"""

from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------------------------------------------------------------------
# CONSTANTS / CONFIGURATION
# ------------------------------------------------------------------------------

VALID_PHASES       = {"baseline", "active"}
# Matches const.py: ACTIONS = {"noop":0, "specific":1, "anomaly":2, "behavioural":3, "normative":4}
VALID_ACTION_NAMES  = {"noop", "specific", "anomaly", "behavioural", "normative"}
VALID_ACTION_SOURCES = {"shadow_explore", "shadow_exploit", "explore", "exploit",
                        "expired"}    # expired = episode expired without action
# Block reasons from decision_agent.py _decide_action()
VALID_BLOCK_REASONS = {"cooldown", "fatigue_threshold", "no_opportunity", "threshold"}
VALID_WEATHER      = {"sunny", "cloudy", "rainy", "windy", "foggy", "snowy", "partly_cloudy"}

# Unix timestamp columns (seconds since epoch) -> convert to UTC datetime
UNIX_TS_COLS = {
    "research_blocked_notifications": ["timestamp"],
    "research_nudge_log":             ["timestamp", "response_timestamp"],
    "research_phase_metadata":        ["start_timestamp", "end_timestamp"],
    "research_rl_episodes":           ["timestamp"],
    "research_task_interactions":     [
        "generation_timestamp", "first_view_timestamp", "completion_timestamp"
    ],
}

# Columns with JSON strings (dictionary or list) -> expand into new columns
JSON_COLS = {
    "research_blocked_notifications": ["action_mask", "state_vector"],
    "research_nudge_log":             ["state_vector"],
    "research_rl_episodes":           ["state_vector", "q_values", "action_mask"],
}


# ------------------------------------------------------------------------------
# SHARED UTILITIES
# ------------------------------------------------------------------------------

class CleaningReport:
    """Accumulates metrics and warnings during cleaning and prints a final report."""

    def __init__(self):
        self.tables: dict[str, dict] = {}

    def register(self, table: str, original_rows: int, cleaned_rows: int,
                 issues: list[str], dropped_cols: list[str],
                 added_cols: list[str]):
        self.tables[table] = {
            "original_rows":  original_rows,
            "cleaned_rows":   cleaned_rows,
            "rows_removed":   original_rows - cleaned_rows,
            "issues":         issues,
            "dropped_cols":   dropped_cols,
            "added_cols":     added_cols,
        }

    def print_summary(self):
        sep = "-" * 80
        print(f"\n{'═' * 80}")
        print("  DATA CLEANING REPORT")
        print(f"{'═' * 80}")
        for table, info in self.tables.items():
            print(f"\n📋  {table}")
            print(sep)
            print(f"    Rows: {info['original_rows']} -> {info['cleaned_rows']}"
                  f"  (removed: {info['rows_removed']})")
            if info["dropped_cols"]:
                print(f"    Removed columns : {', '.join(info['dropped_cols'])}")
            if info["added_cols"]:
                print(f"    Created columns : {', '.join(info['added_cols'])}")
            for issue in info["issues"]:
                print(f"    ⚠  {issue}")
        print(f"\n{'═' * 80}\n")


report = CleaningReport()


def unix_to_datetime(series: pd.Series, col_name: str = "") -> pd.Series:
    """Converts Unix timestamp column (float, seconds) to UTC datetime."""
    return pd.to_datetime(series, unit="s", utc=True, errors="coerce")


def parse_json_col_to_dict(series: pd.Series) -> pd.Series:
    """Attempts to parse JSON in each cell; returns NaN on failure."""
    def _parse(val):
        if pd.isna(val):
            return np.nan
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return np.nan
    return series.apply(_parse)


def expand_state_vector(df: pd.DataFrame, col: str = "state_vector",
                        prefix: str = "sv") -> pd.DataFrame:
    """
    Expands a state_vector (JSON list of 18 floats) into named columns.

    Exact mapping from _build_state_vector() in decision_agent.py:
        [0]  global_power_w          - current total consumption (W)
        [1]  has_power_data          - 1.0 if power > 0
        [2]  top_consumer_w          - consumption of the highest load device (W)
        [3]  has_individual_power    - 1.0 if top_consumer > 0
        [4]  temperature_c           - global average temperature (°C)
        [5]  has_temperature         - 1.0 if temperature sensor present
        [6]  humidity_pct            - global average humidity (%)
        [7]  has_humidity            - 1.0 if humidity sensor present
        [8]  illuminance_lx          - global average illuminance (lx)
        [9]  has_illuminance         - 1.0 if illuminance sensor present
        [10] occupancy               - 1.0 if space is occupied
        [11] has_occupancy_sensors   - 1.0 if occupancy sensors configured
        [12] anomaly_index           - anomaly index (0-1)
        [13] behaviour_index         - behavior index (0-1)
        [14] fatigue_index           - fatigue index (0-1)
        [15] area_anomaly_count      - number of areas with significant anomaly
        [16] time_of_day_norm        - normalized time of day (0-1)
        [17] day_of_week_norm        - normalized day of week (0-1, 0=Monday)
    """
    STATE_NAMES = [
        "global_power_w",
        "has_power_data",
        "top_consumer_w",
        "has_individual_power",
        "temperature_c",
        "has_temperature",
        "humidity_pct",
        "has_humidity",
        "illuminance_lx",
        "has_illuminance",
        "occupancy",
        "has_occupancy_sensors",
        "anomaly_index",
        "behaviour_index",
        "fatigue_index",
        "area_anomaly_count",
        "time_of_day_norm",
        "day_of_week_norm",
    ]
    parsed = parse_json_col_to_dict(df[col])
    rows = []
    for val in parsed:
        if isinstance(val, list) and len(val) == len(STATE_NAMES):
            rows.append(dict(zip(STATE_NAMES, val)))
        else:
            rows.append({n: np.nan for n in STATE_NAMES})
    expanded = pd.DataFrame(rows, index=df.index)
    expanded.columns = [f"{prefix}_{c}" for c in expanded.columns]
    return expanded


def expand_action_mask(df: pd.DataFrame, col: str = "action_mask",
                       prefix: str = "mask") -> pd.DataFrame:
    """
    Expands action_mask (JSON dict {"0": bool, ..., "4": bool}) into 5 boolean columns.
    Exact mapping from const.py ACTIONS:
        0=noop, 1=specific, 2=anomaly, 3=behavioural, 4=normative
    """
    ACTION_LABELS = {
        "0": "noop", "1": "specific",
        "2": "anomaly", "3": "behavioural", "4": "normative"
    }
    parsed = parse_json_col_to_dict(df[col])
    rows = []
    for val in parsed:
        if isinstance(val, dict):
            rows.append({f"{prefix}_{lbl}": val.get(k, np.nan)
                         for k, lbl in ACTION_LABELS.items()})
        else:
            rows.append({f"{prefix}_{lbl}": np.nan
                         for lbl in ACTION_LABELS.values()})
    return pd.DataFrame(rows, index=df.index)


def expand_q_values(df: pd.DataFrame, col: str = "q_values",
                    prefix: str = "q") -> pd.DataFrame:
    """
    Expands q_values (JSON dict {"0": float, ..., "4": float}) into 5 columns.
    Exact mapping from const.py ACTIONS:
        0=noop, 1=specific, 2=anomaly, 3=behavioural, 4=normative
    """
    ACTION_LABELS = {
        "0": "noop", "1": "specific",
        "2": "anomaly", "3": "behavioural", "4": "normative"
    }
    parsed = parse_json_col_to_dict(df[col])
    rows = []
    for val in parsed:
        if isinstance(val, dict):
            rows.append({f"{prefix}_{lbl}": float(val.get(k, np.nan))
                         for k, lbl in ACTION_LABELS.items()})
        else:
            rows.append({f"{prefix}_{lbl}": np.nan
                         for lbl in ACTION_LABELS.values()})
    return pd.DataFrame(rows, index=df.index)


def flag_invalid_categories(df: pd.DataFrame, col: str,
                             valid_set: set, issues: list[str]) -> pd.Series:
    """Marks values outside a valid set as NaN and registers a warning."""
    invalid_mask = ~df[col].isin(valid_set) & df[col].notna()
    n_invalid = invalid_mask.sum()
    if n_invalid:
        issues.append(
            f"Column '{col}': {n_invalid} invalid value(s) -> converted to NaN "
            f"({df.loc[invalid_mask, col].unique().tolist()})"
        )
    return df[col].where(~invalid_mask, other=np.nan)


def coerce_boolean_flag(df: pd.DataFrame, col: str, issues: list[str]) -> pd.Series:
    """
    Safely converts a boolean flag column (0/1) to numeric.

    When the column contains mixed values (e.g., a blank cell causes
    pandas to read the whole column as str), isin([0, 1]) fails for all rows.
    This function first coerces to numeric and then validates.
    """
    numeric = pd.to_numeric(df[col], errors="coerce")
    # Values that were non-null in the original but became NaN after coercion
    # are invalid (e.g., non-numeric strings, spaces)
    originally_present = df[col].notna() & (df[col].astype(str).str.strip() != "")
    became_nan = originally_present & numeric.isna()
    # Numeric values but outside {0, 1}
    out_of_range = numeric.notna() & ~numeric.isin([0, 1])
    n_invalid = (became_nan | out_of_range).sum()
    if n_invalid:
        issues.append(
            f"Column '{col}': {n_invalid} invalid value(s) outside {{0, 1}} -> NaN"
        )
    result = numeric.copy()
    result[became_nan | out_of_range] = np.nan
    return result


def validate_range(df: pd.DataFrame, col: str, lo: float, hi: float,
                   issues: list[str]) -> pd.Series:
    """Converts values outside [lo, hi] to NaN and registers a warning."""
    mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
    n_out = mask.sum()
    if n_out:
        issues.append(
            f"Column '{col}': {n_out} value(s) outside the range [{lo}, {hi}] "
            f"-> converted to NaN"
        )
    return df[col].where(~mask, other=np.nan)


# ------------------------------------------------------------------------------
# CLEANING PER TABLE
# ------------------------------------------------------------------------------

def clean_area_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    research_area_daily_stats
    -------------------------
    Records daily statistics per area (Main Lab, Server Room, …).
    Known issues:
      - All power and environment columns are NaN (missing sensors in the
        first records or hardware integration error).
      - occupancy_percentage can be > 100 due to counting errors.
    """
    issues:       list[str] = []
    dropped_cols: list[str] = []
    added_cols:   list[str] = []
    original_rows = len(df)

    # -- 1. Timestamps --------------------------------------------------------
    df["date"]       = pd.to_datetime(df["date"],       errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    added_cols += ["date (datetime)", "created_at (datetime)"]

    # -- 2. Categories --------------------------------------------------------
    df["phase"] = flag_invalid_categories(df, "phase", VALID_PHASES, issues)

    # -- 3. Numeric ranges ----------------------------------------------------
    df["occupancy_percentage"] = validate_range(
        df, "occupancy_percentage", 0, 100, issues
    )
    for col in ["avg_power_w", "max_power_w", "min_power_w"]:
        if col in df.columns:
            df[col] = validate_range(df, col, 0, 1e6, issues)

    # -- 4. Power min/max consistency -----------------------------------------
    if {"min_power_w", "max_power_w"}.issubset(df.columns):
        bad_minmax = (
            df["min_power_w"].notna()
            & df["max_power_w"].notna()
            & (df["min_power_w"] > df["max_power_w"])
        )
        if bad_minmax.any():
            issues.append(
                f"{bad_minmax.sum()} row(s) with min_power_w > max_power_w -> "
                "values swapped automatically"
            )
            df.loc[bad_minmax, ["min_power_w", "max_power_w"]] = \
                df.loc[bad_minmax, ["max_power_w", "min_power_w"]].values

    # -- 5. 100% NaN columns: keep but flag -----------------------------------
    fully_null = [c for c in df.columns
                  if df[c].isna().all() and c not in ("date", "created_at")]
    if fully_null:
        issues.append(
            f"Columns without any value (100% NaN) - missing sensor data: "
            f"{fully_null}"
        )

    # -- 6. Auxiliary column: has sensor data? --------------------------------
    df["has_sensor_data"] = df["avg_power_w"].notna().astype(int)
    added_cols.append("has_sensor_data")

    # -- 7. Sort --------------------------------------------------------------
    df = df.sort_values(["date", "area_name"]).reset_index(drop=True)

    report.register("research_area_daily_stats", original_rows, len(df),
                    issues, dropped_cols, added_cols)
    return df


def clean_blocked_notifications(df: pd.DataFrame) -> pd.DataFrame:
    """
    research_blocked_notifications
    -------------------------------
    Records notifications that the RL agent decided NOT to send (and why).
    Known issues:
      - timestamp is in Unix float -> convert to datetime.
      - action_mask and state_vector are JSON strings -> expand.
    """
    issues:       list[str] = []
    dropped_cols: list[str] = []
    added_cols:   list[str] = []
    original_rows = len(df)

    # -- 1. Timestamps --------------------------------------------------------
    df["timestamp"]  = unix_to_datetime(df["timestamp"],  "timestamp")
    df["created_at"] = pd.to_datetime(df["created_at"],   errors="coerce")

    # -- 2. Categories --------------------------------------------------------
    df["phase"]        = flag_invalid_categories(df, "phase", VALID_PHASES, issues)
    df["block_reason"] = flag_invalid_categories(
        df, "block_reason", VALID_BLOCK_REASONS, issues
    )

    # -- 3. Ranges ------------------------------------------------------------
    df["opportunity_score"]  = validate_range(df, "opportunity_score",  0, 1,    issues)
    df["fatigue_index"]      = validate_range(df, "fatigue_index",      0, 1,    issues)
    df["behaviour_index"]    = validate_range(df, "behaviour_index",    0, 1,    issues)
    df["anomaly_index"]      = validate_range(df, "anomaly_index",      0, 1,    issues)
    df["current_power"]      = validate_range(df, "current_power",      0, 1e6,  issues)
    df["time_of_day_hour"]   = validate_range(df, "time_of_day_hour",   0, 23,   issues)

    # -- 4. JSON Expansion ----------------------------------------------------
    sv_expanded = expand_state_vector(df, "state_vector", prefix="sv")
    mask_expanded = expand_action_mask(df, "action_mask", prefix="mask")

    df = df.drop(columns=["state_vector", "action_mask"])
    dropped_cols += ["state_vector", "action_mask"]
    df = pd.concat([df, sv_expanded, mask_expanded], axis=1)
    added_cols += list(sv_expanded.columns) + list(mask_expanded.columns)

    # -- 5. Sort --------------------------------------------------------------
    df = df.sort_values("timestamp").reset_index(drop=True)

    report.register("research_blocked_notifications", original_rows, len(df),
                    issues, dropped_cols, added_cols)
    return df


def clean_daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    research_daily_aggregates
    --------------------------
    Aggregates daily energy, tasks, and notifications metrics.
    Known issues:
      - avg_power_off_hours: 100% NaN (possibly not calculated yet).
      - avg_temperature/humidity/illuminance: 100% NaN (missing environment sensors).
      - tasks_completed/verified: NaN when tasks_generated == 0 -> fill with 0.
      - nudges_accepted/dismissed/ignored: NaN when nudges_sent == 0 -> fill with 0.
    """
    issues:       list[str] = []
    dropped_cols: list[str] = []
    added_cols:   list[str] = []
    original_rows = len(df)

    # -- 1. Timestamps --------------------------------------------------------
    df["date"]       = pd.to_datetime(df["date"],       errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # -- 2. Categories --------------------------------------------------------
    df["phase"]             = flag_invalid_categories(df, "phase", VALID_PHASES, issues)
    df["weather_condition"] = flag_invalid_categories(
        df, "weather_condition", VALID_WEATHER, issues
    )

    # -- 3. Count imputation: NaN -> 0 when total is 0 ------------------------
    #   Tasks
    tasks_no_activity = df["tasks_generated"] == 0
    for col in ["tasks_completed", "tasks_verified"]:
        n_filled = (df[col].isna() & tasks_no_activity).sum()
        if n_filled:
            df.loc[tasks_no_activity & df[col].isna(), col] = 0.0
            issues.append(
                f"'{col}': {n_filled} NaN filled with 0 "
                "(tasks_generated == 0)"
            )
    #   Nudges
    nudge_no_activity = df["nudges_sent"] == 0
    for col in ["nudges_accepted", "nudges_dismissed", "nudges_ignored"]:
        n_filled = (df[col].isna() & nudge_no_activity).sum()
        if n_filled:
            df.loc[nudge_no_activity & df[col].isna(), col] = 0.0
            issues.append(
                f"'{col}': {n_filled} NaN filled with 0 "
                "(nudges_sent == 0)"
            )

    # -- 4. Numeric ranges ----------------------------------------------------
    df["avg_occupancy_count"]  = validate_range(df, "avg_occupancy_count",  0, 1e4,  issues)
    df["avg_fatigue_index"]    = validate_range(df, "avg_fatigue_index",    0, 1,    issues)
    df["avg_behaviour_index"]  = validate_range(df, "avg_behaviour_index",  0, 1,    issues)
    df["avg_anomaly_index"]    = validate_range(df, "avg_anomaly_index",    0, 1,    issues)
    df["total_energy_kwh"]     = validate_range(df, "total_energy_kwh",     0, 1e6,  issues)

    # -- 5. Consistency: tasks_completed ≤ tasks_generated --------------------
    if "tasks_completed" in df.columns:
        bad = (
            df["tasks_completed"].notna()
            & (df["tasks_completed"] > df["tasks_generated"])
        )
        if bad.any():
            issues.append(
                f"{bad.sum()} row(s) with tasks_completed > tasks_generated"
            )

    # -- 6. Auxiliary column: nudge acceptance rate ---------------------------
    df["nudge_acceptance_rate"] = np.where(
        df["nudges_sent"] > 0,
        (df["nudges_accepted"].fillna(0) / df["nudges_sent"]).round(4),
        np.nan
    )
    added_cols.append("nudge_acceptance_rate")

    # -- 7. 100% NaN columns: flag --------------------------------------------
    fully_null = [c for c in df.columns
                  if df[c].isna().all() and c not in ("date", "created_at")]
    if fully_null:
        issues.append(f"Columns without any value (100% NaN): {fully_null}")

    # -- 8. Duplicates by date ------------------------------------------------
    dups = df.duplicated(subset=["date"], keep=False)
    if dups.any():
        issues.append(
            f"{dups.sum()} row(s) with duplicated date -> kept the last one by date"
        )
        df = df.sort_values("created_at").drop_duplicates(subset=["date"], keep="last")

    # -- 9. Sort --------------------------------------------------------------
    df = df.sort_values("date").reset_index(drop=True)

    report.register("research_daily_aggregates", original_rows, len(df),
                    issues, dropped_cols, added_cols)
    return df


def clean_nudge_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    research_nudge_log
    ------------------
    Log of all notifications sent to the user.
    Known issues:
      - response_timestamp and response_time_seconds: 100% NaN because no
        notification was responded to (expected behavior at the beginning).
      - state_vector: JSON string -> expand.
      - notification_id might have duplicates (resends).
    """
    issues:       list[str] = []
    dropped_cols: list[str] = []
    added_cols:   list[str] = []
    original_rows = len(df)

    # -- 1. Timestamps --------------------------------------------------------
    df["timestamp"]          = unix_to_datetime(df["timestamp"],          "timestamp")
    df["response_timestamp"] = unix_to_datetime(df["response_timestamp"], "response_timestamp")
    df["created_at"]         = pd.to_datetime(df["created_at"],           errors="coerce")

    # -- 2. Categories --------------------------------------------------------
    df["phase"]       = flag_invalid_categories(df, "phase",       VALID_PHASES,       issues)
    df["action_type"] = flag_invalid_categories(df, "action_type", VALID_ACTION_NAMES, issues)

    # -- 3. Boolean flags -----------------------------------------------------
    for col in ["responded", "accepted"]:
        df[col] = coerce_boolean_flag(df, col, issues)

    # -- 4. Consistency: accepted ≤ responded ---------------------------------
    bad = df["accepted"].notna() & df["responded"].notna() & \
          (df["accepted"] > df["responded"])
    if bad.any():
        issues.append(
            f"{bad.sum()} row(s) with accepted=1 but responded=0 (inconsistent)"
        )

    # -- 5. response_time_seconds ≥ 0 -----------------------------------------
    if "response_time_seconds" in df.columns:
        df["response_time_seconds"] = validate_range(
            df, "response_time_seconds", 0, 86400, issues
        )

    # -- 6. RL Indices --------------------------------------------------------
    df["anomaly_index"]   = validate_range(df, "anomaly_index",   0, 1, issues)
    df["behaviour_index"] = validate_range(df, "behaviour_index", 0, 1, issues)
    df["fatigue_index"]   = validate_range(df, "fatigue_index",   0, 1, issues)

    # -- 7. Expand state_vector -----------------------------------------------
    sv_expanded = expand_state_vector(df, "state_vector", prefix="sv")
    df = df.drop(columns=["state_vector"])
    dropped_cols.append("state_vector")
    df = pd.concat([df, sv_expanded], axis=1)
    added_cols += list(sv_expanded.columns)

    # -- 8. Auxiliary column: response time in minutes ------------------------
    df["response_time_minutes"] = (
        df["response_time_seconds"] / 60
    ).round(2)
    added_cols.append("response_time_minutes")

    # -- 9. Flag response_* 100% NaN ------------------------------------------
    if df["response_timestamp"].isna().all():
        issues.append(
            "response_timestamp: 100% NaN - no notifications were responded to yet "
            "(expected behavior in the initial phase)"
        )

    # -- 10. notification_id duplicates ---------------------------------------
    dups = df.duplicated(subset=["notification_id"], keep=False)
    if dups.any():
        issues.append(
            f"{dups.sum()} rows with duplicated notification_id "
            "(possible resends) - kept all"
        )

    # -- 11. Sort -------------------------------------------------------------
    df = df.sort_values("timestamp").reset_index(drop=True)

    report.register("research_nudge_log", original_rows, len(df),
                    issues, dropped_cols, added_cols)
    return df


def clean_phase_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    research_phase_metadata
    ------------------------
    Metadata of the study phases (baseline -> active).
    Known issues:
      - Two records with phase='active' and different start_timestamps - indicates
        a poorly recorded transition; keep both but flag.
      - end_timestamp: NaN in the ongoing active phase (expected).
      - baseline_occupancy_avg: 100% NaN.
      - baseline_consumption_W: NaN in baseline (not calculated yet).
    """
    issues:       list[str] = []
    dropped_cols: list[str] = []
    added_cols:   list[str] = []
    original_rows = len(df)

    # -- 1. Timestamps --------------------------------------------------------
    df["start_timestamp"] = unix_to_datetime(df["start_timestamp"], "start_timestamp")
    df["end_timestamp"]   = unix_to_datetime(df["end_timestamp"],   "end_timestamp")
    df["created_at"]      = pd.to_datetime(df["created_at"],        errors="coerce")

    # -- 2. Categories --------------------------------------------------------
    df["phase"] = flag_invalid_categories(df, "phase", VALID_PHASES, issues)

    # -- 3. Detect duplicated phases ------------------------------------------
    phase_counts = df["phase"].value_counts()
    for phase, count in phase_counts.items():
        if count > 1:
            issues.append(
                f"Phase '{phase}' appears {count}x - possible duplicate record "
                "of phase transition. Check manually."
            )

    # -- 4. Phase duration (days) ---------------------------------------------
    now_utc = pd.Timestamp.now("UTC")
    df["duration_days"] = (
        df["end_timestamp"].fillna(now_utc) - df["start_timestamp"]
    ).dt.total_seconds() / 86400
    df["duration_days"] = df["duration_days"].round(2)
    df["is_ongoing"] = df["end_timestamp"].isna().astype(int)
    added_cols += ["duration_days", "is_ongoing"]

    # -- 5. Validate baseline_consumption_W ≥ 0 -------------------------------
    df["baseline_consumption_W"] = validate_range(
        df, "baseline_consumption_W", 0, 1e6, issues
    )

    # -- 6. Flag baseline_occupancy_avg 100% NaN ------------------------------
    if df["baseline_occupancy_avg"].isna().all():
        issues.append(
            "baseline_occupancy_avg: 100% NaN - metric not calculated"
        )

    # -- 7. Sort --------------------------------------------------------------
    df = df.sort_values("start_timestamp").reset_index(drop=True)

    report.register("research_phase_metadata", original_rows, len(df),
                    issues, dropped_cols, added_cols)
    return df


def clean_rl_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    research_rl_episodes
    ---------------------
    Episodes from the Reinforcement Learning agent (Q-learning).
    Known issues:
      - opportunity_score: NaN during baseline (agent in shadow mode, without
        opportunity calculation - expected behavior).
      - accepted: 100% NaN (baseline doesn't send real notifications).
      - gamma_used: some NaN in the first episodes.
      - state_vector, q_values, action_mask: JSON strings -> expand.
    """
    issues:       list[str] = []
    dropped_cols: list[str] = []
    added_cols:   list[str] = []
    original_rows = len(df)

    # -- 1. Timestamps --------------------------------------------------------
    df["timestamp"]  = unix_to_datetime(df["timestamp"],  "timestamp")
    df["created_at"] = pd.to_datetime(df["created_at"],   errors="coerce")

    # -- 2. Categories --------------------------------------------------------
    df["phase"]         = flag_invalid_categories(df, "phase",         VALID_PHASES,        issues)
    df["action_name"]   = flag_invalid_categories(df, "action_name",   VALID_ACTION_NAMES,  issues)
    df["action_source"] = flag_invalid_categories(df, "action_source", VALID_ACTION_SOURCES, issues)

    # -- 3. Numeric ranges ----------------------------------------------------
    df["reward"]            = validate_range(df, "reward",            -10, 10,  issues)
    df["epsilon"]           = validate_range(df, "epsilon",           0,   1,   issues)
    df["anomaly_index"]     = validate_range(df, "anomaly_index",     0,   1,   issues)
    df["behaviour_index"]   = validate_range(df, "behaviour_index",   0,   1,   issues)
    df["fatigue_index"]     = validate_range(df, "fatigue_index",     0,   1,   issues)
    df["opportunity_score"] = validate_range(df, "opportunity_score", 0,   1,   issues)
    df["time_of_day_hour"]  = validate_range(df, "time_of_day_hour",  0,   23,  issues)
    df["gamma_used"]        = validate_range(df, "gamma_used",        0,   1,   issues)
    df["current_power"]     = validate_range(df, "current_power",     0,   1e6, issues)

    # -- 4. Phase context: flag shadow mode -----------------------------------
    df["is_shadow_mode"] = (df["action_source"].isin(
        ["shadow_explore", "shadow_exploit"]
    )).astype(int)
    added_cols.append("is_shadow_mode")

    # -- 5. opportunity_score NaN in baseline: expected -----------------------
    opp_null_baseline = (
        df["opportunity_score"].isna() & (df["phase"] == "baseline")
    ).sum()
    if opp_null_baseline:
        issues.append(
            f"opportunity_score: {opp_null_baseline} NaN in the baseline phase "
            "(agent in shadow mode - expected behavior)"
        )

    # -- 6. accepted 100% NaN -------------------------------------------------
    if df["accepted"].isna().all():
        issues.append(
            "accepted: 100% NaN - no real notifications sent yet "
            "(baseline phase/shadow mode)"
        )

    # -- 7. JSON Expansion ----------------------------------------------------
    sv_expanded   = expand_state_vector(df, "state_vector", prefix="sv")
    mask_expanded = expand_action_mask(df,  "action_mask",  prefix="mask")
    q_expanded    = expand_q_values(df,     "q_values",     prefix="q")

    df = df.drop(columns=["state_vector", "action_mask", "q_values"])
    dropped_cols += ["state_vector", "action_mask", "q_values"]
    df = pd.concat([df, sv_expanded, mask_expanded, q_expanded], axis=1)
    added_cols += (
        list(sv_expanded.columns)
        + list(mask_expanded.columns)
        + list(q_expanded.columns)
    )

    # -- 8. Duplicated episodes -----------------------------------------------
    dups = df.duplicated(subset=["episode_number", "phase"], keep=False)
    if dups.any():
        issues.append(
            f"{dups.sum()} duplicated episodes (episode_number + phase) "
            "-> kept the last occurrence"
        )
        df = df.sort_values("timestamp").drop_duplicates(
            subset=["episode_number", "phase"], keep="last"
        )

    # -- 9. Sort --------------------------------------------------------------
    df = df.sort_values(["phase", "episode_number"]).reset_index(drop=True)

    report.register("research_rl_episodes", original_rows, len(df),
                    issues, dropped_cols, added_cols)
    return df


def clean_task_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    research_task_interactions
    ---------------------------
    Daily gamification tasks presented to the user (up to 3/day).
    Known issues:
      - area_name: 100% NaN - area association not implemented yet.
      - first_view_timestamp: 100% NaN - user didn't open the app to view.
      - time_to_view_seconds: 100% NaN (depends on first_view_timestamp).
      - user_feedback: 100% NaN - feedback not collected.
      - completion_value / completion_timestamp: NaN when completed == 0.
    """
    issues:       list[str] = []
    dropped_cols: list[str] = []
    added_cols:   list[str] = []
    original_rows = len(df)

    # -- 1. Timestamps --------------------------------------------------------
    df["date"]                 = pd.to_datetime(df["date"],             errors="coerce")
    df["generation_timestamp"] = unix_to_datetime(df["generation_timestamp"], "generation_timestamp")
    df["first_view_timestamp"] = unix_to_datetime(df["first_view_timestamp"], "first_view_timestamp")
    df["completion_timestamp"] = unix_to_datetime(df["completion_timestamp"], "completion_timestamp")
    df["created_at"]           = pd.to_datetime(df["created_at"],       errors="coerce")

    # -- 2. Categories --------------------------------------------------------
    df["phase"] = flag_invalid_categories(df, "phase", VALID_PHASES, issues)

    VALID_TASK_TYPES = {
        # Matching task_manager.py generators exactly:
        "temperature_reduction", "temperature_increase",
        "power_reduction", "daylight_usage",
        "unoccupied_power", "peak_avoidance",
    }
    df["task_type"] = flag_invalid_categories(
        df, "task_type", VALID_TASK_TYPES, issues
    )

    # -- 3. Boolean flags -----------------------------------------------------
    for col in ["completed", "verified"]:
        df[col] = coerce_boolean_flag(df, col, issues)

    # -- 4. Consistency: verified only if completed ---------------------------
    bad_verify = df["verified"].notna() & df["completed"].notna() & \
                 (df["verified"] > df["completed"])
    if bad_verify.any():
        issues.append(
            f"{bad_verify.sum()} task(s) marked as verified but not completed"
        )

    # -- 5. Imputation: NaN values when not completed -> 0 --------------------
    not_completed = df["completed"] == 0
    for col in ["completion_value"]:
        n_filled = (df[col].isna() & not_completed).sum()
        if n_filled:
            df.loc[not_completed & df[col].isna(), col] = 0.0
            issues.append(
                f"'{col}': {n_filled} NaN filled with 0 (completed == 0)"
            )

    # -- 6. Ranges ------------------------------------------------------------
    df["difficulty_level"]  = validate_range(df, "difficulty_level",  1, 5,    issues)
    df["target_value"]      = validate_range(df, "target_value",      0, 1e6,  issues)
    df["baseline_value"]    = validate_range(df, "baseline_value",    0, 1e6,  issues)
    df["power_at_generation"] = validate_range(
        df, "power_at_generation", 0, 1e6, issues
    )

    # -- 7. Time to complete (in minutes) -------------------------------------
    df["time_to_complete_minutes"] = (
        df["time_to_complete_seconds"] / 60
    ).round(2)
    added_cols.append("time_to_complete_minutes")

    # -- 8. 100% NaN - flag ---------------------------------------------------
    fully_null = [c for c in df.columns
                  if df[c].isna().all() and c not in ("date", "created_at")]
    if fully_null:
        issues.append(f"Columns without any value (100% NaN): {fully_null}")

    # -- 9. Sort --------------------------------------------------------------
    df = df.sort_values(["date", "task_type"]).reset_index(drop=True)

    report.register("research_task_interactions", original_rows, len(df),
                    issues, dropped_cols, added_cols)
    return df


def clean_weekly_challenges(df: pd.DataFrame) -> pd.DataFrame:
    """
    research_weekly_challenges
    ---------------------------
    Records the weekly energy-saving challenges and their outcomes.
    Known issues to handle:
      - Ongoing weeks might have actual_W and savings_W as NaN.
      - Start and end dates might have inconsistencies.
      - Percentages and target limits must be within sensible bounds.
    """
    issues:       list[str] = []
    dropped_cols: list[str] = []
    added_cols:   list[str] = []
    original_rows = len(df)

    # -- 1. Timestamps and Dates ----------------------------------------------
    df["week_start_date"] = pd.to_datetime(df["week_start_date"], errors="coerce")
    df["week_end_date"]   = pd.to_datetime(df["week_end_date"],   errors="coerce")
    df["created_at"]      = pd.to_datetime(df["created_at"],      errors="coerce")

    # -- 2. Date consistency checks -------------------------------------------
    bad_dates = (df["week_start_date"].notna() & 
                 df["week_end_date"].notna() & 
                 (df["week_start_date"] > df["week_end_date"]))
    if bad_dates.any():
        issues.append(
            f"{bad_dates.sum()} row(s) with week_start_date > week_end_date"
        )

    # -- 3. Categories --------------------------------------------------------
    df["phase"] = flag_invalid_categories(df, "phase", VALID_PHASES, issues)

    # -- 4. Boolean Flags -----------------------------------------------------
    df["achieved"] = coerce_boolean_flag(df, "achieved", issues)

    # -- 5. Numeric Ranges ----------------------------------------------------
    # Allowing negative savings percentage in case consumption went up
    df["target_percentage"]  = validate_range(df, "target_percentage",  0,    100,  issues)
    df["savings_percentage"] = validate_range(df, "savings_percentage", -100, 100,  issues)
    df["baseline_W"]         = validate_range(df, "baseline_W",         0,    1e6,  issues)
    df["actual_W"]           = validate_range(df, "actual_W",           0,    1e6,  issues)

    # -- 6. Consistency check: Savings calculation ----------------------------
    if {"baseline_W", "actual_W", "savings_W"}.issubset(df.columns):
        expected_savings = df["baseline_W"] - df["actual_W"]
        # Allow a small float tolerance (e.g., 1 Watt difference due to rounding)
        mismatch = (df["savings_W"].notna() & expected_savings.notna() &
                    (abs(df["savings_W"] - expected_savings) > 1.0))
        if mismatch.any():
            issues.append(
                f"{mismatch.sum()} row(s) where savings_W != (baseline_W - actual_W)"
            )

    # -- 7. Auxiliary column: Is the challenge ongoing? -----------------------
    # If actual_W is missing, the week likely hasn't concluded yet
    df["is_ongoing"] = df["actual_W"].isna().astype(int)
    added_cols.append("is_ongoing")

    # -- 8. Sorting -----------------------------------------------------------
    df = df.sort_values("week_start_date").reset_index(drop=True)

    report.register("research_weekly_challenges", original_rows, len(df),
                    issues, dropped_cols, added_cols)
    return df


# ------------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------------

CLEANERS = {
    "research_area_daily_stats":       clean_area_daily_stats,
    "research_blocked_notifications":  clean_blocked_notifications,
    "research_daily_aggregates":       clean_daily_aggregates,
    "research_nudge_log":              clean_nudge_log,
    "research_phase_metadata":         clean_phase_metadata,
    "research_rl_episodes":            clean_rl_episodes,
    "research_task_interactions":      clean_task_interactions,
    "research_weekly_challenges":      clean_weekly_challenges
}


def run_pipeline(input_dir: Path, output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 80}")
    print("  GREEN SHIFT - DATA CLEANING PIPELINE")
    print(f"{'═' * 80}")
    print(f"  Source      : {input_dir.resolve()}")
    print(f"  Destination : {output_dir.resolve()}")
    print(f"  Date/time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    success_count = 0
    fail_count    = 0

    for table_name, cleaner_fn in CLEANERS.items():
        csv_path = input_dir / f"{table_name}.csv"

        if not csv_path.exists():
            print(f"  ⚠  {table_name:<45} - file not found, ignored")
            continue

        try:
            df_raw   = pd.read_csv(csv_path)
            df_clean = cleaner_fn(df_raw.copy())

            out_path = output_dir / f"{table_name}_clean.csv"
            df_clean.to_csv(out_path, index=False)

            print(f"  ✅ {table_name:<45} "
                  f"({len(df_raw)} -> {len(df_clean)} rows, "
                  f"{len(df_clean.columns)} columns) -> {out_path.name}")
            success_count += 1

        except Exception as exc:
            print(f"  ❌ {table_name:<45} - ERROR: {exc}")
            fail_count += 1

    report.print_summary()

    print(f"  ✅ Success : {success_count} tables")
    if fail_count:
        print(f"  ❌ Failures: {fail_count} tables")
    print()
    return fail_count == 0


def main() -> int:
    if len(sys.argv) > 3:
        print("Usage: python clean_research_data.py [input_dir] [output_dir]")
        return 1

    input_dir  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("cleaned_data")

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1

    ok = run_pipeline(input_dir, output_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())