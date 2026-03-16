"""
health_monitor.py - Green Shift System Diagnostic Tool
=======================================================
Standalone diagnostic script to check the real-time health, logic integrity,
and AI convergence of the Green Shift Home Assistant custom component.

Usage:
    python health_monitor.py [--data-dir PATH]

    Default data directory: config/green_shift_data
    Override with:  python health_monitor.py --data-dir /path/to/green_shift_data
"""

import argparse
import ast
import json
import math
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# ANSI color codes
# ---------------------------------------------------------------------------

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"

# Force ANSI on Windows (requires Python 3.12+ or colorama fallback)
if sys.platform == "win32":
    os.system("")  # Activate VT-100 on Windows console

# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

_issues = []   # Accumulated issues for the final summary

def _log(level: str, check: str, detail: str, extra: str = "") -> None:
    """Print a single result line and record non-OK issues."""
    icons = {"OK": f"{C.GREEN}[OK]      {C.RESET}",
             "WARNING":  f"{C.YELLOW}[WARNING] {C.RESET}",
             "CRITICAL": f"{C.RED}[CRITICAL]{C.RESET}",
             "INFO":     f"{C.CYAN}[INFO]    {C.RESET}",
             "SKIP":     f"{C.DIM}[SKIP]    {C.RESET}"}

    icon = icons.get(level, icons["INFO"])
    line = f"  {icon} {C.BOLD}{check}{C.RESET}: {detail}"
    if extra:
        line += f"\n           {C.DIM}{extra}{C.RESET}"
    print(line)

    if level in ("WARNING", "CRITICAL"):
        _issues.append({"level": level, "check": check, "detail": detail})


def _section(title: str) -> None:
    """Print a section header."""
    width = 74
    print()
    print(f"{C.BLUE}{C.BOLD}{'═' * width}{C.RESET}")
    print(f"{C.BLUE}{C.BOLD}  {title}{C.RESET}")
    print(f"{C.BLUE}{C.BOLD}{'═' * width}{C.RESET}")


def _subsection(title: str) -> None:
    print(f"\n  {C.CYAN}{C.BOLD}── {title}{C.RESET}")


def _summary() -> None:
    """Print the final summary table."""
    _section("DIAGNOSTIC SUMMARY")
    criticals = [i for i in _issues if i["level"] == "CRITICAL"]
    warnings  = [i for i in _issues if i["level"] == "WARNING"]

    if not _issues:
        print(f"\n  {C.GREEN}{C.BOLD}All checks passed - system appears healthy.{C.RESET}\n")
        return

    if criticals:
        print(f"\n  {C.RED}{C.BOLD}{len(criticals)} CRITICAL issue(s) require immediate attention:{C.RESET}")
        for i in criticals:
            print(f"    {C.RED}▸ [{i['check']}] {i['detail']}{C.RESET}")

    if warnings:
        print(f"\n  {C.YELLOW}{C.BOLD}{len(warnings)} WARNING(s) to investigate:{C.RESET}")
        for i in warnings:
            print(f"    {C.YELLOW}▸ [{i['check']}] {i['detail']}{C.RESET}")

    print()


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _open_db(path: Path, readonly: bool = True) -> Optional[sqlite3.Connection]:
    """Open a SQLite connection in WAL mode (read-only by default)."""
    uri = f"file:{path}?mode=ro" if readonly else str(path)
    flags = sqlite3.PARSE_DECLTYPES
    try:
        conn = sqlite3.connect(uri, uri=readonly, detect_types=flags, timeout=3)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError as exc:
        return None


def _q(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list:
    """Execute a query and return all rows (empty list on error)."""
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []


def _q1(conn: sqlite3.Connection, sql: str, params: tuple = ()):
    """Execute a query and return the first row (None on error)."""
    rows = _q(conn, sql, params)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# 1. Database & WAL Health
# ---------------------------------------------------------------------------

def check_database_health(data_dir: Path) -> Dict[str, Optional[sqlite3.Connection]]:
    """Check existence, accessibility, and WAL file sizes of both databases."""
    _section("1 · Database & WAL Health")

    conns = {}  # type: Dict[str, Optional[sqlite3.Connection]]

    for name, filename in [("sensor_data.db", "sensor_data.db"),
                            ("research_data.db", "research_data.db")]:
        db_path  = data_dir / filename
        wal_path = data_dir / f"{filename}-wal"
        shm_path = data_dir / f"{filename}-shm"

        # Existence
        if not db_path.exists():
            _log("CRITICAL", name, "Database file not found.",
                 f"Expected at: {db_path}")
            conns[name] = None
            continue

        db_size_mb = db_path.stat().st_size / (1024 * 1024)
        _log("OK", name, f"File exists ({db_size_mb:.2f} MB)")

        # WAL file size
        if wal_path.exists():
            wal_mb = wal_path.stat().st_size / (1024 * 1024)
            if wal_mb > 50:
                _log("CRITICAL", f"{name} WAL",
                     f"WAL file is {wal_mb:.1f} MB - checkpoint is not running.",
                     "The WAL file should stay below ~10 MB. "
                     "A stuck checkpoint can indicate a long-running read transaction or crash.")
            elif wal_mb > 10:
                _log("WARNING", f"{name} WAL",
                     f"WAL file is {wal_mb:.1f} MB - consider forcing a checkpoint.",
                     "Run: PRAGMA wal_checkpoint(TRUNCATE); from a DB client.")
            else:
                _log("OK", f"{name} WAL", f"WAL file is {wal_mb:.2f} MB (normal)")
        else:
            _log("INFO", f"{name} WAL", "No WAL file found (database may be fully checkpointed)")

        # Connectivity & lock
        conn = _open_db(db_path)
        if conn is None:
            _log("CRITICAL", f"{name} Lock",
                 "Cannot open database - it may be locked by another process.",
                 "Ensure Home Assistant is not holding an exclusive lock.")
        else:
            _log("OK", f"{name} Lock", "Database is accessible (no exclusive lock)")

        conns[name] = conn

    # state.json
    state_path = data_dir / "state.json"
    if not state_path.exists():
        _log("CRITICAL", "state.json", "File not found.",
             f"Expected at: {state_path}")
    else:
        sz_kb = state_path.stat().st_size / 1024
        if sz_kb > 5000:
            _log("WARNING", "state.json",
                 f"File is {sz_kb:.0f} KB - Q-table may be exploding.",
                 "A Q-table larger than 5 MB suggests the state space is too granular "
                 "and may slow persistence or cause memory pressure.")
        else:
            _log("OK", "state.json", f"File exists ({sz_kb:.1f} KB)")

    return conns


# ---------------------------------------------------------------------------
# 2. Real-Time Data Freshness
# ---------------------------------------------------------------------------

def check_data_freshness(conn: Optional[sqlite3.Connection]) -> None:
    _section("2 · Real-Time Data Freshness")

    if conn is None:
        _log("SKIP", "Data Freshness", "sensor_data.db unavailable - skipping.")
        return

    now_ts = time.time()

    # Latest record
    row = _q1(conn, "SELECT timestamp, power, temperature FROM sensor_history "
                    "ORDER BY timestamp DESC LIMIT 1")
    if row is None:
        _log("CRITICAL", "Latest Snapshot", "sensor_history is completely empty.",
             "No data has ever been written. Check if Home Assistant is running and sensors are discovered.")
        return

    age_seconds = now_ts - row["timestamp"]
    if age_seconds < 45:
        _log("OK", "Latest Snapshot",
             f"Last record {age_seconds:.0f}s ago - data is fresh")
    elif age_seconds < 120:
        _log("WARNING", "Latest Snapshot",
             f"Last record {age_seconds:.0f}s ago (expected < 45s).",
             "Home Assistant may be under load or the collection loop is delayed.")
    else:
        _log("CRITICAL", "Latest Snapshot",
             f"Last record {age_seconds:.0f}s ago - data is STALE.",
             "Possible causes: HA is stopped, Green Shift component crashed, or the "
             "async_track_time_interval was not registered. Check HA logs.")

    # Total row count
    total = _q1(conn, "SELECT COUNT(*) AS n FROM sensor_history")["n"]
    _log("INFO", "Sensor History", f"{total:,} total snapshots stored")

    # Consecutive zero / NULL power readings (last 10 records)
    recent = _q(conn,
                "SELECT power, temperature FROM sensor_history "
                "ORDER BY timestamp DESC LIMIT 10")
    zero_power = sum(1 for r in recent if r["power"] is None or r["power"] == 0.0)
    zero_temp  = sum(1 for r in recent if r["temperature"] is None or r["temperature"] == 0.0)

    if zero_power >= 8:
        _log("CRITICAL", "Power Sensor Dropout",
             f"{zero_power}/10 recent readings are NULL or 0 W.",
             "The main power sensor appears to have dropped out. "
             "Check the sensor entity in Home Assistant for unavailable/unknown state.")
    elif zero_power >= 5:
        _log("WARNING", "Power Sensor Dropout",
             f"{zero_power}/10 recent readings are NULL or 0 W.",
             "Intermittent power sensor readings. Verify hardware connectivity.")
    else:
        _log("OK", "Power Sensor",
             f"Only {zero_power}/10 recent readings are zero/NULL (acceptable)")

    if zero_temp >= 8:
        _log("WARNING", "Temperature Sensor Dropout",
             f"{zero_temp}/10 recent temperature readings are NULL or 0.",
             "Temperature sensor may be offline - this can affect task generation "
             "and the temperature-based anomaly index.")
    else:
        _log("OK", "Temperature Sensor",
             f"Only {zero_temp}/10 recent temperature readings are zero/NULL")

    # Data gap detection - any gap > 5 minutes in last 2 hours
    two_hours_ago = now_ts - 7200
    rows_2h = _q(conn,
                 "SELECT timestamp FROM sensor_history "
                 "WHERE timestamp >= ? ORDER BY timestamp ASC",
                 (two_hours_ago,))
    if len(rows_2h) >= 2:
        timestamps = [r["timestamp"] for r in rows_2h]
        max_gap = max(timestamps[i+1] - timestamps[i]
                      for i in range(len(timestamps) - 1))
        if max_gap > 300:
            _log("WARNING", "Data Gap Detected",
                 f"Largest gap in last 2 hours: {max_gap:.0f}s.",
                 "A gap > 5 minutes suggests the collection loop was interrupted. "
                 "Check Home Assistant uptime and component error logs.")
        else:
            _log("OK", "Data Continuity",
                 f"No gaps > 5 min in the last 2 hours (max gap: {max_gap:.0f}s)")
    elif len(rows_2h) == 0:
        _log("CRITICAL", "Data Gap Detected",
             "No records in the last 2 hours.",
             "The data collection loop appears to have stopped entirely.")
    else:
        _log("INFO", "Data Gap", "Insufficient data points to compute gap analysis (<2h of data)")

    # Area coverage
    areas = _q(conn, "SELECT DISTINCT area_name FROM area_sensor_history")
    area_count = len(areas)
    if area_count == 0:
        _log("WARNING", "Area Coverage",
             "No area-specific data found.",
             "Sensors may not have been assigned to areas during setup. "
             "The AI cannot identify per-room anomalies without area data.")
    else:
        area_names = ", ".join(r["area_name"] for r in areas[:8])
        _log("OK", "Area Coverage",
             f"{area_count} area(s) tracked: {area_names}")


# ---------------------------------------------------------------------------
# 3. AI Core & Q-Learning Health (state.json)
# ---------------------------------------------------------------------------

def check_ai_health(data_dir: Path) -> Optional[dict]:
    _section("3 · AI Core & Q-Learning Health")

    state_path = data_dir / "state.json"
    if not state_path.exists():
        _log("SKIP", "AI Health", "state.json not found - skipping all AI checks.")
        return None

    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except json.JSONDecodeError as exc:
        _log("CRITICAL", "state.json Parse", f"JSON is malformed: {exc}",
             "The file may have been corrupted during a write. Restore from a backup.")
        return None

    phase = state.get("phase", "unknown")
    active_since = state.get("active_since")
    _log("INFO", "System Phase", f"Current phase: {C.BOLD}{phase}{C.RESET}")

    # Phase / active_since consistency
    if phase == "active" and not active_since:
        _log("WARNING", "Phase Consistency",
             "Phase is 'active' but active_since is null.",
             "This may cause incorrect savings calculations. "
             "Consider re-triggering the phase transition.")
    elif phase == "baseline" and active_since:
        _log("WARNING", "Phase Consistency",
             "Phase is 'baseline' but active_since is set - conflicting state.",
             "A previous transition may have been partially written.")
    else:
        _log("OK", "Phase Consistency", "phase and active_since are coherent")

    # Baseline consumption
    baseline_w = state.get("baseline_consumption", 0.0)
    if baseline_w <= 0.0 and phase == "active":
        _log("CRITICAL", "Baseline Consumption",
             f"Baseline is {baseline_w} W in active phase.",
             "Without a valid baseline the savings sensor will always read 0. "
             "All recommendations will lack a reference. Re-run baseline phase.")
    elif baseline_w <= 0.0:
        _log("INFO", "Baseline Consumption",
             f"Baseline is {baseline_w} W (normal during baseline phase)")
    else:
        _log("OK", "Baseline Consumption", f"Baseline = {baseline_w:.1f} W")

    # Q-table
    q_table_raw = state.get("q_table", {})
    q_table_size = len(q_table_raw)
    episode_number = state.get("episode_number", 0)
    shadow_episodes = state.get("shadow_episode_number", 0)
    feedback_episodes = state.get("feedback_episode_number", 0)

    _log("INFO", "Q-Table", f"{q_table_size} state entries "
         f"| episodes: {episode_number} total / "
         f"{shadow_episodes} shadow / {feedback_episodes} real-feedback")

    if q_table_size == 0:
        if phase == "active":
            _log("CRITICAL", "Q-Table Empty",
                 "Q-table is completely empty in the active phase.",
                 "Shadow learning during baseline should have pre-populated it. "
                 "The agent is flying blind - check if shadow RL was disabled.")
        else:
            _log("INFO", "Q-Table Empty",
                 "Q-table is empty (expected early in baseline phase)")
    elif q_table_size < 10 and phase == "active":
        _log("WARNING", "Q-Table Size",
             f"Only {q_table_size} state entries after {episode_number} episodes.",
             "State space is very small - the agent may not be exploring diverse "
             "conditions. Verify that discretisation is producing varied state keys.")
    else:
        _log("OK", "Q-Table Size",
             f"{q_table_size} state entries - exploration appears active")

    # Q-value distribution health
    if q_table_raw:
        all_q_vals = [float(v) for actions in q_table_raw.values()
                      for v in actions.values()
                      if v is not None and not math.isnan(float(v))]
        if all_q_vals:
            q_min = min(all_q_vals)
            q_max = max(all_q_vals)
            q_mean = sum(all_q_vals) / len(all_q_vals)
            if q_max == q_min:
                _log("WARNING", "Q-Value Variance",
                     f"All Q-values are identical ({q_mean:.4f}) - no learning signal.",
                     "The reward function may be returning constant values, or "
                     "the agent is not receiving any user feedback.")
            elif any(math.isnan(v) for v in all_q_vals):
                _log("CRITICAL", "Q-Value NaN",
                     "NaN Q-values detected in the Q-table.",
                     "This will cause the argmax policy to malfunction silently. "
                     "Check for division by zero or float overflow in the reward function.")
            else:
                _log("OK", "Q-Value Distribution",
                     f"min={q_min:.4f} / mean={q_mean:.4f} / max={q_max:.4f}")

    # Epsilon
    epsilon = state.get("epsilon", None)
    if epsilon is None:
        _log("INFO", "Epsilon", "Not stored in state.json (will use default on next start)")
    elif phase == "active":
        if epsilon > 0.5:
            _log("WARNING", "Epsilon (Exploration)",
                 f"ε = {epsilon:.4f} - agent is still highly exploratory after "
                 f"{feedback_episodes} real-feedback episodes.",
                 "Expected ε < 0.2 after several dozen real interactions. "
                 "Check if feedback_episode_number is incrementing correctly.")
        elif epsilon <= 0.05:
            _log("WARNING", "Epsilon (Exploitation)",
                 f"ε = {epsilon:.4f} - agent is almost fully greedy.",
                 "With < 5% exploration the agent will rarely try novel actions. "
                 "Consider a small epsilon reset if study data is sparse.")
        else:
            _log("OK", "Epsilon",
                 f"ε = {epsilon:.4f} - healthy exploration/exploitation balance")

    # AI indices
    for key, label, low_warn, high_warn in [
        ("anomaly_index",   "Anomaly Index",   None, 1.0),
        ("behaviour_index", "Behaviour Index", 0.0,  1.0),
        ("fatigue_index",   "Fatigue Index",   None, 1.1),
    ]:
        val = state.get(key, None)
        if val is None:
            _log("WARNING", label, "Key not present in state.json.")
            continue

        try:
            val = float(val)
        except (ValueError, TypeError):
            _log("CRITICAL", label, f"Value '{val}' is not numeric.",
                 "A corrupted index will cause the AI to make incorrect decisions.")
            continue

        if math.isnan(val) or math.isinf(val):
            _log("CRITICAL", label, f"Value is {val} (NaN/Inf) - this will corrupt AI decisions.",
                 "Reset the index via the set_test_indices service or restore from backup.")
        elif val < 0:
            _log("CRITICAL", label, f"Value is {val:.4f} - negative, which is invalid.",
                 "Index calculations have a sign error. Check for inverted subtraction.")
        elif high_warn and val > high_warn:
            _log("WARNING", label, f"Value is {val:.4f} - above expected maximum of {high_warn}.",
                 "Index normalisation may have broken. Check the calculation pipeline.")
        elif key == "anomaly_index" and val == 0.0 and phase == "active":
            _log("WARNING", label,
                 "Anomaly index has been 0.0 throughout the active phase.",
                 "Either no anomalies have occurred (unlikely) or the rolling "
                 "std-dev calculation is not receiving enough variance. "
                 "Check power history buffer length.")
        elif key == "behaviour_index" and val == 0.0 and phase == "active":
            _log("WARNING", label,
                 "Behaviour index is 0.0 in active phase.",
                 "No engagement has been recorded. Tasks or notifications "
                 "may not be completing, or the engagement_history deque is empty.")
        else:
            _log("OK", label, f"Value = {val:.4f}")

    # Engagement history
    eng_hist = state.get("engagement_history", [])
    _log("INFO", "Engagement History",
         f"{len(eng_hist)} entries (max 100)")
    if len(eng_hist) == 0 and phase == "active":
        _log("WARNING", "Engagement History",
             "Engagement history is empty during the active phase.",
             "No user interactions (task completions, nudge responses) have been "
             "recorded. The behaviour index will remain uninformative.")

    # Pending episodes (stale check)
    pending = state.get("pending_episodes", {})
    if pending:
        now_dt = datetime.now()
        stale_count = 0
        for notif_id, ep in pending.items():
            ts_raw = ep.get("timestamp")
            if isinstance(ts_raw, str):
                try:
                    ep_ts = datetime.fromisoformat(ts_raw)
                    if (now_dt - ep_ts).total_seconds() > 86400:
                        stale_count += 1
                except ValueError:
                    stale_count += 1
        if stale_count:
            _log("WARNING", "Pending Episodes",
                 f"{stale_count}/{len(pending)} pending episode(s) older than 24 hours.",
                 "These episodes are waiting for user response that will never come. "
                 "They should already have been expired by the AI cycle cleanup, but a high count means "
                 "users are ignoring notification prompts entirely.")
        else:
            _log("OK", "Pending Episodes",
                 f"{len(pending)} pending episode(s), none stale")

    # Area baselines
    area_baselines = state.get("area_baselines", {})
    if area_baselines:
        _log("OK", "Area Baselines",
             f"{len(area_baselines)} area baseline(s) loaded: "
             f"{', '.join(list(area_baselines.keys())[:5])}")
    elif phase == "active":
        _log("WARNING", "Area Baselines",
             "No area baselines in active phase.",
             "Area-specific anomaly detection will be disabled. "
             "Sensors may not have been assigned to areas.")

    return state


# ---------------------------------------------------------------------------
# 4. RL Episodes & Nudges
# ---------------------------------------------------------------------------

def check_rl_and_nudges(conn: Optional[sqlite3.Connection]) -> None:
    _section("4 · RL Episodes & Nudges")

    if conn is None:
        _log("SKIP", "RL/Nudges", "research_data.db unavailable - skipping.")
        return

    # --- Nudge log ---
    _subsection("Nudge Log (research_nudge_log)")

    total_nudges = _q1(conn, "SELECT COUNT(*) AS n FROM research_nudge_log")["n"]
    _log("INFO", "Total Nudges", f"{total_nudges} nudge(s) recorded")

    if total_nudges == 0:
        _log("WARNING", "Nudge Log",
             "No nudges have been sent yet.",
             "If the system is in the active phase this is unexpected. "
             "Check if the action mask is blocking all notification types.")
    else:
        # Last 10 responded nudges
        last_10 = _q(conn,
                     "SELECT responded, accepted FROM research_nudge_log "
                     "WHERE responded = 1 ORDER BY timestamp DESC LIMIT 10")
        if len(last_10) == 0:
            _log("WARNING", "Nudge Response Rate",
                 "No nudges have been responded to yet.",
                 "Users have not interacted with any notification. "
                 "Consider reviewing notification timing or messaging.")
        else:
            acceptance_rate = sum(r["accepted"] for r in last_10) / len(last_10) * 100
            if acceptance_rate == 0:
                _log("WARNING", "Nudge Acceptance Rate",
                     f"0% acceptance over last {len(last_10)} responded nudge(s).",
                     "Users are consistently rejecting notifications. Possible causes: "
                     "bad timing, irrelevant suggestions, or high fatigue index. "
                     "Review notification templates and the action mask logic.")
            elif acceptance_rate < 20:
                _log("WARNING", "Nudge Acceptance Rate",
                     f"{acceptance_rate:.0f}% acceptance over last {len(last_10)} nudge(s) - very low.",
                     "Low acceptance degrades the Q-table reward signal and "
                     "can lock the agent in a non-feedback loop.")
            else:
                _log("OK", "Nudge Acceptance Rate",
                     f"{acceptance_rate:.0f}% over last {len(last_10)} responded nudge(s)")

        # Average response time
        avg_resp = _q1(conn,
                       "SELECT AVG(response_time_seconds) AS avg_rt "
                       "FROM research_nudge_log WHERE responded = 1")
        if avg_resp and avg_resp["avg_rt"] is not None:
            _log("INFO", "Avg Response Time",
                 f"{avg_resp['avg_rt']:.0f}s average user response latency")

        # Nudge action type distribution
        dist = _q(conn,
                  "SELECT action_type, COUNT(*) AS n FROM research_nudge_log "
                  "GROUP BY action_type ORDER BY n DESC")
        if dist:
            dist_str = " | ".join(f"{r['action_type']}: {r['n']}" for r in dist)
            _log("INFO", "Nudge Action Distribution", dist_str)
            # Check if all nudges are a single type (agent stuck on one action)
            if len(dist) == 1 and total_nudges >= 20:
                _log("WARNING", "Action Diversity",
                     f"All {total_nudges} nudges use only '{dist[0]['action_type']}' action.",
                     "The Q-table may have converged too early and is exploiting a "
                     "single action. Consider reviewing the Q-initialisation or reward weights.")

    # Unresponded (pending) nudges older than 24 hours
    cutoff_ts = time.time() - 86400
    stale_nudges = _q1(conn,
                       "SELECT COUNT(*) AS n FROM research_nudge_log "
                       "WHERE responded = 0 AND timestamp < ?",
                       (cutoff_ts,))["n"]
    if stale_nudges > 0:
        _log("WARNING", "Stale Unresponded Nudges",
             f"{stale_nudges} nudge(s) unresponded for > 24 hours.",
             "These will never receive feedback. They should have been expired by "
             "the agent's pending_episodes cleanup. Check if the expiry logic runs correctly.")
    else:
        _log("OK", "Stale Nudges", "No nudges unresponded for > 24 hours")

    # --- RL episodes ---
    _subsection("RL Episodes (research_rl_episodes)")

    total_ep = _q1(conn, "SELECT COUNT(*) AS n FROM research_rl_episodes")["n"]
    _log("INFO", "Total RL Episodes", f"{total_ep} episode(s) recorded")

    if total_ep == 0:
        _log("INFO", "RL Episodes",
             "No RL episodes logged yet (expected early in baseline phase)")
        return

    # Recent 50 episodes - expired ratio
    recent_ep = _q(conn,
                   "SELECT action_source, reward, action_name, epsilon "
                   "FROM research_rl_episodes "
                   "ORDER BY timestamp DESC LIMIT 50")
    expired_count   = sum(1 for r in recent_ep if r["action_source"] == "expired")
    shadow_count    = sum(1 for r in recent_ep if r["action_source"] in ("shadow_exploit", "shadow_explore"))
    real_count      = sum(1 for r in recent_ep
                          if r["action_source"] not in ("expired", "shadow_exploit", "shadow_explore", None))

    expired_pct = expired_count / len(recent_ep) * 100 if recent_ep else 0

    if expired_pct >= 80:
        _log("CRITICAL", "Episode Expiry Rate",
             f"{expired_pct:.0f}% of recent {len(recent_ep)} episodes expired (user ignoring AI).",
             "Users are not responding to notifications before the Q-update window closes. "
             "Consider extending the pending episode TTL or increasing notification persistence.")
    elif expired_pct >= 40:
        _log("WARNING", "Episode Expiry Rate",
             f"{expired_pct:.0f}% of recent {len(recent_ep)} episodes expired.",
             "A significant number of episodes lack user feedback - the Q-table "
             "is being updated with missing reward signals.")
    else:
        _log("OK", "Episode Expiry Rate",
             f"{expired_pct:.0f}% expired | {shadow_count} shadow | {real_count} real-feedback "
             f"(from last {len(recent_ep)} episodes)")

    # Reward trend
    rewards = [r["reward"] for r in recent_ep
               if r["reward"] is not None and not math.isnan(r["reward"])]
    if rewards:
        avg_reward = sum(rewards) / len(rewards)
        neg_pct = sum(1 for r in rewards if r < 0) / len(rewards) * 100
        if avg_reward < -0.5:
            _log("WARNING", "Reward Trend",
                 f"Mean reward over last {len(rewards)} episodes: {avg_reward:.4f} (mostly negative).",
                 "The agent is receiving poor reward signals - users may be consistently "
                 "rejecting nudges, or the reward function weights need tuning.")
        elif avg_reward > 0:
            _log("OK", "Reward Trend",
                 f"Mean reward: {avg_reward:.4f} | {neg_pct:.0f}% negative episodes")
        else:
            _log("INFO", "Reward Trend",
                 f"Mean reward: {avg_reward:.4f} (near-zero, agent is learning gradually)")

    # Action distribution in recent episodes
    recent_actions = _q(conn,
                        "SELECT action_name, COUNT(*) AS n "
                        "FROM research_rl_episodes "
                        "WHERE action_source NOT IN ('shadow_exploit', 'shadow_explore', 'expired') "
                        "GROUP BY action_name ORDER BY n DESC LIMIT 10")
    if recent_actions:
        dist_str = " | ".join(f"{r['action_name']}: {r['n']}" for r in recent_actions)
        _log("INFO", "Active Episode Action Dist.", dist_str)

    # Blocked notifications
    blocked_count = _q1(conn, "SELECT COUNT(*) AS n FROM research_blocked_notifications")
    if blocked_count:
        total_blocked = blocked_count["n"]
        # Ratio of blocked to sent in the last 24h
        cutoff = time.time() - 86400
        blocked_24h = _q1(conn,
                          "SELECT COUNT(*) AS n FROM research_blocked_notifications "
                          "WHERE timestamp >= ?", (cutoff,))["n"]
        nudges_24h  = _q1(conn,
                          "SELECT COUNT(*) AS n FROM research_nudge_log "
                          "WHERE timestamp >= ?", (cutoff,))["n"]
        if nudges_24h + blocked_24h > 0:
            block_ratio = blocked_24h / (nudges_24h + blocked_24h) * 100
            if block_ratio >= 90:
                _log("WARNING", "Notification Block Rate",
                     f"{block_ratio:.0f}% of notification attempts blocked in last 24h "
                     f"({blocked_24h} blocked, {nudges_24h} sent).",
                     "The system is heavily throttling itself. Check fatigue index, "
                     "cooldown periods, and whether opportunity_score thresholds are too high.")
            else:
                _log("OK", "Notification Block Rate",
                     f"{block_ratio:.0f}% blocked in last 24h "
                     f"({blocked_24h} blocked / {nudges_24h} sent)")

        # Top block reasons
        reasons = _q(conn,
                     "SELECT block_reason, COUNT(*) AS n "
                     "FROM research_blocked_notifications "
                     "GROUP BY block_reason ORDER BY n DESC LIMIT 5")
        if reasons:
            reasons_str = " | ".join(f"{r['block_reason']}: {r['n']}" for r in reasons)
            _log("INFO", "Top Block Reasons", reasons_str)


# ---------------------------------------------------------------------------
# 5. Gamification & Task Integrity
# ---------------------------------------------------------------------------

def check_gamification(conn_sensor: Optional[sqlite3.Connection],
                        state: Optional[dict]) -> None:
    _section("5 · Gamification & Task Integrity")

    if conn_sensor is None:
        _log("SKIP", "Tasks", "sensor_data.db unavailable - skipping task checks.")
    else:
        today_str = date.today().isoformat()

        # Today's tasks
        today_tasks = _q(conn_sensor,
                         "SELECT task_id, task_type, completed, verified "
                         "FROM daily_tasks WHERE date = ?", (today_str,))
        if not today_tasks:
            _log("CRITICAL", "Today's Tasks",
                 f"No tasks generated for today ({today_str}).",
                 "Possible causes: task generation at 06:00 was missed (HA was down), "
                 "the TaskManager raised an uncaught exception, or no suitable sensors "
                 "are available to generate verifiable tasks.")
        else:
            completed = sum(1 for t in today_tasks if t["completed"])
            verified  = sum(1 for t in today_tasks if t["verified"])
            _log("OK", "Today's Tasks",
                 f"{len(today_tasks)} task(s) for today | "
                 f"{completed} completed | {verified} verified")

        # Last 3 days - verification engine health
        three_days_ago = (date.today() - timedelta(days=3)).isoformat()
        old_tasks = _q(conn_sensor,
                       "SELECT date, task_id, completed, verified "
                       "FROM daily_tasks WHERE date >= ? AND date < ?",
                       (three_days_ago, today_str))
        if old_tasks:
            # Only flag if there was a significant number of tasks and none ever verified
            total_old   = len(old_tasks)
            never_verified = sum(1 for t in old_tasks
                                 if t["completed"] == 0 and t["verified"] == 0)
            if total_old >= 3 and never_verified == total_old:
                _log("CRITICAL", "Verification Engine",
                     f"0/{total_old} tasks from the last 3 days were ever verified or completed.",
                     "The periodic verify_tasks call (every 15 min) appears to have never "
                     "succeeded. Check async_track_time_interval registration and "
                     "TaskManager._verify_single_task for silent exceptions.")
            else:
                verified_3d = total_old - never_verified
                _log("OK", "Verification Engine",
                     f"{verified_3d}/{total_old} tasks from last 3 days reached "
                     "verified or completed state")
        else:
            _log("INFO", "Verification Engine",
                 "No past tasks found (may be first days of study)")

        # Task type diversity in last 7 days
        week_ago = (date.today() - timedelta(days=7)).isoformat()
        task_types = _q(conn_sensor,
                        "SELECT task_type, COUNT(*) AS n FROM daily_tasks "
                        "WHERE date >= ? GROUP BY task_type ORDER BY n DESC",
                        (week_ago,))
        if task_types:
            type_str = " | ".join(f"{r['task_type']}: {r['n']}" for r in task_types)
            _log("INFO", "Task Type Distribution (7d)", type_str)
            if len(task_types) == 1:
                _log("WARNING", "Task Diversity",
                     f"Only one task type ('{task_types[0]['task_type']}') in last 7 days.",
                     "Task variety is important for user engagement. "
                     "Check if certain task generators are failing silently "
                     "or if sensors for other task types are unavailable.")

        # Difficulty history
        diff_extremes = _q(conn_sensor,
                           "SELECT feedback, COUNT(*) AS n FROM task_difficulty_history "
                           "GROUP BY feedback ORDER BY n DESC")
        if diff_extremes:
            diff_str = " | ".join(f"{r['feedback']}: {r['n']}" for r in diff_extremes)
            _log("INFO", "Difficulty Feedback Distribution", diff_str)

    # Streaks (from state.json)
    if state is None:
        _log("SKIP", "Streaks", "state.json unavailable - skipping streak checks.")
        return

    for key, label in [("task_streak", "Task Streak"),
                        ("weekly_streak", "Weekly Streak")]:
        val = state.get(key, None)
        if val is None:
            _log("INFO", label, "Not present in state.json")
            continue
        try:
            val = int(val)
        except (ValueError, TypeError):
            _log("CRITICAL", label, f"Value '{val}' is not an integer - corrupted.",
                 "Restore from backup or reset the streak counter via service call.")
            continue

        if val < 0:
            _log("CRITICAL", label, f"Streak is {val} (negative) - corrupted.",
                 "A negative streak breaks the gamification display. "
                 "Reset with: state.json → set task_streak to 0.")
        else:
            _log("OK", label, f"Current streak: {val} day(s)/week(s)")

    # Streak last date sanity
    last_streak_date = state.get("task_streak_last_date")
    if last_streak_date:
        try:
            last_dt = date.fromisoformat(last_streak_date)
            days_since = (date.today() - last_dt).days
            if days_since > 7:
                _log("WARNING", "Task Streak Last Date",
                     f"Streak last credited {days_since} day(s) ago ({last_streak_date}).",
                     "If the task streak counter is non-zero but was last credited long ago, "
                     "the streak should have been reset. Check update_task_streak() call site.")
            else:
                _log("OK", "Task Streak Last Date",
                     f"Last streak credit: {last_streak_date} ({days_since}d ago)")
        except ValueError:
            _log("WARNING", "Task Streak Last Date",
                 f"Cannot parse date '{last_streak_date}'.")


# ---------------------------------------------------------------------------
# 6. Aggregation & Research Data
# ---------------------------------------------------------------------------

def check_research_aggregates(conn: Optional[sqlite3.Connection]) -> None:
    _section("6 · Aggregation & Research Data")

    if conn is None:
        _log("SKIP", "Aggregates", "research_data.db unavailable - skipping.")
        return

    yesterday = (date.today() - timedelta(days=1)).isoformat()

    # Yesterday's aggregate
    yesterday_agg = _q1(conn,
                         "SELECT date, total_energy_kwh, avg_power_w, tasks_generated, "
                         "nudges_sent, avg_anomaly_index "
                         "FROM research_daily_aggregates WHERE date = ?",
                         (yesterday,))
    if yesterday_agg is None:
        _log("CRITICAL", "Daily Aggregate (Yesterday)",
             f"No aggregate row for {yesterday}.",
             "The midnight daily aggregation job did not run or failed silently. "
             "This is a critical gap in the academic study data. "
             "Check the async_track_time_change listener registered at midnight "
             "and look for exceptions in the HA log around 00:00.")
    else:
        energy = yesterday_agg["total_energy_kwh"]
        power  = yesterday_agg["avg_power_w"]
        tasks  = yesterday_agg["tasks_generated"]
        nudges = yesterday_agg["nudges_sent"]

        _log("OK", "Daily Aggregate (Yesterday)",
             f"{yesterday}: {energy:.3f} kWh | avg {power:.1f} W | "
             f"{tasks} tasks | {nudges} nudges")

        if energy is None or energy <= 0:
            _log("WARNING", "Energy in Aggregate",
                 f"Yesterday's total_energy_kwh is {energy} - missing or zero.",
                 "The energy sensor may not have been available during aggregation.")

    # Aggregate completeness (last 14 days)
    two_weeks_ago = (date.today() - timedelta(days=14)).isoformat()
    agg_rows = _q(conn,
                  "SELECT date FROM research_daily_aggregates "
                  "WHERE date >= ? ORDER BY date ASC",
                  (two_weeks_ago,))
    agg_dates = {r["date"] for r in agg_rows}
    missing = []
    for i in range(1, 15):
        d = (date.today() - timedelta(days=i)).isoformat()
        if d not in agg_dates:
            missing.append(d)

    if missing:
        _log("WARNING" if len(missing) <= 3 else "CRITICAL",
             "Aggregate Completeness (14d)",
             f"{len(missing)} day(s) missing: {', '.join(sorted(missing))}",
             "Each missing day is a permanent gap in the study. "
             "Investigate HA uptime and the aggregation job scheduler.")
    else:
        _log("OK", "Aggregate Completeness (14d)",
             "All 14 days have aggregate rows")

    # Total aggregate rows
    total_agg = _q1(conn, "SELECT COUNT(*) AS n FROM research_daily_aggregates")["n"]
    _log("INFO", "Total Aggregate Days", f"{total_agg} aggregate row(s) stored")

    # Research phase metadata
    phase_meta = _q(conn, "SELECT phase, start_timestamp, end_timestamp "
                          "FROM research_phase_metadata ORDER BY start_timestamp ASC")
    if not phase_meta:
        _log("WARNING", "Phase Metadata",
             "research_phase_metadata is empty.",
             "Phase transitions are not being logged. "
             "The trigger_phase_transition_notification function may not persist metadata.")
    else:
        for pm in phase_meta:
            end_str = datetime.fromtimestamp(pm["end_timestamp"]).strftime("%Y-%m-%d %H:%M") \
                      if pm["end_timestamp"] else "ongoing"
            start_str = datetime.fromtimestamp(pm["start_timestamp"]).strftime("%Y-%m-%d %H:%M")
            _log("INFO", f"Phase: {pm['phase']}", f"Start: {start_str} | End: {end_str}")

    # Weekly challenges
    total_wc = _q1(conn, "SELECT COUNT(*) AS n FROM research_weekly_challenges")["n"]
    if total_wc > 0:
        achieved_wc = _q1(conn,
                          "SELECT COUNT(*) AS n FROM research_weekly_challenges "
                          "WHERE achieved = 1")["n"]
        pct = achieved_wc / total_wc * 100 if total_wc else 0
        _log("INFO", "Weekly Challenges",
             f"{achieved_wc}/{total_wc} achieved ({pct:.0f}% success rate)")
    else:
        _log("INFO", "Weekly Challenges", "No weekly challenge records yet")

    # Area daily stats coverage
    area_stats = _q1(conn, "SELECT COUNT(*) AS n FROM research_area_daily_stats")["n"]
    _log("INFO", "Area Daily Stats", f"{area_stats} area-day record(s) stored")

    # Task interaction records
    task_interactions = _q1(conn,
                             "SELECT COUNT(*) AS n FROM research_task_interactions")["n"]
    _log("INFO", "Task Interactions", f"{task_interactions} task interaction record(s)")


# ---------------------------------------------------------------------------
# 7. Additional Critical Checks
# ---------------------------------------------------------------------------

def check_additional(conn_sensor: Optional[sqlite3.Connection],
                     conn_research: Optional[sqlite3.Connection],
                     state: Optional[dict],
                     data_dir: Path) -> None:
    _section("7 · Additional Integrity & Study Checks")

    # --- Notification history consistency ---
    _subsection("Notification History")
    if state:
        notif_hist = state.get("notification_history", [])
        _log("INFO", "In-Memory Notification History",
             f"{len(notif_hist)} notification(s) stored in state.json")

        # Check for duplicate notification IDs
        if notif_hist:
            ids = [
                n.get("notification_id") or n.get("id")
                for n in notif_hist
                if isinstance(n, dict) and ("notification_id" in n or "id" in n)
            ]
            if len(ids) != len(set(ids)):
                _log("WARNING", "Duplicate Notification IDs",
                     f"{len(ids) - len(set(ids))} duplicate ID(s) in notification_history.",
                     "Duplicate IDs will cause the notification selector dropdown to "
                     "show stale entries and confuse Q-table lookups.")
            else:
                _log("OK", "Notification IDs", "No duplicate notification IDs")

    # --- Research DB data volume for a meaningful study ---
    _subsection("Study Data Volume")
    if conn_research:
        total_rl = _q1(conn_research, "SELECT COUNT(*) AS n FROM research_rl_episodes")["n"]
        total_nudges_db = _q1(conn_research, "SELECT COUNT(*) AS n FROM research_nudge_log")["n"]
        total_agg = _q1(conn_research, "SELECT COUNT(*) AS n FROM research_daily_aggregates")["n"]

        if total_agg < 7:
            _log("WARNING", "Study Data Maturity",
                 f"Only {total_agg} day(s) of aggregate data - study is in early stage.",
                 "Minimum ~14 days of baseline + active data needed for analysis.")
        elif total_agg < 14:
            _log("INFO", "Study Data Maturity",
                 f"{total_agg} day(s) of data - approaching minimum for analysis")
        else:
            _log("OK", "Study Data Maturity",
                 f"{total_agg} day(s) of aggregate data - sufficient for analysis")

        if total_rl < 50 and total_agg > 7:
            _log("WARNING", "RL Episode Volume",
                 f"Only {total_rl} RL episodes after {total_agg} study days.",
                 "The Q-table has very little training data. "
                 "Shadow learning may not be producing episodes at the expected rate.")
        else:
            _log("INFO", "RL Episode Volume",
                 f"{total_rl} total RL episodes across {total_agg} study days")

    # --- Sensor data rolling DB health ---
    _subsection("Sensor DB Data Retention")
    if conn_sensor:
        oldest_row = _q1(conn_sensor,
                         "SELECT MIN(timestamp) AS oldest FROM sensor_history")
        newest_row = _q1(conn_sensor,
                         "SELECT MAX(timestamp) AS newest FROM sensor_history")
        if oldest_row and oldest_row["oldest"] and newest_row and newest_row["newest"]:
            oldest_dt = datetime.fromtimestamp(oldest_row["oldest"])
            newest_dt = datetime.fromtimestamp(newest_row["newest"])
            span_days = (newest_dt - oldest_dt).days
            _log("INFO", "Sensor DB Span",
                 f"Data spans {span_days} day(s): {oldest_dt.strftime('%Y-%m-%d')} → "
                 f"{newest_dt.strftime('%Y-%m-%d')}")
            if span_days > 16:
                _log("WARNING", "Sensor DB Retention",
                     f"Sensor history spans {span_days} days (purge threshold is 14 days).",
                     "The scheduled cleanup may not be running. "
                     "Check _cleanup_old_data() execution in StorageManager.")

    # --- Backup health ---
    _subsection("Backup Health")
    backup_dir = data_dir / "backups"
    if not backup_dir.exists():
        _log("WARNING", "Backup Directory",
             "backups/ directory does not exist.",
             "Backups have never been created. "
             "If HA has been running for > 6h, the automatic backup job should have triggered.")
    else:
        for subdir, label in [("auto", "Auto Backups"),
                               ("startup", "Startup Backups"),
                               ("shutdown", "Shutdown Backups")]:
            bdir = backup_dir / subdir
            if bdir.exists():
                entries = list(bdir.iterdir())
                db_backups = [e for e in entries
                              if e.is_dir() or e.suffix in (".db", ".json")]
                if not db_backups:
                    _log("WARNING", label, f"backup/{subdir}/ exists but is empty.")
                else:
                    newest_backup = max(
                        (e.stat().st_mtime for e in bdir.rglob("*") if e.is_file()),
                        default=0
                    )
                    age_h = (time.time() - newest_backup) / 3600
                    if age_h > 12:
                        _log("WARNING", label,
                             f"Newest backup in {subdir}/ is {age_h:.1f}h old "
                             f"(expected every 6h).",
                             "The BackupManager periodic task may have failed or HA restarted "
                             "without completing a backup cycle.")
                    else:
                        _log("OK", label,
                             f"{len(db_backups)} backup set(s), newest {age_h:.1f}h ago")

    # --- Configuration integrity ---
    _subsection("Configuration Integrity")
    if state:
        start_date_raw = state.get("start_date")
        if start_date_raw:
            try:
                start_dt = datetime.fromisoformat(start_date_raw)
                study_days = (datetime.now() - start_dt).days
                _log("INFO", "Study Duration",
                     f"Started: {start_dt.strftime('%Y-%m-%d')} "
                     f"({study_days} day(s) ago)")
                if study_days > 90:
                    _log("WARNING", "Study Duration",
                         f"Study is {study_days} days old - approaching/exceeding the 90-day target.",
                         "Ensure data export and analysis pipeline is ready.")
            except ValueError:
                _log("WARNING", "Study Start Date",
                     f"Cannot parse start_date: '{start_date_raw}'.")

        # Notification count sanity
        notif_today = state.get("notification_count_today", 0)
        last_notif_date_raw = state.get("last_notification_date")
        if last_notif_date_raw:
            try:
                last_notif_date_obj = date.fromisoformat(last_notif_date_raw)
                if last_notif_date_obj < date.today() and notif_today > 0:
                    _log("WARNING", "Notification Count",
                         f"notification_count_today = {notif_today} but "
                         f"last_notification_date = {last_notif_date_raw} (yesterday or older).",
                         "The daily counter is not resetting at midnight. "
                         "Check the date-reset logic inside process_ai_model().")
                else:
                    _log("OK", "Notification Count",
                         f"{notif_today} notification(s) today (date: {last_notif_date_raw})")
            except ValueError:
                pass

    # --- WAL checkpoint health ---
    _subsection("WAL Checkpoint Status")
    for label, path in [("sensor_data.db", data_dir / "sensor_data.db"),
                         ("research_data.db", data_dir / "research_data.db")]:
        if not path.exists():
            continue
        conn_tmp = _open_db(path)
        if conn_tmp:
            try:
                row = _q1(conn_tmp,
                          "PRAGMA wal_checkpoint(PASSIVE)")
                # Returns (busy, log, checkpointed)
                if row:
                    busy, log_frames, ckpt_frames = row[0], row[1], row[2]
                    if busy > 0:
                        _log("WARNING", f"WAL Checkpoint ({label})",
                             f"busy={busy} writer(s) blocking checkpoint "
                             f"(log={log_frames}, checkpointed={ckpt_frames}).",
                             "An active write transaction is blocking the checkpoint. "
                             "This is usually transient but repeated occurrences indicate lock contention.")
                    else:
                        _log("OK", f"WAL Checkpoint ({label})",
                             f"Checkpoint OK (log={log_frames} frames, "
                             f"checkpointed={ckpt_frames} frames)")
            except Exception:
                pass
            conn_tmp.close()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Green Shift - System Health Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python health_monitor.py\n"
               "  python health_monitor.py --data-dir /config/green_shift_data"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join("config", "green_shift_data"),
        help="Path to the green_shift_data directory "
             "(default: config/green_shift_data relative to cwd)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()

    # Header
    width = 74
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dir_str = str(data_dir)
    h_top   = "╔" + "═" * (width - 2) + "╗"
    h_bot   = "╚" + "═" * (width - 2) + "╝"
    h_title = "║" + " GREEN SHIFT \u2014 SYSTEM HEALTH MONITOR ".center(width - 2) + "║"
    h_time  = "║" + (" Run at " + now_str + " ").center(width - 2) + "║"
    h_dir   = "║" + (" Data directory: " + dir_str + " ").center(width - 2) + "║"
    print()
    print(f"{C.MAGENTA}{C.BOLD}{h_top}{C.RESET}")
    print(f"{C.MAGENTA}{C.BOLD}{h_title}{C.RESET}")
    print(f"{C.MAGENTA}{C.BOLD}{h_time}{C.RESET}")
    print(f"{C.MAGENTA}{C.BOLD}{h_dir}{C.RESET}")
    print(f"{C.MAGENTA}{C.BOLD}{h_bot}{C.RESET}")

    if not data_dir.exists():
        print(f"\n{C.RED}{C.BOLD}ERROR: Data directory not found: {data_dir}{C.RESET}")
        print(f"{C.DIM}Run with --data-dir to specify a custom path.{C.RESET}\n")
        sys.exit(1)

    # Run all checks
    conns = check_database_health(data_dir)
    conn_sensor   = conns.get("sensor_data.db")
    conn_research = conns.get("research_data.db")

    check_data_freshness(conn_sensor)
    state = check_ai_health(data_dir)  # type: Optional[dict]
    check_rl_and_nudges(conn_research)
    check_gamification(conn_sensor, state)
    check_research_aggregates(conn_research)
    check_additional(conn_sensor, conn_research, state, data_dir)

    # Close connections
    for conn in [conn_sensor, conn_research]:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    _summary()


if __name__ == "__main__":
    main()
