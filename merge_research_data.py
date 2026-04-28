"""
merge_research_data.py
======================
Merges cleaned per-environment CSV exports (output of clean_research_data.py) from any number of environment folders into a single set of combined CSVs ready for analysis (e.g. in SPSS).

Per-environment tables (must end with _clean.csv):
    research_area_daily_stats_clean.csv
    research_blocked_notifications_clean.csv
    research_daily_aggregates_clean.csv
    research_nudge_log_clean.csv
    research_phase_metadata_clean.csv
    research_rl_episodes_clean.csv
    research_task_interactions_clean.csv
    research_weekly_challenges_clean.csv

Global table (no environment_id added):
    research_participant_survey_clean.csv

Usage:
    python merge_research_data.py ENV_DIR [ENV_DIR ...] [options]

Arguments:
    ENV_DIR   One or more paths to cleaned-data folders (one per environment).
              The first path -> environment_id=1, second -> 2, and so on.

Options:
    --output DIR          Output folder (default: ./spss_data)
    --survey PATH         Explicit path to research_participant_survey_clean.csv.
                          If omitted the script looks for it in each ENV folder
                          and uses the first one it finds.

Examples:
    python merge_research_data.py ./env_A/cleaned_data ./env_B/cleaned_data

    python merge_research_data.py ./env_A/cleaned_data ./env_B/cleaned_data \\
                                  ./env_C/cleaned_data ./env_D/cleaned_data \\
                                  --output ./spss_data \\
                                  --survey ./global/research_participant_survey_clean.csv
"""

from __future__ import annotations

import sys
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Tables that receive an environment_id column (one file per environment folder)
PER_ENV_TABLES = [
    "research_area_daily_stats_clean",
    "research_blocked_notifications_clean",
    "research_daily_aggregates_clean",
    "research_nudge_log_clean",
    "research_phase_metadata_clean",
    "research_rl_episodes_clean",
    "research_task_interactions_clean",
    "research_weekly_challenges_clean",
]

# Table that is global across all environments (no environment_id added)
GLOBAL_TABLE = "research_participant_survey_clean"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

class MergeReport:
    """Collects merge results and prints a formatted summary at the end."""

    def __init__(self):
        self._entries: list[dict] = []

    def add(self, table: str, status: str, detail: str = ""):
        self._entries.append({"table": table, "status": status, "detail": detail})

    def print_summary(self):
        sep = "-" * 80
        print(f"\n{'═' * 80}")
        print("  MERGE REPORT")
        print(f"{'═' * 80}")
        for e in self._entries:
            icon = "✅" if e["status"] == "ok" else ("⚠ " if e["status"] == "warn" else "❌")
            print(f"  {icon}  {e['table']:<50}  {e['detail']}")
        print(f"{'═' * 80}\n")


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Reads a CSV and returns a DataFrame, or None on failure."""
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"     ⚠  Could not read {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# MERGE LOGIC
# ---------------------------------------------------------------------------

def merge_per_env_table(
    table_name: str,
    env_dirs: list[Path],
    output_dir: Path,
    report: MergeReport,
) -> None:
    """
    Loads `table_name`.csv from each environment folder, prepends
    `environment_id`, concatenates all frames and writes the result.
    """
    frames: list[pd.DataFrame] = []
    found_in: list[int] = []
    missing_in: list[int] = []

    for env_id, folder in enumerate(env_dirs, start=1):
        csv_path = folder / f"{table_name}.csv"
        if not csv_path.exists():
            missing_in.append(env_id)
            continue

        df = read_csv_safe(csv_path)
        if df is None:
            missing_in.append(env_id)
            continue

        # Prepend environment_id as the first column
        df.insert(0, "environment_id", env_id)
        frames.append(df)
        found_in.append(env_id)

    if not frames:
        report.add(table_name, "error", "No data found in any environment folder — skipped")
        print(f"  ❌ {table_name:<50}  no source files found")
        return

    merged = pd.concat(frames, ignore_index=True, sort=False)
    out_path = output_dir / f"{table_name}.csv"
    merged.to_csv(out_path, index=False)

    detail_parts = [f"{len(merged)} rows from env(s) {found_in}"]
    if missing_in:
        detail_parts.append(f"missing in env(s) {missing_in}")
    detail = " | ".join(detail_parts)

    status = "warn" if missing_in else "ok"
    report.add(table_name, status, detail)
    print(f"  {'✅' if status == 'ok' else '⚠ '}  {table_name:<50}  {detail}")


def copy_global_table(
    survey_path: Path | None,
    env_dirs: list[Path],
    output_dir: Path,
    report: MergeReport,
) -> None:
    """
    Finds and copies the global participant survey CSV to the output folder
    without modification (no environment_id is added).

    Search order:
      1. Explicit --survey path (if provided)
      2. Each env folder in input order
    """
    table_name = GLOBAL_TABLE
    resolved_path: Path | None = None

    if survey_path is not None:
        if survey_path.exists():
            resolved_path = survey_path
        else:
            print(f"  ⚠  --survey path not found: {survey_path}")

    if resolved_path is None:
        for folder in env_dirs:
            candidate = folder / f"{table_name}.csv"
            if candidate.exists():
                resolved_path = candidate
                print(f"     ℹ  Survey file found in: {folder}")
                break

    if resolved_path is None:
        report.add(table_name, "warn", "File not found in any location — skipped")
        print(f"  ⚠   {table_name:<50}  not found — skipped")
        return

    df = read_csv_safe(resolved_path)
    if df is None:
        report.add(table_name, "error", f"Could not read {resolved_path}")
        return

    out_path = output_dir / f"{table_name}.csv"
    df.to_csv(out_path, index=False)

    detail = f"{len(df)} rows (global, no environment_id) from {resolved_path.name}"
    report.add(table_name, "ok", detail)
    print(f"  ✅  {table_name:<50}  {detail}")


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def run_merge(
    env_dirs: list[Path],
    output_dir: Path,
    survey_path: Path | None,
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = MergeReport()

    print(f"\n{'═' * 80}")
    print("  GREEN SHIFT — ENVIRONMENT MERGE PIPELINE")
    print(f"{'═' * 80}")
    print(f"  Output folder : {output_dir.resolve()}")
    print(f"  Date/time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    for i, d in enumerate(env_dirs, start=1):
        print(f"  Environment {i} : {d.resolve()}")
    print()

    # -- Per-environment tables -----------------------------------------------
    print("  Merging per-environment tables …")
    for table in PER_ENV_TABLES:
        merge_per_env_table(table, env_dirs, output_dir, report)

    # -- Global survey table --------------------------------------------------
    print("\n  Copying global survey table …")
    copy_global_table(survey_path, env_dirs, output_dir, report)

    report.print_summary()
    return True


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="merge_research_data.py",
        description="Green Shift — merge cleaned per-environment CSVs for SPSS analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python merge_research_data.py ./env_A/cleaned ./env_B/cleaned\n\n"
            "  python merge_research_data.py ./env_A/cleaned ./env_B/cleaned "
            "./env_C/cleaned ./env_D/cleaned \\\n"
            "      --output ./spss_data \\\n"
            "      --survey ./global/research_participant_survey_clean.csv\n"
        ),
    )

    parser.add_argument(
        "env_dirs",
        nargs="+",
        metavar="ENV_DIR",
        help=(
            "Paths to the cleaned-data folders, one per environment (at least one). "
            "The first path becomes environment_id=1, the second environment_id=2, etc."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default="spss_data",
        help="Output folder for the merged CSVs (default: ./spss_data)",
    )
    parser.add_argument(
        "--survey",
        metavar="PATH",
        default=None,
        help=(
            "Explicit path to research_participant_survey_clean.csv. "
            "If omitted, the script searches for it inside each ENV folder "
            "and uses the first match."
        ),
    )

    args = parser.parse_args()

    env_dirs: list[Path] = []
    for raw in args.env_dirs:
        p = Path(raw)
        if not p.exists():
            print(f"❌ Environment folder not found: {p}")
            return 1
        if not p.is_dir():
            print(f"❌ Path is not a directory: {p}")
            return 1
        env_dirs.append(p)

    output_dir   = Path(args.output)
    survey_path  = Path(args.survey) if args.survey else None

    run_merge(env_dirs, output_dir, survey_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())