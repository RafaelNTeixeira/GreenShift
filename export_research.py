"""
Export Research Data Script
Exports all research data from research_data.db to CSV files for analysis.

Usage:
    python export_research.py <db_path> [output_dir]

Example:
    python export_research.py ./config/green_shift_data/research_data.db ./research_export
    python export_research.py /path/to/research_data.db /path/to/output
"""

import sqlite3
import sys
import argparse
from pathlib import Path

def export_research_data(db_path: str, output_dir: str):
    """Export all research data to CSV files."""
    
    db_path = Path(db_path)
    output_path = Path(output_dir)
    
    # Verify database exists
    if not db_path.exists():
        print(f"❌ Research database not found: {db_path}")
        return False
    
    print(f"✅ Found database: {db_path}")
    print(f"   Size: {db_path.stat().st_size / 1024:.2f} KB\n")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_path.absolute()}\n")
    
    # Check if pandas is available
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas not installed!")
        print("   Install with: pip install pandas")
        return False
    
    # Tables to export
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
    
    conn = sqlite3.connect(str(db_path))
    
    print("Exporting tables...")
    print("=" * 80)
    
    successful = 0
    failed = 0
    
    for table in tables:
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            if row_count == 0:
                print(f"⚠️  {table:<40} (0 rows - skipped)")
                continue
            
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            file_path = output_path / f"{table}.csv"
            df.to_csv(file_path, index=False)
            
            print(f"✅ {table:<40} ({row_count:>6} rows) → {file_path.name}")
            successful += 1
            
        except Exception as e:
            print(f"❌ {table:<40} ERROR: {e}")
            failed += 1
    
    conn.close()
    
    print("=" * 80)
    print(f"\n📊 Export Summary")
    print(f"   ✅ Successful: {successful} tables")
    if failed > 0:
        print(f"   ❌ Failed: {failed} tables")
    print(f"   📁 Output: {output_path.absolute()}\n")
    
    return failed == 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export research data from SQLite database to CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "db_path",
        metavar="DB_PATH",
        help="Path to the research_data.db SQLite database"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="research_export",
        metavar="OUTPUT_DIR",
        help="Directory to write CSV files to (default: ./research_export)"
    )

    args = parser.parse_args()

    success = export_research_data(args.db_path, args.output_dir)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())