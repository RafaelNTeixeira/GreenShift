"""
Export Research Data Script
Exports all research data from research_data.db to CSV files for analysis.

Usage:
    python export_research.py [output_dir]

Example:
    python export_research.py ./research_export
    python export_research.py /path/to/output
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

def export_research_data(db_path: str, output_dir: str = None):
    """Export all research data to CSV files."""
    
    if output_dir is None:
        output_dir = "research_export"
    
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
            # Check if table exists and has data
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            if row_count == 0:
                print(f"⚠️  {table:<40} (0 rows - skipped)")
                continue
            
            # Export to CSV
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
    if len(sys.argv) > 2:
        print("❌ Too many arguments!")
        print(__doc__)
        return 1
    
    # Determine database path
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "research_export"
    
    # Find research database
    possible_paths = [
        Path("config/green_shift_data/research_data.db"),
        Path("green_shift_data/research_data.db"),
        Path.home() / "config/green_shift_data/research_data.db"
    ]
    
    db_path = None
    for path in possible_paths:
        if path.exists():
            db_path = path
            break
    
    if db_path is None:
        print("❌ Could not find research_data.db!")
        print("\n   Searched in:")
        for path in possible_paths:
            print(f"     - {path.absolute()}")
        print("\n   Please specify the database path:")
        print("   python export_research.py /path/to/research_data.db output_dir")
        return 1
    
    # Run export
    success = export_research_data(str(db_path), output_dir)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
