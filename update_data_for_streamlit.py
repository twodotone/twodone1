#!/usr/bin/env python3
"""
NFL Data Update Script for Streamlit Deployment
Updates play-by-play data, schedules, and betting lines
"""

import argparse
import nfl_data_py as nfl
import pandas as pd
import os
from datetime import datetime
import json

def create_data_dir():
    """Create data directory if it doesn't exist"""
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created data directory")

def update_pbp_data(years, force_update=False):
    """Update play-by-play data for specified years"""
    print(f"Updating play-by-play data for years: {years}")
    
    for year in years:
        file_path = f"data/pbp_{year}.parquet"
        
        # Check if file exists and if we should skip
        if os.path.exists(file_path) and not force_update:
            print(f"PBP {year} already exists, skipping...")
            continue
            
        try:
            print(f"Downloading PBP data for {year}...")
            pbp = nfl.import_pbp_data([year])
            
            if not pbp.empty:
                pbp.to_parquet(file_path)
                print(f"✅ Saved PBP {year}: {len(pbp):,} plays")
            else:
                print(f"⚠️ No PBP data available for {year}")
                
        except Exception as e:
            print(f"❌ Error downloading PBP {year}: {e}")

def update_schedule_data(years, force_update=True):
    """Update schedule data for specified years (includes betting lines)"""
    print(f"Updating schedule data for years: {years}")
    
    for year in years:
        file_path = f"data/schedule_{year}.parquet"
        
        try:
            print(f"Downloading schedule data for {year}...")
            schedule = nfl.import_schedules([year])
            
            if not schedule.empty:
                schedule.to_parquet(file_path)
                print(f"✅ Saved schedule {year}: {len(schedule):,} games")
            else:
                print(f"⚠️ No schedule data available for {year}")
                
        except Exception as e:
            print(f"❌ Error downloading schedule {year}: {e}")

def update_team_data():
    """Update team information"""
    try:
        print("Updating team data...")
        teams = nfl.import_team_desc()
        
        if not teams.empty:
            teams.to_parquet("data/teams.parquet")
            print(f"✅ Saved team data: {len(teams)} teams")
        else:
            print("⚠️ No team data available")
            
    except Exception as e:
        print(f"❌ Error downloading team data: {e}")

def update_current_season_stats(current_year=2025):
    """Update current season play-by-play data"""
    print(f"Updating {current_year} season stats...")
    
    try:
        # Try to get current season data
        pbp_current = nfl.import_pbp_data([current_year])
        
        if not pbp_current.empty:
            pbp_current.to_parquet(f"data/pbp_{current_year}.parquet")
            print(f"✅ Updated {current_year} season: {len(pbp_current):,} plays")
        else:
            print(f"⚠️ No {current_year} season data available yet")
            
    except Exception as e:
        print(f"❌ Error updating {current_year} season: {e}")

def update_betting_lines(current_year=2025):
    """Update betting lines (part of schedule data)"""
    print(f"Updating betting lines for {current_year}...")
    update_schedule_data([current_year], force_update=True)

def create_update_log(update_type):
    """Create update log with timestamp"""
    log_data = {
        'last_update': datetime.utcnow().isoformat() + 'Z',
        'update_type': update_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('data/update_log.json', 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"✅ Created update log: {update_type}")

def main():
    parser = argparse.ArgumentParser(description='Update NFL data for Streamlit app')
    parser.add_argument('--update-2025-stats', action='store_true', 
                       help='Update 2025 season play-by-play data')
    parser.add_argument('--lines-only', action='store_true',
                       help='Update betting lines only')
    parser.add_argument('--full-update', action='store_true',
                       help='Update all data (historical + current + lines)')
    
    args = parser.parse_args()
    
    # Create data directory
    create_data_dir()
    
    if args.lines_only:
        print("🎯 LINES-ONLY UPDATE (Daily)")
        update_betting_lines(2025)
        create_update_log('lines_only')
        
    elif args.update_2025_stats:
        print("📊 FULL UPDATE WITH 2025 STATS (Tuesday)")
        update_current_season_stats(2025)
        update_betting_lines(2025)
        update_team_data()
        create_update_log('full_with_2025')
        
    elif args.full_update:
        print("🔄 COMPLETE DATA UPDATE")
        # Update historical data
        update_pbp_data([2022, 2023, 2024])
        update_schedule_data([2022, 2023, 2024, 2025])
        update_team_data()
        # Update current season
        update_current_season_stats(2025)
        create_update_log('full_complete')
        
    else:
        # Default: Update lines only
        print("🎯 DEFAULT: LINES-ONLY UPDATE")
        update_betting_lines(2025)
        create_update_log('default_lines')
    
    print("\n✅ Data update complete!")
    
    # Show summary
    if os.path.exists('data'):
        files = [f for f in os.listdir('data') if f.endswith('.parquet')]
        print(f"\n📁 Data files available: {len(files)}")
        for file in sorted(files):
            file_path = os.path.join('data', file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  • {file} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
