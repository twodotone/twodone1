import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import sys
import nfl_data_py as nfl
from stats_calculator import generate_stable_matchup_line, calculate_granular_epa_stats, calculate_weighted_stats, get_last_n_games_pbp
import time

# Constants - these are the same values we're using in our model
RECENT_GAMES_WINDOW = 8
RECENT_WEIGHT = 0.30  # The optimized weight we discovered

def load_data(season):
    """
    Load NFL data for a specific season
    """
    try:
        print(f"Loading data for {season}...")
        
        # First try to load from local parquet files
        file_path = os.path.join("data", f"pbp_{season}.parquet")
        if os.path.exists(file_path):
            pbp_data = pd.read_parquet(file_path)
            print(f"Loaded {len(pbp_data)} rows from local parquet file.")
        else:
            # Fall back to nfl_data_py
            pbp_data = nfl.import_pbp_data([season], downcast=True)
            print(f"Loaded {len(pbp_data)} rows from nfl_data_py API.")
            
        schedule_data = nfl.import_schedules([season])
        print(f"Loaded {len(schedule_data)} schedule rows.")
        
        return pbp_data, schedule_data
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_data_leakage(season):
    """
    Analyze potential data leakage for a specific season by checking:
    1. Data availability and timing
    2. Season-to-season result correlations
    3. Model accuracy consistency
    
    Args:
        season (int): Season year to analyze
    """
    print(f"Analyzing potential data leakage for {season} season...")
    
    try:
        # Load data
        pbp_df, schedule_df = load_data(season)
        
        if pbp_df is None or schedule_df is None:
            print(f"Data not available for {season}")
            return
            
        # 1. Check for data beyond the game date
        print("\n1. Checking for data timing issues...")
        schedule_df = schedule_df[schedule_df['game_type'] == 'REG']
        
        for _, game in schedule_df.iterrows():
            game_date = game['gameday']
            home_team = game['home_team']
            away_team = game['away_team']
            
            if pd.isna(game_date) or pd.isna(home_team) or pd.isna(away_team):
                continue
                
            # Check if we have PBP data for games after this game date
            game_date_obj = pd.to_datetime(game_date)
            future_games = pbp_df[pd.to_datetime(pbp_df['game_date']) > game_date_obj]
            
            if not future_games.empty:
                # This is potentially correct behavior since we're using all season data
                # But it's worth checking if we're using future data inappropriately
                game_key = f"{home_team} vs {away_team} on {game_date}"
                future_game_count = len(future_games['game_id'].unique())
                print(f"  - For {game_key}, model has access to {future_game_count} future games")
        
        # 2. Check for home vs away model consistency
        print("\n2. Checking for home vs away model consistency...")
        inconsistency_count = 0
        total_checks = 0
        
        for _, game in schedule_df.iterrows():
            game_date = game['gameday']
            home_team = game['home_team']
            away_team = game['away_team']
            
            if pd.isna(game_date) or pd.isna(home_team) or pd.isna(away_team):
                continue
                
            try:
                # Get our model's prediction in both directions
                # First calculate stats for the home team
                home_full_season_stats = calculate_granular_epa_stats(pbp_df, home_team)
                home_recent_games_pbp = get_last_n_games_pbp(pbp_df, home_team, RECENT_GAMES_WINDOW)
                home_recent_stats = calculate_granular_epa_stats(home_recent_games_pbp, home_team)
                
                # Calculate stats for the away team
                away_full_season_stats = calculate_granular_epa_stats(pbp_df, away_team)
                away_recent_games_pbp = get_last_n_games_pbp(pbp_df, away_team, RECENT_GAMES_WINDOW)
                away_recent_stats = calculate_granular_epa_stats(away_recent_games_pbp, away_team)
                
                # Calculate weighted stats
                full_season_weight = 1.0 - RECENT_WEIGHT
                home_weighted_stats = calculate_weighted_stats(home_full_season_stats, home_recent_stats, full_season_weight, RECENT_WEIGHT)
                away_weighted_stats = calculate_weighted_stats(away_full_season_stats, away_recent_stats, full_season_weight, RECENT_WEIGHT)
                
                # Get normal direction prediction
                try:
                    line_normal, _, _, _ = generate_stable_matchup_line(
                        home_weighted_stats, 
                        away_weighted_stats, 
                        return_weights=True,
                        pbp_df=pbp_df,
                        home_team=home_team,
                        away_team=away_team
                    )
                except ValueError:
                    line_normal, _, _ = generate_stable_matchup_line(
                        home_weighted_stats, 
                        away_weighted_stats, 
                        return_weights=True,
                        pbp_df=pbp_df,
                        home_team=home_team,
                        away_team=away_team
                    )
                
                # Now reversed direction
                try:
                    line_reversed, _, _, _ = generate_stable_matchup_line(
                        away_weighted_stats, 
                        home_weighted_stats, 
                        return_weights=True,
                        pbp_df=pbp_df,
                        home_team=away_team,
                        away_team=home_team
                    )
                except ValueError:
                    line_reversed, _, _ = generate_stable_matchup_line(
                        away_weighted_stats, 
                        home_weighted_stats, 
                        return_weights=True,
                        pbp_df=pbp_df,
                        home_team=away_team,
                        away_team=home_team
                    )
                
                # Check if they're roughly opposites (allowing for small floating point differences)
                if abs(line_normal + line_reversed) > 0.5:  # More than 0.5 point difference
                    print(f"  - Inconsistency: {home_team} vs {away_team}")
                    print(f"    Home perspective: {line_normal:.1f}, Away perspective: {line_reversed:.1f}")
                    print(f"    Difference: {abs(line_normal + line_reversed):.1f} points\n")
                    inconsistency_count += 1
                    
                total_checks += 1
                
            except Exception as e:
                print(f"Error processing game {home_team} vs {away_team}: {str(e)}")
                continue
                
        if total_checks > 0:
            print(f"  Found {inconsistency_count} inconsistencies out of {total_checks} checks ({inconsistency_count/total_checks*100:.1f}%)")
        else:
            print("  No consistency checks performed")
            
        # 3. Check for correlation with final scores
        print("\n3. Checking correlation between model predictions and actual results...")
        predictions = []
        actuals = []
        
        for _, game in schedule_df.iterrows():
            if pd.isna(game['spread_line']):
                continue
                
            vegas_spread = float(game['spread_line'])
            home_team = game['home_team']
            away_team = game['away_team']
            game_date = game['gameday']
            
            if pd.isna(game_date) or pd.isna(home_team) or pd.isna(away_team):
                continue
                
            # Get actual score differential
            home_score = game['home_score'] 
            away_score = game['away_score']
            
            if pd.isna(home_score) or pd.isna(away_score):
                continue
                
            actual_diff = home_score - away_score
            
            try:
                # Get our model's prediction
                # First calculate stats for the home team
                home_full_season_stats = calculate_granular_epa_stats(pbp_df, home_team)
                home_recent_games_pbp = get_last_n_games_pbp(pbp_df, home_team, RECENT_GAMES_WINDOW)
                home_recent_stats = calculate_granular_epa_stats(home_recent_games_pbp, home_team)
                
                # Calculate stats for the away team
                away_full_season_stats = calculate_granular_epa_stats(pbp_df, away_team)
                away_recent_games_pbp = get_last_n_games_pbp(pbp_df, away_team, RECENT_GAMES_WINDOW)
                away_recent_stats = calculate_granular_epa_stats(away_recent_games_pbp, away_team)
                
                # Calculate weighted stats
                full_season_weight = 1.0 - RECENT_WEIGHT
                home_weighted_stats = calculate_weighted_stats(home_full_season_stats, home_recent_stats, full_season_weight, RECENT_WEIGHT)
                away_weighted_stats = calculate_weighted_stats(away_full_season_stats, away_recent_stats, full_season_weight, RECENT_WEIGHT)
                
                # Get prediction
                try:
                    model_line, _, _, _ = generate_stable_matchup_line(
                        home_weighted_stats, 
                        away_weighted_stats, 
                        return_weights=True,
                        pbp_df=pbp_df,
                        home_team=home_team,
                        away_team=away_team
                    )
                except ValueError:
                    model_line, _, _ = generate_stable_matchup_line(
                        home_weighted_stats, 
                        away_weighted_stats, 
                        return_weights=True,
                        pbp_df=pbp_df,
                        home_team=home_team,
                        away_team=away_team
                    )
                
                predictions.append(model_line)
                actuals.append(actual_diff)
                
            except Exception as e:
                print(f"Error processing game {home_team} vs {away_team}: {str(e)}")
                continue
                
        if predictions and actuals:
            correlation = np.corrcoef(predictions, actuals)[0, 1]
            print(f"  Correlation between model predictions and actual results: {correlation:.3f}")
            
            # Calculate Mean Absolute Error
            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            print(f"  Mean Absolute Error (MAE): {mae:.2f} points")
        else:
            print("  Not enough data to calculate correlation")
            
    except Exception as e:
        print(f"Error analyzing data for {season}: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_week_by_week_data(season):
    """
    Analyze data availability on a week-by-week basis to check for data leakage
    """
    print(f"\nAnalyzing week-by-week data for {season} season...")
    
    pbp_df, schedule_df = load_data(season)
    
    if pbp_df is None or schedule_df is None:
        print("No data available")
        return
        
    # Filter for regular season games
    schedule_df = schedule_df[schedule_df['game_type'] == 'REG']
    
    # Check data availability by week
    for week in sorted(schedule_df['week'].unique()):
        print(f"\nWeek {week}:")
        
        # Get games for this week
        week_games = schedule_df[schedule_df['week'] == week]
        
        # Data that should be available up to this week
        pbp_up_to_week = pbp_df[pbp_df['week'] < week]
        pbp_this_week = pbp_df[pbp_df['week'] == week]
        
        games_before = pbp_up_to_week['game_id'].nunique()
        games_this_week = pbp_this_week['game_id'].nunique()
        
        print(f"  Available games before Week {week}: {games_before}")
        print(f"  Games in Week {week}: {games_this_week}")
        
        # Check for games in this week that shouldn't be in the "up to week" data
        if not pbp_this_week.empty and not pbp_up_to_week.empty:
            this_week_ids = set(pbp_this_week['game_id'].unique())
            up_to_week_ids = set(pbp_up_to_week['game_id'].unique())
            
            overlap = this_week_ids.intersection(up_to_week_ids)
            
            if overlap:
                print(f"  DATA LEAKAGE DETECTED: {len(overlap)} games from Week {week} are in the prior data")
                for game_id in overlap:
                    game_data = pbp_this_week[pbp_this_week['game_id'] == game_id]
                    home_team = game_data['home_team'].iloc[0] if 'home_team' in game_data.columns else "Unknown"
                    away_team = game_data['away_team'].iloc[0] if 'away_team' in game_data.columns else "Unknown"
                    print(f"    Leakage: {home_team} vs {away_team} (Game ID: {game_id})")
            else:
                print("  No data leakage detected")

def main():
    # Analyze each season for potential data leakage
    for season in [2022, 2023, 2024]:
        print("\n" + "=" * 60)
        print(f"ANALYZING SEASON {season}")
        print("=" * 60)
        
        # Perform standard leakage checks
        analyze_data_leakage(season)
        
        # Check week-by-week data availability
        analyze_week_by_week_data(season)

if __name__ == "__main__":
    main()
