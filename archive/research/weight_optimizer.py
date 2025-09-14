import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import sys
import traceback
from data_loader import load_pbp_data, load_schedule
from stats_calculator import generate_stable_matchup_line
import time

def test_weight_window(season, recent_games_window, recent_weight, value_threshold=5.0):
    """
    Test a specific weighting window configuration for a given season
    
    Args:
        season (int): Season year to test
        recent_games_window (int): Number of recent games to use in window
        recent_weight (float): Weight to apply to recent games (0.0-1.0)
        value_threshold (float): Value threshold for picks
        
    Returns:
        tuple: (wins, losses, total_picks)
    """
    try:
        print(f"  - Loading PBP and Schedule data for {season}...")
        
        # Set global variables for stats_calculator.py
        global RECENT_GAMES_WINDOW
        global RECENT_WEIGHT
        
        # Override the default window size and weight
        import stats_calculator
        stats_calculator.RECENT_GAMES_WINDOW = recent_games_window
        stats_calculator.RECENT_WEIGHT = recent_weight
        
        # Load data
        pbp_df = load_pbp_data(season)
        schedule_df = load_schedule(season)
        
        if pbp_df is None or schedule_df is None:
            print(f"Data not available for {season}")
            return 0, 0, 0
            
        print(f"{season} done.")
        
        # Filter schedule to regular season games with spreads
        schedule_df = schedule_df[
            (schedule_df['game_type'] == 'REG') & 
            (~schedule_df['spread_line'].isna())
        ]
        
        wins = 0
        losses = 0
        total_picks = 0
        
        for _, game in schedule_df.iterrows():
            if pd.isna(game['spread_line']):
                continue
                
            vegas_spread = float(game['spread_line'])
            home_team = game['home_team']
            away_team = game['away_team']
            game_date = game['gameday']
            
            if pd.isna(game_date) or pd.isna(home_team) or pd.isna(away_team):
                continue
                
            try:
                # Get our model's prediction
                matchup_line, _ = generate_stable_matchup_line(
                    pbp_df, 
                    home_team, 
                    away_team, 
                    game_date,
                    return_weights=True
                )
                
                # Calculate the value difference
                value_diff = abs(vegas_spread - matchup_line)
                
                # Only count picks that meet our value threshold
                if value_diff >= value_threshold:
                    total_picks += 1
                    
                    # Determine if our pick is home or away team
                    our_pick = "home" if matchup_line < vegas_spread else "away"
                    
                    # Get the actual game result
                    home_score = game['home_score']
                    away_score = game['away_score']
                    
                    if pd.isna(home_score) or pd.isna(away_score):
                        total_picks -= 1  # Don't count this game if no score
                        continue
                        
                    # Determine if our pick won
                    actual_diff = home_score - away_score
                    
                    if (our_pick == "home" and actual_diff > vegas_spread) or \
                       (our_pick == "away" and actual_diff < vegas_spread):
                        wins += 1
                    else:
                        losses += 1
                        
            except Exception as e:
                print(f"Error processing game {home_team} vs {away_team}: {str(e)}")
                continue
                
        win_rate = (wins / total_picks * 100) if total_picks > 0 else 0
        print(f"  Season {season}: {wins}-{losses} ({win_rate:.1f}%) - {total_picks} picks")
        return wins, losses, total_picks
        
    except Exception as e:
        print(f"Error loading data for {season}: {str(e)}")
        return 0, 0, 0

def main():
    # Define the ranges to test
    seasons = [2024, 2023, 2022]
    window_sizes = [4, 6, 8, 10, 12]
    window_weights = [0.10, 0.15, 0.20, 0.25, 0.30]
    value_thresholds = [5.0]  # Using 5.0 as our primary threshold based on results
    
    results = []
    
    print("Testing different weighting windows...")
    print("=" * 80)
    
    # Test all combinations
    for window_size in window_sizes:
        for weight in window_weights:
            for threshold in value_thresholds:
                print(f"\nWindow Size: {window_size} games, Weight: {weight*100:.0f}%, Threshold: {threshold}")
                print("-" * 60)
                
                total_wins = 0
                total_losses = 0
                total_picks = 0
                season_results = {}
                
                for season in seasons:
                    try:
                        wins, losses, picks = test_weight_window(
                            season, 
                            window_size, 
                            weight, 
                            threshold
                        )
                        
                        total_wins += wins
                        total_losses += losses
                        total_picks += picks
                        
                        if picks > 0:
                            season_results[season] = {
                                'wins': wins,
                                'losses': losses,
                                'picks': picks,
                                'win_rate': (wins / picks * 100) if picks > 0 else 0
                            }
                    except Exception as e:
                        print(f"Error testing {season}: {str(e)}")
                        traceback.print_exc()
                
                if total_picks > 0:
                    win_rate = (total_wins / total_picks * 100)
                    print(f"\n  OVERALL: {total_wins}-{total_losses} ({win_rate:.1f}%) - {total_picks} picks")
                    
                    results.append({
                        'window_size': window_size,
                        'weight': weight,
                        'threshold': threshold,
                        'wins': total_wins,
                        'losses': total_losses,
                        'picks': total_picks,
                        'win_rate': win_rate,
                        'season_results': season_results
                    })
                else:
                    print("\n  OVERALL: No picks identified")
    
    # Sort results by win rate and print top configurations
    if results:
        results.sort(key=lambda x: x['win_rate'], reverse=True)
        
        print("\n\n")
        print("=" * 80)
        print("TOP PERFORMING WEIGHT CONFIGURATIONS")
        print("=" * 80)
        
        for i, result in enumerate(results[:10]):
            print(f"{i+1}. Window Size: {result['window_size']} games, Weight: {result['weight']*100:.0f}%, Threshold: {result['threshold']}")
            print(f"   Record: {result['wins']}-{result['losses']} ({result['win_rate']:.1f}%)")
            print(f"   Total Picks: {result['picks']}")
            
            for season, season_data in result['season_results'].items():
                print(f"     {season}: {season_data['wins']}-{season_data['losses']} ({season_data['win_rate']:.1f}%)")
                
            print()

if __name__ == "__main__":
    main()
