# simple_backtest.py

import pandas as pd
import nfl_data_py as nfl
from stats_calculator import (
    get_last_n_games_pbp,
    calculate_granular_epa_stats,
    calculate_weighted_stats,
    generate_stable_matchup_line
)

def run_backtest(year, weeks, value_threshold):
    """
    Runs a simplified backtest for a single season with the dynamic weighting model.
    """
    print(f"  - Testing {year} season with {value_threshold} point threshold...")
    
    # Load data
    pbp_data = nfl.import_pbp_data([year], downcast=True)
    schedule_data = nfl.import_schedules([year])
    
    # Track results
    wins = 0
    losses = 0
    
    # Set model parameters
    recent_games_window = 8
    recent_form_weight = 0.20
    
    for week in weeks:
        week_schedule = schedule_data[schedule_data['week'] == week]
        pbp_upto_week = pbp_data[pbp_data['week'] < week]
        
        if week_schedule.empty or pbp_upto_week.empty:
            continue

        for game in week_schedule.itertuples():
            home_team, away_team = game.home_team, game.away_team
            if pd.isna(game.result): 
                continue

            # Get Vegas spread
            home_ml = getattr(game, 'home_moneyline', None)
            away_ml = getattr(game, 'away_moneyline', None)
            spread_magnitude = abs(getattr(game, 'spread_line', 0))
            
            if home_ml is not None and away_ml is not None:
                if home_ml < away_ml: 
                    home_spread_vegas = -spread_magnitude
                else: 
                    home_spread_vegas = spread_magnitude
            else:
                continue
            
            # Calculate model prediction
            try:
                # Get full season stats
                away_stats_std = calculate_granular_epa_stats(pbp_upto_week, away_team)
                home_stats_std = calculate_granular_epa_stats(pbp_upto_week, home_team)
                
                # Get recent stats
                pbp_away_recent = get_last_n_games_pbp(pbp_upto_week, away_team, recent_games_window)
                pbp_home_recent = get_last_n_games_pbp(pbp_upto_week, home_team, recent_games_window)
                away_stats_recent = calculate_granular_epa_stats(pbp_away_recent, away_team)
                home_stats_recent = calculate_granular_epa_stats(pbp_home_recent, home_team)
                
                # Skip if we don't have enough data
                if not all([home_stats_std, away_stats_std, home_stats_recent, away_stats_recent]): 
                    continue
                
                # Calculate weighted stats
                away_stats_w = calculate_weighted_stats(away_stats_std, away_stats_recent, 1 - recent_form_weight, recent_form_weight)
                home_stats_w = calculate_weighted_stats(home_stats_std, home_stats_recent, 1 - recent_form_weight, recent_form_weight)
                
                # Get model prediction
                model_line, _ = generate_stable_matchup_line(home_stats_w, away_stats_w, return_weights=True)
                model_home_spread = -model_line
                
                # Calculate edge and make pick
                model_edge = home_spread_vegas - model_home_spread
                
                if abs(model_edge) >= value_threshold:
                    pick = home_team if model_edge > 0 else away_team
                    
                    # Evaluate pick
                    actual_home_margin = game.result
                    if (actual_home_margin + home_spread_vegas) > 0:
                        covering_team = home_team
                    elif (actual_home_margin + home_spread_vegas) < 0:
                        covering_team = away_team
                    else:
                        covering_team = "Push"
                    
                    if covering_team != "Push":
                        if pick == covering_team:
                            wins += 1
                        else:
                            losses += 1
            except Exception as e:
                continue
    
    return wins, losses

if __name__ == "__main__":
    # Set up parameters
    YEARS = [2024, 2023, 2022]
    WEEKS = range(4, 18)  # Weeks 4-17
    THRESHOLDS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS WITH DYNAMIC WEIGHTING MODEL")
    print("="*50)
    
    # Test each year separately
    for year in YEARS:
        print(f"\nSEASON {year}:")
        print("-" * 40)
        
        for threshold in THRESHOLDS:
            wins, losses = run_backtest(year, WEEKS, threshold)
            total = wins + losses
            
            if total > 0:
                win_rate = (wins / total) * 100
                print(f"  {threshold} points: {wins}-{losses} ({win_rate:.1f}%) - {total} picks")
            else:
                print(f"  {threshold} points: No qualifying picks")
    
    # Test all years combined for each threshold
    print("\n" + "="*50)
    print("COMBINED RESULTS (ALL SEASONS):")
    print("="*50)
    
    for threshold in THRESHOLDS:
        total_wins = 0
        total_losses = 0
        
        for year in YEARS:
            wins, losses = run_backtest(year, WEEKS, threshold)
            total_wins += wins
            total_losses += losses
        
        total = total_wins + total_losses
        if total > 0:
            win_rate = (total_wins / total) * 100
            print(f"  {threshold} points: {total_wins}-{total_losses} ({win_rate:.1f}%) - {total} picks")
        else:
            print(f"  {threshold} points: No qualifying picks")
