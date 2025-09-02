# threshold_backtest.py

import pandas as pd
import nfl_data_py as nfl
from stats_calculator import (
    get_last_n_games_pbp,
    calculate_granular_epa_stats,
    calculate_weighted_stats,
    generate_stable_matchup_line
)
from dynamic_hfa import calculate_dynamic_hfa

def run_backtest(year, weeks, value_threshold, recent_games_window, recent_form_weight):
    """
    Runs a backtest for a single season with the updated dynamic weighting model.
    """
    print(f"  - Loading PBP and Schedule data for {year}...")
    try:
        pbp_data = nfl.import_pbp_data([year], downcast=True)
        schedule_data = nfl.import_schedules([year])
    except Exception as e:
        print(f"Error loading data for {year}: {e}")
        return 0, 0, 0  # Return wins, losses, and total picks
        
    all_picks = []

    for week in weeks:
        week_schedule = schedule_data[schedule_data['week'] == week]
        pbp_upto_week = pbp_data[pbp_data['week'] < week]
        if week_schedule.empty or pbp_upto_week.empty:
            continue

        for game in week_schedule.itertuples():
            home_team, away_team = game.home_team, game.away_team
            if pd.isna(game.result): continue

            # Establish the correct Vegas Spread using Moneyline
            home_ml = getattr(game, 'home_moneyline', None)
            away_ml = getattr(game, 'away_moneyline', None)
            spread_magnitude = abs(getattr(game, 'spread_line', 0))
            
            if home_ml is not None and away_ml is not None:
                if home_ml < away_ml: home_spread_vegas = -spread_magnitude
                else: home_spread_vegas = spread_magnitude
            else:
                continue # Skip if no moneyline to confirm favorite

            # Calculate the Model's Predicted Spread using dynamic weighting
            away_stats_std = calculate_granular_epa_stats(pbp_upto_week, away_team)
            home_stats_std = calculate_granular_epa_stats(pbp_upto_week, home_team)
            pbp_away_recent = get_last_n_games_pbp(pbp_upto_week, away_team, recent_games_window)
            pbp_home_recent = get_last_n_games_pbp(pbp_upto_week, home_team, recent_games_window)
            away_stats_recent = calculate_granular_epa_stats(pbp_away_recent, away_team)
            home_stats_recent = calculate_granular_epa_stats(pbp_home_recent, home_team)
            
            if not all([home_stats_std, away_stats_std, home_stats_recent, away_stats_recent]): continue

            away_stats_w = calculate_weighted_stats(away_stats_std, away_stats_recent, 1 - recent_form_weight, recent_form_weight)
            home_stats_w = calculate_weighted_stats(home_stats_std, home_stats_recent, 1 - recent_form_weight, recent_form_weight)
            
            # Create game info dictionary for HFA calculation
            game_info = {
                'is_primetime': getattr(game, 'gameday', '').lower() in ['monday', 'thursday', 'sunday night'],
                'day_of_week': getattr(game, 'gameday', '').lower().split()[0] if hasattr(game, 'gameday') else ''
            }
            
            # Get both the prediction, weights used, and HFA value
            model_line, weights, hfa_value = generate_stable_matchup_line(
                home_stats_w, away_stats_w, 
                return_weights=True,
                pbp_df=pbp_upto_week,
                home_team=home_team,
                away_team=away_team,
                game_info=game_info
            )
            model_home_spread = -model_line
            
            # Calculate the Value Edge and Make a Pick
            model_edge = home_spread_vegas - model_home_spread
            
            if abs(model_edge) >= value_threshold:
                pick = home_team if model_edge > 0 else away_team
                
                # Evaluate the Pick Against the Actual Result
                actual_home_margin = game.result
                if (actual_home_margin + home_spread_vegas) > 0:
                    covering_team = home_team
                elif (actual_home_margin + home_spread_vegas) < 0:
                    covering_team = away_team
                else:
                    covering_team = "Push"
                
                if covering_team != "Push":
                    is_win = 1 if pick == covering_team else 0
                    bet_data = {
                        'year': year,
                        'week': week,
                        'home_team': home_team,
                        'away_team': away_team,
                        'vegas_spread': home_spread_vegas,
                        'model_spread': model_home_spread,
                        'edge': model_edge,
                        'pick': pick,
                        'actual_margin': actual_home_margin,
                        'covering_team': covering_team,
                        'is_win': is_win,
                        'home_off_weight': weights['home_off_weight'],
                        'away_off_weight': weights['away_off_weight'],
                        'hfa_value': hfa_value
                    }
                    all_picks.append(bet_data)

    if not all_picks:
        return 0, 0, 0

    df = pd.DataFrame(all_picks)
    wins = df['is_win'].sum()
    losses = len(df) - wins
    return wins, losses, len(df)

if __name__ == '__main__':
    YEARS_TO_TEST = [2024, 2023, 2022]
    WEEKS_TO_TEST = range(4, 18)  # Weeks 4-17
    
    # Model parameters
    RECENT_GAMES_WINDOW = 8
    RECENT_FORM_WEIGHT = 0.20
    
    # Test different thresholds
    THRESHOLDS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    print("="*80)
    print(f"--- Running Threshold Backtest with Dynamic Weighting and HFA Model ---")
    print(f"Recent form settings: {RECENT_GAMES_WINDOW} games window with {RECENT_FORM_WEIGHT*100:.0f}% weight")
    print("Using team-specific dynamic HFA calculations")
    print("="*80)
    
    # Initialize a summary DataFrame to store results
    summary_data = []
    
    # Run the backtest for each year and threshold combination
    for year in YEARS_TO_TEST:
        print(f"\n{'='*30} SEASON {year} {'='*30}")
        
        for threshold in THRESHOLDS:
            wins, losses, total_picks = run_backtest(
                year, 
                WEEKS_TO_TEST, 
                threshold, 
                RECENT_GAMES_WINDOW, 
                RECENT_FORM_WEIGHT
            )
            
            if total_picks > 0:
                win_pct = (wins / total_picks) * 100
                print(f"\nValue Threshold: {threshold} points")
                print(f"  Record (W-L): {wins}-{losses}")
                print(f"  Total Picks: {total_picks}")
                print(f"  Win Rate: {win_pct:.1f}%")
                
                # Add result to summary data
                summary_data.append({
                    'Year': year,
                    'Threshold': threshold,
                    'Wins': wins,
                    'Losses': losses,
                    'Total': total_picks,
                    'Win Rate': win_pct
                })
            else:
                print(f"\nValue Threshold: {threshold} points")
                print("  No picks identified at this threshold.")
    
    # Print overall summary by threshold (all years combined)
    print("\n" + "="*80)
    print("--- OVERALL SUMMARY BY THRESHOLD ---")
    
    summary_df = pd.DataFrame(summary_data)
    
    # Group by threshold and calculate aggregate stats
    if not summary_df.empty:
        threshold_summary = summary_df.groupby('Threshold').agg({
            'Wins': 'sum',
            'Losses': 'sum',
            'Total': 'sum'
        }).reset_index()
        
        threshold_summary['Win Rate'] = (threshold_summary['Wins'] / threshold_summary['Total']) * 100
        
        # Print the threshold summary
        for _, row in threshold_summary.iterrows():
            print(f"\nValue Threshold: {row['Threshold']} points")
            print(f"  Record (W-L): {int(row['Wins'])}-{int(row['Losses'])}")
            print(f"  Total Picks: {int(row['Total'])}")
            print(f"  Win Rate: {row['Win Rate']:.1f}%")
    
    print("="*80)
