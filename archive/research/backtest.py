# backtest.py

import pandas as pd
import nfl_data_py as nfl
from stats_calculator import (
    get_last_n_games_pbp,
    calculate_granular_epa_stats,
    calculate_weighted_stats,
    generate_stable_matchup_line
)

def run_backtest(year, weeks, value_threshold, recent_games_window, recent_form_weight):
    """
    Runs a backtest for a single season without QB adjustments.
    """
    print(f"  - Loading PBP and Schedule data for {year}...")
    try:
        pbp_data = nfl.import_pbp_data([year], downcast=True)
        schedule_data = nfl.import_schedules([year])
    except Exception as e:
        print(f"Error loading data for {year}: {e}")
        return 0, 0
        
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

            # Calculate the Model's Predicted Spread (No QB Adjustment)
            away_stats_std = calculate_granular_epa_stats(pbp_upto_week, away_team)
            home_stats_std = calculate_granular_epa_stats(pbp_upto_week, home_team)
            pbp_away_recent = get_last_n_games_pbp(pbp_upto_week, away_team, recent_games_window)
            pbp_home_recent = get_last_n_games_pbp(pbp_upto_week, home_team, recent_games_window)
            away_stats_recent = calculate_granular_epa_stats(pbp_away_recent, away_team)
            home_stats_recent = calculate_granular_epa_stats(pbp_home_recent, home_team)
            
            if not all([home_stats_std, away_stats_std, home_stats_recent, away_stats_recent]): continue

            away_stats_w = calculate_weighted_stats(away_stats_std, away_stats_recent, 1 - recent_form_weight, recent_form_weight)
            home_stats_w = calculate_weighted_stats(home_stats_std, home_stats_recent, 1 - recent_form_weight, recent_form_weight)
            
            # Note: We are intentionally calling this without qb_adj parameters
            model_line, _ = generate_stable_matchup_line(home_stats_w, away_stats_w, return_weights=True)
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
                        'pick': pick,
                        'is_win': is_win
                    }
                    all_picks.append(bet_data)

    if not all_picks:
        return 0, 0

    df = pd.DataFrame(all_picks)
    wins = df['is_win'].sum()
    losses = len(df) - wins
    return wins, losses

if __name__ == '__main__':
    YEARS_TO_TEST = [2024, 2023, 2022]
    WEEKS_TO_TEST = range(4, 18)
    
    # Model parameters
    RECENT_GAMES_WINDOW = 8
    RECENT_FORM_WEIGHT = 0.20
    VALUE_THRESHOLD = 5.0
    
    total_wins = 0
    total_losses = 0
    
    print("="*60)
    print(f"--- Running Multi-Year Backtest (No HFA, No QB Adj.) ---")
    print(f"Seasons: {YEARS_TO_TEST}")
    print(f"Value Threshold: {VALUE_THRESHOLD} points")
    print("="*60)
    
    for year in YEARS_TO_TEST:
        wins, losses = run_backtest(
            year, 
            WEEKS_TO_TEST, 
            VALUE_THRESHOLD, 
            RECENT_GAMES_WINDOW, 
            RECENT_FORM_WEIGHT
        )
        total_games = wins + losses
        win_pct = (wins / total_games) * 100 if total_games > 0 else 0
        print(f"\n--- Season: {year} ---")
        print(f"  Record (W-L): {wins} - {losses}")
        print(f"  Win Percentage: {win_pct:.2f}%")
        total_wins += wins
        total_losses += losses

    grand_total_games = total_wins + total_losses
    overall_win_pct = (total_wins / grand_total_games) * 100 if grand_total_games > 0 else 0
    
    print("\n" + "="*60)
    print("--- Overall Backtest Summary ---")
    print(f"  Total Record (W-L): {total_wins} - {total_losses}")
    print(f"  Total Games: {grand_total_games}")
    print(f"  Overall Win Percentage: {overall_win_pct:.2f}%")
    print("="*60)