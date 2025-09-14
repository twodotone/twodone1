import pandas as pd
import nfl_data_py as nfl
from stats_calculator import (
    get_last_n_games_pbp,
    calculate_granular_epa_stats,
    calculate_weighted_stats,
    generate_stable_matchup_line
)

def run_backtest(year, weeks, value_threshold, recent_games_window, recent_form_weight, min_hfa=None, max_hfa=None):
    """
    Runs a backtest for a single season with the dynamic HFA model,
    with optional HFA filtering.
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
            try:
                # Handle the new return format with 4 values
                model_line, weights, hfa_value, _ = generate_stable_matchup_line(
                    home_stats_w, away_stats_w, 
                    return_weights=True,
                    pbp_df=pbp_upto_week,
                    home_team=home_team,
                    away_team=away_team,
                    game_info=game_info
                )
            except ValueError:
                # Fallback for older version if needed
                model_line, weights, hfa_value = generate_stable_matchup_line(
                    home_stats_w, away_stats_w, 
                    return_weights=True,
                    pbp_df=pbp_upto_week,
                    home_team=home_team,
                    away_team=away_team,
                    game_info=game_info
                )
                
            model_home_spread = -model_line
            
            # Skip if HFA is outside our target range (if specified)
            if (min_hfa is not None and hfa_value < min_hfa) or (max_hfa is not None and hfa_value > max_hfa):
                continue
            
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
    
    # Fixed values
    RECENT_GAMES_WINDOW = 8  # Fixed at 8 games based on previous testing
    
    # HFA filtering options
    HFA_RANGES = [(None, None, "All HFA values")]
    
    # Threshold for testing
    THRESHOLD = 5.0  # Fixed at 5.0 based on previous testing
    
    # Higher recency weights to test - test one at a time to save memory
    RECENCY_WEIGHTS = [0.60, 0.70, 0.80]  # Start with 60% since we had issues there
    
    print("="*80)
    print(f"--- Testing Higher Recency Weights with {RECENT_GAMES_WINDOW}-Game Window ---")
    print(f"Weights to test: {', '.join([f'{w*100:.0f}%' for w in RECENCY_WEIGHTS])}")
    print("="*80)
    
    # Try to load previous results if they exist
    import os
    results_file = "recency_test_results.csv"
    if os.path.exists(results_file):
        print(f"Loading previous results from {results_file}")
        summary_data = pd.read_csv(results_file).to_dict('records')
    else:
        summary_data = []
    
    # Test each recency weight
    for recency_weight in RECENCY_WEIGHTS:
        print(f"\n{'='*30} Testing {recency_weight*100:.0f}% Recency Weight {'='*30}")
        yearly_results = []
        
        # Skip weights we've already tested
        existing_weights = set(item['Recency_Weight'] for item in summary_data if 'Recency_Weight' in item)
        if recency_weight in existing_weights:
            print(f"Skipping {recency_weight*100:.0f}% - already tested")
            continue
        
        for year in YEARS_TO_TEST:
            try:
                wins, losses, total_picks = run_backtest(
                    year, 
                    WEEKS_TO_TEST, 
                    THRESHOLD, 
                    RECENT_GAMES_WINDOW, 
                    recency_weight,
                    HFA_RANGES[0][0],
                    HFA_RANGES[0][1]
                )
                
                if total_picks > 0:
                    win_pct = (wins / total_picks) * 100
                    print(f"  Season {year}: {wins}-{losses} ({win_pct:.1f}%) - {total_picks} picks")
                    
                    # Add result to summary data
                    summary_data.append({
                        'Year': year,
                        'Recency_Weight': recency_weight,
                        'Wins': wins,
                        'Losses': losses,
                        'Total': total_picks,
                        'Win Rate': win_pct
                    })
                    
                    yearly_results.append((wins, losses, total_picks))
                else:
                    print(f"  Season {year}: No picks identified")
            except Exception as e:
                print(f"Error processing {year}: {str(e)}")
                continue
        
        # Calculate combined stats for this weight across years
        total_wins = sum([r[0] for r in yearly_results])
        total_losses = sum([r[1] for r in yearly_results])
        total_picks = sum([r[2] for r in yearly_results])
        
        if total_picks > 0:
            overall_win_pct = (total_wins / total_picks) * 100
            print(f"\n  OVERALL: {total_wins}-{total_losses} ({overall_win_pct:.1f}%) - {total_picks} picks")
        else:
            print(f"\n  OVERALL: No picks identified")
            
        # Save progress after each weight
        pd.DataFrame(summary_data).to_csv(results_file, index=False)
        print(f"Saved results to {results_file}")
        
        # Clear memory
        import gc
        gc.collect()
    
    # Print the best performing recency weights
    print("\n" + "="*80)
    print("--- TOP PERFORMING RECENCY WEIGHTS ---")
    
    # Load the complete results
    if os.path.exists(results_file):
        summary_df = pd.read_csv(results_file)
    else:
        summary_df = pd.DataFrame(summary_data)
    
    if not summary_df.empty:
        # Group by recency weight to find the best combinations
        grouped = summary_df.groupby(['Recency_Weight']).agg({
            'Wins': 'sum',
            'Losses': 'sum',
            'Total': 'sum'
        }).reset_index()
        
        grouped['Win Rate'] = (grouped['Wins'] / grouped['Total']) * 100
        
        # Sort by win rate (descending) and total picks (descending)
        grouped = grouped.sort_values(['Win Rate', 'Total'], ascending=[False, False])
        
        # Print all weights
        for i, row in grouped.iterrows():
            print(f"\n{i+1}. {row['Recency_Weight']*100:.0f}% Recency Weight:")
            print(f"   Record: {int(row['Wins'])}-{int(row['Losses'])} ({row['Win Rate']:.1f}%)")
            print(f"   Total Picks: {int(row['Total'])}")
            
            # Show yearly breakdown for this weight
            combo_df = summary_df[summary_df['Recency_Weight'] == row['Recency_Weight']]
            
            for _, yr_row in combo_df.iterrows():
                print(f"     {int(yr_row['Year'])}: {int(yr_row['Wins'])}-{int(yr_row['Losses'])} ({yr_row['Win Rate']:.1f}%)")
    
    print("="*80)
