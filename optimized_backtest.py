import pandas as pd
import nfl_data_py as nfl
from stats_calculator import (
    get_last_n_games_pbp,
    calculate_granular_epa_stats,
    calculate_weighted_stats,
    generate_stable_matchup_line
)
from dynamic_hfa import calculate_dynamic_hfa

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
    
    # Get parameters from command line if provided
    import sys
    
    # Default model parameters
    RECENT_GAMES_WINDOW = 8
    RECENT_FORM_WEIGHT = 0.20
    TEST_RECENCY = False
    
    # Check for command line args
    if len(sys.argv) > 1 and sys.argv[1] == 'test_recency':
        TEST_RECENCY = True
        print("Testing different recency window sizes and weights")
    
    # Test different thresholds
    THRESHOLDS = [4.0, 4.5, 5.0, 5.5]
    
    # HFA filtering options
    HFA_RANGES = [
        (None, None, "All HFA values"),
        (1.0, 2.0, "HFA 1-2 points"),
        (0.0, 1.0, "HFA 0-1 points")
    ]
    
    # Recency parameters to test (only used if TEST_RECENCY is True)
    WINDOW_SIZES = [4, 6, 8, 10, 12] if TEST_RECENCY else [RECENT_GAMES_WINDOW]
    WINDOW_WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.30] if TEST_RECENCY else [RECENT_FORM_WEIGHT]
    
    print("="*80)
    if TEST_RECENCY:
        print(f"--- Running Recency Parameter Optimization Backtest ---")
        print(f"Testing window sizes: {WINDOW_SIZES}")
        print(f"Testing weights: [{', '.join([f'{w*100:.0f}%' for w in WINDOW_WEIGHTS])}]")
    else:
        print(f"--- Running Optimized Dynamic HFA Model Backtest ---")
        print(f"Recent form settings: {RECENT_GAMES_WINDOW} games window with {RECENT_FORM_WEIGHT*100:.0f}% weight")
        print("Testing optimal HFA ranges and thresholds")
    print("="*80)
    
    # Initialize a summary DataFrame to store results
    summary_data = []
    
    if TEST_RECENCY:
        # For recency testing, just use All HFA values and 5.0 threshold (our best performer)
        hfa_config = HFA_RANGES[0]  # All HFA values
        threshold = 5.0
        
        print(f"\n{'='*30} Testing Recency Parameters {'='*30}")
        
        for window_size in WINDOW_SIZES:
            for weight in WINDOW_WEIGHTS:
                print(f"\nWindow Size: {window_size} games, Weight: {weight*100:.0f}%")
                yearly_results = []
                
                for year in YEARS_TO_TEST:
                    wins, losses, total_picks = run_backtest(
                        year, 
                        WEEKS_TO_TEST, 
                        threshold, 
                        window_size,  # Use the current window size
                        weight,       # Use the current weight
                        hfa_config[0],
                        hfa_config[1]
                    )
                    
                    if total_picks > 0:
                        win_pct = (wins / total_picks) * 100
                        print(f"  Season {year}: {wins}-{losses} ({win_pct:.1f}%) - {total_picks} picks")
                        
                        # Add result to summary data
                        summary_data.append({
                            'Year': year,
                            'Threshold': threshold,
                            'HFA_Range': hfa_config[2],
                            'Window_Size': window_size,
                            'Weight': weight,
                            'Wins': wins,
                            'Losses': losses,
                            'Total': total_picks,
                            'Win Rate': win_pct
                        })
                        
                        yearly_results.append((wins, losses, total_picks))
                    else:
                        print(f"  Season {year}: No picks identified")
                
                # Calculate combined stats for this configuration across years
                total_wins = sum([r[0] for r in yearly_results])
                total_losses = sum([r[1] for r in yearly_results])
                total_picks = sum([r[2] for r in yearly_results])
                
                if total_picks > 0:
                    overall_win_pct = (total_wins / total_picks) * 100
                    print(f"\n  OVERALL: {total_wins}-{total_losses} ({overall_win_pct:.1f}%) - {total_picks} picks")
                else:
                    print(f"\n  OVERALL: No picks identified")
    else:
        # Run backtest for each combination of HFA and threshold parameters
        for min_hfa, max_hfa, hfa_desc in HFA_RANGES:
            print(f"\n{'='*30} {hfa_desc} {'='*30}")
            
            for threshold in THRESHOLDS:
                print(f"\nValue Threshold: {threshold} points")
                yearly_results = []
                
                for year in YEARS_TO_TEST:
                    wins, losses, total_picks = run_backtest(
                        year, 
                        WEEKS_TO_TEST, 
                        threshold, 
                        RECENT_GAMES_WINDOW, 
                        RECENT_FORM_WEIGHT,
                        min_hfa,
                        max_hfa
                    )
                    
                    if total_picks > 0:
                        win_pct = (wins / total_picks) * 100
                        print(f"  Season {year}: {wins}-{losses} ({win_pct:.1f}%) - {total_picks} picks")
                        
                        # Add result to summary data
                        summary_data.append({
                            'Year': year,
                            'Threshold': threshold,
                            'HFA_Range': hfa_desc,
                            'Window_Size': RECENT_GAMES_WINDOW,
                            'Weight': RECENT_FORM_WEIGHT,
                            'Wins': wins,
                            'Losses': losses,
                            'Total': total_picks,
                            'Win Rate': win_pct
                        })
                        
                        yearly_results.append((wins, losses, total_picks))
                    else:
                        print(f"  Season {year}: No picks identified")
                
                # Calculate combined stats for this threshold across years
                total_wins = sum([r[0] for r in yearly_results])
                total_losses = sum([r[1] for r in yearly_results])
                total_picks = sum([r[2] for r in yearly_results])
                
                if total_picks > 0:
                    overall_win_pct = (total_wins / total_picks) * 100
                    print(f"\n  OVERALL: {total_wins}-{total_losses} ({overall_win_pct:.1f}%) - {total_picks} picks")
                else:
                    print(f"\n  OVERALL: No picks identified")
    
    # Print the best performing combinations
    print("\n" + "="*80)
    if TEST_RECENCY:
        print("--- TOP PERFORMING RECENCY PARAMETERS ---")
    else:
        print("--- TOP PERFORMING MODELS ---")
    
    summary_df = pd.DataFrame(summary_data)
    
    if not summary_df.empty:
        if TEST_RECENCY:
            # Group by window size and weight to find the best combinations
            grouped = summary_df.groupby(['Window_Size', 'Weight']).agg({
                'Wins': 'sum',
                'Losses': 'sum',
                'Total': 'sum'
            }).reset_index()
            
            grouped['Win Rate'] = (grouped['Wins'] / grouped['Total']) * 100
            
            # Sort by win rate (descending) and total picks (descending)
            grouped = grouped.sort_values(['Win Rate', 'Total'], ascending=[False, False])
            
            # Print the top 5 combinations
            for i, row in grouped.head(5).iterrows():
                print(f"\n{i+1}. Window Size: {int(row['Window_Size'])} games, Weight: {row['Weight']*100:.0f}%")
                print(f"   Record: {int(row['Wins'])}-{int(row['Losses'])} ({row['Win Rate']:.1f}%)")
                print(f"   Total Picks: {int(row['Total'])}")
                
                # Show yearly breakdown for this combination
                combo_df = summary_df[(summary_df['Window_Size'] == row['Window_Size']) & 
                                    (summary_df['Weight'] == row['Weight'])]
                
                for _, yr_row in combo_df.iterrows():
                    print(f"     {int(yr_row['Year'])}: {int(yr_row['Wins'])}-{int(yr_row['Losses'])} ({yr_row['Win Rate']:.1f}%)")
        else:
            # Group by HFA range and threshold to find the best combinations
            grouped = summary_df.groupby(['HFA_Range', 'Threshold']).agg({
                'Wins': 'sum',
                'Losses': 'sum',
                'Total': 'sum'
            }).reset_index()
            
            grouped['Win Rate'] = (grouped['Wins'] / grouped['Total']) * 100
            
            # Sort by win rate (descending) and total picks (descending)
            grouped = grouped.sort_values(['Win Rate', 'Total'], ascending=[False, False])
            
            # Print the top 5 combinations
            for i, row in grouped.head(5).iterrows():
                print(f"\n{i+1}. {row['HFA_Range']} with {row['Threshold']} point threshold:")
                print(f"   Record: {int(row['Wins'])}-{int(row['Losses'])} ({row['Win Rate']:.1f}%)")
                print(f"   Total Picks: {int(row['Total'])}")
                
                # Show yearly breakdown for this combination
                combo_df = summary_df[(summary_df['HFA_Range'] == row['HFA_Range']) & 
                                    (summary_df['Threshold'] == row['Threshold'])]
                
                for _, yr_row in combo_df.iterrows():
                    print(f"     {int(yr_row['Year'])}: {int(yr_row['Wins'])}-{int(yr_row['Losses'])} ({yr_row['Win Rate']:.1f}%)")
    
    print("="*80)
