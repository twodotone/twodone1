import pandas as pd
import nfl_data_py as nfl
import matplotlib.pyplot as plt
from dynamic_hfa import analyze_hfa_impact
from stats_calculator import (
    get_last_n_games_pbp,
    calculate_granular_epa_stats,
    calculate_weighted_stats,
    generate_stable_matchup_line
)

def run_hfa_analysis(year, weeks, recent_games_window, recent_form_weight):
    """
    Runs a full analysis of the HFA impact on model performance for a single season.
    """
    print(f"  - Loading PBP and Schedule data for {year}...")
    try:
        pbp_data = nfl.import_pbp_data([year], downcast=True)
        schedule_data = nfl.import_schedules([year])
    except Exception as e:
        print(f"Error loading data for {year}: {e}")
        return pd.DataFrame()
        
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

            # Calculate the Model's Predicted Spread with dynamic HFA
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
            
            # Get prediction with dynamic HFA
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
            
            # Calculate the Model Edge
            model_edge = home_spread_vegas - model_home_spread
            
            # Evaluate the pick against the actual result for all games
            actual_home_margin = game.result
            if (actual_home_margin + home_spread_vegas) > 0:
                covering_team = home_team
            elif (actual_home_margin + home_spread_vegas) < 0:
                covering_team = away_team
            else:
                covering_team = "Push"
            
            if covering_team != "Push":
                # Determine which side the model would pick based on the edge
                pick = home_team if model_edge > 0 else away_team
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
        return pd.DataFrame()

    return pd.DataFrame(all_picks)

def plot_hfa_performance(df):
    """
    Create visualizations for HFA impact on model performance
    """
    if df.empty:
        print("No data to plot.")
        return
    
    # Create HFA ranges for analysis
    df['hfa_range'] = pd.cut(
        df['hfa_value'],
        bins=[-0.1, 1.0, 2.0, 3.0, 4.0],
        labels=['0-1', '1-2', '2-3', '3+']
    )
    
    # Group by HFA range and calculate win rates
    hfa_performance = df.groupby('hfa_range').agg({
        'is_win': ['count', 'sum', 'mean']
    })
    
    hfa_performance.columns = ['count', 'wins', 'win_rate']
    hfa_performance = hfa_performance.reset_index()
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot win rate by HFA range
    ax1 = plt.subplot(1, 2, 1)
    bars = ax1.bar(hfa_performance['hfa_range'], hfa_performance['win_rate'] * 100)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('HFA Range (points)')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Win Rate by HFA Range')
    ax1.axhline(y=50, color='r', linestyle='--')
    
    # Add labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Plot count by HFA range
    ax2 = plt.subplot(1, 2, 2)
    bars = ax2.bar(hfa_performance['hfa_range'], hfa_performance['count'])
    ax2.set_xlabel('HFA Range (points)')
    ax2.set_ylabel('Number of Picks')
    ax2.set_title('Pick Frequency by HFA Range')
    
    # Add labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Plot home vs away performance
    home_picks = df[df['pick'] == df['home_team']]
    away_picks = df[df['pick'] == df['away_team']]
    
    home_win_rate = home_picks['is_win'].mean() * 100 if not home_picks.empty else 0
    away_win_rate = away_picks['is_win'].mean() * 100 if not away_picks.empty else 0
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Home Team Picks', 'Away Team Picks'], [home_win_rate, away_win_rate])
    plt.ylim(0, 100)
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rate for Home vs Away Team Picks')
    plt.axhline(y=50, color='r', linestyle='--')
    
    # Add labels to the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Add pick counts
    plt.text(0, 5, f'n = {len(home_picks)}', ha='center')
    plt.text(1, 5, f'n = {len(away_picks)}', ha='center')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    YEARS_TO_TEST = [2024, 2023, 2022]
    WEEKS_TO_TEST = range(4, 18)  # Weeks 4-17
    
    # Model parameters
    RECENT_GAMES_WINDOW = 8
    RECENT_FORM_WEIGHT = 0.20
    
    print("="*80)
    print(f"--- Running HFA Impact Analysis ---")
    print(f"Seasons: {YEARS_TO_TEST}")
    print("="*80)
    
    all_results = []
    
    for year in YEARS_TO_TEST:
        print(f"\n--- Analyzing Season {year} ---")
        year_results = run_hfa_analysis(
            year, 
            WEEKS_TO_TEST, 
            RECENT_GAMES_WINDOW, 
            RECENT_FORM_WEIGHT
        )
        
        if not year_results.empty:
            all_results.append(year_results)
            
            # Print basic statistics for the year
            total_games = len(year_results)
            wins = year_results['is_win'].sum()
            win_rate = (wins / total_games) * 100
            
            print(f"\nSeason {year} Overview:")
            print(f"  Total games analyzed: {total_games}")
            print(f"  Overall win rate: {win_rate:.1f}%")
            
            # Calculate HFA statistics
            avg_hfa = year_results['hfa_value'].mean()
            min_hfa = year_results['hfa_value'].min()
            max_hfa = year_results['hfa_value'].max()
            
            print(f"\nHFA Statistics:")
            print(f"  Average HFA: {avg_hfa:.2f} points")
            print(f"  HFA Range: {min_hfa:.1f} to {max_hfa:.1f} points")
            
            # Analyze impact by HFA range
            print("\nPerformance by HFA Range:")
            hfa_ranges = pd.cut(
                year_results['hfa_value'],
                bins=[-0.1, 1.0, 2.0, 3.0, 4.0],
                labels=['0-1', '1-2', '2-3', '3+']
            )
            
            for hfa_range in hfa_ranges.unique():
                range_df = year_results[hfa_ranges == hfa_range]
                if len(range_df) > 0:
                    range_wins = range_df['is_win'].sum()
                    range_win_rate = (range_wins / len(range_df)) * 100
                    print(f"  HFA {hfa_range} points: {range_wins}-{len(range_df)-range_wins} ({range_win_rate:.1f}%)")
    
    # Combine all results for overall analysis
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        print("\n" + "="*80)
        print("--- OVERALL HFA IMPACT ANALYSIS ---")
        
        total_games = len(combined_results)
        wins = combined_results['is_win'].sum()
        win_rate = (wins / total_games) * 100
        
        print(f"\nOverall Results:")
        print(f"  Total games analyzed: {total_games}")
        print(f"  Overall win rate: {win_rate:.1f}%")
        
        # Calculate overall HFA statistics
        avg_hfa = combined_results['hfa_value'].mean()
        
        print(f"\nHFA Statistics:")
        print(f"  Average HFA: {avg_hfa:.2f} points")
        
        # Analyze impact by HFA range
        print("\nOverall Performance by HFA Range:")
        hfa_ranges = pd.cut(
            combined_results['hfa_value'],
            bins=[-0.1, 1.0, 2.0, 3.0, 4.0],
            labels=['0-1', '1-2', '2-3', '3+']
        )
        
        for hfa_range in sorted(hfa_ranges.unique()):
            range_df = combined_results[hfa_ranges == hfa_range]
            if len(range_df) > 0:
                range_wins = range_df['is_win'].sum()
                range_win_rate = (range_wins / len(range_df)) * 100
                print(f"  HFA {hfa_range} points: {range_wins}-{len(range_df)-range_wins} ({range_win_rate:.1f}%)")
        
        # Analyze home vs away picks
        home_picks = combined_results[combined_results['pick'] == combined_results['home_team']]
        away_picks = combined_results[combined_results['pick'] == combined_results['away_team']]
        
        home_win_rate = home_picks['is_win'].mean() * 100 if not home_picks.empty else 0
        away_win_rate = away_picks['is_win'].mean() * 100 if not away_picks.empty else 0
        
        print(f"\nHome vs Away Team Picks:")
        print(f"  Home team picks: {len(home_picks)} games ({home_win_rate:.1f}% win rate)")
        print(f"  Away team picks: {len(away_picks)} games ({away_win_rate:.1f}% win rate)")
        
        print("="*80)
    else:
        print("\nNo results to analyze.")
