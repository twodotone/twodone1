import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nfl_data_py as nfl
from stats_calculator import (
    get_last_n_games_pbp,
    calculate_granular_epa_stats,
    calculate_weighted_stats,
    generate_stable_matchup_line
)

def run_edge_analysis(years, weeks, recent_games_window=8, recent_form_weight=0.3):
    """
    Comprehensive analysis of the relationship between model edge and actual win rate.
    Generates confidence parameters for different edge magnitudes.
    """
    print(f"Analyzing model edge confidence for {len(years)} seasons...")
    print(f"Using {recent_games_window} game window with {recent_form_weight*100:.0f}% recency weight")
    print("="*80)
    
    all_picks = []
    
    for year in years:
        print(f"  - Loading PBP and Schedule data for {year}...")
        try:
            pbp_data = nfl.import_pbp_data([year], downcast=True)
            schedule_data = nfl.import_schedules([year])
        except Exception as e:
            print(f"Error loading data for {year}: {e}")
            continue
            
        for week in weeks:
            week_schedule = schedule_data[schedule_data['week'] == week]
            pbp_upto_week = pbp_data[pbp_data['week'] < week]
            if week_schedule.empty or pbp_upto_week.empty:
                continue

            # Process each game to get model projections vs actual results
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

                # Calculate the Model's Predicted Spread
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
                
                # Get model prediction and HFA value
                try:
                    model_line, weights, hfa_value, _ = generate_stable_matchup_line(
                        home_stats_w, away_stats_w, 
                        return_weights=True,
                        pbp_df=pbp_upto_week,
                        home_team=home_team,
                        away_team=away_team,
                        game_info=game_info
                    )
                except ValueError:
                    model_line, weights, hfa_value = generate_stable_matchup_line(
                        home_stats_w, away_stats_w, 
                        return_weights=True,
                        pbp_df=pbp_upto_week,
                        home_team=home_team,
                        away_team=away_team,
                        game_info=game_info
                    )
                    
                model_home_spread = -model_line
                
                # Calculate the Value Edge
                model_edge = home_spread_vegas - model_home_spread
                
                # Evaluate the Pick Against the Actual Result
                actual_home_margin = game.result
                if (actual_home_margin + home_spread_vegas) > 0:
                    covering_team = home_team
                elif (actual_home_margin + home_spread_vegas) < 0:
                    covering_team = away_team
                else:
                    covering_team = "Push"
                
                if covering_team != "Push":
                    # Determine the correct pick based on the edge
                    pick = home_team if model_edge > 0 else away_team
                    is_win = 1 if pick == covering_team else 0
                    
                    # Collect data for all games, not just those exceeding a threshold
                    game_data = {
                        'year': year,
                        'week': week,
                        'home_team': home_team,
                        'away_team': away_team,
                        'vegas_spread': home_spread_vegas,
                        'model_spread': model_home_spread,
                        'edge': model_edge,
                        'edge_abs': abs(model_edge),
                        'pick': pick,
                        'actual_margin': actual_home_margin,
                        'covering_team': covering_team,
                        'is_win': is_win,
                        'hfa_value': hfa_value
                    }
                    all_picks.append(game_data)

    if not all_picks:
        print("No valid games found for analysis.")
        return None
        
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_picks)
    print(f"\nAnalyzed {len(results_df)} total games across {len(years)} seasons")
    
    # Group the data by absolute edge magnitude
    edge_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    results_df['edge_bin'] = pd.cut(results_df['edge_abs'], bins=edge_bins, right=False)
    
    # Calculate win rates for each edge bin
    edge_analysis = results_df.groupby('edge_bin').agg({
        'is_win': ['count', 'sum', 'mean'],
        'edge_abs': ['mean', 'min', 'max']
    }).reset_index()
    
    # Calculate confidence intervals (95%)
    edge_analysis['error_margin'] = 1.96 * np.sqrt(
        (edge_analysis[('is_win', 'mean')] * (1 - edge_analysis[('is_win', 'mean')])) / 
        edge_analysis[('is_win', 'count')]
    )
    
    edge_analysis['conf_lower'] = np.maximum(0, edge_analysis[('is_win', 'mean')] - edge_analysis['error_margin'])
    edge_analysis['conf_upper'] = np.minimum(1, edge_analysis[('is_win', 'mean')] + edge_analysis['error_margin'])
    
    # Rename columns for clarity
    edge_analysis.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in edge_analysis.columns]
    edge_analysis = edge_analysis.rename(columns={
        'is_win_count': 'samples',
        'is_win_sum': 'wins',
        'is_win_mean': 'win_rate',
        'edge_abs_mean': 'avg_edge',
        'edge_abs_min': 'min_edge',
        'edge_abs_max': 'max_edge'
    })
    
    # Add expected value calculation (assuming -110 odds, 52.4% break-even)
    edge_analysis['expected_value'] = (edge_analysis['win_rate'] * 0.91) - (1 - edge_analysis['win_rate'])
    
    # Plot the results
    plot_edge_confidence(edge_analysis)
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("EDGE MAGNITUDE CONFIDENCE ANALYSIS")
    print("="*80)
    
    for _, row in edge_analysis.iterrows():
        bin_range = str(row['edge_bin']).replace('[', '').replace(')', '')
        print(f"\nEdge {bin_range} points:")
        print(f"  Samples: {int(row['samples'])}")
        print(f"  Win Rate: {row['win_rate']*100:.1f}% (95% CI: {row['conf_lower']*100:.1f}%-{row['conf_upper']*100:.1f}%)")
        print(f"  Expected Value: {row['expected_value']*100:.2f}%")
        print(f"  Average Edge: {row['avg_edge']:.2f} points")
    
    # Find the minimum profitable edge threshold
    profitable_edges = edge_analysis[edge_analysis['win_rate'] > 0.524]
    if not profitable_edges.empty:
        min_profitable = profitable_edges.iloc[0]
        print("\n" + "="*80)
        print(f"MINIMUM PROFITABLE EDGE: {min_profitable['edge_bin']}")
        print(f"Win Rate: {min_profitable['win_rate']*100:.1f}% with {int(min_profitable['samples'])} samples")
        print("="*80)
    
    # Generate a model confidence function based on the data
    confidence_model = generate_confidence_model(results_df)
    
    return edge_analysis, confidence_model

def plot_edge_confidence(edge_analysis):
    """Plot the relationship between edge magnitude and win rate with confidence intervals"""
    plt.figure(figsize=(12, 8))
    
    # Extract midpoint of each bin for x-axis
    x = [float(str(b).split(',')[0].replace('[', '')) + 0.5 for b in edge_analysis['edge_bin']]
    
    # Plot win rate with error bars
    plt.errorbar(x, edge_analysis['win_rate'], 
                 yerr=edge_analysis['error_margin'],
                 fmt='o-', ecolor='gray', capsize=5, markersize=8)
    
    # Add sample size as text annotations
    for i, row in edge_analysis.iterrows():
        plt.text(x[i], row['win_rate'] + 0.03, f"n={int(row['samples'])}", 
                 ha='center', va='bottom', fontsize=8)
    
    # Add break-even line
    plt.axhline(y=0.524, color='r', linestyle='--', alpha=0.7, label='Break-even (52.4%)')
    
    # Add profit/loss zones
    plt.axhspan(0.524, 1.0, alpha=0.1, color='green', label='Profit Zone')
    plt.axhspan(0, 0.524, alpha=0.1, color='red', label='Loss Zone')
    
    plt.title('Win Rate by Edge Magnitude with 95% Confidence Intervals', fontsize=16)
    plt.xlabel('Edge Magnitude (points)', fontsize=14)
    plt.ylabel('Win Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('edge_confidence_analysis.png', dpi=300)
    print("\nSaved visualization to 'edge_confidence_analysis.png'")
    
def generate_confidence_model(results_df):
    """
    Generate a mathematical model to predict win probability based on edge magnitude
    Returns a callable function
    """
    # Create simplified dataset for modeling
    edges = results_df['edge_abs'].values
    outcomes = results_df['is_win'].values
    
    # Use logistic regression as a simple model
    from sklearn.linear_model import LogisticRegression
    
    # Reshape the input features
    X = edges.reshape(-1, 1)
    
    # Fit the model
    model = LogisticRegression(random_state=42)
    model.fit(X, outcomes)
    
    # Create a function that returns win probability for a given edge
    def predict_win_probability(edge_magnitude):
        """Returns estimated win probability for a given edge magnitude"""
        if edge_magnitude < 0:
            edge_magnitude = abs(edge_magnitude)
        return model.predict_proba(np.array([[edge_magnitude]]))[0][1]
    
    # Print model coefficients for reference
    print("\n" + "="*80)
    print("WIN PROBABILITY MODEL")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    print(f"Coefficient: {model.coef_[0][0]:.4f}")
    print("="*80)
    
    # Test the model on some example edges
    print("\nWin probability estimates:")
    for edge in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        prob = predict_win_probability(edge)
        print(f"  {edge} point edge: {prob*100:.1f}% win probability")
    
    return predict_win_probability

if __name__ == "__main__":
    # Set parameters
    YEARS = [2024, 2023, 2022]
    WEEKS = range(4, 18)  # Weeks 4-17
    WINDOW_SIZE = 8
    RECENCY_WEIGHT = 0.3  # Our optimized 30% weight
    
    # Run the analysis
    edge_analysis, confidence_model = run_edge_analysis(
        YEARS, WEEKS, 
        recent_games_window=WINDOW_SIZE, 
        recent_form_weight=RECENCY_WEIGHT
    )
    
    # Save analysis to CSV
    if edge_analysis is not None:
        edge_analysis.to_csv('edge_confidence_analysis.csv', index=False)
        print("\nSaved detailed analysis to 'edge_confidence_analysis.csv'")
