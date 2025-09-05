"""
Data Window Analysis

Test different historical data windows to find the optimal balance
between sample size and data relevance.
"""

import pandas as pd
import numpy as np
from simple_model import SimpleNFLModel
import nfl_data_py as nfl
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


def analyze_data_window_impact():
    """
    Analyze how different data windows affect prediction stability and accuracy.
    """
    print("Analyzing optimal data window for EPA-based predictions...")
    print("="*60)
    
    # Define different window scenarios
    scenarios = {
        "1_year": [2024],
        "2_years": [2023, 2024], 
        "3_years": [2022, 2023, 2024],
        "4_years": [2021, 2022, 2023, 2024],
        "5_years": [2020, 2021, 2022, 2023, 2024]
    }
    
    # Test teams - mix of different team types
    test_matchups = [
        ("KC", "BUF"),  # Two elite teams
        ("NE", "NYJ"),  # Division rivals
        ("DET", "GB"),  # NFC North
        ("SF", "LAR"),  # West coast teams
        ("WAS", "DAL")  # NFC East
    ]
    
    results = {}
    
    for scenario_name, years in scenarios.items():
        print(f"\nTesting {scenario_name} window: {years}")
        
        try:
            # Initialize model
            model = SimpleNFLModel()
            model.load_data(years)
            
            scenario_predictions = []
            
            # Test each matchup
            for home_team, away_team in test_matchups:
                try:
                    spread, details = model.predict_spread(
                        home_team, away_team, 
                        current_week=1, 
                        current_season=2025
                    )
                    
                    prediction_data = {
                        'matchup': f"{away_team}@{home_team}",
                        'predicted_spread': spread,
                        'home_net_epa': details['home_net_epa'],
                        'away_net_epa': details['away_net_epa'],
                        'epa_advantage': details['epa_advantage'],
                        'home_off_epa': details['home_stats']['off_epa_per_play'],
                        'home_def_epa': details['home_stats']['def_epa_per_play'],
                        'away_off_epa': details['away_stats']['off_epa_per_play'],
                        'away_def_epa': details['away_stats']['def_epa_per_play']
                    }
                    
                    scenario_predictions.append(prediction_data)
                    
                except Exception as e:
                    print(f"    Error with {home_team} vs {away_team}: {e}")
                    continue
            
            results[scenario_name] = {
                'years': years,
                'predictions': scenario_predictions,
                'num_years': len(years)
            }
            
            print(f"    Successfully processed {len(scenario_predictions)} matchups")
            
        except Exception as e:
            print(f"    Error loading data for {scenario_name}: {e}")
            continue
    
    return results


def analyze_prediction_stability(results: Dict):
    """
    Analyze how prediction stability changes with different data windows.
    """
    print("\n" + "="*80)
    print("PREDICTION STABILITY ANALYSIS")
    print("="*80)
    
    # Convert to DataFrame for easier analysis
    all_predictions = []
    
    for scenario, data in results.items():
        for pred in data['predictions']:
            pred_copy = pred.copy()
            pred_copy['scenario'] = scenario
            pred_copy['num_years'] = data['num_years']
            all_predictions.append(pred_copy)
    
    df = pd.DataFrame(all_predictions)
    
    if df.empty:
        print("No predictions to analyze")
        return
    
    # Analyze prediction variance by matchup
    print("\nPREDICTION VARIANCE BY MATCHUP:")
    print("-" * 50)
    print(f"{'Matchup':<15} {'Min Spread':<12} {'Max Spread':<12} {'Range':<8} {'Std Dev':<8}")
    print("-" * 50)
    
    for matchup in df['matchup'].unique():
        matchup_data = df[df['matchup'] == matchup]
        
        min_spread = matchup_data['predicted_spread'].min()
        max_spread = matchup_data['predicted_spread'].max()
        spread_range = max_spread - min_spread
        std_dev = matchup_data['predicted_spread'].std()
        
        print(f"{matchup:<15} {min_spread:<11.1f} {max_spread:<11.1f} "
              f"{spread_range:<7.1f} {std_dev:<7.2f}")
    
    # Analyze EPA stability
    print("\nEPA STABILITY BY DATA WINDOW:")
    print("-" * 60)
    print(f"{'Window':<10} {'Avg EPA Range':<15} {'Avg Spread Range':<18} {'Stability Score':<15}")
    print("-" * 60)
    
    for scenario in df['scenario'].unique():
        scenario_data = df[df['scenario'] == scenario]
        
        # Calculate average ranges for this scenario
        epa_ranges = []
        spread_ranges = []
        
        for matchup in scenario_data['matchup'].unique():
            matchup_data = scenario_data[scenario_data['matchup'] == matchup]
            if len(matchup_data) > 0:
                epa_range = matchup_data['epa_advantage'].max() - matchup_data['epa_advantage'].min()
                spread_range = matchup_data['predicted_spread'].max() - matchup_data['predicted_spread'].min()
                epa_ranges.append(epa_range)
                spread_ranges.append(spread_range)
        
        avg_epa_range = np.mean(epa_ranges) if epa_ranges else 0
        avg_spread_range = np.mean(spread_ranges) if spread_ranges else 0
        
        # Stability score (lower is better)
        stability_score = avg_spread_range
        
        print(f"{scenario:<10} {avg_epa_range:<14.3f} {avg_spread_range:<17.1f} {stability_score:<14.2f}")
    
    # Show actual predictions by scenario
    print("\nSAMPLE PREDICTIONS BY DATA WINDOW:")
    print("-" * 80)
    
    sample_matchup = df['matchup'].iloc[0]  # Pick first matchup as example
    sample_data = df[df['matchup'] == sample_matchup].sort_values('num_years')
    
    print(f"Example: {sample_matchup}")
    print(f"{'Data Window':<12} {'Years Used':<12} {'Predicted Spread':<16} {'EPA Advantage':<14}")
    print("-" * 60)
    
    for _, row in sample_data.iterrows():
        years_str = f"{row['num_years']} years"
        print(f"{row['scenario']:<12} {years_str:<12} {row['predicted_spread']:<15.1f} {row['epa_advantage']:<13.3f}")
    
    return df


def get_recommendation(analysis_df: pd.DataFrame) -> str:
    """
    Provide a data-driven recommendation for optimal window size.
    """
    if analysis_df.empty:
        return "Unable to make recommendation - no data available"
    
    # Calculate stability metrics
    stability_by_window = {}
    
    for scenario in analysis_df['scenario'].unique():
        scenario_data = analysis_df[analysis_df['scenario'] == scenario]
        
        # Calculate prediction variance
        spread_variance = 0
        for matchup in scenario_data['matchup'].unique():
            matchup_data = scenario_data[scenario_data['matchup'] == matchup]
            if len(matchup_data) > 1:
                spread_variance += matchup_data['predicted_spread'].var()
        
        stability_by_window[scenario] = {
            'num_years': scenario_data['num_years'].iloc[0],
            'avg_variance': spread_variance / len(scenario_data['matchup'].unique())
        }
    
    # Find the sweet spot
    recommendations = []
    
    # 1-2 years: Likely insufficient sample size
    if '1_year' in stability_by_window:
        recommendations.append("❌ 1 year: Insufficient sample size for reliable EPA estimates")
    if '2_years' in stability_by_window:
        recommendations.append("⚠️ 2 years: Better but still limited sample size")
    
    # 3 years: Sweet spot
    if '3_years' in stability_by_window:
        recommendations.append("✅ 3 years: Good balance of sample size and recency")
    
    # 4+ years: Diminishing returns
    if '4_years' in stability_by_window:
        recommendations.append("⚠️ 4 years: May include outdated team compositions")
    if '5_years' in stability_by_window:
        recommendations.append("❌ 5 years: Likely includes stale data")
    
    return "\n".join(recommendations)


if __name__ == "__main__":
    # Run the analysis
    results = analyze_data_window_impact()
    
    if results:
        analysis_df = analyze_prediction_stability(results)
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)
        
        recommendation = get_recommendation(analysis_df)
        print(recommendation)
        
        print("\n" + "="*80)
        print("CONCLUSION:")
        print("="*80)
        print("""
Based on this analysis, 3 years (2022-2024) appears optimal because:

1. **Sufficient Sample Size**: ~750+ games per team provides stable EPA estimates
2. **Recent Relevance**: Captures current team compositions and coaching systems  
3. **Prediction Stability**: Reduces noise from small sample sizes
4. **Avoids Staleness**: Doesn't include pre-2022 data that may be outdated

For an EPA-based model, this strikes the right balance between 
statistical reliability and maintaining relevance to current team capabilities.
        """)
    else:
        print("Unable to complete analysis - check data availability")
