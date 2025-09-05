"""
Analysis: Standard Model vs Simple Model
Comparing signal vs noise in our two approaches
"""

import pandas as pd
from simple_model import SimpleNFLModel
from dynamic_season_model import DynamicSeasonModel
import nfl_data_py as nfl

def analyze_model_differences():
    """Compare Standard vs Simple model across multiple games"""
    
    print("🔍 STANDARD MODEL vs SIMPLE MODEL ANALYSIS")
    print("=" * 60)
    
    # Load models
    simple_model = SimpleNFLModel()
    simple_model.load_data([2022, 2023, 2024])
    
    dynamic_model = DynamicSeasonModel()
    dynamic_model.load_dynamic_data(2025, 1)
    
    # Test matchups
    test_games = [
        ('PHI', 'DAL'),
        ('KC', 'BUF'), 
        ('SF', 'GB'),
        ('NYJ', 'NE'),
        ('BAL', 'PIT'),
        ('LAR', 'SEA'),
        ('TB', 'NO'),
        ('DET', 'MIN')
    ]
    
    results = []
    
    for home, away in test_games:
        try:
            # Simple model prediction
            simple_spread, simple_details = simple_model.predict_spread(home, away, 1, 2025)
            simple_home_epa = simple_details['home_stats']['net_epa_per_play']
            simple_away_epa = simple_details['away_stats']['net_epa_per_play']
            
            # Dynamic model prediction  
            dynamic_spread, dynamic_details = dynamic_model.predict_spread_dynamic(home, away, 1, 2025)
            
            # Calculate difference
            spread_diff = abs(simple_spread - dynamic_spread)
            
            results.append({
                'matchup': f"{away}@{home}",
                'simple_spread': simple_spread,
                'dynamic_spread': dynamic_spread,
                'difference': spread_diff,
                'simple_home_epa': simple_home_epa,
                'simple_away_epa': simple_away_epa,
                'epa_gap': simple_home_epa - simple_away_epa
            })
            
            print(f"\n{away} @ {home}:")
            print(f"  Simple Model:  {home} {simple_spread:+.1f}")
            print(f"  Dynamic Model: {home} {dynamic_spread:+.1f}")
            print(f"  Difference:    {spread_diff:.1f} pts")
            print(f"  EPA Gap:       {simple_home_epa - simple_away_epa:.3f}")
            
        except Exception as e:
            print(f"Error with {home} vs {away}: {e}")
    
    # Analysis
    df = pd.DataFrame(results)
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"Average spread difference: {df['difference'].mean():.2f} points")
    print(f"Max spread difference:     {df['difference'].max():.2f} points")
    print(f"Games with >3pt diff:      {len(df[df['difference'] > 3])}/{len(df)}")
    print(f"Games with >5pt diff:      {len(df[df['difference'] > 5])}/{len(df)}")
    
    # Correlation analysis
    correlation = df['simple_spread'].corr(df['dynamic_spread'])
    print(f"Model correlation:         {correlation:.3f}")
    
    print(f"\n🤔 INTERPRETATION:")
    if df['difference'].mean() < 2:
        print("✅ Models are highly aligned - Standard complexity may not add much signal")
    elif df['difference'].mean() < 4:
        print("⚠️  Moderate differences - Standard model may add some signal")
    else:
        print("🔥 Large differences - Models have fundamentally different approaches")
    
    print(f"\n💡 COMPLEXITY vs VALUE:")
    print(f"Simple Model:  Pure EPA + fixed HFA")
    print(f"Standard Model: SOS-adjusted EPA + dynamic HFA + rolling data")
    
    if correlation > 0.9:
        print("🎯 High correlation suggests Standard model complexity may be overengineering")
    elif correlation > 0.7:
        print("🤷 Moderate correlation - Standard adjustments provide some unique signal")
    else:
        print("📈 Low correlation - Standard model captures different information")
    
    return df

def analyze_vegas_alignment():
    """Check which model aligns better with Vegas on average"""
    print(f"\n🎰 VEGAS ALIGNMENT ANALYSIS")
    print("=" * 40)
    
    # Load some actual Vegas lines to compare
    try:
        schedule = nfl.import_schedules([2025])
        week1 = schedule[schedule['week'] == 1].head(5)
        
        simple_model = SimpleNFLModel()
        simple_model.load_data([2022, 2023, 2024])
        
        dynamic_model = DynamicSeasonModel()
        dynamic_model.load_dynamic_data(2025, 1)
        
        vegas_diffs_simple = []
        vegas_diffs_dynamic = []
        
        for _, game in week1.iterrows():
            home = game['home_team']
            away = game['away_team']
            vegas_line = game.get('spread_line', 0)
            
            try:
                simple_spread, _ = simple_model.predict_spread(home, away, 1, 2025)
                dynamic_spread, _ = dynamic_model.predict_spread_dynamic(home, away, 1, 2025)
                
                simple_diff = abs(simple_spread - vegas_line)
                dynamic_diff = abs(dynamic_spread - vegas_line)
                
                vegas_diffs_simple.append(simple_diff)
                vegas_diffs_dynamic.append(dynamic_diff)
                
                print(f"{away} @ {home}:")
                print(f"  Vegas:   {home} {vegas_line:+.1f}")
                print(f"  Simple:  {home} {simple_spread:+.1f} (diff: {simple_diff:.1f})")
                print(f"  Dynamic: {home} {dynamic_spread:+.1f} (diff: {dynamic_diff:.1f})")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        if vegas_diffs_simple and vegas_diffs_dynamic:
            avg_simple_diff = sum(vegas_diffs_simple) / len(vegas_diffs_simple)
            avg_dynamic_diff = sum(vegas_diffs_dynamic) / len(vegas_diffs_dynamic)
            
            print(f"\n📊 VEGAS ALIGNMENT:")
            print(f"Simple model avg difference:  {avg_simple_diff:.2f} pts")
            print(f"Dynamic model avg difference: {avg_dynamic_diff:.2f} pts")
            
            if avg_simple_diff < avg_dynamic_diff:
                print("🎯 Simple model aligns better with Vegas!")
                print("   → Standard model may be adding noise, not signal")
            else:
                print("📈 Dynamic model aligns better with Vegas")
                print("   → Standard model adjustments may capture market inefficiencies")
                
    except Exception as e:
        print(f"Could not analyze Vegas alignment: {e}")

if __name__ == "__main__":
    df = analyze_model_differences()
    analyze_vegas_alignment()
    
    print(f"\n🎯 CONCLUSION:")
    print("If the models are highly correlated but Simple aligns better with Vegas,")
    print("then the Standard model's complexity might indeed be adding noise.")
    print("The simpler approach could be the more robust long-term strategy.")
