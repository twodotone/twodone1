"""
COMPREHENSIVE MODEL COMPARISON: Standard vs Simple

This script compares the methodologies between our previous "standard model" 
(used in app.py with stats_calculator.py) and our current "simple model" 
to understand why we're getting different results.
"""

import pandas as pd
from simple_model import SimpleNFLModel
from stats_calculator import (
    calculate_granular_epa_stats, 
    calculate_weighted_stats,
    generate_stable_matchup_line
)
from data_loader import load_rolling_data
import warnings
warnings.filterwarnings('ignore')

def compare_model_methodologies():
    """
    Compare the core methodologies between Standard and Simple models.
    """
    print("📊 COMPREHENSIVE MODEL METHODOLOGY COMPARISON")
    print("="*80)
    
    print("\n🏗️  ARCHITECTURAL DIFFERENCES:")
    print("-" * 50)
    
    print("STANDARD MODEL (app.py + stats_calculator.py):")
    print("  📋 Data Processing:")
    print("     • Uses load_rolling_data() for smart season selection")
    print("     • Applies SOS (Strength of Schedule) adjustments by default")
    print("     • Calculates granular EPA stats with opponent adjustments")
    print("     • Separate offensive/defensive EPA for rush/pass")
    print("  🧮 Calculation Method:")
    print("     • Weighted combination of offense vs defense EPA")
    print("     • Considers play type percentages (rush vs pass)")
    print("     • Dynamic pace of play calculations")
    print("     • Dynamic HFA with multiple components")
    print("  📐 Formula: (Off_EPA * weight + Def_EPA * weight) * plays_per_game + HFA")
    
    print("\nSIMPLE MODEL (simple_model.py):")
    print("  📋 Data Processing:")
    print("     • Fixed multi-year data loading (2022-2024)")
    print("     • Raw EPA calculations (no SOS adjustments)")
    print("     • Combined net EPA per play (offense - defense)")
    print("     • Recent form weighting with configurable window")
    print("  🧮 Calculation Method:")
    print("     • Direct EPA advantage calculation")
    print("     • Fixed scaling factor (25 points per EPA)")
    print("     • Constant HFA (2.5 points)")
    print("  📐 Formula: -(epa_advantage * 25) + 2.5")
    
    print("\n🔍 KEY METHODOLOGICAL DIFFERENCES:")
    print("-" * 50)
    print("1. SOS ADJUSTMENT:")
    print("   Standard: YES - Adjusts EPA based on opponent strength")
    print("   Simple:   NO  - Uses raw EPA values")
    
    print("\n2. EPA GRANULARITY:")
    print("   Standard: Separate rush/pass EPA for offense/defense (4 metrics)")
    print("   Simple:   Combined net EPA per play (1 metric)")
    
    print("\n3. WEIGHTING SYSTEM:")
    print("   Standard: Complex weights for off/def with play type percentages")
    print("   Simple:   Simple recent games weighting (default 30%)")
    
    print("\n4. HOME FIELD ADVANTAGE:")
    print("   Standard: Dynamic HFA based on team, venue, travel, etc.")
    print("   Simple:   Fixed 2.5 points for all teams")
    
    print("\n5. SCALING METHODOLOGY:")
    print("   Standard: EPA per play * plays per game")
    print("   Simple:   EPA per play * fixed multiplier (25)")
    
    print("\n6. DATA WINDOWS:")
    print("   Standard: Smart rolling data with recent form emphasis")
    print("   Simple:   Fixed 3-year window with recent form overlay")

def test_eagles_cowboys_both_models():
    """
    Test Eagles vs Cowboys using both methodologies.
    """
    print("\n\n🦅 vs 🤠 EAGLES vs COWBOYS - BOTH MODELS")
    print("="*60)
    
    # SIMPLE MODEL TEST
    print("\n1. SIMPLE MODEL PREDICTION:")
    print("-" * 30)
    simple_model = SimpleNFLModel()
    simple_model.load_data([2022, 2023, 2024])
    
    spread_simple, details_simple = simple_model.predict_spread('PHI', 'DAL', 1, 2025)
    
    print(f"   Prediction: Eagles {spread_simple:+.1f}")
    print(f"   Eagles Net EPA: {details_simple['home_stats']['net_epa_per_play']:.3f}")
    print(f"   Cowboys Net EPA: {details_simple['away_stats']['net_epa_per_play']:.3f}")
    print(f"   EPA Advantage: {details_simple['epa_advantage']:.3f}")
    print(f"   Raw Spread: {details_simple['predicted_spread_raw']:.1f}")
    print(f"   HFA: +{details_simple['home_field_advantage']:.1f}")
    
    # STANDARD MODEL TEST (using the stats_calculator functions)
    print("\n2. STANDARD MODEL PREDICTION:")
    print("-" * 30)
    
    try:
        # Load data using the standard model's method
        pbp_data, team_stats_df = load_rolling_data(
            CURRENT_YEAR=2025,
            LOAD_SEASONS=[2022, 2023, 2024],
            exclude_playoffs=True
        )
        
        # Calculate stats for both teams using standard model
        eagles_stats = calculate_granular_epa_stats(pbp_data, 'PHI', use_sos_adjustment=True)
        cowboys_stats = calculate_granular_epa_stats(pbp_data, 'DAL', use_sos_adjustment=True)
        
        # Apply recent form weighting
        eagles_recent_pbp = pbp_data[
            ((pbp_data['home_team'] == 'PHI') | (pbp_data['away_team'] == 'PHI'))
        ].sort_values(['season', 'week']).tail(500)  # Approximate last 8 games worth of plays
        
        cowboys_recent_pbp = pbp_data[
            ((pbp_data['home_team'] == 'DAL') | (pbp_data['away_team'] == 'DAL'))
        ].sort_values(['season', 'week']).tail(500)
        
        eagles_recent_stats = calculate_granular_epa_stats(eagles_recent_pbp, 'PHI', use_sos_adjustment=True)
        cowboys_recent_stats = calculate_granular_epa_stats(cowboys_recent_pbp, 'DAL', use_sos_adjustment=True)
        
        # Combine with weighting (70% season, 30% recent)
        eagles_weighted = calculate_weighted_stats(eagles_stats, eagles_recent_stats, 0.7, 0.3)
        cowboys_weighted = calculate_weighted_stats(cowboys_stats, cowboys_recent_stats, 0.7, 0.3)
        
        # Generate prediction
        game_info = {'current_season': 2025}
        spread_standard, weights, hfa, hfa_components = generate_stable_matchup_line(
            eagles_weighted, cowboys_weighted, return_weights=True,
            pbp_df=pbp_data, home_team='PHI', away_team='DAL', game_info=game_info
        )
        
        # Note: standard model returns positive = home favored, but we use negative convention
        spread_standard = -spread_standard
        
        print(f"   Prediction: Eagles {spread_standard:+.1f}")
        print(f"   Eagles Off Rush EPA: {eagles_weighted.get('Off_Rush_EPA', 0):.3f}")
        print(f"   Eagles Off Pass EPA: {eagles_weighted.get('Off_Pass_EPA', 0):.3f}")
        print(f"   Eagles Def Rush EPA: {eagles_weighted.get('Def_Rush_EPA', 0):.3f}")
        print(f"   Eagles Def Pass EPA: {eagles_weighted.get('Def_Pass_EPA', 0):.3f}")
        print(f"   Cowboys Off Rush EPA: {cowboys_weighted.get('Off_Rush_EPA', 0):.3f}")
        print(f"   Cowboys Off Pass EPA: {cowboys_weighted.get('Off_Pass_EPA', 0):.3f}")
        print(f"   Cowboys Def Rush EPA: {cowboys_weighted.get('Def_Rush_EPA', 0):.3f}")
        print(f"   Cowboys Def Pass EPA: {cowboys_weighted.get('Def_Pass_EPA', 0):.3f}")
        print(f"   Dynamic HFA: {hfa:.1f}")
        if hfa_components:
            for component, value in hfa_components.items():
                print(f"     • {component}: {value:.2f}")
        
        # COMPARISON
        print("\n3. COMPARISON & ANALYSIS:")
        print("-" * 30)
        diff = spread_simple - spread_standard
        print(f"   Spread Difference: {diff:+.1f} points")
        print(f"   (Simple - Standard = {spread_simple:+.1f} - {spread_standard:+.1f})")
        
        if abs(diff) > 3:
            print(f"   🚨 SIGNIFICANT DIFFERENCE! Analyzing causes...")
            
            print(f"\n   POTENTIAL CAUSES:")
            if abs(hfa - 2.5) > 0.5:
                print(f"     • HFA difference: {hfa:.1f} vs 2.5 = {hfa-2.5:+.1f}")
            
            # Calculate simple net EPA for comparison
            eagles_simple_net = details_simple['home_stats']['net_epa_per_play']
            cowboys_simple_net = details_simple['away_stats']['net_epa_per_play']
            
            # Calculate approximate net EPA from standard model
            eagles_std_net = (eagles_weighted.get('Off_Rush_EPA', 0) + eagles_weighted.get('Off_Pass_EPA', 0)) - \
                           (eagles_weighted.get('Def_Rush_EPA', 0) + eagles_weighted.get('Def_Pass_EPA', 0))
            cowboys_std_net = (cowboys_weighted.get('Off_Rush_EPA', 0) + cowboys_weighted.get('Off_Pass_EPA', 0)) - \
                            (cowboys_weighted.get('Def_Rush_EPA', 0) + cowboys_weighted.get('Def_Pass_EPA', 0))
            
            print(f"     • Eagles Net EPA: Simple={eagles_simple_net:.3f}, Standard≈{eagles_std_net:.3f}")
            print(f"     • Cowboys Net EPA: Simple={cowboys_simple_net:.3f}, Standard≈{cowboys_std_net:.3f}")
            
            if abs(eagles_simple_net - eagles_std_net) > 0.05:
                print(f"     • SOS adjustments significantly affected Eagles EPA")
            if abs(cowboys_simple_net - cowboys_std_net) > 0.05:
                print(f"     • SOS adjustments significantly affected Cowboys EPA")
                
        else:
            print(f"   ✅ Models are reasonably aligned")
            
    except Exception as e:
        print(f"   ❌ Error testing standard model: {e}")
        print(f"   Note: Standard model may require additional setup")

def methodology_recommendations():
    """
    Provide recommendations on when to use each model.
    """
    print("\n\n🎯 METHODOLOGY RECOMMENDATIONS:")
    print("="*50)
    
    print("USE SIMPLE MODEL WHEN:")
    print("  ✅ You want transparent, interpretable results")
    print("  ✅ You prefer consistent methodology across all games")
    print("  ✅ You want to avoid potential overfitting from adjustments")
    print("  ✅ You're doing rapid prototyping or backtesting")
    print("  ✅ You want to easily understand why a prediction was made")
    
    print("\nUSE STANDARD MODEL WHEN:")
    print("  ✅ You want maximum accuracy with sophisticated adjustments")
    print("  ✅ You need to account for strength of schedule differences")
    print("  ✅ You want team-specific and venue-specific HFA")
    print("  ✅ You're making high-stakes predictions")
    print("  ✅ You want to separate rush/pass performance analysis")
    
    print("\n🔄 HYBRID APPROACH:")
    print("  • Use Simple Model for general predictions and understanding")
    print("  • Use Standard Model for final validation on important games")
    print("  • Compare both models - large differences indicate uncertainty")
    print("  • Simple Model spread +/- 2 points ≈ Standard Model range")

if __name__ == "__main__":
    compare_model_methodologies()
    test_eagles_cowboys_both_models()
    methodology_recommendations()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("Both models use EPA but with fundamentally different approaches.")
    print("Differences in predictions help identify model uncertainty.")
    print("="*80)
