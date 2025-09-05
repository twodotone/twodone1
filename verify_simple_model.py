"""
COMPREHENSIVE SIMPLE MODEL VERIFICATION

Test to ensure the simple model is correctly:
1. Calculating EPA advantages
2. Converting to spreads with correct signs
3. Displaying favorites vs underdogs properly
4. Handling home field advantage correctly
"""

from simple_model import SimpleNFLModel
from dynamic_season_model import DynamicSeasonModel

def test_spread_sign_logic():
    """
    Test the fundamental spread sign logic with clear examples.
    """
    print("🔍 TESTING SPREAD SIGN LOGIC")
    print("="*60)
    
    # Load simple model
    model = SimpleNFLModel()
    model.load_data([2022, 2023, 2024])
    
    # Test a clear favorite vs underdog scenario
    # Eagles (good team) vs Bears (weaker team)
    print("\n📊 TEST CASE: Eagles (home) vs Bears (away)")
    print("-" * 50)
    
    spread, details = model.predict_spread('PHI', 'CHI', 1, 2025)
    
    eagles_epa = details['home_stats']['net_epa_per_play']
    bears_epa = details['away_stats']['net_epa_per_play']
    epa_advantage = details['epa_advantage']
    raw_spread = details['predicted_spread_raw']
    hfa = details['home_field_advantage']
    final_spread = details['predicted_spread_final']
    
    print(f"Eagles EPA: {eagles_epa:.3f}")
    print(f"Bears EPA:  {bears_epa:.3f}")
    print(f"EPA Advantage (Eagles - Bears): {epa_advantage:.3f}")
    print(f"Raw spread (before HFA): {raw_spread:.1f}")
    print(f"HFA: +{hfa:.1f}")
    print(f"Final spread: {final_spread:+.1f}")
    
    # Logic check
    print(f"\n🧮 LOGIC CHECK:")
    print(f"1. EPA Advantage: {epa_advantage:.3f}")
    print(f"2. Raw calculation: -{epa_advantage:.3f} * 25 = {-epa_advantage * 25:.1f}")
    print(f"3. With HFA: {-epa_advantage * 25:.1f} - {hfa:.1f} = {(-epa_advantage * 25) - hfa:.1f}")
    
    # Interpretation
    if eagles_epa > bears_epa:
        expected_favorite = "Eagles"
        expected_sign = "negative"
    else:
        expected_favorite = "Bears"
        expected_sign = "positive"
    
    actual_favorite = "Eagles" if final_spread < 0 else "Bears"
    actual_sign = "negative" if final_spread < 0 else "positive"
    
    print(f"\n✅ INTERPRETATION:")
    print(f"Expected favorite: {expected_favorite} (better EPA)")
    print(f"Expected spread sign: {expected_sign}")
    print(f"Actual favorite: {actual_favorite}")
    print(f"Actual spread sign: {actual_sign}")
    print(f"Result: Eagles {final_spread:+.1f} means {'Eagles favored by ' + str(abs(final_spread)) if final_spread < 0 else 'Bears favored by ' + str(abs(final_spread))} points")
    
    # Check if logic is correct
    if (eagles_epa > bears_epa and final_spread < 0) or (eagles_epa < bears_epa and final_spread > 0):
        print(f"✅ LOGIC CORRECT: Better EPA team is favored (negative spread)")
    else:
        print(f"❌ LOGIC ERROR: Better EPA team should be favored!")
        
    return final_spread, details

def test_multiple_scenarios():
    """
    Test multiple team combinations to verify consistency.
    """
    print(f"\n\n🎯 TESTING MULTIPLE SCENARIOS")
    print("="*60)
    
    model = SimpleNFLModel()
    model.load_data([2022, 2023, 2024])
    
    # Test scenarios: [home_team, away_team, expected_home_advantage]
    test_cases = [
        ('KC', 'NYJ', 'KC should be heavily favored'),  # Strong home vs weak away
        ('BUF', 'MIA', 'BUF should be favored'),        # Strong home vs decent away  
        ('SF', 'LA', 'Close game, SF slight edge'),     # Two good teams (LA not LAR)
        ('NYG', 'DAL', 'DAL should be favored'),        # Weak home vs decent away
        ('DET', 'GB', 'Close NFC North rivalry')        # Divisional matchup
    ]
    
    for home, away, expectation in test_cases:
        spread, details = model.predict_spread(home, away, 1, 2025)
        
        home_epa = details['home_stats']['net_epa_per_play']
        away_epa = details['away_stats']['net_epa_per_play']
        
        print(f"\n{away}@{home}:")
        print(f"  {home} EPA: {home_epa:.3f}")
        print(f"  {away} EPA: {away_epa:.3f}")
        print(f"  Spread: {home} {spread:+.1f}")
        print(f"  Interpretation: {home if spread < 0 else away} favored by {abs(spread):.1f}")
        print(f"  Expectation: {expectation}")
        
        # Sanity check
        if home_epa > away_epa and spread > 0:
            print(f"  ⚠️  WARNING: {home} has better EPA but is underdog!")
        elif home_epa < away_epa and spread < 0:
            print(f"  ⚠️  WARNING: {home} has worse EPA but is favorite!")

def test_away_team_perspective():
    """
    Test how spreads look from away team perspective.
    """
    print(f"\n\n📍 TESTING AWAY TEAM PERSPECTIVE")
    print("="*60)
    
    model = SimpleNFLModel()
    model.load_data([2022, 2023, 2024])
    
    # Test KC at PHI
    home_spread, _ = model.predict_spread('PHI', 'KC', 1, 2025)
    
    print(f"KC @ PHI:")
    print(f"  From home perspective: PHI {home_spread:+.1f}")
    print(f"  From away perspective: KC {-home_spread:+.1f}")
    print(f"  Interpretation: {('PHI favored by ' + str(abs(home_spread))) if home_spread < 0 else ('KC favored by ' + str(abs(home_spread)))}")
    
    # Test flipped
    away_spread, _ = model.predict_spread('KC', 'PHI', 1, 2025)
    
    print(f"\nPHI @ KC:")
    print(f"  From home perspective: KC {away_spread:+.1f}")
    print(f"  From away perspective: PHI {-away_spread:+.1f}")
    print(f"  Interpretation: {('KC favored by ' + str(abs(away_spread))) if away_spread < 0 else ('PHI favored by ' + str(abs(away_spread)))}")
    
    # Check consistency
    print(f"\n🔄 CONSISTENCY CHECK:")
    print(f"  PHI home vs KC: PHI {home_spread:+.1f}")
    print(f"  KC home vs PHI: KC {away_spread:+.1f}")
    
    # Should be roughly opposite accounting for HFA difference
    expected_diff = 2 * 2.5  # 2x HFA since one loses HFA and other gains it
    actual_diff = away_spread - (-home_spread)
    print(f"  Expected HFA difference: ~{expected_diff:.1f}")
    print(f"  Actual difference: {actual_diff:.1f}")
    print(f"  {'✅ Consistent' if abs(actual_diff - expected_diff) < 1 else '❌ Inconsistent'}")

def test_dynamic_vs_simple():
    """
    Compare simple vs dynamic model for consistency.
    """
    print(f"\n\n⚖️  SIMPLE vs DYNAMIC MODEL COMPARISON")
    print("="*60)
    
    simple_model = SimpleNFLModel()
    simple_model.load_data([2022, 2023, 2024])
    
    dynamic_model = DynamicSeasonModel()
    dynamic_model.load_dynamic_data(2025, 1)
    
    # Test Eagles vs Cowboys
    simple_spread, simple_details = simple_model.predict_spread('PHI', 'DAL', 1, 2025)
    dynamic_spread, dynamic_details = dynamic_model.predict_spread_dynamic('PHI', 'DAL', 1, 2025)
    
    print(f"Eagles vs Cowboys:")
    print(f"  Simple Model:  PHI {simple_spread:+.1f}")
    print(f"  Dynamic Model: PHI {dynamic_spread:+.1f}")
    print(f"  Difference: {abs(simple_spread - dynamic_spread):.1f} points")
    
    # Check if both agree on favorite
    simple_favorite = "PHI" if simple_spread < 0 else "DAL"
    dynamic_favorite = "PHI" if dynamic_spread < 0 else "DAL"
    
    print(f"  Simple picks: {simple_favorite}")
    print(f"  Dynamic picks: {dynamic_favorite}")
    print(f"  {'✅ Agreement on favorite' if simple_favorite == dynamic_favorite else '❌ Disagree on favorite'}")

if __name__ == "__main__":
    print("🧪 COMPREHENSIVE SIMPLE MODEL VERIFICATION")
    print("="*80)
    
    # Run all tests
    test_spread_sign_logic()
    test_multiple_scenarios()
    test_away_team_perspective()
    test_dynamic_vs_simple()
    
    print(f"\n" + "="*80)
    print("VERIFICATION COMPLETE!")
    print("Check above for any logic errors or sign inconsistencies.")
    print("="*80)
