#!/usr/bin/env python3
"""
Test both directions of Eagles vs Cowboys to identify the issue
"""

from simple_model import SimpleNFLModel
from dynamic_season_model import DynamicSeasonModel

def test_both_directions():
    print("🔍 TESTING BOTH HOME/AWAY CONFIGURATIONS")
    print("="*60)
    
    # Load models
    simple_model = SimpleNFLModel()
    simple_model.load_data([2022, 2023, 2024])
    
    dynamic_model = DynamicSeasonModel() 
    dynamic_model.load_dynamic_data(2025, 1)
    
    print("1. EAGLES AT HOME vs Cowboys:")
    spread1, _ = simple_model.predict_spread('PHI', 'DAL', 1, 2025)
    spread1_dyn, _ = dynamic_model.predict_spread_dynamic('PHI', 'DAL', 1, 2025)
    print(f"   Simple Model: Eagles {spread1:+.1f}")
    print(f"   Dynamic Model: Eagles {spread1_dyn:+.1f}")
    
    print("\n2. COWBOYS AT HOME vs Eagles:")
    spread2, _ = simple_model.predict_spread('DAL', 'PHI', 1, 2025)
    spread2_dyn, _ = dynamic_model.predict_spread_dynamic('DAL', 'PHI', 1, 2025)
    print(f"   Simple Model: Cowboys {spread2:+.1f}")
    print(f"   Dynamic Model: Cowboys {spread2_dyn:+.1f}")
    
    print("\n📊 INTERPRETATION:")
    print(f"   Eagles at home: Eagles favored by {spread1:.1f} (simple) / {spread1_dyn:.1f} (dynamic)")
    print(f"   Cowboys at home: Cowboys favored by {spread2:.1f} (simple) / {spread2_dyn:.1f} (dynamic)")
    
    # Check if there's a major discrepancy
    if abs(spread1 - spread1_dyn) > 5 or abs(spread2 - spread2_dyn) > 5:
        print(f"\n🚨 MAJOR DISCREPANCY DETECTED!")
        print(f"   Difference at PHI: {abs(spread1 - spread1_dyn):.1f} points")
        print(f"   Difference at DAL: {abs(spread2 - spread2_dyn):.1f} points")
    
    # Also test with most recent data only to see 2024 performance
    print(f"\n3. TESTING 2024-ONLY MODEL (to check recent form):")
    recent_model = SimpleNFLModel()
    recent_model.load_data([2024])
    
    spread_recent_phi, details_recent = recent_model.predict_spread('PHI', 'DAL', 1, 2025)
    spread_recent_dal, _ = recent_model.predict_spread('DAL', 'PHI', 1, 2025)
    
    print(f"   Eagles at home (2024 only): Eagles {spread_recent_phi:+.1f}")
    print(f"   Cowboys at home (2024 only): Cowboys {spread_recent_dal:+.1f}")
    
    print(f"\n   2024 Eagles EPA: {details_recent['home_stats']['net_epa_per_play']:.3f}")
    print(f"   2024 Cowboys EPA: {details_recent['away_stats']['net_epa_per_play']:.3f}")

if __name__ == "__main__":
    test_both_directions()
