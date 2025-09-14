#!/usr/bin/env python3
"""
Quick test to check Eagles vs Cowboys prediction
"""

from simple_model import SimpleNFLModel
from dynamic_season_model import DynamicSeasonModel

def test_eagles_cowboys():
    print("🦅 vs 🤠 EAGLES vs COWBOYS TEST")
    print("="*50)
    
    # Test with simple model (our original)
    print("1. SIMPLE MODEL (2022-2024 data):")
    simple_model = SimpleNFLModel()
    simple_model.load_data([2022, 2023, 2024])
    
    spread, details = simple_model.predict_spread('PHI', 'DAL', 1, 2025)
    print(f"   Eagles vs Cowboys: {spread:+.1f}")
    print(f"   Eagles EPA: {details['home_stats']['net_epa_per_play']:.3f}")
    print(f"   Cowboys EPA: {details['away_stats']['net_epa_per_play']:.3f}")
    print(f"   EPA Advantage: {details['epa_advantage']:.3f}")
    
    print("\n2. DYNAMIC MODEL (Week 1 of 2025):")
    dynamic_model = DynamicSeasonModel()
    dynamic_model.load_dynamic_data(2025, 1)
    
    spread2, details2 = dynamic_model.predict_spread_dynamic('PHI', 'DAL', 1, 2025)
    print(f"   Eagles vs Cowboys: {spread2:+.1f}")
    print(f"   Eagles EPA: {details2['home_stats']['net_epa_per_play']:.3f}")
    print(f"   Cowboys EPA: {details2['away_stats']['net_epa_per_play']:.3f}")
    print(f"   EPA Advantage: {details2['epa_advantage']:.3f}")
    print(f"   Season Weights: {details2['season_weights']}")
    
    print("\n3. DIFFERENCE ANALYSIS:")
    print(f"   Spread difference: {spread2 - spread:+.1f} points")
    print(f"   Eagles EPA difference: {details2['home_stats']['net_epa_per_play'] - details['home_stats']['net_epa_per_play']:+.3f}")
    print(f"   Cowboys EPA difference: {details2['away_stats']['net_epa_per_play'] - details['away_stats']['net_epa_per_play']:+.3f}")
    
    if abs(spread2 - spread) > 5:
        print(f"\n🚨 WARNING: Large difference detected! Investigating...")
        
        print(f"\n   Simple model years: 2022, 2023, 2024")
        print(f"   Dynamic model years: {list(details2['season_weights'].keys())}")
        print(f"   Dynamic model weights: {details2['season_weights']}")

if __name__ == "__main__":
    test_eagles_cowboys()
