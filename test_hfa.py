import pandas as pd
import numpy as np
import nfl_data_py as nfl
from dynamic_hfa import calculate_dynamic_hfa

def test_hfa_model():
    print("Testing updated dynamic HFA model...")
    print("="*80)
    
    # Load some sample data
    print("Loading sample data...")
    year = 2024
    try:
        pbp_data = nfl.import_pbp_data([year], downcast=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Test HFA for all teams
    print("\nHFA Values for all teams (should be in 0-1 range):")
    print("-"*60)
    teams = sorted(list(set(pbp_data['home_team'].dropna().unique().tolist())))
    
    for home_team in teams:
        for away_team in teams:
            if home_team == away_team:
                continue
                
            # Test different game contexts
            contexts = [
                {"is_primetime": False, "day_of_week": "sunday"},
                {"is_primetime": True, "day_of_week": "sunday"},
                {"is_primetime": True, "day_of_week": "monday"},
                {"is_primetime": True, "day_of_week": "thursday"}
            ]
            
            for context in contexts:
                # Calculate HFA with components
                hfa_value, components = calculate_dynamic_hfa(
                    pbp_data, 
                    home_team, 
                    away_team, 
                    game_info=context,
                    return_components=True
                )
                
                # Only print the first regular game for each team to keep output manageable
                if context["is_primetime"] == False:
                    print(f"{home_team} vs {away_team}: {hfa_value} points")
                    print(f"  Raw base HFA: {components['raw_base_hfa']}, Scaled: {components['scaled_base_hfa']}")
                
                # Verify HFA is in the 0-1 range
                assert 0.0 <= hfa_value <= 1.0, f"HFA outside 0-1 range: {hfa_value}"
                
                # For primetime games, verify the values
                if context["is_primetime"]:
                    if context["day_of_week"] == "monday":
                        assert components["primetime_factor"] > 1.0, f"Monday primetime factor incorrect: {components['primetime_factor']}"
                    elif context["day_of_week"] == "thursday":
                        assert components["primetime_factor"] > 1.0, f"Thursday primetime factor incorrect: {components['primetime_factor']}"
            
            # Only test one away team per home team to avoid excessive output
            break
    
    print("\nHFA model verification complete. All values within 0-1 range as expected.")
    
if __name__ == "__main__":
    test_hfa_model()
