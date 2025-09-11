#!/usr/bin/env python3
"""
Debug EPA calculations for Eagles and Cowboys
"""

from simple_model import SimpleNFLModel
import pandas as pd

def debug_epa_calculations():
    print("🔍 DEBUGGING EPA CALCULATIONS")
    print("="*50)
    
    model = SimpleNFLModel()
    model.load_data([2024])  # Just 2024 to see recent performance
    
    # Check Eagles 2024 performance directly
    eagles_pbp = model.pbp_data[
        (model.pbp_data['posteam'] == 'PHI') | (model.pbp_data['defteam'] == 'PHI')
    ].copy()
    
    cowboys_pbp = model.pbp_data[
        (model.pbp_data['posteam'] == 'DAL') | (model.pbp_data['defteam'] == 'DAL')  
    ].copy()
    
    print(f"Eagles games in 2024: {eagles_pbp['game_id'].nunique()} unique games")
    print(f"Cowboys games in 2024: {cowboys_pbp['game_id'].nunique()} unique games")
    
    # Check offensive EPA for Eagles
    eagles_off = eagles_pbp[eagles_pbp['posteam'] == 'PHI']
    eagles_def = eagles_pbp[eagles_pbp['defteam'] == 'PHI']
    
    print(f"\nEAGLES 2024:")
    print(f"  Offensive plays: {len(eagles_off)}")
    print(f"  Offensive EPA/play: {eagles_off['epa'].mean():.3f}")
    print(f"  Defensive plays: {len(eagles_def)}")  
    print(f"  Defensive EPA/play allowed: {eagles_def['epa'].mean():.3f}")
    print(f"  Net EPA/play: {eagles_off['epa'].mean() - eagles_def['epa'].mean():.3f}")
    
    # Check Cowboys
    cowboys_off = cowboys_pbp[cowboys_pbp['posteam'] == 'DAL']
    cowboys_def = cowboys_pbp[cowboys_pbp['defteam'] == 'DAL']
    
    print(f"\nCOWBOYS 2024:")
    print(f"  Offensive plays: {len(cowboys_off)}")
    print(f"  Offensive EPA/play: {cowboys_off['epa'].mean():.3f}")
    print(f"  Defensive plays: {len(cowboys_def)}")
    print(f"  Defensive EPA/play allowed: {cowboys_def['epa'].mean():.3f}")
    print(f"  Net EPA/play: {cowboys_off['epa'].mean() - cowboys_def['epa'].mean():.3f}")
    
    # Check team records
    print(f"\n📊 TEAM RECORDS CHECK:")
    
    # Eagles win/loss record
    eagles_home = model.pbp_data[(model.pbp_data['home_team'] == 'PHI')]
    eagles_away = model.pbp_data[(model.pbp_data['away_team'] == 'PHI')]
    
    eagles_home_wins = 0
    eagles_away_wins = 0
    
    for game_id in eagles_home['game_id'].unique():
        game_data = eagles_home[eagles_home['game_id'] == game_id]
        if len(game_data) > 0:
            final_score_diff = game_data['score_differential'].iloc[-1]
            if final_score_diff > 0:  # Home team won
                eagles_home_wins += 1
                
    for game_id in eagles_away['game_id'].unique():
        game_data = eagles_away[eagles_away['game_id'] == game_id]
        if len(game_data) > 0:
            final_score_diff = game_data['score_differential'].iloc[-1]
            if final_score_diff < 0:  # Away team won
                eagles_away_wins += 1
    
    eagles_total_games = len(eagles_home['game_id'].unique()) + len(eagles_away['game_id'].unique())
    eagles_wins = eagles_home_wins + eagles_away_wins
    
    print(f"Eagles 2024: {eagles_wins}-{eagles_total_games - eagles_wins}")
    
    # Let's also manually calculate the prediction
    print(f"\n🧮 MANUAL PREDICTION CALCULATION:")
    eagles_stats = model.calculate_team_epa_stats('PHI', model.pbp_data)
    cowboys_stats = model.calculate_team_epa_stats('DAL', model.pbp_data)
    
    print(f"Eagles net EPA: {eagles_stats['net_epa_per_play']:.3f}")
    print(f"Cowboys net EPA: {cowboys_stats['net_epa_per_play']:.3f}")
    
    epa_diff = eagles_stats['net_epa_per_play'] - cowboys_stats['net_epa_per_play']
    raw_spread = epa_diff * 25
    final_spread = raw_spread + 2.5  # HFA
    
    print(f"EPA difference: {epa_diff:.3f}")
    print(f"Raw spread: {raw_spread:.1f}")
    print(f"With HFA: {final_spread:.1f}")
    print(f"Interpretation: Eagles {final_spread:+.1f} (negative = Eagles favored)")

if __name__ == "__main__":
    debug_epa_calculations()
