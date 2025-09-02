import pandas as pd
from hfa_data import HFA_DATA

def calculate_dynamic_hfa(pbp_df, home_team, away_team, game_info=None, return_components=False):
    """
    Calculate a dynamic home field advantage based on:
    1. Base HFA values from historical data (scaled down to optimize performance)
    2. Recent home vs away performance differential
    3. Game context (if provided) like primetime status and travel distance
    
    Parameters:
    -----------
    pbp_df : DataFrame
        Play-by-play data to use for analysis
    home_team : str
        Home team abbreviation
    away_team : str
        Away team abbreviation
    game_info : dict, optional
        Additional game context information like:
        - is_primetime: bool
        - travel_distance: float (in miles)
        - day_of_week: str
    return_components : bool, optional
        If True, return a dictionary of the HFA components for analysis
    
    Returns:
    --------
    float or tuple
        Dynamic HFA value in points, or if return_components=True, 
        a tuple of (hfa_value, components_dict)
    """
    # Get base HFA from the reference data but scale it down to target the 0-1 point range
    # that showed optimal performance in backtesting
    raw_base_hfa = HFA_DATA.get(home_team, 1.5)  # Reduced default from 2.0 to 1.5
    base_hfa = raw_base_hfa * 0.5  # Scale down to target the optimal 0-1 range
    
    # Initialize dynamic factors and components tracking
    recent_performance_factor = 1.0
    primetime_factor = 1.0
    travel_factor = 1.0
    
    components = {
        'raw_base_hfa': raw_base_hfa,
        'scaled_base_hfa': base_hfa,
        'home_epa_diff': 0,
        'is_primetime': False,
        'travel_distance': 0,
        'day_of_week': None
    }
    
    # Calculate recent home/away performance differential (last 16 games)
    if not pbp_df.empty:
        # Filter for regular season games only
        reg_games = pbp_df[pbp_df['season_type'] == 'REG']
        
        # Get home team's performance at home vs away
        home_team_games = reg_games[(reg_games['home_team'] == home_team) | (reg_games['away_team'] == home_team)]
        if not home_team_games.empty:
            # Create game ID and home/away status columns
            home_team_games = home_team_games.assign(
                is_home=lambda x: x['home_team'] == home_team
            )
            
            # Group by game_id and is_home to get per-game stats
            home_team_per_game = home_team_games.groupby(['game_id', 'is_home']).agg({
                'epa': 'mean'
            }).reset_index()
            
            # Compare home vs away EPA
            home_games = home_team_per_game[home_team_per_game['is_home']]['epa'].mean()
            away_games = home_team_per_game[~home_team_per_game['is_home']]['epa'].mean()
            
            # If the team performs better at home, increase the HFA factor
            if not (pd.isna(home_games) or pd.isna(away_games)):
                home_away_diff = home_games - away_games
                components['home_epa_diff'] = home_away_diff
                # Scale to a reasonable factor between 0.8 and 1.2
                recent_performance_factor = 1.0 + (min(max(home_away_diff, -0.2), 0.2))
    
    # Apply primetime and travel adjustments if game info is provided
    if game_info:
        # Primetime games typically have stronger HFA
        is_primetime = game_info.get('is_primetime', False)
        components['is_primetime'] = is_primetime
        if is_primetime:
            primetime_factor = 1.15
        
        # Travel distance impact (diminishing returns)
        travel_distance = game_info.get('travel_distance', 0)
        components['travel_distance'] = travel_distance
        if travel_distance > 0:
            # More impact for cross-country travel (>1500 miles)
            if travel_distance > 1500:
                travel_factor = 1.15
            # Moderate impact for medium distance (500-1500 miles)
            elif travel_distance > 500:
                travel_factor = 1.10
            # Small impact for short travel (<500 miles)
            else:
                travel_factor = 1.05
                
        # Day of week adjustment (Monday/Thursday games can have different dynamics)
        day_of_week = game_info.get('day_of_week', '')
        components['day_of_week'] = day_of_week
        if day_of_week.lower() in ['monday', 'thursday']:
            # Thursday games often favor home teams due to short rest impact on away teams
            if day_of_week.lower() == 'thursday':
                primetime_factor *= 1.1
            # Monday night games have strong home field advantage
            elif day_of_week.lower() == 'monday':
                primetime_factor *= 1.05
    
    # Calculate final dynamic HFA
    components['recent_performance_factor'] = recent_performance_factor
    components['primetime_factor'] = primetime_factor
    components['travel_factor'] = travel_factor
    
    # Calculate dynamic HFA with all factors
    dynamic_hfa = base_hfa * recent_performance_factor * primetime_factor * travel_factor
    
    # Cap the HFA to stay within our optimal range (0-1 points)
    dynamic_hfa = min(max(dynamic_hfa, 0.0), 1.0)
    
    # Round to 1 decimal place for clarity
    dynamic_hfa = round(dynamic_hfa, 1)
    
    # Return either just the HFA value or the value with components
    if return_components:
        components['final_hfa'] = dynamic_hfa
        return dynamic_hfa, components
    else:
        return dynamic_hfa

def analyze_hfa_impact(picks_df):
    """
    Analyze the impact of HFA on model performance
    
    Parameters:
    -----------
    picks_df : DataFrame
        DataFrame containing pick records with HFA values
        
    Returns:
    --------
    dict
        Dictionary with HFA impact analysis results
    """
    results = {}
    
    # Group by HFA ranges
    if 'hfa_value' in picks_df.columns:
        # Create HFA ranges
        picks_df['hfa_range'] = pd.cut(
            picks_df['hfa_value'],
            bins=[-0.1, 1.0, 2.0, 3.0, 4.0],
            labels=['0-1', '1-2', '2-3', '3+']
        )
        
        # Analyze performance by HFA range
        hfa_performance = picks_df.groupby('hfa_range').agg({
            'is_win': ['count', 'sum', 'mean']
        })
        
        # Format the results
        results['hfa_performance'] = hfa_performance.reset_index()
        
        # Check if HFA helped more for home or away picks
        home_picks = picks_df[picks_df['pick'] == picks_df['home_team']]
        away_picks = picks_df[picks_df['pick'] == picks_df['away_team']]
        
        if not home_picks.empty:
            results['home_picks_win_rate'] = home_picks['is_win'].mean()
            results['home_picks_count'] = len(home_picks)
        
        if not away_picks.empty:
            results['away_picks_win_rate'] = away_picks['is_win'].mean()
            results['away_picks_count'] = len(away_picks)
    
    return results
