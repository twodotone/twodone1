"""
Simple NFL Spread Prediction Model

A clean, straightforward EPA-based model for predicting NFL spreads.
Focuses on simplicity, clarity, and proper validation.
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SimpleNFLModel:
    """
    A simple EPA-based NFL spread prediction model.
    
    Key principles:
    1. Use EPA as the primary metric (most predictive stat available)
    2. Apply recency weighting (recent games matter more)
    3. Adjust for opponent strength (strength of schedule)
    4. Use a simple, constant home field advantage
    5. Keep everything transparent and explainable
    """
    
    def __init__(self, 
                 recent_games_window: int = 8,
                 recent_weight: float = 0.3,
                 home_field_advantage: float = 2.5):
        """
        Initialize the model with key parameters.
        
        Args:
            recent_games_window: Number of recent games to emphasize
            recent_weight: Weight given to recent games (0-1)
            home_field_advantage: Points added for home team
        """
        self.recent_games_window = recent_games_window
        self.recent_weight = recent_weight
        self.home_field_advantage = home_field_advantage
        self.pbp_data = None
        
    def load_data(self, years: list) -> None:
        """Load play-by-play data for the specified years."""
        print(f"Loading data for years: {years}")
        self.pbp_data = nfl.import_pbp_data(years, downcast=True)
        print(f"Loaded {len(self.pbp_data)} plays")
        
    def get_team_games(self, team: str, pbp_df: pd.DataFrame, 
                      min_week: int = None) -> pd.DataFrame:
        """Get all games for a team from the play-by-play data."""
        # Filter for regular season games involving this team
        team_games = pbp_df[
            ((pbp_df['home_team'] == team) | (pbp_df['away_team'] == team)) &
            (pbp_df['season_type'] == 'REG')
        ].copy()
        
        if min_week is not None:
            team_games = team_games[team_games['week'] >= min_week]
            
        return team_games
    
    def calculate_team_epa_stats(self, team: str, pbp_df: pd.DataFrame, 
                                opponent_adj: bool = True) -> Dict[str, float]:
        """
        Calculate EPA statistics for a team.
        
        Args:
            team: Team abbreviation
            pbp_df: Play-by-play DataFrame
            opponent_adj: Whether to adjust for opponent strength
            
        Returns:
            Dictionary of EPA statistics
        """
        # Get plays where team is on offense and defense
        offensive_plays = pbp_df[
            (pbp_df['posteam'] == team) & 
            (pbp_df['play_type'].isin(['pass', 'run'])) &
            (~pbp_df['epa'].isna())
        ].copy()
        
        defensive_plays = pbp_df[
            (pbp_df['defteam'] == team) & 
            (pbp_df['play_type'].isin(['pass', 'run'])) &
            (~pbp_df['epa'].isna())
        ].copy()
        
        if len(offensive_plays) == 0 or len(defensive_plays) == 0:
            return {}
            
        stats = {}
        
        # Basic EPA stats
        stats['off_epa_per_play'] = offensive_plays['epa'].mean()
        stats['def_epa_per_play'] = defensive_plays['epa'].mean()  # Lower is better for defense
        stats['net_epa_per_play'] = stats['off_epa_per_play'] - stats['def_epa_per_play']
        
        # Play type breakdown
        pass_plays = offensive_plays[offensive_plays['play_type'] == 'pass']
        run_plays = offensive_plays[offensive_plays['play_type'] == 'run']
        
        if len(pass_plays) > 0:
            stats['off_pass_epa'] = pass_plays['epa'].mean()
        else:
            stats['off_pass_epa'] = 0
            
        if len(run_plays) > 0:
            stats['off_run_epa'] = run_plays['epa'].mean()
        else:
            stats['off_run_epa'] = 0
            
        # Defensive breakdown
        def_pass_plays = defensive_plays[defensive_plays['play_type'] == 'pass']
        def_run_plays = defensive_plays[defensive_plays['play_type'] == 'run']
        
        if len(def_pass_plays) > 0:
            stats['def_pass_epa'] = def_pass_plays['epa'].mean()
        else:
            stats['def_pass_epa'] = 0
            
        if len(def_run_plays) > 0:
            stats['def_run_epa'] = def_run_plays['epa'].mean()
        else:
            stats['def_run_epa'] = 0
            
        # Opponent adjustment (simple strength of schedule)
        if opponent_adj and len(offensive_plays) > 10:
            # Calculate league averages for context
            league_off_epa = pbp_df[
                (pbp_df['play_type'].isin(['pass', 'run'])) &
                (~pbp_df['epa'].isna())
            ]['epa'].mean()
            
            # Get unique opponents faced
            opponents = set()
            for _, play in offensive_plays.iterrows():
                opponents.add(play['defteam'])
            for _, play in defensive_plays.iterrows():
                opponents.add(play['posteam'])
                
            # Simple opponent adjustment - could be enhanced later
            stats['opponents_faced'] = len(opponents)
            stats['opponent_adjusted'] = True
        else:
            stats['opponent_adjusted'] = False
            
        return stats
    
    def get_recent_form(self, team: str, pbp_df: pd.DataFrame) -> Dict[str, float]:
        """Get recent form statistics for a team."""
        # Get unique game IDs for this team, sorted by date
        team_games = self.get_team_games(team, pbp_df)
        
        if team_games.empty:
            return {}
            
        # Get the most recent N games
        unique_games = team_games[['game_id', 'season', 'week']].drop_duplicates()
        recent_games = unique_games.sort_values(['season', 'week'], ascending=False).head(self.recent_games_window)
        
        if recent_games.empty:
            return {}
            
        # Get plays from recent games only
        recent_pbp = pbp_df[pbp_df['game_id'].isin(recent_games['game_id'])]
        
        return self.calculate_team_epa_stats(team, recent_pbp, opponent_adj=False)
    
    def combine_stats(self, season_stats: Dict[str, float], 
                     recent_stats: Dict[str, float]) -> Dict[str, float]:
        """Combine season and recent statistics with weighting."""
        if not season_stats or not recent_stats:
            return season_stats or recent_stats or {}
            
        combined = {}
        
        # Weight the key statistics
        key_stats = ['off_epa_per_play', 'def_epa_per_play', 'net_epa_per_play',
                    'off_pass_epa', 'off_run_epa', 'def_pass_epa', 'def_run_epa']
        
        for stat in key_stats:
            if stat in season_stats and stat in recent_stats:
                combined[stat] = (
                    season_stats[stat] * (1 - self.recent_weight) + 
                    recent_stats[stat] * self.recent_weight
                )
            elif stat in season_stats:
                combined[stat] = season_stats[stat]
            elif stat in recent_stats:
                combined[stat] = recent_stats[stat]
                
        return combined
    
    def predict_spread(self, home_team: str, away_team: str, 
                      current_week: int, current_season: int) -> Tuple[float, Dict]:
        """
        Predict the spread for a matchup.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation  
            current_week: Current week of season
            current_season: Current season year
            
        Returns:
            Tuple of (predicted_spread, model_details)
            Spread is from home team perspective (negative = home favored)
        """
        if self.pbp_data is None:
            raise ValueError("Must load data first using load_data()")
            
        # Filter data to only include games before current week
        pbp_for_prediction = self.pbp_data[
            (self.pbp_data['season'] < current_season) | 
            ((self.pbp_data['season'] == current_season) & (self.pbp_data['week'] < current_week))
        ].copy()
        
        if pbp_for_prediction.empty:
            raise ValueError("No historical data available for prediction")
            
        # Calculate season stats for both teams
        home_season_stats = self.calculate_team_epa_stats(home_team, pbp_for_prediction)
        away_season_stats = self.calculate_team_epa_stats(away_team, pbp_for_prediction)
        
        # Calculate recent form
        home_recent_stats = self.get_recent_form(home_team, pbp_for_prediction)
        away_recent_stats = self.get_recent_form(away_team, pbp_for_prediction)
        
        # Combine stats with recency weighting
        home_combined = self.combine_stats(home_season_stats, home_recent_stats)
        away_combined = self.combine_stats(away_season_stats, away_recent_stats)
        
        if not home_combined or not away_combined:
            raise ValueError("Unable to calculate team statistics")
            
        # Simple spread calculation based on net EPA difference
        home_net_epa = home_combined.get('net_epa_per_play', 0)
        away_net_epa = away_combined.get('net_epa_per_play', 0)
        
        # EPA difference per play
        epa_advantage = home_net_epa - away_net_epa
        
        # Convert EPA per play to point spread (empirical scaling factor)
        # Typical NFL game has ~130 plays, EPA of 0.1 per play ≈ 3-4 points
        points_per_epa = 25  # This could be calibrated from historical data
        
        # Negative spread = home team favored, positive = home team underdog
        predicted_spread = -(epa_advantage * points_per_epa) - self.home_field_advantage
        
        # Model details for transparency
        model_details = {
            'home_net_epa': home_net_epa,
            'away_net_epa': away_net_epa,
            'epa_advantage': epa_advantage,
            'home_field_advantage': self.home_field_advantage,
            'predicted_spread_raw': -(epa_advantage * points_per_epa),
            'predicted_spread_final': predicted_spread,
            'home_stats': home_combined,
            'away_stats': away_combined,
            'recent_weight_used': self.recent_weight,
            'recent_games_window': self.recent_games_window
        }
        
        return round(predicted_spread, 1), model_details

    def predict_total(self, home_team: str, away_team: str, week: int, year: int) -> Tuple[float, Dict]:
        """
        Predict the total points for a game.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation  
            week: Week number
            year: Season year
            
        Returns:
            Tuple of (predicted_total, model_details)
        """
        if self.pbp_data is None:
            raise ValueError("No play-by-play data loaded")
            
        # Filter data up to the prediction point
        pbp_for_prediction = self.pbp_data[
            (self.pbp_data['season'] < year) | 
            ((self.pbp_data['season'] == year) & (self.pbp_data['week'] < week))
        ].copy()
        
        if pbp_for_prediction.empty:
            raise ValueError("No historical data available for prediction")
            
        # Calculate season stats for both teams
        home_season_stats = self.calculate_team_epa_stats(home_team, pbp_for_prediction)
        away_season_stats = self.calculate_team_epa_stats(away_team, pbp_for_prediction)
        
        # Calculate recent form
        home_recent_stats = self.get_recent_form(home_team, pbp_for_prediction)
        away_recent_stats = self.get_recent_form(away_team, pbp_for_prediction)
        
        # Combine stats with recency weighting
        home_combined = self.combine_stats(home_season_stats, home_recent_stats)
        away_combined = self.combine_stats(away_season_stats, away_recent_stats)
        
        if not home_combined or not away_combined:
            raise ValueError("Unable to calculate team statistics")
            
        # Calculate expected points for each team
        # Using offensive EPA vs defensive EPA matchups
        home_off_epa = home_combined.get('off_epa_per_play', 0)
        home_def_epa = home_combined.get('def_epa_per_play', 0)
        away_off_epa = away_combined.get('off_epa_per_play', 0)
        away_def_epa = away_combined.get('def_epa_per_play', 0)
        
        # Expected EPA for each team (offense vs opponent defense)
        home_expected_epa = (home_off_epa - away_def_epa) / 2
        away_expected_epa = (away_off_epa - home_def_epa) / 2
        
        # Convert EPA to points using NFL-calibrated scaling
        # Average NFL team scores ~23 points, average EPA ~0.02
        # Better teams might have 0.1+ EPA and score 27+ points
        plays_per_team = 65
        
        # Scale EPA relative to league average to get points above/below average
        league_avg_epa = 0.02  # Approximate league average EPA per play
        league_avg_points = 23  # Approximate league average points per game
        
        home_epa_above_avg = home_expected_epa - league_avg_epa
        away_epa_above_avg = away_expected_epa - league_avg_epa
        
        # Convert EPA advantage to points (more conservative scaling)
        epa_to_points_factor = 30  # Points per EPA unit above average
        
        home_expected_points = league_avg_points + (home_epa_above_avg * epa_to_points_factor)
        away_expected_points = league_avg_points + (away_epa_above_avg * epa_to_points_factor)
        
        predicted_total = home_expected_points + away_expected_points
        
        # Model details for transparency
        total_details = {
            'home_off_epa': home_off_epa,
            'home_def_epa': home_def_epa,
            'away_off_epa': away_off_epa,
            'away_def_epa': away_def_epa,
            'home_expected_epa': home_expected_epa,
            'away_expected_epa': away_expected_epa,
            'home_expected_points': home_expected_points,
            'away_expected_points': away_expected_points,
            'predicted_total': predicted_total,
            'league_avg_epa': league_avg_epa,
            'league_avg_points': league_avg_points,
            'epa_to_points_factor': epa_to_points_factor
        }
        
        return round(predicted_total, 1), total_details


def calculate_edge_and_confidence(model_spread: float, vegas_spread: float) -> Tuple[float, str]:
    """
    Calculate the edge and provide a simple confidence assessment.
    
    Args:
        model_spread: Model's predicted spread
        vegas_spread: Vegas/market spread
        
    Returns:
        Tuple of (edge_magnitude, confidence_level)
    """
    edge = abs(model_spread - vegas_spread)
    
    # Simple confidence levels based on edge size
    if edge < 2:
        confidence = "Low"
    elif edge < 4:
        confidence = "Moderate"  
    elif edge < 6:
        confidence = "High"
    else:
        confidence = "Very High"
        
    return edge, confidence


if __name__ == "__main__":
    # Example usage
    model = SimpleNFLModel()
    
    # Load recent years of data
    model.load_data([2022, 2023, 2024])
    
    # Example prediction
    try:
        spread, details = model.predict_spread('KC', 'BUF', current_week=1, current_season=2025)
        print(f"Predicted spread: KC {spread:+.1f}")
        print(f"Model details: {details}")
        
        # Compare to hypothetical Vegas line
        vegas_line = -3.0
        edge, confidence = calculate_edge_and_confidence(spread, vegas_line)
        print(f"Edge: {edge:.1f} points ({confidence} confidence)")
        
    except Exception as e:
        print(f"Error making prediction: {e}")
