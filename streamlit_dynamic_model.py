"""
Streamlit-compatible Dynamic Season Model
Uses local parquet files instead of nfl_data_py downloads
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class StreamlitDynamicSeasonModel:
    """
    Streamlit-compatible version of DynamicSeasonModel using local parquet files
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.pbp_data = None
        self.current_week = None
        self.current_year = None
        self.season_weights = {}
        
    def get_dynamic_years(self, current_year: int, current_week: int) -> Tuple[List[int], Dict[int, float]]:
        """
        Determine which years to include and their weights based on season progress.
        """
        if current_week <= 4:
            # Early season: Include 2022, fade it out gradually
            years = [2022, 2023, 2024]
            weights = {
                2022: 0.25,  # Still use some old data
                2023: 0.35,  # More weight on 2023
                2024: 0.4    # Most weight on most recent
            }
        elif current_week <= 8:
            # Mid season: Transition away from 2022
            years = [2022, 2023, 2024]
            weights = {
                2022: 0.15,  # Reduced 2022 weight
                2023: 0.35,
                2024: 0.5    # Increased recent weight
            }
        elif current_week <= 12:
            # Late season: Phase out 2022 completely
            years = [2023, 2024]
            weights = {
                2023: 0.4,
                2024: 0.6
            }
        else:
            # Very late season: Focus on recent years only
            years = [2023, 2024]
            weights = {
                2023: 0.3,
                2024: 0.7
            }
        
        # If we have current year data, include it
        if current_year > 2024:
            years.append(current_year)
            # Redistribute weights to include current year
            total_weight = sum(weights.values())
            current_year_weight = min(0.1 + (current_week - 1) * 0.02, 0.3)  # Up to 30% by week 10
            
            # Scale down other weights
            scale_factor = (1 - current_year_weight) / total_weight
            weights = {year: weight * scale_factor for year, weight in weights.items()}
            weights[current_year] = current_year_weight
        
        return years, weights
        
    def load_dynamic_data(self, current_year: int, current_week: int) -> None:
        """Load data with dynamic weighting based on season progress."""
        import os
        
        print(f"Loading dynamic data for Week {current_week} of {current_year} season")
        
        self.current_week = current_week
        self.current_year = current_year
        
        # Determine years and weights
        years, self.season_weights = self.get_dynamic_years(current_year, current_week)
        
        print(f"Years: {years}")
        print(f"Weights: {self.season_weights}")
        
        # Load play-by-play data from parquet files
        dfs = []
        for year in years:
            file_path = os.path.join(self.data_dir, f"pbp_{year}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                if len(df) > 0:  # Only add if not empty
                    dfs.append(df)
                    print(f"Loaded {year}: {len(df):,} plays")
                else:
                    print(f"Warning: {year} data is empty")
            else:
                print(f"Warning: {file_path} not found")
        
        if dfs:
            self.pbp_data = pd.concat(dfs, ignore_index=True)
            print(f"✅ Successfully loaded weighted data")
            print(f"Total plays: {len(self.pbp_data):,}")
        else:
            raise FileNotFoundError("No play-by-play data files found")
            
    def calculate_weighted_epa_stats(self, team: str, pbp_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate EPA statistics with season weighting."""
        stats_by_season = {}
        
        # Calculate stats for each season
        for season in self.season_weights.keys():
            season_data = pbp_df[pbp_df['season'] == season]
            if len(season_data) == 0:
                continue
                
            # Get plays where team is on offense and defense
            offensive_plays = season_data[
                (season_data['posteam'] == team) & 
                (season_data['play_type'].isin(['pass', 'run'])) &
                (~season_data['epa'].isna())
            ]
            
            defensive_plays = season_data[
                (season_data['defteam'] == team) & 
                (season_data['play_type'].isin(['pass', 'run'])) &
                (~season_data['epa'].isna())
            ]
            
            if len(offensive_plays) > 0 and len(defensive_plays) > 0:
                stats_by_season[season] = {
                    'off_epa_per_play': offensive_plays['epa'].mean(),
                    'def_epa_per_play': defensive_plays['epa'].mean(),
                }
                stats_by_season[season]['net_epa_per_play'] = (
                    stats_by_season[season]['off_epa_per_play'] - 
                    stats_by_season[season]['def_epa_per_play']
                )
        
        # Weight the stats by season importance
        if not stats_by_season:
            return {}
            
        weighted_stats = {}
        stat_names = ['off_epa_per_play', 'def_epa_per_play', 'net_epa_per_play']
        
        for stat_name in stat_names:
            weighted_value = 0
            total_weight = 0
            
            for season, weight in self.season_weights.items():
                if season in stats_by_season and stat_name in stats_by_season[season]:
                    weighted_value += stats_by_season[season][stat_name] * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_stats[stat_name] = weighted_value / total_weight
        
        # Add season weights for transparency
        weighted_stats['season_weights'] = self.season_weights.copy()
        
        return weighted_stats
        
    def get_team_games(self, team: str, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """Get all games for a specific team."""
        team_games = pbp_df[
            ((pbp_df['home_team'] == team) | (pbp_df['away_team'] == team)) &
            (~pbp_df['week'].isna())
        ].copy()
        
        return team_games.drop_duplicates(subset=['game_id'])
        
    def calculate_strength_of_schedule_adjustment(self, team: str, pbp_df: pd.DataFrame) -> float:
        """Calculate strength of schedule adjustment for a team."""
        team_games = self.get_team_games(team, pbp_df)
        
        if len(team_games) == 0:
            return 0.0
        
        # Get opponents
        opponents = []
        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                opponents.append(game['away_team'])
            else:
                opponents.append(game['home_team'])
        
        # Calculate average opponent strength
        opponent_strengths = []
        for opponent in opponents:
            if opponent != team:  # Don't include self
                opp_stats = self.calculate_weighted_epa_stats(opponent, pbp_df)
                if opp_stats and 'net_epa_per_play' in opp_stats:
                    opponent_strengths.append(opp_stats['net_epa_per_play'])
        
        if not opponent_strengths:
            return 0.0
        
        avg_opponent_strength = np.mean(opponent_strengths)
        
        # Adjustment factor: positive means faced tougher opponents
        # Scale by 0.1 to avoid over-adjusting
        return avg_opponent_strength * 0.1
        
    def calculate_dynamic_hfa(self, home_team: str, away_team: str, week: int, year: int) -> float:
        """Calculate dynamic home field advantage."""
        # Base HFA
        base_hfa = 2.5
        
        # Week adjustment (HFA slightly lower late in season)
        week_adjustment = max(0, (18 - week) * 0.05)  # Up to 0.85 pts extra early season
        
        # Team-specific adjustments (simplified for Streamlit)
        strong_home_teams = ['GB', 'KC', 'SEA', 'NO', 'DEN']  # Historical strong home teams
        weak_home_teams = ['LAC', 'LV', 'JAX']  # Teams with weaker home advantages
        
        team_adjustment = 0
        if home_team in strong_home_teams:
            team_adjustment = 0.5
        elif home_team in weak_home_teams:
            team_adjustment = -0.5
        
        return base_hfa + week_adjustment + team_adjustment
        
    def predict_spread_dynamic(self, home_team: str, away_team: str, week: int, year: int) -> Tuple[float, Dict]:
        """Predict spread using dynamic season weighting and SOS adjustments."""
        if self.pbp_data is None:
            raise ValueError("No play-by-play data loaded")
            
        # Filter data up to prediction point
        pbp_for_prediction = self.pbp_data[
            (self.pbp_data['season'] < year) | 
            ((self.pbp_data['season'] == year) & (self.pbp_data['week'] < week))
        ].copy()
        
        if pbp_for_prediction.empty:
            raise ValueError("No historical data available for prediction")
        
        # Calculate weighted stats for both teams
        home_stats = self.calculate_weighted_epa_stats(home_team, pbp_for_prediction)
        away_stats = self.calculate_weighted_epa_stats(away_team, pbp_for_prediction)
        
        if not home_stats or not away_stats:
            raise ValueError("Unable to calculate team statistics")
        
        # Get base EPA values
        home_net_epa = home_stats.get('net_epa_per_play', 0)
        away_net_epa = away_stats.get('net_epa_per_play', 0)
        
        # Calculate SOS adjustments
        home_sos_adj = self.calculate_strength_of_schedule_adjustment(home_team, pbp_for_prediction)
        away_sos_adj = self.calculate_strength_of_schedule_adjustment(away_team, pbp_for_prediction)
        
        # Apply SOS adjustments
        home_adjusted_epa = home_net_epa + home_sos_adj
        away_adjusted_epa = away_net_epa + away_sos_adj
        
        # Calculate EPA advantage
        epa_advantage = home_adjusted_epa - away_adjusted_epa
        
        # Calculate dynamic HFA
        dynamic_hfa = self.calculate_dynamic_hfa(home_team, away_team, week, year)
        
        # Convert to spread (negative = home favored)
        points_per_epa = 25
        predicted_spread = -(epa_advantage * points_per_epa) - dynamic_hfa
        
        # Model details
        model_details = {
            'home_net_epa': home_net_epa,
            'away_net_epa': away_net_epa,
            'home_sos_adjustment': home_sos_adj,
            'away_sos_adjustment': away_sos_adj,
            'home_adjusted_epa': home_adjusted_epa,
            'away_adjusted_epa': away_adjusted_epa,
            'epa_advantage': epa_advantage,
            'dynamic_hfa': dynamic_hfa,
            'predicted_spread_raw': -(epa_advantage * points_per_epa),
            'predicted_spread_final': predicted_spread,
            'season_weights': self.season_weights,
            'week': week,
            'year': year
        }
        
        return round(predicted_spread, 1), model_details
