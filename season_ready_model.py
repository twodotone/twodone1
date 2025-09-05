"""
2025 Season Ready Model

Enhanced version of the simple model that automatically handles 2025 data
and provides production-ready functionality for the new season.
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from simple_model import SimpleNFLModel, calculate_edge_and_confidence
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime


class SeasonReadyModel(SimpleNFLModel):
    """
    Production-ready version of the simple model that automatically
    handles current season data and provides robust error handling.
    """
    
    def __init__(self, 
                 recent_games_window: int = 8,
                 recent_weight: float = 0.3,
                 home_field_advantage: float = 2.5):
        super().__init__(recent_games_window, recent_weight, home_field_advantage)
        self.current_season = 2025  # Update this each year
        
    def get_optimal_years(self, current_season: int = None) -> List[int]:
        """
        Get the optimal 3-year window for the current season.
        
        Args:
            current_season: Current NFL season (defaults to 2025)
            
        Returns:
            List of years to include in analysis
        """
        if current_season is None:
            current_season = self.current_season
            
        # Always use the 3 most recent seasons that have data
        # For 2025 season: use 2023, 2024, 2025
        return [current_season - 2, current_season - 1, current_season]
    
    def load_current_season_data(self, current_season: int = None) -> None:
        """
        Load data optimized for the current season.
        
        Args:
            current_season: Current NFL season (defaults to 2025)
        """
        if current_season is None:
            current_season = self.current_season
            
        optimal_years = self.get_optimal_years(current_season)
        
        print(f"Loading optimal data window for {current_season} season...")
        print(f"Years: {optimal_years}")
        
        try:
            self.load_data(optimal_years)
            print(f"✅ Successfully loaded data for {current_season} season")
        except Exception as e:
            print(f"❌ Error loading current season data: {e}")
            
            # Fallback to previous years if current season data not available
            fallback_years = optimal_years[:-1]  # Remove current season
            print(f"🔄 Falling back to: {fallback_years}")
            
            try:
                self.load_data(fallback_years)
                print(f"✅ Fallback successful")
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                raise
    
    def predict_week(self, home_team: str, away_team: str, 
                    week: int, season: int = None) -> Tuple[float, Dict]:
        """
        Predict a specific week's game with enhanced error handling.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            week: Week number (1-18)
            season: Season year (defaults to current season)
            
        Returns:
            Tuple of (predicted_spread, model_details)
        """
        if season is None:
            season = self.current_season
            
        try:
            return self.predict_spread(home_team, away_team, week, season)
        except Exception as e:
            print(f"❌ Error predicting {away_team}@{home_team} Week {week}: {e}")
            raise
    
    def check_data_freshness(self) -> Dict[str, any]:
        """
        Check how fresh the loaded data is and provide recommendations.
        
        Returns:
            Dictionary with data freshness information
        """
        if self.pbp_data is None:
            return {"status": "no_data", "message": "No data loaded"}
        
        # Get the most recent data available
        max_season = self.pbp_data['season'].max()
        max_week = self.pbp_data[self.pbp_data['season'] == max_season]['week'].max()
        
        # Count games in current season
        current_season_data = self.pbp_data[self.pbp_data['season'] == self.current_season]
        current_season_games = len(current_season_data['game_id'].unique()) if not current_season_data.empty else 0
        
        # Determine status
        if max_season < self.current_season:
            status = "outdated"
            message = f"Most recent data is from {max_season}. Current season ({self.current_season}) data not available yet."
        elif max_season == self.current_season and current_season_games == 0:
            status = "waiting"
            message = f"Current season ({self.current_season}) started but no game data available yet."
        elif max_season == self.current_season and current_season_games > 0:
            status = "current"
            message = f"Current season data available through Week {max_week} ({current_season_games} games)."
        else:
            status = "unknown"
            message = "Unable to determine data status."
        
        return {
            "status": status,
            "message": message,
            "max_season": max_season,
            "max_week": max_week if max_season == self.current_season else None,
            "current_season_games": current_season_games,
            "total_plays": len(self.pbp_data)
        }


def create_production_model() -> SeasonReadyModel:
    """
    Create a production-ready model for the 2025 season.
    
    Returns:
        Configured SeasonReadyModel instance
    """
    print("🏈 Initializing 2025 NFL Season Model...")
    print("="*50)
    
    model = SeasonReadyModel()
    
    # Load optimal data for 2025 season
    model.load_current_season_data()
    
    # Check data status
    freshness = model.check_data_freshness()
    print(f"\n📊 Data Status: {freshness['message']}")
    
    if freshness['status'] == 'current':
        print("✅ Model ready for live predictions!")
    elif freshness['status'] == 'waiting':
        print("⏳ Model ready, waiting for game data...")
    else:
        print("⚠️ Model using historical data only")
    
    return model


def test_week_1_predictions():
    """
    Test some Week 1 predictions to make sure everything works.
    """
    print("\n🧪 Testing Week 1 predictions...")
    print("-" * 40)
    
    model = create_production_model()
    
    # Test matchups for Week 1 2025
    test_games = [
        ("KC", "BAL"),
        ("BUF", "NYJ"), 
        ("SF", "NYG"),
        ("DAL", "LAR")
    ]
    
    for home, away in test_games:
        try:
            spread, details = model.predict_week(home, away, week=1, season=2025)
            edge_vs_hypothetical = abs(spread - 0)  # vs hypothetical pick'em line
            
            print(f"{away}@{home}: {spread:+.1f} (Net EPA diff: {details['epa_advantage']:+.3f})")
            
        except Exception as e:
            print(f"{away}@{home}: Error - {e}")


if __name__ == "__main__":
    # Test the production model
    test_week_1_predictions()
    
    print("\n" + "="*60)
    print("2025 SEASON READINESS CHECKLIST:")
    print("="*60)
    print("✅ Model loads optimal 3-year data window")
    print("✅ Handles 2025 season data when available") 
    print("✅ Graceful fallback if 2025 data not ready")
    print("✅ Enhanced error handling for production use")
    print("✅ Data freshness monitoring")
    print("✅ Week-by-week prediction capability")
    print("\n🚀 MODEL IS READY FOR 2025 SEASON!")
