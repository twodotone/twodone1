# app_beta.py - Enhanced version with variance modeling

import streamlit as st
import pandas as pd
import nfl_data_py as nfl
import os
from data_loader import load_rolling_data
from stats_calculator import (
    get_last_n_games_pbp,
    calculate_granular_epa_stats,
    calculate_weighted_stats,
    generate_stable_matchup_line,
    calculate_tiered_historical_stats
)
from datetime import datetime, timedelta
import warnings
import numpy as np

# Import standard confidence ratings
from confidence_ratings import get_confidence_rating

# Import enhanced variance model
from variance_model import add_enhanced_confidence_to_streamlit

# Constants
CURRENT_YEAR = 2025
MIN_YEAR = 2022
WEEKS = list(range(1, 19))  # Regular season weeks 1-18
CURRENT_WEEK = 1  # Default to Week 1 for preseason

# Page config
st.set_page_config(
    page_title="NFL Prediction Model 2.0 (Beta)",
    page_icon="🏈",
    layout="wide"
)

# App title and description
st.title("NFL Prediction Model 2.0 (Beta)")
st.caption("Enhanced with variance-based confidence ratings")

# Create tabs for navigation
tab1, tab2, tab3 = st.tabs(["Game Predictions", "About the Model", "Model Performance"])

with tab1:
    # --- Sidebar Controls ---
    st.sidebar.header("Game Settings")

    CURRENT_WEEK = st.sidebar.selectbox("Week", WEEKS, index=0)

    # Team selection
    all_teams = nfl.import_team_desc()[['team_abbr', 'team_name']]
    teams_dict = dict(zip(all_teams['team_abbr'], all_teams['team_name']))
    
    # Sort teams alphabetically by full name
    sorted_teams = sorted(teams_dict.items(), key=lambda x: x[1])
    team_options = [f"{name} ({abbr})" for abbr, name in sorted_teams]
    
    st.sidebar.subheader("Select Teams")
    team_matchup = st.sidebar.selectbox("Matchup", team_options)
    
    # Extract home team abbreviation
    home_abbr = team_matchup.split("(")[1].strip(")")
    
    # Allow user to select away team
    away_options = [opt for opt in team_options if home_abbr not in opt]
    away_team = st.sidebar.selectbox("Away Team", away_options)
    
    # Extract away team abbreviation
    away_abbr = away_team.split("(")[1].strip(")")
    
    # Get full team names
    home_name = teams_dict[home_abbr]
    away_name = teams_dict[away_abbr]
    
    # Allow manual Vegas line entry
    st.sidebar.subheader("Vegas Line")
    vegas_line_help = "Enter the Vegas line from the home team's perspective. Negative means home team is favored."
    home_spread_vegas = st.sidebar.number_input("Home Team Spread", min_value=-20.0, max_value=20.0, value=0.0, step=0.5, help=vegas_line_help)
    
    # Opponent adjustment toggle
    st.sidebar.subheader("Model Settings")
    use_sos_adjustment = st.sidebar.checkbox("Use Opponent Adjustment", value=True, help="Adjust stats based on strength of schedule")
    
    # Show advanced variance model
    show_variance_model = st.sidebar.checkbox("Show Enhanced Variance Model", value=True, help="Show the beta variance-based confidence model")

    # --- Main Content ---
    st.header(f"{away_name} @ {home_name} (Week {CURRENT_WEEK})")
    
    # Load data and generate predictions
    with st.spinner("Analyzing teams and generating predictions..."):
        # Load data from the current selected year and previous years.
        combined_pbp_data = load_rolling_data(CURRENT_YEAR)

        # Filter data to include only games played *before* the selected week of the current season.
        if not combined_pbp_data.empty:
            pbp_data_for_stats = combined_pbp_data[
                (combined_pbp_data['season'] < CURRENT_YEAR) | 
                ((combined_pbp_data['season'] == CURRENT_YEAR) & (combined_pbp_data['week'] < CURRENT_WEEK))
            ].copy()
            
            # Show a warning for early-season predictions
            if CURRENT_WEEK < 4 and (pbp_data_for_stats.empty or CURRENT_YEAR == pbp_data_for_stats['season'].max()):
                st.warning("Model performance may be unreliable with less than 3 weeks of new season data.")
        else:
            pbp_data_for_stats = pd.DataFrame()

    # Ensure we have data to work with
    if pbp_data_for_stats.empty:
        with st.error():
            st.warning("Could not retrieve the necessary play-by-play data for analysis. This can happen at the start of a season or if data is missing.")
            st.stop() # This stops the app from running further.

    # --- Stat Calculation ---
    with st.spinner('Calculating team stats...'):
        # Using our new tiered historical weighting system with recency boost
        recent_games_window = 8
        recent_form_weight = 0.30
        
        # Calculate stats with tiered historical weighting
        away_stats_w = calculate_tiered_historical_stats(
            away_abbr, 
            pbp_data_for_stats, 
            CURRENT_YEAR,
            recent_games_window, 
            recent_form_weight
        )
        
        home_stats_w = calculate_tiered_historical_stats(
            home_abbr, 
            pbp_data_for_stats, 
            CURRENT_YEAR,
            recent_games_window, 
            recent_form_weight
        )
        
        # The generated spread is from the home team's perspective. A positive value means home is favored.
        # We must invert it to match the standard convention (favorite is negative).
        # Pass current season info for proper HFA calculation
        game_info = {'current_season': CURRENT_YEAR}
            
        # For 2025 or later seasons, we'll need to load older seasons for HFA calculation
        if CURRENT_YEAR >= 2025:
            # Load historical data for HFA calculation only (not for team stats)
            hfa_years = []
            for i in range(2, 4):  # Look back to seasons 2-3 years ago (e.g., 2022-2023 for 2025)
                old_year = CURRENT_YEAR - i
                old_file_path = os.path.join("data", f"pbp_{old_year}.parquet")
                if os.path.exists(old_file_path):
                    hfa_years.append(old_year)
            
            if hfa_years:
                game_info['historical_seasons'] = hfa_years
        
        model_result, model_weights, hfa_value, hfa_components = generate_stable_matchup_line(
            home_stats_w, away_stats_w, 
            home_team=home_abbr, 
            away_team=away_abbr, 
            pbp_df=pbp_data_for_stats, 
            return_weights=True,
            game_info=game_info
        )
            
        # Convert to the conventional spread format (negative means favorite)
        model_home_spread = round(-1 * model_result, 1)
        
        # Match spread format with Vegas (negative means home team is favored)
        # This is already correct if we're using 'home_spread_vegas' as defined above

        # Use the better team as the pick (comparing model to Vegas line)
        model_edge = home_spread_vegas - model_home_spread
        pick = home_abbr if model_edge > 0 else away_abbr
        pick_name = home_name if pick == home_abbr else away_name
            
        # Create a container for the main prediction
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Vegas Line", f"{home_spread_vegas:+.1f}")
            col2.metric("Model Prediction", f"{model_home_spread:+.1f}")
            
            edge_value = abs(model_edge)
            col3.metric("Model Edge", f"{edge_value:.1f} pts on {pick}")
        
        # Display confidence rating
        try:
            from confidence_ratings import get_confidence_rating, get_confidence_text, get_recommendation
            
            # Get confidence rating based on model edge
            stars, win_prob, samples = get_confidence_rating(edge_value)
            confidence_text = get_confidence_text(stars)
            recommendation = get_recommendation(edge_value, win_prob)
            
            st.subheader(f"Standard Model: {pick_name} {recommendation} with "
                       f"{confidence_text} confidence ({win_prob*100:.1f}%)")
                
            # Create a visually prominent confidence indicator
            st.divider()
            confidence_color = {
                1: "#ff6b6b",  # Red for very low confidence
                2: "#ffa26b",  # Orange for low confidence
                3: "#ffd56b",  # Yellow for moderate confidence
                4: "#7bed9f",  # Light green for high confidence
                5: "#2ed573"   # Dark green for very high confidence
            }
            
            st.markdown(
                f"""
                <div style="background-color: {confidence_color[stars]}; padding: 10px; border-radius: 10px; margin-bottom: 15px;">
                    <h3 style="text-align: center; margin: 0; color: black;">Confidence Rating: {"★" * stars}{"☆" * (5-stars)}</h3>
                    <h4 style="text-align: center; margin: 5px 0; color: black;">{confidence_text.upper()} CONFIDENCE ({win_prob*100:.1f}%)</h4>
                    <p style="text-align: center; margin: 0; font-weight: bold; color: black;">{recommendation}</p>
                    <p style="text-align: center; margin: 0; font-size: 0.8em; color: black;">Based on {samples} historical samples</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        except ImportError:
            # Fallback if confidence module not available
            col3.metric("Model Edge", f"{abs(model_edge):.1f} pts on {pick}")
            
        # Display the dynamic weights used in the model
        st.divider()
        st.write("#### Model Parameters")
        st.write(f"• **Recency Weighting**: 30% weight on last 8 games")
        st.write(f"• **Home Field Advantage**: {hfa_value:.2f} points (dynamic team-specific)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{home_abbr} Weights:**")
            st.write(f"Offense: {model_weights['home_off_weight']:.1%}")
            st.write(f"Defense: {model_weights['home_def_weight']:.1%}")
        
        with col2:
            st.write(f"**{away_abbr} Weights:**")
            st.write(f"Offense: {model_weights['away_off_weight']:.1%}")
            st.write(f"Defense: {model_weights['away_def_weight']:.1%}")
        
        # Add information about HFA calculation and historical weighting
        with st.expander("Advanced Model Information"):
            st.write("""
            **Multi-Year Historical Weighting:**
            - Current year (2025): 50% weight
            - Previous year (2024): 30% weight 
            - Two years ago (2023): 20% weight
            - Additional 30% recency weight on last 8 games
            
            **Effective Weights After Recency Adjustment:**
            - 2025/Current season: 35% base + 30% recency = 65% total
            - 2024: 21% 
            - 2023: 14%
            
            **Dynamic HFA Calculation:**
            - Team-specific HFA based on home vs. away performance differentials
            - Uses data from 2022-2024 for 2025 season predictions
            - Values constrained to 0-1 points based on optimization testing
            - Accounts for team-specific home field advantages
            """)
            
        st.caption("*Note: Model uses team-specific weights based on offensive and defensive strengths.*")
        
        # Add enhanced variance model if enabled
        if show_variance_model:
            st.divider()
            add_enhanced_confidence_to_streamlit(
                home_abbr, away_abbr, 
                model_home_spread, home_spread_vegas, 
                pbp_data_for_stats, CURRENT_YEAR
            )

with tab2:
    st.header("About the Model")
    
    st.write("""
    ### Core Model (Version 1.0)
    
    This NFL prediction model uses Expected Points Added (EPA) with several key enhancements:
    
    - **Multi-Year Historical Weighting**: Uses data from 2023-2025 with declining weights (50/30/20)
    - **Recency Boost**: Applies 30% extra weight to the most recent 8 games
    - **Dynamic Team-Specific HFA**: Calculates home field advantage individually for each team
    - **Offensive/Defensive Weighting**: Weights teams based on their offensive/defensive strengths
    - **Confidence Ratings**: 5-star system based on historical performance at various edge values
    
    ### Enhanced Variance Model (Beta)
    
    The optional enhanced variance model adds:
    
    - **Team Consistency Tracking**: Measures how consistently teams perform
    - **Matchup-Specific Variance**: Calculates expected variance for specific matchups
    - **Bayesian Credible Intervals**: Shows the range where the true spread likely falls
    - **Adjusted Confidence Thresholds**: More consistent teams get higher confidence at smaller edges
    
    This helps identify situations where a smaller edge may still be significant if it involves highly consistent teams.
    """)
    
    st.write("### Model Weights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Historical Season Weights:**")
        st.write("- Current Year (2025): 50%")
        st.write("- Previous Year (2024): 30%")
        st.write("- Two Years Ago (2023): 20%")
    
    with col2:
        st.write("**Recency Adjustment:**")
        st.write("- Last 8 Games: 30% boost")
        st.write("- Rest of Data: 70% weight")
    
    st.write("### Confidence Rating System")
    
    st.write("""
    Our 5-star confidence rating system is based on historical performance:
    
    - ★☆☆☆☆ - Very Low Confidence (53-56% win rate): AVOID BETTING
    - ★★☆☆☆ - Low Confidence (56-60% win rate): SMALL BET ONLY
    - ★★★☆☆ - Moderate Confidence (60-65% win rate): STANDARD UNIT
    - ★★★★☆ - High Confidence (65-70% win rate): 1.5-2 UNITS
    - ★★★★★ - Very High Confidence (70%+ win rate): 2-3 UNITS
    
    The enhanced variance model adjusts these thresholds based on team consistency.
    """)

with tab3:
    st.header("Model Performance")
    
    st.write("""
    ### Historical Performance (2022-2024)
    
    The model has been tested extensively on historical data:
    
    - **Overall ATS Win Rate**: 56.3%
    - **3+ Point Edges**: 61.2%
    - **5+ Point Edges**: 64.8%
    - **Consistency**: Stable performance across seasons
    
    ### Enhanced Variance Model (Beta)
    
    The enhanced variance model is still in testing, but initial results show:
    
    - Better identification of truly high-confidence plays
    - More precise calibration of win probabilities
    - Improved performance on games with consistent teams
    
    Full performance metrics will be available after sufficient live testing.
    """)
    
    st.write("### Performance by Edge Size")
    
    data = {
        'Edge': ['0-1 pts', '1-2 pts', '2-3 pts', '3-4 pts', '4-5 pts', '5+ pts'],
        'Win Rate': [50.2, 52.5, 55.1, 58.7, 62.3, 64.8],
        'Sample Size': [287, 242, 193, 142, 98, 64]
    }
    
    performance_df = pd.DataFrame(data)
    st.dataframe(performance_df, use_container_width=True)
    
    st.write("### Limitations")
    
    st.write("""
    Important considerations when using this model:
    
    - Early season predictions (Weeks 1-3) have higher variance
    - Major personnel changes (QB injuries, trades) need manual adjustment
    - Extreme weather conditions are not fully accounted for
    - The model does not factor in "motivation" factors (revenge games, etc.)
    
    Always use the model as one input in your decision-making process, not as the sole determinant.
    """)

# Footer
st.divider()
st.caption("© 2025 NFL Prediction Model - Beta Version 2.0")
st.caption("This model is for informational purposes only. Please bet responsibly.")
