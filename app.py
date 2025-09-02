# app.py

import streamlit as st
import pandas as pd
import nfl_data_py as nfl
import os
from data_loader import load_rolling_data
from stats_calculator import (
    get_last_n_games_pbp,
    calculate_granular_epa_stats,
    calculate_weighted_stats,
    generate_stable_matchup_line
)

# --- Page & Sidebar Configuration ---
st.set_page_config(page_title="NFL Matchup Analyzer", layout="wide")
st.title('🏈 NFL Matchup Analyzer')
st.sidebar.header('Season Settings')
CURRENT_YEAR = st.sidebar.selectbox('Year', [2025, 2024, 2023, 2022], index=0)
CURRENT_WEEK = st.sidebar.number_input('Week', min_value=1, max_value=18, value=1, step=1)

st.sidebar.header('Model Settings')
use_sos_adjustment = st.sidebar.checkbox('Apply Strength of Schedule Adjustment', value=True)

# --- Data Loading ---
try:
    team_desc = nfl.import_team_desc()
    schedule_data = nfl.import_schedules([CURRENT_YEAR])
except Exception as e:
    st.error(f"Could not load schedule or team data for {CURRENT_YEAR}. Error: {e}")
    st.stop()

# --- Main Page: Matchup Selection ---
st.header(f'Week {CURRENT_WEEK} Matchups for the {CURRENT_YEAR} Season')
week_schedule = schedule_data[schedule_data['week'] == CURRENT_WEEK].copy()
if week_schedule.empty:
    st.warning(f"No schedule found for Week {CURRENT_WEEK} of the {CURRENT_YEAR} season.")
else:
    week_schedule['game_description'] = week_schedule['away_team'] + ' @ ' + week_schedule['home_team']
    game_options = ["Select a Game to Analyze"] + week_schedule['game_description'].tolist()
    selected_game_str = st.selectbox('Choose a matchup:', game_options)

    if selected_game_str != "Select a Game to Analyze":
        game_details = week_schedule[week_schedule['game_description'] == selected_game_str].iloc[0]
        away_abbr, home_abbr = game_details['away_team'], game_details['home_team']
        
        # Display Betting Odds Banner
        st.subheader("Betting Odds & Game Info")
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            away_logo = team_desc.loc[team_desc['team_abbr'] == away_abbr, 'team_logo_espn'].values[0]
            home_logo = team_desc.loc[team_desc['team_abbr'] == home_abbr, 'team_logo_espn'].values[0]
            
            home_ml = game_details.get('home_moneyline')
            away_ml = game_details.get('away_moneyline')
            spread_magnitude = abs(game_details.get('spread_line', 0))

            if home_ml is not None and away_ml is not None:
                if home_ml < away_ml:
                    home_spread_vegas = -spread_magnitude
                    away_spread_vegas = spread_magnitude
                else:
                    home_spread_vegas = spread_magnitude
                    away_spread_vegas = -spread_magnitude
            else:
                # Fallback if moneyline is not available
                home_spread_vegas = game_details.get('spread_line', 0)
                away_spread_vegas = -home_spread_vegas
            
            total_line = game_details.get('total_line', 0)
            
            col1.image(away_logo, width=70)
            col2.metric("Away Spread", f"{away_spread_vegas:+.1f}")
            col3.metric("Over/Under", f"{total_line:.1f}")
            col4.metric("Home Spread", f"{home_spread_vegas:+.1f}")
            col5.image(home_logo, width=70)

        # --- Data Prep using Rolling Window ---
        st.info("Using a rolling data window including the previous season for statistical analysis.")

        # Load data from the current selected year and the year prior.
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


        # --- Add this check to gracefully stop if data loading failed ---
        if pbp_data_for_stats.empty:
            st.warning("Could not retrieve the necessary play-by-play data for analysis. This can happen at the start of a season or if data is missing.")
            st.stop() # This stops the app from running further.

        # --- Stat Calculation ---
        with st.spinner('Calculating team stats...'):
            # Using our new tiered historical weighting system with recency boost
            recent_games_window = 8
            recent_form_weight = 0.30
            
            # Calculate stats with tiered historical weighting
            from stats_calculator import calculate_tiered_historical_stats
            
            # Apply our new tiered weighting system that incorporates:
            # 1. Year-based weighting (70% current year, 20% previous year, 10% two years ago)
            # 2. 30% recency weighting on last 8 games
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
                home_stats_w, away_stats_w, return_weights=True, 
                pbp_df=pbp_data_for_stats, home_team=home_abbr, away_team=away_abbr, game_info=game_info
            )
            model_home_spread = -model_result
            model_away_spread = -model_home_spread

        st.subheader("Prediction Engine")
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Vegas Line (Home Spread)", f"{home_spread_vegas:+.1f}")
            col2.metric("Model Spread (Home)", f"{model_home_spread:+.1f}")
            
            model_edge = home_spread_vegas - model_home_spread
            pick = home_abbr if model_edge > 0 else away_abbr
            
            # Calculate confidence rating for this edge
            try:
                from confidence_ratings import get_confidence_rating, get_confidence_text, get_recommendation
                stars, win_prob, samples = get_confidence_rating(abs(model_edge), use_2025_model=True)
                confidence_text = get_confidence_text(stars)
                recommendation = get_recommendation(abs(model_edge), win_prob)
                
                # Display edge with confidence info
                col3.metric("Model Edge", f"{abs(model_edge):.1f} pts on {pick}", 
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

        # --- Historical Backtest Display ---
        if CURRENT_YEAR < 2025:
            st.subheader("Historical Game Backtest")
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Vegas Line (Home)", f"{home_spread_vegas:+.1f}")
                col2.metric("Model's Line (Home)", f"{model_home_spread:+.1f}")
                
                model_edge = home_spread_vegas - model_home_spread
                pick = home_abbr if model_edge > 0 else away_abbr
                col3.metric("Model Edge", f"{abs(model_edge):.1f} pts on {pick}")

                actual_home_margin = game_details['result']
                final_score = f"Final: {game_details['away_score']} - {game_details['home_score']}"
                col4.metric("Actual Margin (Home)", f"{actual_home_margin:+.0f}", final_score)
                
                st.divider()

                if (actual_home_margin + home_spread_vegas) > 0:
                    covering_team = home_abbr
                elif (actual_home_margin + home_spread_vegas) < 0:
                    covering_team = away_abbr
                else:
                    covering_team = "Push"
                
                if covering_team == "Push":
                    st.warning(f"**PUSH.** The model identified value on **{pick}**, but the game landed on the number.")
                elif pick == covering_team:
                    st.success(f"**MODEL WIN.** The model identified value on **{pick}** and they **COVERED** the spread.")
                else:
                    st.error(f"**MODEL LOSS.** The model identified value on **{pick}**, but the **{covering_team}** covered the spread.")