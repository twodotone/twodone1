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
from simple_model import SimpleNFLModel
from dynamic_season_model import DynamicSeasonModel

# --- Page & Sidebar Configuration ---
st.set_page_config(page_title="NFL Matchup Analyzer", layout="wide")
st.title('🏈 NFL Matchup Analyzer')
st.sidebar.header('Season Settings')
CURRENT_YEAR = st.sidebar.selectbox('Year', [2025, 2024, 2023, 2022], index=0)
CURRENT_WEEK = st.sidebar.number_input('Week', min_value=1, max_value=18, value=1, step=1)

st.sidebar.header('Model Settings')
use_sos_adjustment = st.sidebar.checkbox('Apply Strength of Schedule Adjustment', value=True)

# Model comparison settings
st.sidebar.subheader('Model Comparison')
show_simple_model = st.sidebar.checkbox('Show Simple Model', value=True, 
                                       help="Shows transparent EPA-based model alongside standard model")
simple_model_type = st.sidebar.selectbox(
    'Simple Model Type',
    ['Fixed Window', 'Dynamic Season'],
    help="Fixed: 3-year window. Dynamic: Adjusts weights based on season progress"
)

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
            away_moneyline = game_details.get('away_moneyline', 0)
            home_moneyline = game_details.get('home_moneyline', 0)
            
            # Format moneylines
            away_ml_str = f"+{int(away_moneyline)}" if away_moneyline > 0 else f"{int(away_moneyline)}"
            home_ml_str = f"+{int(home_moneyline)}" if home_moneyline > 0 else f"{int(home_moneyline)}"
            
            col1.image(away_logo, width=70)
            col1.markdown(f"<p style='text-align: center; margin: 0; font-weight: bold; color: #1f77b4;'>{away_ml_str}</p>", unsafe_allow_html=True)
            col2.metric("Away Spread", f"{away_spread_vegas:+.1f}")
            col3.metric("Over/Under", f"{total_line:.1f}")
            col4.metric("Home Spread", f"{home_spread_vegas:+.1f}")
            col5.image(home_logo, width=70)
            col5.markdown(f"<p style='text-align: center; margin: 0; font-weight: bold; color: #1f77b4;'>{home_ml_str}</p>", unsafe_allow_html=True)

        # --- Data Prep using Rolling Window ---
        st.info("Using a rolling data window including the previous season for statistical analysis.")

        # Load data from the current selected year and the year prior.
        combined_pbp_data = load_rolling_data(CURRENT_YEAR)

        # Initialize simple models if requested
        simple_model = None
        simple_model_spread = None
        simple_model_details = None
        simple_model_total = None
        simple_total_details = None
        total_edge = 0
        total_pick = "N/A"
        
        if show_simple_model:
            with st.spinner("Loading simple model..."):
                try:
                    if simple_model_type == 'Dynamic Season':
                        simple_model = DynamicSeasonModel()
                        simple_model.load_dynamic_data(CURRENT_YEAR, CURRENT_WEEK)
                    else:
                        simple_model = SimpleNFLModel()
                        # Use 3-year window: current year-2, current year-1, and current year if available
                        years_to_load = [CURRENT_YEAR-2, CURRENT_YEAR-1]
                        if CURRENT_YEAR <= 2024:  # For historical years, include current year
                            years_to_load.append(CURRENT_YEAR)
                        simple_model.load_data(years_to_load)
                        
                    # Generate simple model prediction
                    if simple_model_type == 'Dynamic Season':
                        simple_model_spread, simple_model_details = simple_model.predict_spread_dynamic(
                            home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR
                        )
                    else:
                        simple_model_spread, simple_model_details = simple_model.predict_spread(
                            home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR
                        )
                    
                    # Generate total prediction
                    try:
                        simple_model_total, simple_total_details = simple_model.predict_total(
                            home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR
                        )
                    except Exception as e:
                        simple_model_total = None
                        simple_total_details = None
                        st.warning(f"Total prediction failed: {e}")
                        
                except Exception as e:
                    st.warning(f"Simple model failed to load: {e}")
                    show_simple_model = False
                    simple_model_total = None

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
        
        if show_simple_model and simple_model_spread is not None:
            # Two-model comparison layout
            st.subheader("📊 Model Comparison")
            
            # Create tabs for easy comparison
            tab1, tab2 = st.tabs(["📈 Model Overview", "🔍 Detailed Analysis"])
            
            with tab1:
                # Main comparison metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Vegas Line", f"{home_abbr} {home_spread_vegas:+.1f}")
                    st.metric("Vegas Total", f"{total_line:.1f}")
                
                with col2:
                    standard_edge = home_spread_vegas - model_home_spread
                    standard_pick = home_abbr if standard_edge > 0 else away_abbr
                    standard_edge_strength = "STRONG" if abs(standard_edge) >= 3 else "WEAK"
                    
                    # Color coding for edge strength
                    if abs(standard_edge) >= 3:
                        edge_color = "🔥"
                        edge_text = f"**{standard_pick}** {edge_color}"
                    elif abs(standard_edge) >= 1:
                        edge_color = "💡"
                        edge_text = f"**{standard_pick}** {edge_color}"
                    else:
                        edge_text = f"{standard_pick}"
                    
                    st.metric("Standard Model", f"{home_abbr} {model_home_spread:+.1f}", 
                             delta=f"Edge: {edge_text}")
                    
                with col3:
                    simple_edge = home_spread_vegas - simple_model_spread
                    simple_pick = home_abbr if simple_edge > 0 else away_abbr
                    
                    # Color coding for edge strength
                    if abs(simple_edge) >= 3:
                        edge_color = "🔥"
                        edge_text = f"**{simple_pick}** {edge_color}"
                    elif abs(simple_edge) >= 1:
                        edge_color = "💡" 
                        edge_text = f"**{simple_pick}** {edge_color}"
                    else:
                        edge_text = f"{simple_pick}"
                    
                    st.metric("Simple Model", f"{home_abbr} {simple_model_spread:+.1f}",
                             delta=f"Edge: {edge_text}")
                    
                with col4:
                    if simple_model_total is not None:
                        total_edge = simple_model_total - total_line
                        total_pick = "OVER" if total_edge > 0 else "UNDER"
                        
                        # Color coding for total edge
                        if abs(total_edge) >= 3:
                            total_color = "🔥"
                            total_text = f"**{total_pick}** {total_color}"
                        elif abs(total_edge) >= 1:
                            total_color = "💡"
                            total_text = f"**{total_pick}** {total_color}"
                        else:
                            total_text = f"{total_pick}"
                            
                        st.metric("Model Total", f"{simple_model_total:.1f}",
                                 delta=f"Edge: {total_text}")
                    else:
                        st.metric("Model Total", "N/A")
                    
                with col5:
                    model_difference = abs(model_home_spread - simple_model_spread)
                    st.metric("Model Difference", f"{model_difference:.1f} pts")
                    
                # Agreement/Disagreement indicator with Recommendations
                st.divider()
                
                # Determine consensus and edge strength
                models_agree_on_pick = (standard_pick == simple_pick)
                model_difference = abs(model_home_spread - simple_model_spread)
                
                # Primary agreement logic: both team pick AND spread difference
                models_closely_aligned = (models_agree_on_pick and model_difference <= 2)
                
                if models_closely_aligned:
                    consensus_team = standard_pick
                    avg_edge = (abs(standard_edge) + abs(simple_edge)) / 2
                    
                    if avg_edge >= 3:
                        rec_color = "#ff4757"  # Red for strong edge
                        rec_icon = "🔥🔥"
                        rec_strength = "STRONG EDGE"
                    elif avg_edge >= 1.5:
                        rec_color = "#ffa502"  # Orange for moderate edge
                        rec_icon = "🔥"
                        rec_strength = "MODERATE EDGE"
                    else:
                        rec_color = "#70a1ff"  # Blue for weak edge
                        rec_icon = "�"
                        rec_strength = "WEAK EDGE"
                        
                    st.markdown(
                        f"""
                        <div style="background-color: {rec_color}; padding: 20px; border-radius: 15px; margin: 15px 0; border: 3px solid #2f3542;">
                            <h2 style="text-align: center; margin: 0; color: white;">{rec_icon} {rec_strength} {rec_icon}</h2>
                            <h1 style="text-align: center; margin: 10px 0; color: white; font-size: 2.5em;">TAKE {consensus_team}</h1>
                            <p style="text-align: center; margin: 5px 0; color: white; font-size: 1.2em;">
                                Both models agree • Average edge: {avg_edge:.1f} points • Spread diff: {model_difference:.1f} pts
                            </p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                elif models_agree_on_pick and model_difference > 2:
                    # Same pick but different spreads - moderate confidence
                    consensus_team = standard_pick
                    avg_edge = (abs(standard_edge) + abs(simple_edge)) / 2
                    
                    st.markdown(
                        f"""
                        <div style="background-color: #ffd56b; padding: 20px; border-radius: 15px; margin: 15px 0; border: 3px solid #2f3542;">
                            <h2 style="text-align: center; margin: 0; color: black;">🟡 SAME PICK, DIFFERENT SPREADS</h2>
                            <h2 style="text-align: center; margin: 10px 0; color: black; font-size: 2em;">LEAN {consensus_team}</h2>
                            <p style="text-align: center; margin: 5px 0; color: black; font-size: 1.1em;">
                                Same team pick • Average edge: {avg_edge:.1f} points • But {model_difference:.1f} pts spread difference
                            </p>
                            <p style="text-align: center; margin: 5px 0; color: black; font-weight: bold;">
                                Consider smaller bet size due to spread uncertainty
                            </p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    # Models disagree completely
                    st.markdown(
                        f"""
                        <div style="background-color: #ff6b6b; padding: 20px; border-radius: 15px; margin: 15px 0; border: 3px solid #2f3542;">
                            <h2 style="text-align: center; margin: 0; color: white;">⚠️ MODELS DISAGREE ⚠️</h2>
                            <div style="display: flex; justify-content: space-around; margin: 15px 0;">
                                <div style="text-align: center; background-color: white; padding: 10px; border-radius: 10px;">
                                    <strong>Standard Model:</strong><br>
                                    <span style="font-size: 1.5em; color: #2f3542;">{standard_pick}</span><br>
                                    <span style="color: #57606f;">{abs(standard_edge):.1f} pt edge</span>
                                </div>
                                <div style="text-align: center; background-color: white; padding: 10px; border-radius: 10px;">
                                    <strong>Simple Model:</strong><br>
                                    <span style="font-size: 1.5em; color: #2f3542;">{simple_pick}</span><br>
                                    <span style="color: #57606f;">{abs(simple_edge):.1f} pt edge</span>
                                </div>
                            </div>
                            <p style="text-align: center; margin: 5px 0; color: white; font-weight: bold;">
                                Avoid this game or use very small bet size
                            </p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Total Points Recommendation
                if simple_model_total is not None:
                    st.divider()
                    
                    if abs(total_edge) >= 2:
                        if abs(total_edge) >= 4:
                            total_rec_color = "#ff4757"  # Red for strong edge
                            total_rec_icon = "🔥🔥"
                            total_rec_strength = "STRONG TOTAL EDGE"
                        else:
                            total_rec_color = "#ffa502"  # Orange for moderate edge
                            total_rec_icon = "🔥"
                            total_rec_strength = "MODERATE TOTAL EDGE"
                            
                        st.markdown(
                            f"""
                            <div style="background-color: {total_rec_color}; padding: 15px; border-radius: 15px; margin: 10px 0; border: 2px solid #2f3542;">
                                <h3 style="text-align: center; margin: 0; color: white;">{total_rec_icon} {total_rec_strength} {total_rec_icon}</h3>
                                <h2 style="text-align: center; margin: 10px 0; color: white; font-size: 2em;">TAKE THE {total_pick}</h2>
                                <p style="text-align: center; margin: 5px 0; color: white; font-size: 1.1em;">
                                    Model: {simple_model_total:.1f} • Vegas: {total_line:.1f} • Edge: {abs(total_edge):.1f} pts
                                </p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                
            with tab2:
                # Detailed model breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏗️ Standard Model Details")
                    st.write(f"**Method:** SOS-adjusted EPA with dynamic HFA")
                    st.write(f"**Home Spread:** {model_home_spread:+.1f}")
                    st.write(f"**Edge vs Vegas:** {abs(standard_edge):.1f} pts on {standard_pick}")
                    st.write(f"**HFA Used:** {hfa_value:.1f} points")
                    if hfa_components:
                        st.write("**HFA Breakdown:**")
                        for component, value in hfa_components.items():
                            if isinstance(value, (int, float)):
                                st.write(f"  • {component}: {value:.2f}")
                            else:
                                st.write(f"  • {component}: {value}")
                    
                with col2:
                    st.subheader("⚡ Simple Model Details")
                    model_name = "Dynamic Season" if simple_model_type == 'Dynamic Season' else "Fixed Window"
                    st.write(f"**Method:** {model_name} EPA (no SOS)")
                    st.write(f"**Home Spread:** {simple_model_spread:+.1f}")
                    st.write(f"**Edge vs Vegas:** {abs(simple_edge):.1f} pts on {simple_pick}")
                    st.write(f"**{home_abbr} EPA:** {simple_model_details['home_stats']['net_epa_per_play']:.3f}")
                    st.write(f"**{away_abbr} EPA:** {simple_model_details['away_stats']['net_epa_per_play']:.3f}")
                    st.write(f"**EPA Advantage:** {simple_model_details['epa_advantage']:.3f}")
                    st.write(f"**HFA Used:** {simple_model_details['home_field_advantage']:.1f} points")
                    
                    # Total prediction details
                    if simple_model_total is not None and simple_total_details is not None:
                        st.write("---")
                        st.write("**Total Points Prediction:**")
                        st.write(f"**Model Total:** {simple_model_total:.1f}")
                        st.write(f"**Vegas Total:** {total_line:.1f}")
                        st.write(f"**Edge:** {abs(total_edge):.1f} pts ({total_pick})")
                        st.write(f"**{home_abbr} Expected:** {simple_total_details['home_expected_points']:.1f}")
                        st.write(f"**{away_abbr} Expected:** {simple_total_details['away_expected_points']:.1f}")
                    
                    if 'season_weights' in simple_model_details:
                        st.write("**Season Weights:**")
                        for year, weight in simple_model_details['season_weights'].items():
                            st.write(f"  • {year}: {weight:.1%}")
                            
        else:
            # Standard single-model layout
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
                    
        # Common confidence analysis section (for both single and dual model views)
        if show_simple_model and simple_model_spread is not None:
            # Dual model confidence analysis
            st.subheader("📊 Confidence Analysis")
            
            # Use model agreement/disagreement for confidence
            model_difference = abs(model_home_spread - simple_model_spread)
            
            if model_difference <= 2:
                confidence_level = "HIGH"
                confidence_desc = "Both models closely agree - high confidence in prediction"
                confidence_color = "#2ed573"
            elif model_difference <= 5:
                confidence_level = "MODERATE" 
                confidence_desc = "Models show some disagreement - moderate confidence"
                confidence_color = "#ffd56b"
            else:
                confidence_level = "LOW"
                confidence_desc = "Models significantly disagree - low confidence, proceed with caution"
                confidence_color = "#ff6b6b"
                
            st.markdown(
                f"""
                <div style="background-color: {confidence_color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="text-align: center; margin: 0; color: black;">CONFIDENCE LEVEL: {confidence_level}</h3>
                    <p style="text-align: center; margin: 5px 0; color: black; font-weight: bold;">{confidence_desc}</p>
                    <p style="text-align: center; margin: 0; color: black;">Model difference: {model_difference:.1f} points</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Display additional model parameters section for single model view
        if not show_simple_model or simple_model_spread is None:
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