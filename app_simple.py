"""
Simple NFL Prediction App

A clean, straightforward interface for EPA-based NFL spread predictions.
No overcomplicated features - just solid fundamentals.
"""

import streamlit as st
import pandas as pd
import nfl_data_py as nfl
from simple_model import SimpleNFLModel, calculate_edge_and_confidence
from dynamic_season_model import DynamicSeasonModel
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Simple NFL Prediction Model",
    page_icon="🏈",
    layout="wide"
)

# Initialize session state for model caching
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# App title
st.title("🏈 Simple NFL Prediction Model")
st.caption("Straightforward EPA-based spread predictions")

# Sidebar for settings
st.sidebar.header("Model Settings")

# Data loading section
st.sidebar.subheader("Data")

# Week selector for dynamic weighting
current_week = st.sidebar.selectbox(
    "Current Week (for dynamic weighting)",
    options=list(range(1, 19)),
    index=0,
    help="Determines how much historical vs current season data to use"
)

# Quick load for 2025 season with dynamic weighting
if st.sidebar.button("🚀 Load 2025 Season Data", type="primary"):
    with st.spinner(f"Loading dynamic data for Week {current_week} of 2025..."):
        try:
            st.session_state.model = DynamicSeasonModel()
            st.session_state.model.load_dynamic_data(2025, current_week)
            st.session_state.data_loaded = True
            st.session_state.current_week = current_week
            
            # Show weighting info
            years, weights = st.session_state.model.get_dynamic_years(2025, current_week)
            weight_info = ", ".join([f"{year}: {weight:.0%}" for year, weight in weights.items()])
            st.sidebar.success(f"✅ Data loaded with weights: {weight_info}")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error loading 2025 data: {e}")
            st.session_state.data_loaded = False

st.sidebar.markdown("---")
st.sidebar.markdown("**Manual Data Selection:**")

years_to_load = st.sidebar.multiselect(
    "Select years to include", 
    [2020, 2021, 2022, 2023, 2024, 2025], 
    default=[2023, 2024, 2025]
)

if st.sidebar.button("Load Manual Selection") or not st.session_state.data_loaded:
    if years_to_load:
        with st.spinner(f"Loading data for {years_to_load}..."):
            try:
                st.session_state.model = SimpleNFLModel()
                st.session_state.model.load_data(years_to_load)
                st.session_state.data_loaded = True
                st.sidebar.success(f"Data loaded for {years_to_load}")
            except Exception as e:
                st.sidebar.error(f"Error loading data: {e}")
                st.session_state.data_loaded = False
    else:
        st.sidebar.warning("Please select at least one year")

# Model parameters
st.sidebar.subheader("Model Parameters")
recent_games = st.sidebar.slider("Recent games window", 4, 12, 8)
recent_weight = st.sidebar.slider("Recent games weight", 0.1, 0.5, 0.3, 0.05)
hfa = st.sidebar.slider("Home field advantage", 1.0, 4.0, 2.5, 0.5)

# Update model parameters if they changed
if st.session_state.model is not None:
    st.session_state.model.recent_games_window = recent_games
    st.session_state.model.recent_weight = recent_weight
    st.session_state.model.home_field_advantage = hfa

# Main content
if not st.session_state.data_loaded:
    st.info("👆 Please load data using the sidebar to get started")
    st.stop()

# Show data status with dynamic weighting info
if st.session_state.data_loaded and hasattr(st.session_state.model, 'season_weights'):
    current_week = st.session_state.get('current_week', 1)
    years, weights = st.session_state.model.get_dynamic_years(2025, current_week)
    weight_display = " | ".join([f"{year}: {weight:.0%}" for year, weight in weights.items()])
    
    st.success(f"📊 Week {current_week} Dynamic Weighting: {weight_display}")
    
    with st.expander("📈 Weighting Strategy"):
        if current_week <= 3:
            st.info("**Historical Focus:** Using full 3-year data window with emphasis on complete seasons")
        elif current_week <= 7:
            st.info("**Transition Phase:** Gradually reducing 2022 weight as 2025 data builds")
        elif current_week <= 11:
            st.info("**Balanced Mix:** 2022 eliminated, balanced between 2023/2024/2025")
        else:
            st.info("**Current Focus:** Emphasizing recent seasons with growing 2025 weight")
        
        st.write("**Benefits:**")
        st.write("• Maintains adequate sample size throughout season")
        st.write("• Automatically adapts to data availability")
        st.write("• No manual intervention required")
elif st.session_state.data_loaded:
    # Fallback for non-dynamic models
    st.info("📊 Standard model loaded - using fixed historical data window")

st.divider()

# Team selection
st.header("Game Prediction")

# Get team list
try:
    teams_df = nfl.import_team_desc()
    teams_dict = dict(zip(teams_df['team_abbr'], teams_df['team_name']))
    team_options = sorted([(abbr, name) for abbr, name in teams_dict.items()], key=lambda x: x[1])
    team_display = [f"{name} ({abbr})" for abbr, name in team_options]
except:
    st.error("Unable to load team data")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    away_selection = st.selectbox("Away Team", team_display, key="away")
    away_abbr = away_selection.split("(")[1].strip(")")

with col2:
    home_selection = st.selectbox("Home Team", team_display, key="home")
    home_abbr = home_selection.split("(")[1].strip(")")

# Game info - use the week from session state if available
col1, col2 = st.columns(2)
with col1:
    current_season = st.number_input("Season", 2022, 2025, 2025)
with col2:
    # Use the week from the dynamic loading if available
    display_week = st.session_state.get('current_week', 1)
    current_week = st.number_input("Week", 1, 18, display_week)

# Vegas line input
vegas_spread = st.number_input(
    "Vegas spread (home team perspective, negative = home favored)", 
    -20.0, 20.0, 0.0, 0.5,
    help="Enter the current Vegas line from the home team's perspective"
)

# Make prediction
if st.button("Generate Prediction", type="primary"):
    if home_abbr == away_abbr:
        st.error("Please select different teams")
    else:
        try:
            with st.spinner("Calculating prediction..."):
                # Use dynamic prediction if available, otherwise standard
                if hasattr(st.session_state.model, 'predict_spread_dynamic'):
                    predicted_spread, details = st.session_state.model.predict_spread_dynamic(
                        home_abbr, away_abbr, current_week, current_season
                    )
                else:
                    predicted_spread, details = st.session_state.model.predict_spread(
                        home_abbr, away_abbr, current_week, current_season
                    )
                
                # Calculate edge
                edge, confidence = calculate_edge_and_confidence(predicted_spread, vegas_spread)
                
                # Display results
                st.divider()
                st.subheader("Prediction Results")
                
                # Main metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model Spread", f"{predicted_spread:+.1f}")
                
                with col2:
                    st.metric("Vegas Spread", f"{vegas_spread:+.1f}")
                
                with col3:
                    edge_color = "normal"
                    if edge >= 4:
                        edge_color = "inverse"
                    st.metric("Edge", f"{edge:.1f} pts", delta=None, delta_color=edge_color)
                
                # Confidence and recommendation
                st.markdown(f"**Confidence Level:** {confidence}")
                
                # Determine pick
                if edge < 2:
                    recommendation = "❌ **PASS** - Edge too small"
                    rec_color = "#ff6b6b"
                elif edge < 4:
                    recommendation = "⚠️ **SMALL BET** - Modest edge"
                    rec_color = "#ffd93d"
                else:
                    pick_team = home_abbr if predicted_spread < vegas_spread else away_abbr
                    pick_name = teams_dict[pick_team]
                    recommendation = f"✅ **BET {pick_name.upper()}** - Strong edge"
                    rec_color = "#6bcf7f"
                
                st.markdown(
                    f"<div style='padding: 10px; background-color: {rec_color}; border-radius: 5px; text-align: center; color: black; font-weight: bold;'>{recommendation}</div>",
                    unsafe_allow_html=True
                )
                
                # Model details
                with st.expander("📊 Model Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"{teams_dict[home_abbr]} (Home)")
                        home_stats = details['home_stats']
                        st.write(f"Net EPA/play: {home_stats.get('net_epa_per_play', 0):.3f}")
                        st.write(f"Offensive EPA/play: {home_stats.get('off_epa_per_play', 0):.3f}")
                        st.write(f"Defensive EPA/play: {home_stats.get('def_epa_per_play', 0):.3f}")
                    
                    with col2:
                        st.subheader(f"{teams_dict[away_abbr]} (Away)")
                        away_stats = details['away_stats']
                        st.write(f"Net EPA/play: {away_stats.get('net_epa_per_play', 0):.3f}")
                        st.write(f"Offensive EPA/play: {away_stats.get('off_epa_per_play', 0):.3f}")
                        st.write(f"Defensive EPA/play: {away_stats.get('def_epa_per_play', 0):.3f}")
                    
                    st.subheader("Calculation Breakdown")
                    st.write(f"EPA Advantage: {details['epa_advantage']:.3f} per play")
                    st.write(f"Raw Spread (no HFA): {details['predicted_spread_raw']:.1f}")
                    st.write(f"Home Field Advantage: +{details['home_field_advantage']:.1f}")
                    st.write(f"Final Prediction: {details['predicted_spread_final']:+.1f}")
                    
                    st.subheader("Model Settings Used")
                    st.write(f"Recent games window: {details['recent_games_window']}")
                    st.write(f"Recent games weight: {details['recent_weight_used']:.1%}")
                    
                    # Show dynamic weighting if available
                    if 'season_weights' in details and details['season_weights']:
                        st.subheader("Dynamic Season Weighting")
                        for year, weight in details['season_weights'].items():
                            st.write(f"{year}: {weight:.1%}")
                        st.caption(f"Weights automatically adjusted for Week {details.get('current_week', 'N/A')} of {details.get('current_season', 'N/A')}")
                
                # Visualization
                with st.expander("📈 Team Comparison"):
                    # Create comparison chart
                    metrics = ['off_epa_per_play', 'def_epa_per_play', 'net_epa_per_play']
                    metric_names = ['Offensive EPA/Play', 'Defensive EPA/Play', 'Net EPA/Play']
                    
                    home_values = [details['home_stats'].get(m, 0) for m in metrics]
                    away_values = [details['away_stats'].get(m, 0) for m in metrics]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name=teams_dict[home_abbr],
                        x=metric_names,
                        y=home_values,
                        marker_color='lightblue'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name=teams_dict[away_abbr],
                        x=metric_names,
                        y=away_values,
                        marker_color='lightcoral'
                    ))
                    
                    fig.update_layout(
                        title="Team EPA Comparison",
                        xaxis_title="Metric",
                        yaxis_title="EPA per Play",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error generating prediction: {e}")

# Information section
st.divider()
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    ### Simple EPA-Based Model
    
    This model predicts NFL spreads using a straightforward approach:
    
    **Core Methodology:**
    1. **EPA (Expected Points Added)** - Uses offensive and defensive EPA per play as the primary metric
    2. **Recency Weighting** - Emphasizes recent games over full season stats
    3. **Opponent Adjustment** - Basic strength of schedule consideration
    4. **Home Field Advantage** - Simple constant adjustment for playing at home
    
    **Key Features:**
    - Transparent calculations (all details shown)
    - Adjustable parameters for experimentation  
    - Simple confidence levels based on edge size
    - No overcomplicated variance models or theoretical adjustments
    
    **Confidence Levels:**
    - **Low** (< 2 pt edge): Pass
    - **Moderate** (2-4 pt edge): Small bet consideration
    - **High** (4-6 pt edge): Standard bet
    - **Very High** (6+ pt edge): Strong bet
    
    **Important Notes:**
    - This is for entertainment purposes only
    - NFL betting markets are highly efficient
    - Past performance doesn't guarantee future results
    - Always bet responsibly
    """)

st.caption("© 2025 Simple NFL Model - For entertainment purposes only")
