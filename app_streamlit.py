# app_streamlit.py - Production Streamlit App

import streamlit as st
import pandas as pd
import os
from streamlit_data_loader import StreamlitDataLoader, check_data_freshness
from streamlit_simple_model import StreamlitSimpleNFLModel
from streamlit_real_standard_model import StreamlitRealStandardModel
from data_loader import load_rolling_data
from stats_calculator import (
    get_last_n_games_pbp,
    calculate_granular_epa_stats,
    calculate_weighted_stats,
    generate_stable_matchup_line
)

# --- Page & Sidebar Configuration ---
st.set_page_config(page_title="NFL Matchup Analyzer", layout="wide")
st.title('🏈 NFL Matchup Analyzer - Production')

# --- Data Freshness Check ---
with st.sidebar:
    st.header('📊 Data Status')
    data_freshness = check_data_freshness()
    
    for file_name, last_updated in data_freshness.items():
        if last_updated == "Missing":
            st.error(f"❌ {file_name}: Missing")
        else:
            st.success(f"✅ {file_name}")
            st.caption(f"Updated: {last_updated.strftime('%Y-%m-%d %H:%M')}")

# --- Settings ---
st.sidebar.header('⚙️ Settings')
CURRENT_YEAR = st.sidebar.selectbox('Year', [2025, 2024, 2023, 2022], index=0)
CURRENT_WEEK = st.sidebar.number_input('Week', min_value=1, max_value=18, value=1, step=1)

st.sidebar.header('🔧 Model Settings')
show_standard_model = st.sidebar.checkbox('Show Standard Model', value=True, 
                                         help="Shows complex SOS-adjusted model")
show_simple_model = st.sidebar.checkbox('Show Simple Model', value=True, 
                                       help="Shows transparent EPA-based model")

# --- Data Loading ---
@st.cache_data
def load_data():
    """Load all necessary data with caching"""
    loader = StreamlitDataLoader()
    
    try:
        # Load team data
        team_desc = loader.load_team_data()
        
        # Load schedule data
        schedule_data = loader.load_schedule_data([CURRENT_YEAR])
        
        return team_desc, schedule_data
        
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

@st.cache_data
def load_simple_model():
    """Load and cache the simple model"""
    simple_model = StreamlitSimpleNFLModel(data_dir="data")
    simple_model.load_data_from_parquet([2022, 2023, 2024])
    return simple_model

@st.cache_data  
def load_standard_model(current_year, current_week):
    """Load and cache the real standard model"""
    standard_model = StreamlitRealStandardModel(data_dir="data")
    standard_model.load_standard_data(current_year, current_week)
    return standard_model

team_desc, schedule_data = load_data()

# --- Main Page: Matchup Selection ---
st.header(f'Week {CURRENT_WEEK} Matchups for the {CURRENT_YEAR} Season')
week_schedule = schedule_data[schedule_data['week'] == CURRENT_WEEK].copy()

if week_schedule.empty:
    st.warning(f"No schedule found for Week {CURRENT_WEEK} of the {CURRENT_YEAR} season.")
    st.stop()

week_schedule['game_description'] = week_schedule['away_team'] + ' @ ' + week_schedule['home_team']
selected_game_str = st.selectbox('Select a Game:', week_schedule['game_description'].tolist())

if selected_game_str:
    with st.spinner("Loading game analysis..."):
        game_details = week_schedule[week_schedule['game_description'] == selected_game_str].iloc[0]
        away_abbr, home_abbr = selected_game_str.split(' @ ')
        
        # Display Betting Odds Banner
        st.subheader("🎰 Betting Odds & Game Info")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Get team logos
        away_logo = team_desc.loc[team_desc['team_abbr'] == away_abbr, 'team_logo_espn'].values[0] if len(team_desc.loc[team_desc['team_abbr'] == away_abbr]) > 0 else ""
        home_logo = team_desc.loc[team_desc['team_abbr'] == home_abbr, 'team_logo_espn'].values[0] if len(team_desc.loc[team_desc['team_abbr'] == home_abbr]) > 0 else ""
        
        # Determine spread display based on betting convention
        spread_magnitude = abs(game_details.get('spread_line', 0))
        home_moneyline = game_details.get('home_moneyline', 0)
        
        if home_moneyline < 0:  # Home team favored
            home_spread_vegas = -spread_magnitude
            away_spread_vegas = spread_magnitude
        elif home_moneyline > 0:  # Away team favored
            home_spread_vegas = spread_magnitude
            away_spread_vegas = -spread_magnitude
        else:
            home_spread_vegas = game_details.get('spread_line', 0)
            away_spread_vegas = -home_spread_vegas
        
        total_line = game_details.get('total_line', 0)
        away_moneyline = game_details.get('away_moneyline', 0)
        
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

        # --- Model Predictions ---
        st.header("🤖 Model Predictions")
        
        # Initialize model prediction variables
        simple_model_spread = None
        simple_model_details = None
        simple_model_total = None
        simple_total_details = None
        model_home_spread = None
        dynamic_model_details = None
        total_edge = 0
        total_pick = "N/A"
        
        # Load Simple Model
        if show_simple_model:
            with st.spinner("Loading Simple Model..."):
                try:
                    simple_model = load_simple_model()
                    
                    # Generate predictions
                    simple_model_spread, simple_model_details = simple_model.predict_spread(
                        home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR
                    )
                    
                    # Generate total prediction
                    try:
                        simple_model_total, simple_total_details = simple_model.predict_total(
                            home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR
                        )
                    except Exception as e:
                        st.warning(f"Total prediction failed: {e}")
                        
                except Exception as e:
                    st.error(f"Simple model failed to load: {e}")
                    show_simple_model = False

        # Load Standard Model (Real Standard Model with tiered historical stats)
        if show_standard_model:
            with st.spinner("Loading Standard Model..."):
                try:
                    # Initialize the Real Standard Model
                    standard_model = load_standard_model(CURRENT_YEAR, CURRENT_WEEK)
                    
                    # Generate predictions
                    model_home_spread, dynamic_model_details = standard_model.predict_spread_standard(
                        home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR
                    )
                    
                except Exception as e:
                    st.warning(f"Standard model failed, using simplified approach: {e}")
                    model_home_spread = simple_model_spread + 0.5 if simple_model_spread else 0
                    dynamic_model_details = None

        # --- Display Results ---
        if show_simple_model and simple_model_spread is not None:
            st.subheader("📊 Model Analysis")
            
            # Create tabs for organized display
            tab1, tab2 = st.tabs(["📈 Predictions", "🔍 Model Details"])
            
            with tab1:
                # Main metrics
                if show_standard_model:
                    col1, col2, col3, col4, col5 = st.columns(5)
                else:
                    col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Vegas Line", f"{home_abbr} {home_spread_vegas:+.1f}")
                    st.metric("Vegas Total", f"{total_line:.1f}")
                
                with col2:
                    if show_standard_model and model_home_spread is not None:
                        standard_edge = home_spread_vegas - model_home_spread
                        standard_pick = home_abbr if standard_edge > 0 else away_abbr
                        
                        if abs(standard_edge) >= 3:
                            edge_text = f"**{standard_pick}** 🔥"
                        elif abs(standard_edge) >= 1:
                            edge_text = f"**{standard_pick}** 💡"
                        else:
                            edge_text = f"{standard_pick}"
                        
                        st.metric("Standard Model", f"{home_abbr} {model_home_spread:+.1f}", 
                                 delta=f"Edge: {edge_text}")
                
                with col3:
                    simple_edge = home_spread_vegas - simple_model_spread
                    simple_pick = home_abbr if simple_edge > 0 else away_abbr
                    
                    if abs(simple_edge) >= 3:
                        edge_text = f"**{simple_pick}** 🔥"
                    elif abs(simple_edge) >= 1:
                        edge_text = f"**{simple_pick}** 💡"
                    else:
                        edge_text = f"{simple_pick}"
                    
                    st.metric("Simple Model", f"{home_abbr} {simple_model_spread:+.1f}",
                             delta=f"Edge: {edge_text}")
                
                if show_simple_model and len(st.columns(5)) > 3:
                    with st.columns(5)[3]:
                        if simple_model_total is not None:
                            total_edge = simple_model_total - total_line
                            total_pick = "OVER" if total_edge > 0 else "UNDER"
                            
                            if abs(total_edge) >= 3:
                                total_text = f"**{total_pick}** 🔥"
                            elif abs(total_edge) >= 1:
                                total_text = f"**{total_pick}** 💡"
                            else:
                                total_text = f"{total_pick}"
                                
                            st.metric("Model Total", f"{simple_model_total:.1f}",
                                     delta=f"Edge: {total_text}")
                        else:
                            st.metric("Model Total", "N/A")
                
                # Recommendations
                st.divider()
                
                # Spread recommendation
                if abs(simple_edge) >= 2:
                    if abs(simple_edge) >= 4:
                        rec_color = "#ff4757"
                        rec_icon = "🔥🔥"
                        rec_strength = "STRONG EDGE"
                    else:
                        rec_color = "#ffa502"
                        rec_icon = "🔥"
                        rec_strength = "MODERATE EDGE"
                        
                    st.markdown(
                        f"""
                        <div style="background-color: {rec_color}; padding: 20px; border-radius: 15px; margin: 15px 0; border: 3px solid #2f3542;">
                            <h2 style="text-align: center; margin: 0; color: white;">{rec_icon} {rec_strength} {rec_icon}</h2>
                            <h1 style="text-align: center; margin: 10px 0; color: white; font-size: 2.5em;">TAKE {simple_pick}</h1>
                            <p style="text-align: center; margin: 5px 0; color: white; font-size: 1.2em;">
                                Model edge: {abs(simple_edge):.1f} points
                            </p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Total recommendation
                if simple_model_total is not None and abs(total_edge) >= 2:
                    if abs(total_edge) >= 4:
                        total_rec_color = "#ff4757"
                        total_rec_icon = "🔥🔥"
                        total_rec_strength = "STRONG TOTAL EDGE"
                    else:
                        total_rec_color = "#ffa502"
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
                # Model details
                if show_simple_model and simple_model_details:
                    st.subheader("⚡ Simple Model Details")
                    st.write(f"**Method:** Pure EPA Analysis")
                    st.write(f"**Home Spread:** {simple_model_spread:+.1f}")
                    st.write(f"**Edge vs Vegas:** {abs(simple_edge):.1f} pts on {simple_pick}")
                    st.write(f"**{home_abbr} EPA:** {simple_model_details['home_stats']['net_epa_per_play']:.3f}")
                    st.write(f"**{away_abbr} EPA:** {simple_model_details['away_stats']['net_epa_per_play']:.3f}")
                    st.write(f"**EPA Advantage:** {simple_model_details['epa_advantage']:.3f}")
                    st.write(f"**HFA Used:** {simple_model_details['home_field_advantage']:.1f} points")
                    
                    if simple_model_total is not None and simple_total_details is not None:
                        st.write("---")
                        st.write("**Total Points Prediction:**")
                        st.write(f"**Model Total:** {simple_model_total:.1f}")
                        st.write(f"**Vegas Total:** {total_line:.1f}")
                        st.write(f"**Edge:** {abs(total_edge):.1f} pts ({total_pick})")
                        st.write(f"**{home_abbr} Expected:** {simple_total_details['home_expected_points']:.1f}")
                        st.write(f"**{away_abbr} Expected:** {simple_total_details['away_expected_points']:.1f}")
                
                if show_standard_model and dynamic_model_details:
                    st.write("---")
                    st.subheader("🎯 Standard Model Details")
                    st.write(f"**Method:** {dynamic_model_details.get('method', 'Tiered Historical Stats')}")
                    st.write(f"**Home Spread:** {model_home_spread:+.1f}")
                    if 'recent_games_window' in dynamic_model_details:
                        st.write(f"**Recent Games Window:** {dynamic_model_details['recent_games_window']}")
                        st.write(f"**Recent Form Weight:** {dynamic_model_details['recent_form_weight']:.1%}")
                    if 'hfa_value' in dynamic_model_details:
                        st.write(f"**Home Field Advantage:** {dynamic_model_details['hfa_value']:.1f} points")
                    if 'model_weights' in dynamic_model_details:
                        st.write(f"**Model Components:** EPA, Recent Form, SOS, HFA")
                    st.write(f"**Raw Model Result:** {dynamic_model_details.get('model_result_raw', 'N/A'):.1f}")
                    st.write(f"**Final Prediction:** {dynamic_model_details.get('predicted_spread', model_home_spread):+.1f}")

        else:
            st.warning("No model predictions available. Please check data files.")

# --- Footer ---
st.divider()
st.markdown("### 📈 About the Models")
st.markdown("""
- **Simple Model**: Pure EPA-based predictions using team offensive/defensive efficiency
- **Standard Model**: Complex model with strength-of-schedule adjustments and dynamic factors
- **Total Predictions**: Expected points based on offensive vs defensive EPA matchups
- **Edge Analysis**: Difference between our model and Vegas lines to identify betting opportunities
""")

st.caption("Data updated daily. Model predictions for entertainment purposes only.")
