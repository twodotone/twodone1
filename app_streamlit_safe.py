# app_streamlit_safe.py - Safe production version with error handling

import streamlit as st
import pandas as pd
import os
import traceback

# Configure page first
st.set_page_config(page_title="NFL Matchup Analyzer", layout="wide")

# Safe import function
def safe_import(module_name, description):
    try:
        if module_name == "streamlit_data_loader":
            from streamlit_data_loader import StreamlitDataLoader, check_data_freshness
            return StreamlitDataLoader, check_data_freshness
        elif module_name == "streamlit_simple_model":
            from streamlit_simple_model import StreamlitSimpleNFLModel
            return StreamlitSimpleNFLModel
        elif module_name == "streamlit_real_standard_model":
            from streamlit_real_standard_model import StreamlitRealStandardModel
            return StreamlitRealStandardModel
        else:
            return None
    except Exception as e:
        st.error(f"Failed to import {description}: {str(e)}")
        st.exception(e)
        return None

# Title
st.title('🏈 NFL Matchup Analyzer - Safe Mode')

# Safe imports with error handling
st.write("🔄 Loading components...")

# Import data loader
data_loader_result = safe_import("streamlit_data_loader", "Data Loader")
if data_loader_result:
    StreamlitDataLoader, check_data_freshness = data_loader_result
    st.success("✅ Data loader imported")
else:
    st.stop()

# Import simple model
SimpleModel = safe_import("streamlit_simple_model", "Simple Model")
if SimpleModel:
    st.success("✅ Simple model imported")
else:
    st.error("❌ Simple model failed to import")

# Import standard model
StandardModel = safe_import("streamlit_real_standard_model", "Standard Model")
if StandardModel:
    st.success("✅ Standard model imported")
else:
    st.warning("⚠️ Standard model failed to import")

# Sidebar settings
st.sidebar.header('⚙️ Settings')
CURRENT_YEAR = st.sidebar.selectbox('Year', [2025, 2024, 2023, 2022], index=0)
CURRENT_WEEK = st.sidebar.number_input('Week', min_value=1, max_value=18, value=1, step=1)

show_simple_model = st.sidebar.checkbox('Show Simple Model', value=True)
show_standard_model = st.sidebar.checkbox('Show Standard Model', value=StandardModel is not None)

# Data loading with safety
@st.cache_data
def safe_load_data():
    """Safely load data with error handling"""
    try:
        loader = StreamlitDataLoader()
        
        # Check if data directory exists
        if not os.path.exists("data"):
            st.error("Data directory not found")
            return None, None
        
        # Load team data
        team_desc = loader.load_team_data()
        st.success(f"✅ Loaded {len(team_desc)} teams")
        
        # Load schedule data
        schedule_data = loader.load_schedule_data([CURRENT_YEAR])
        st.success(f"✅ Loaded schedule for {CURRENT_YEAR}")
        
        return team_desc, schedule_data
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.exception(e)
        return None, None

@st.cache_data
def safe_load_simple_model():
    """Safely load simple model"""
    try:
        if not SimpleModel:
            return None
            
        model = SimpleModel(data_dir="data")
        model.load_data_from_parquet([2022, 2023, 2024])
        st.success("✅ Simple model loaded")
        return model
    except Exception as e:
        st.error(f"Simple model loading failed: {str(e)}")
        st.exception(e)
        return None

@st.cache_data  
def safe_load_standard_model(current_year, current_week):
    """Safely load standard model"""
    try:
        if not StandardModel:
            return None
            
        model = StandardModel(data_dir="data")
        model.load_standard_data(current_year, current_week)
        st.success("✅ Standard model loaded")
        return model
    except Exception as e:
        st.error(f"Standard model loading failed: {str(e)}")
        st.exception(e)
        return None

# Load data safely
with st.spinner("Loading data..."):
    team_desc, schedule_data = safe_load_data()

if team_desc is None or schedule_data is None:
    st.error("Cannot proceed without data")
    st.stop()

# Load models safely
simple_model = None
standard_model = None

if show_simple_model:
    with st.spinner("Loading Simple Model..."):
        simple_model = safe_load_simple_model()

if show_standard_model and StandardModel:
    with st.spinner("Loading Standard Model..."):
        standard_model = safe_load_standard_model(CURRENT_YEAR, CURRENT_WEEK)

# Game selection
st.header(f'Week {CURRENT_WEEK} Matchups for the {CURRENT_YEAR} Season')
week_schedule = schedule_data[schedule_data['week'] == CURRENT_WEEK].copy()

if week_schedule.empty:
    st.warning(f"No schedule found for Week {CURRENT_WEEK} of the {CURRENT_YEAR} season.")
    st.stop()

week_schedule['game_description'] = week_schedule['away_team'] + ' @ ' + week_schedule['home_team']
selected_game_str = st.selectbox('Select a Game:', week_schedule['game_description'].tolist())

if selected_game_str:
    game_details = week_schedule[week_schedule['game_description'] == selected_game_str].iloc[0]
    away_abbr, home_abbr = selected_game_str.split(' @ ')
    
    # Display game info
    st.subheader("🎰 Game Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Away Team:** {away_abbr}")
    with col2:
        st.write(f"**Home Team:** {home_abbr}")
    with col3:
        st.write(f"**Week:** {CURRENT_WEEK}")
    
    # Safe prediction generation
    st.header("🤖 Model Predictions")
    
    # Simple Model Predictions
    if simple_model and show_simple_model:
        try:
            with st.spinner("Generating Simple Model prediction..."):
                simple_spread, simple_details = simple_model.predict_spread(
                    home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR
                )
                
                st.success(f"✅ Simple Model: {home_abbr} {simple_spread:+.1f}")
                
                # Show details
                with st.expander("Simple Model Details"):
                    st.write(f"**Home EPA:** {simple_details['home_stats']['net_epa_per_play']:.3f}")
                    st.write(f"**Away EPA:** {simple_details['away_stats']['net_epa_per_play']:.3f}")
                    st.write(f"**EPA Advantage:** {simple_details['epa_advantage']:.3f}")
                    st.write(f"**Home Field Advantage:** {simple_details['home_field_advantage']:.1f}")
                
                # Try total prediction
                try:
                    simple_total, total_details = simple_model.predict_total(
                        home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR
                    )
                    st.success(f"✅ Total Prediction: {simple_total:.1f}")
                except Exception as e:
                    st.warning(f"Total prediction failed: {str(e)}")
                
        except Exception as e:
            st.error(f"Simple model prediction failed: {str(e)}")
            st.exception(e)
    
    # Standard Model Predictions
    if standard_model and show_standard_model:
        try:
            with st.spinner("Generating Standard Model prediction..."):
                standard_spread, standard_details = standard_model.predict_spread_standard(
                    home_abbr, away_abbr, CURRENT_WEEK, CURRENT_YEAR
                )
                
                st.success(f"✅ Standard Model: {home_abbr} {standard_spread:+.1f}")
                
                # Show details
                with st.expander("Standard Model Details"):
                    if standard_details:
                        for key, value in standard_details.items():
                            if isinstance(value, (int, float)):
                                st.write(f"**{key}:** {value:.3f}")
                            else:
                                st.write(f"**{key}:** {value}")
                
        except Exception as e:
            st.error(f"Standard model prediction failed: {str(e)}")
            st.exception(e)
    
    # Summary
    st.write("---")
    st.write("**Status Summary:**")
    st.write(f"✅ Data loaded successfully")
    st.write(f"{'✅' if simple_model else '❌'} Simple model: {'Ready' if simple_model else 'Failed'}")
    st.write(f"{'✅' if standard_model else '❌'} Standard model: {'Ready' if standard_model else 'Failed'}")

st.write("---")
st.caption("Safe mode version - enhanced error handling and graceful degradation")
