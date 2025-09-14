# debug_team_selection.py - Test team changes to identify crash cause

import streamlit as st
import pandas as pd
from streamlit_simple_model import StreamlitSimpleNFLModel
from streamlit_data_loader import StreamlitDataLoader

st.title("🔍 Team Selection Debug Tool")

# Load data safely
try:
    loader = StreamlitDataLoader()
    team_desc = loader.load_team_data()
    schedule_data = loader.load_schedule_data([2025])
    st.success(f"✅ Loaded {len(team_desc)} teams and {len(schedule_data)} games")
except Exception as e:
    st.error(f"❌ Data loading failed: {e}")
    st.stop()

# Load model safely
try:
    model = StreamlitSimpleNFLModel(data_dir="data")
    
    # Try loading different year combinations
    st.subheader("🔧 Model Loading Test")
    
    years_options = {
        "Historical only (2022-2024)": [2022, 2023, 2024],
        "With 2025 (2022-2025)": [2022, 2023, 2024, 2025],
        "Recent only (2023-2024)": [2023, 2024]
    }
    
    selected_years = st.selectbox("Select data years:", list(years_options.keys()))
    years_to_load = years_options[selected_years]
    
    if st.button("Load Model"):
        try:
            model.load_data_from_parquet(years_to_load)
            st.success(f"✅ Model loaded with {len(model.pbp_data)} plays")
            
            # Show available teams
            available_teams = sorted(set(model.pbp_data['posteam'].dropna().unique()) | 
                                   set(model.pbp_data['defteam'].dropna().unique()))
            st.write(f"**Available teams ({len(available_teams)}):** {available_teams}")
            
        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            st.exception(e)
            st.stop()

except Exception as e:
    st.error(f"❌ Model initialization failed: {e}")
    st.stop()

# Team selection test
if hasattr(model, 'pbp_data') and model.pbp_data is not None:
    st.subheader("🏈 Team Selection Test")
    
    available_teams = sorted(set(model.pbp_data['posteam'].dropna().unique()) | 
                           set(model.pbp_data['defteam'].dropna().unique()))
    
    home_team = st.selectbox("Home Team:", available_teams)
    away_team = st.selectbox("Away Team:", available_teams)
    week = st.number_input("Week:", min_value=1, max_value=18, value=2)
    year = st.number_input("Year:", min_value=2022, max_value=2025, value=2025)
    
    if st.button("Test Prediction"):
        try:
            st.write(f"🔍 Testing: {away_team} @ {home_team}, Week {week} {year}")
            
            # Test spread prediction
            with st.spinner("Testing spread prediction..."):
                spread, spread_details = model.predict_spread(home_team, away_team, week, year)
                st.success(f"✅ Spread: {home_team} {spread:+.1f}")
                
                # Show details
                with st.expander("Spread Details"):
                    st.json(spread_details)
            
            # Test total prediction
            with st.spinner("Testing total prediction..."):
                total, total_details = model.predict_total(home_team, away_team, week, year)
                st.success(f"✅ Total: {total:.1f}")
                
                # Show details
                with st.expander("Total Details"):
                    st.json(total_details)
                    
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)
            
            # Show diagnostic info
            st.subheader("🔍 Diagnostic Information")
            
            # Check team data availability
            home_off = model.pbp_data[model.pbp_data['posteam'] == home_team]
            home_def = model.pbp_data[model.pbp_data['defteam'] == home_team]
            away_off = model.pbp_data[model.pbp_data['posteam'] == away_team]
            away_def = model.pbp_data[model.pbp_data['defteam'] == away_team]
            
            st.write(f"**{home_team} offensive plays:** {len(home_off)}")
            st.write(f"**{home_team} defensive plays:** {len(home_def)}")
            st.write(f"**{away_team} offensive plays:** {len(away_off)}")
            st.write(f"**{away_team} defensive plays:** {len(away_def)}")
            
            # Check data for prediction period
            pred_data = model.pbp_data[
                (model.pbp_data['season'] < year) | 
                ((model.pbp_data['season'] == year) & (model.pbp_data['week'] < week))
            ]
            st.write(f"**Data available for prediction:** {len(pred_data)} plays")

else:
    st.warning("⚠️ Model data not loaded - click 'Load Model' first")
