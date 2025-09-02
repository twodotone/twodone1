import streamlit as st
import pandas as pd
import os

@st.cache_data(show_spinner="Loading play-by-play data for {current_year}...")
def load_rolling_data(current_year):
    """
    Loads play-by-play data for the given year and up to 3 previous years,
    then combines them into a single DataFrame.
    """
    pbp_dfs = []
    years_loaded = []
    
    # Load current year data
    current_file_path = os.path.join("data", f"pbp_{current_year}.parquet")
    try:
        pbp_current = pd.read_parquet(current_file_path)
        pbp_dfs.append(pbp_current)
        years_loaded.append(current_year)
    except FileNotFoundError:
        st.info(f"No data file found for the current season ({current_year}). This is expected early in the season.")
    except Exception as e:
        st.error(f"Failed to load data for {current_year}. Error: {e}")

    # Load previous year data (maintain backward compatibility by always loading just previous year)
    previous_year = current_year - 1
    previous_file_path = os.path.join("data", f"pbp_{previous_year}.parquet")
    try:
        pbp_previous = pd.read_parquet(previous_file_path)
        pbp_dfs.append(pbp_previous)
        years_loaded.append(previous_year)
    except FileNotFoundError:
        st.warning(f"Could not find data for the previous season ({previous_year}). Predictions will be based on current season data only.")
    except Exception as e:
        st.error(f"Failed to load data for {previous_year}. Error: {e}")
        
    # For HFA calculation, we also need data from older seasons
    # But this data is NOT used for team stats calculation
    hfa_years = []
    for i in range(2, 4):  # Look back 2-3 years (years 2 and 3)
        older_year = current_year - i
        older_file_path = os.path.join("data", f"pbp_{older_year}.parquet")
        try:
            # Check if the file exists, but don't load it here
            if os.path.exists(older_file_path):
                hfa_years.append(older_year)
        except Exception as e:
            st.error(f"Error checking data for {older_year}. Error: {e}")

    if not pbp_dfs:
        st.error("No play-by-play data could be loaded. The application cannot proceed.")
        return pd.DataFrame()

    # Combine the dataframes and return
    combined_df = pd.concat(pbp_dfs, ignore_index=True)
    # Ensure data is sorted chronologically for rolling calculations
    combined_df = combined_df.sort_values(by=['season', 'week', 'game_id']).reset_index(drop=True)
    
    # Show what years were loaded
    st.info(f"Loaded stats data for seasons: {', '.join(map(str, sorted(years_loaded)))}")
    if hfa_years:
        st.info(f"Additional HFA historical data available for: {', '.join(map(str, sorted(hfa_years)))}")
    
    return combined_df

@st.cache_data(show_spinner="Loading play-by-play data for {year}...")
def load_full_season_pbp(year):
    """
    Loads the full play-by-play dataset for a given year from a local parquet file.
    This function is kept for potential single-season analysis but is not used by the main rolling model.
    """
    file_path = os.path.join("data", f"pbp_{year}.parquet")
    
    try:
        pbp_df = pd.read_parquet(file_path)
        return pbp_df
    except FileNotFoundError:
        st.error(f"Data file not found for {year} at '{file_path}'. Please ensure the data has been downloaded.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load data for {year}. The file may be corrupt. Error: {e}")
        return pd.DataFrame()