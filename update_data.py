import nfl_data_py as nfl
import pandas as pd
import os
from datetime import datetime

# --- Configuration ---
# Get the current year dynamically
CURRENT_YEAR = datetime.now().year
DATA_DIR = "data"
FILE_PATH = os.path.join(DATA_DIR, f"pbp_{CURRENT_YEAR}.parquet")

def update_pbp_data():
    """
    Fetches the latest play-by-play data for the current season
    and saves it as a parquet file.
    """
    print(f"Fetching play-by-play data for the {CURRENT_YEAR} season...")
    
    try:
        # nfl_data_py can handle fetching data for an ongoing season
        pbp_df = nfl.import_pbp_data([CURRENT_YEAR])
        
        if pbp_df.empty:
            print("No data returned. The season may not have started yet.")
            return

        # Ensure the data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Save the data to a parquet file
        pbp_df.to_parquet(FILE_PATH, compression="gzip")
        print(f"Successfully saved data to {FILE_PATH}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    update_pbp_data()