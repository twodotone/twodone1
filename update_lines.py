import nfl_data_py as nfl
import pandas as pd
import os
from datetime import datetime

# --- Configuration ---
CURRENT_YEAR = datetime.now().year
DATA_DIR = "data"
FILE_PATH = os.path.join(DATA_DIR, f"odds_{CURRENT_YEAR}.parquet")

def get_current_nfl_week():
    """
    Estimates the current NFL week based on the date.
    Note: This is a simplified calculation.
    """
    today = datetime.now()
    # The NFL season typically starts on the first Thursday of September.
    season_start_month = 9
    first_day_of_sept = datetime(today.year, season_start_month, 1)
    days_to_thursday = (3 - first_day_of_sept.weekday() + 7) % 7
    season_start_date = first_day_of_sept.replace(day=1 + days_to_thursday)

    if today < season_start_date:
        return 1  # Before the season starts, default to Week 1

    days_since_start = (today - season_start_date).days
    current_week = (days_since_start // 7) + 1
    
    # A regular season has 18 weeks.
    return min(current_week, 18)

def update_betting_lines():
    """
    Fetches the latest betting lines for the current week and appends them 
    to a parquet file for the current season.
    """
    current_week = get_current_nfl_week()
    print(f"Fetching betting lines for {CURRENT_YEAR}, Week {current_week}...")

    try:
        # Fetch the latest odds for the current week
        weekly_odds_df = nfl.import_weekly_odds(years=[CURRENT_YEAR], weeks=[current_week])

        if weekly_odds_df.empty:
            print(f"No odds data found for Week {current_week}. It might be too early.")
            return

        os.makedirs(DATA_DIR, exist_ok=True)

        # If an odds file already exists, load it and append the new data.
        if os.path.exists(FILE_PATH):
            existing_odds_df = pd.read_parquet(FILE_PATH)
            # Combine data and remove duplicates, keeping the newest entry for each game.
            # This prevents duplicate rows if the script runs multiple times for the same week.
            combined_df = pd.concat([existing_odds_df, weekly_odds_df]).drop_duplicates(
                subset=['game_id', 'provider'], keep='last'
            )
        else:
            combined_df = weekly_odds_df

        # Save the updated data
        combined_df.to_parquet(FILE_PATH, compression="gzip")
        print(f"Successfully updated and saved betting lines to {FILE_PATH}")

    except Exception as e:
        print(f"An error occurred while updating betting lines: {e}")

if __name__ == "__main__":
    update_betting_lines()
