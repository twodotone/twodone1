import nfl_data_py as nfl
import os

# --- Configuration ---
YEARS_TO_DOWNLOAD = [2022, 2023, 2024]
DATA_FOLDER = "data"

# --- Main Execution ---
if __name__ == "__main__":
    # Create the data directory if it doesn't exist
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Created directory: {DATA_FOLDER}")

    # Download and save the data for each year
    for year in YEARS_TO_DOWNLOAD:
        print(f"Downloading play-by-play data for {year}...")
        try:
            pbp_df = nfl.import_pbp_data([year])
            if not pbp_df.empty:
                file_path = os.path.join(DATA_FOLDER, f"pbp_{year}.parquet")
                pbp_df.to_parquet(file_path, compression='gzip')
                print(f"Successfully saved data to {file_path}")
            else:
                print(f"Warning: No data returned for {year}.")
        except Exception as e:
            print(f"Error downloading data for {year}: {e}")

    print("\nData download process complete.")