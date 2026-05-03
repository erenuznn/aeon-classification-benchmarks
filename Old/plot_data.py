import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_combined_data(file_paths: list, plant_id: str):
    """
    Reads multiple time-series CSV files, combines them, sorts the data,
    and plots the 'metric_value' (EP reading) against the 'timestamp'.

    Args:
        file_paths (list): A list of full paths to the CSV files to be combined.
        plant_id (str): The identifier for the plant, used for the plot title.
    """
    print(f"STATUS: Reading and combining {len(file_paths)} files for Plant ID: {plant_id}...")

    # 1. Read and combine the CSV files
    all_data_frames = []
    try:
        for file_path in file_paths:
            # Read CSV. The column names 'timestamp' and 'metric_value' are expected.
            df = pd.read_csv(file_path)
            all_data_frames.append(df)

        if not all_data_frames:
            print("ERROR: No data files were loaded. Terminating operation.")
            return

        combined_df = pd.concat(all_data_frames, ignore_index=True)

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to read or combine data files. Exception: {e}")
        return

    # 2. Data Cleaning and Preparation
    # Ensure 'timestamp' is in datetime format and sort the data
    try:
        # The timestamp format 'YYYY-MM-DD HH:MM:SS' is standard and will be parsed correctly.
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        # Drop duplicates and sort by time
        combined_df = combined_df.sort_values(by='timestamp').drop_duplicates(subset='timestamp')
    except Exception as e:
        print(f"ERROR: Timestamp conversion or sorting failed. Check column names and data types. Exception: {e}")
        return

    total_rows = len(combined_df)
    print(f"STATUS: Data successfully combined. Total data points for plot: {total_rows:,}")

    if total_rows == 0:
        print("WARNING: Combined DataFrame contains zero rows after processing. Plotting aborted.")
        return

    # 3. Data Visualization (Plotting)
    # Configure figure size for better visualization of time series data
    plt.figure(figsize=(15, 7))

    # Plotting the time series data
    # Use a thin line width given the high density of 1-second data points
    plt.plot(combined_df['timestamp'], combined_df['metric_value'],
             label='EP Reading (Metric Value)', color='#1f77b4', linewidth=0.75)  # Using a standard blue color

    # Configure plot aesthetics
    plt.title(f'Time Series Analysis of EP Readings for Plant ID: {plant_id}', fontsize=16)
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('EP Reading Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Customize x-axis to show major time units (e.g., daily or weekly ticks)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d\n%H:%M'))
    plt.gcf().autofmt_xdate()  # Auto-formats the date labels to prevent overlap

    plt.tight_layout()  # Adjust plot to fit all elements

    # Display the plot
    plt.show()
    print("STATUS: Plot generation complete.")


# --- Execution Block ---

# --- CONFIGURATION REQUIRED ---
# NOTE: The user must update these paths to the actual locations of the generated CSV files.

# 1. Replace "25072203-1" with the actual plant ID from your file names.
PLANT_IDENTIFIER = "25072205-1"

# 2. Replace the path below with the directory containing the plant_id subfolders.
# This should be the 'combined_csv_12day_chunks' folder created in the previous step.
BASE_DIR = Path("/Users/erenuzun/Desktop/Thesis/ML/DATA/test_data/vivent46/1sec/combined_csv_12day_chunks")

# 3. Define the full paths to the chunked CSV files for the specified plant.
# Adjust the dates and number of chunks (1, 2, 3, etc.) as necessary.
input_files = [
    BASE_DIR / PLANT_IDENTIFIER / f"{PLANT_IDENTIFIER}_chunk1_20240814_to_20240825.csv",
    BASE_DIR / PLANT_IDENTIFIER / f"{PLANT_IDENTIFIER}_chunk2_20240826_to_20240906.csv",
    BASE_DIR / PLANT_IDENTIFIER / f"{PLANT_IDENTIFIER}_chunk3_20240907_to_20240918.csv",
    BASE_DIR / PLANT_IDENTIFIER / f"{PLANT_IDENTIFIER}_chunk4_20240919_to_20240919.csv",
    # Including the one you uploaded as an example
]

# --- Script Execution ---

# Filter out paths that do not exist before attempting to plot
existing_files = [f for f in input_files if f.exists()]

if not existing_files:
    # Attempt to use rglob to find all chunks if exact file names are unknown
    print(f"WARNING: Specific file names not found. Searching for all CSV files in the plant's output directory...")

    plant_output_dir = BASE_DIR / PLANT_IDENTIFIER
    if plant_output_dir.exists():
        existing_files = list(plant_output_dir.rglob("*.csv"))
        if not existing_files:
            print(
                f"CRITICAL ERROR: No CSV files found in the path: {plant_output_dir}. Please verify the BASE_DIR and PLANT_IDENTIFIER.")
    else:
        print(f"CRITICAL ERROR: The plant output directory does not exist: {plant_output_dir}.")

if existing_files:
    plot_combined_data(existing_files, PLANT_IDENTIFIER)
else:
    print("OPERATION ABORTED: Cannot proceed without valid input files.")