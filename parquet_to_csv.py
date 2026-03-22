import pandas as pd
import pyarrow
from pathlib import Path
import numpy as np

# --- Setup ---
root = Path("YOUR PATH")
output_dir = root / "combined_csv_12day_chunks"
output_dir.mkdir(exist_ok=True)

plant_data = {}

# --- Parquet File Processing and Aggregation ---
for parquet_file in root.rglob("*.parquet"):
    plant_id = parquet_file.parent.name
    df = pd.read_parquet(parquet_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    if "timezone" in df.columns:
        df["timestamp"] = df["timestamp"] + pd.to_timedelta(df["timezone"], unit="h")
    df = df[["timestamp", "metric_value"]]

    if plant_id not in plant_data:
        plant_data[plant_id] = df
    else:
        plant_data[plant_id] = pd.concat([plant_data[plant_id], df], ignore_index=True)

# --- CSV Output with 12-Day Chunking ---

# Calculate the size of a 12-day chunk in terms of rows (1 second interval)
# 12 days * 24 hours/day * 60 minutes/hour * 60 seconds/minute = 1,036,800 rows
ROWS_PER_CHUNK = 12 * 24 * 60 * 60

print(f"INFO: Each output CSV file will contain approximately {ROWS_PER_CHUNK:,} rows.")

for plant_id, df in plant_data.items():
    # Final data cleaning and sorting
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")

    total_rows = len(df)

    # Calculate the number of chunks required
    num_chunks = int(np.ceil(total_rows / ROWS_PER_CHUNK))

    print(f"Processing Plant ID: {plant_id}. Total rows: {total_rows:,}. Creating {num_chunks} files.")

    # Create a subdirectory for the plant's files
    plant_output_dir = output_dir / plant_id
    plant_output_dir.mkdir(exist_ok=True)

    # Loop through and create the chunks
    for i in range(num_chunks):
        start_index = i * ROWS_PER_CHUNK
        end_index = (i + 1) * ROWS_PER_CHUNK

        # Slice the DataFrame for the current chunk
        chunk_df = df.iloc[start_index:end_index]

        # Get the start and end dates for file naming
        start_date = chunk_df["timestamp"].iloc[0].strftime('%Y%m%d')
        end_date = chunk_df["timestamp"].iloc[-1].strftime('%Y%m%d')

        # Define the output file name
        output_file = plant_output_dir / f"{plant_id}_chunk{i + 1}_{start_date}_to_{end_date}.csv"

        # Write the chunk to a CSV file
        chunk_df.to_csv(output_file, index=False)
        print(f"  -> Created file {i + 1}/{num_chunks}: {output_file.name} (Rows: {len(chunk_df):,})")

print("Partitioned CSV files successfully created for each plant!")