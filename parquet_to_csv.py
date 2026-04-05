import pandas as pd
import pyarrow
from pathlib import Path
import numpy as np

def execute_chunking(raw_data_path, days_per_chunk):
    if not (1 <= days_per_chunk <= 12):
        raise ValueError("Termination: The parameter DAYS_PER_CHUNK must be between 1 and 12.")

    root = Path(raw_data_path)
    output_dir = root / f"combined_csv_{days_per_chunk}day_chunks"
    output_dir.mkdir(exist_ok=True)

    plant_data = {}

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

    ROWS_PER_CHUNK = days_per_chunk * 24 * 60 * 60
    print(f"INFO: Parameter set to {days_per_chunk} day(s). Each CSV chunk will contain up to {ROWS_PER_CHUNK:,} rows.")

    for plant_id, df in plant_data.items():
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
        total_rows = len(df)
        num_chunks = int(np.ceil(total_rows / ROWS_PER_CHUNK))

        print(f"Processing Plant ID: {plant_id}. Total rows: {total_rows:,}. Creating {num_chunks} files.")

        plant_output_dir = output_dir / plant_id
        plant_output_dir.mkdir(exist_ok=True)

        for i in range(num_chunks):
            start_index = i * ROWS_PER_CHUNK
            end_index = (i + 1) * ROWS_PER_CHUNK
            chunk_df = df.iloc[start_index:end_index]

            start_date = chunk_df["timestamp"].iloc[0].strftime('%Y%m%d')
            end_date = chunk_df["timestamp"].iloc[-1].strftime('%Y%m%d')

            output_file = plant_output_dir / f"{plant_id}_chunk{i + 1}_{start_date}_to_{end_date}.csv"

            # File existence verification
            if output_file.exists():
                print(f"  -> File exists. Skipping {i + 1}/{num_chunks}: {output_file.name}")
                continue

            chunk_df.to_csv(output_file, index=False)
            print(f"  -> Created file {i + 1}/{num_chunks}: {output_file.name} (Rows: {len(chunk_df):,})")

        print(f"Partitioned {days_per_chunk}-day CSV chunks successfully created for each plant!")

        return output_dir

    if __name__ == "__main__":
        execute_chunking("/Users/erenuzun/Desktop/Thesis/ML/DATA/test_data/vivent46/1sec", 12)