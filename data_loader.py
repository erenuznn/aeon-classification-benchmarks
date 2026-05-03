import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy.signal import savgol_filter

def _fetch_plant_dataframe(base_path, plant_id):
    search_pattern = f"**/plant_id={plant_id}/*.parquet"
    parquet_files = list(base_path.rglob(search_pattern))

    if not parquet_files:
        return None

    dfs = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file, columns=['timestamp', 'metric_value'])
            dfs.append(df)
        except Exception as e:
            print(f"WARNING: Failed to read {file.name}: {e}")

    if not dfs:
        return None

    full_df = pd.concat(dfs, ignore_index=True)
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], unit='ms')
    full_df = full_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    full_df = full_df.dropna(subset=['metric_value']).reset_index(drop=True)

    return full_df

def load_unilateral_data(base_directory, plant_id_list, window_size=3600, savgol_window=201, savgol_poly=1):
    X_list = []
    time_list = []
    id_list = []

    base_path = Path(base_directory)
    print(f"Base path {base_path}")

    for plant_id in plant_id_list:
        df = _fetch_plant_dataframe(base_path, plant_id)

        if df is None or df.empty:
            print(f"WARNING: Insufficient valid data for {plant_id}. Skipping.")
            continue

        raw_signal = df['metric_value'].values
        raw_times = df['timestamp'].values

        if len(raw_signal) > savgol_window:
            processed_unilateral_signal = savgol_filter(raw_signal, savgol_window, savgol_poly)
        else:
            processed_unilateral_signal = raw_signal

        n_windows = len(processed_unilateral_signal) // window_size

        for i in range(n_windows):
            start = i * window_size
            end = start + window_size

            X_list.append(processed_unilateral_signal[start:end])
            time_list.append(raw_times[start])
            id_list.append(plant_id)

    if not X_list:
        print("Zero valid windows generated. Terminating pipeline.")
        sys.exit(1)

    X_arr = np.array(X_list).reshape(len(X_list), 1, window_size)
    time_series = pd.Series(time_list)
    id_arr = np.array(id_list)

    print(f"Total Temporal Windows Extracted: {X_arr.shape[0]}")
    print(f"Matrix Dimension: {X_arr.shape}")

    return X_arr, time_series, id_arr