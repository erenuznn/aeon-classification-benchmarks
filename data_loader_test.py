import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.graph_objects as go
from scipy.signal import savgol_filter


def _remove_spikes(signal, threshold_std):
    """
    Helper Function: Identifies and interpolates anomalous unilateral voltage spikes.
    """
    mean_val = np.mean(signal)
    std_val = np.std(signal)

    if std_val == 0:
        return signal

    z_scores = np.abs((signal - mean_val) / std_val)
    cleaned = np.copy(signal)
    spike_mask = z_scores > threshold_std

    if np.any(spike_mask):
        x_indices = np.arange(len(signal))
        cleaned[spike_mask] = np.interp(x_indices[spike_mask], x_indices[~spike_mask], signal[~spike_mask])
        print(f"STATUS: Removed {np.sum(spike_mask)} spikes using unilateral Z-score threshold {threshold_std}.")
    else:
        print("STATUS: No spikes detected above threshold.")

    return cleaned


def _filter_signal(signal, threshold_std, savgol_window, savgol_poly, apply_norm):
    """
    Helper Function: Executes unilateral mathematical pipeline.
    Sequence: Despike -> [Median-Center] -> Savitzky-Golay Smooth.
    """
    # Phase 1: Despiking
    processed = _remove_spikes(signal, threshold_std)

    # Phase 2: Baseline Correction
    if apply_norm:
        print("STATUS: Applying Median-Centering to correct unilateral baseline drift...")
        processed = processed - np.median(processed)

    # Phase 3: Polynomial Smoothing
    if len(processed) > savgol_window:
        print(f"STATUS: Applying Savitzky-Golay filter (Window={savgol_window}, Poly={savgol_poly})...")
        smooth = savgol_filter(processed, savgol_window, savgol_poly)
    else:
        print("WARNING: Unilateral signal too short for Savitzky-Golay window. Skipping smoothing.")
        smooth = processed

    return smooth


def _fetch_plant_dataframe(base_path, plant_id, preview_lines=False):
    """
    Helper Function: Locates Parquet files, formats timestamps, and drops missing values.
    """
    search_pattern = f"**/plant_id={plant_id}/*.parquet"
    parquet_files = list(base_path.rglob(search_pattern))

    if not parquet_files:
        return None

    dfs = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file, columns=['timestamp', 'metric_value'])
            if preview_lines:
                print(f"\n--- Preview: {file.name} ---\n{df.head()}\n")
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


def _render_full_timeseries(plant_data_dict, output_dir):
    """
    Helper Function: Generates an interactive Plotly HTML file for the continuous timelines.
    """
    print("STATUS: Rendering complete unilateral timeline verification graph via Plotly...")
    fig = go.Figure()

    for plant_id, data in plant_data_dict.items():
        fig.add_trace(
            go.Scattergl(
                x=data['timestamp'],
                y=data['metric_value'],
                mode='lines',
                name=f"Subject: {plant_id}",
                opacity=0.8
            )
        )

    fig.update_layout(
        title="Complete Unilateral Time Series Verification (Filtered Amplitude)",
        xaxis_title="Timestamp",
        yaxis_title="Amplitude (µV)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    output_file = Path(output_dir) / "full_timeseries_verification.html"
    fig.write_html(str(output_file), auto_open=True)
    print(f"STATUS: Interactive plot saved and launched from {output_file}")


def load_unilateral_data(base_directory, plant_id_list, window_size=300,
                         threshold_std=4.0, savgol_window=201, savgol_poly=1, apply_norm=True,
                         preview_lines=False, plot_full=False):
    """
    Master Ingestion Function: Orchestrates data loading, unilateral filtering,
    and matrix segmentation.
    """
    X_list = []
    time_list = []
    id_list = []

    base_path = Path(base_directory)
    print("-" * 50)
    print(f"STATUS: Initiating Data Ingestion at {base_path}")

    plot_registry = {}

    for plant_id in plant_id_list:
        df = _fetch_plant_dataframe(base_path, plant_id, preview_lines)

        if df is None or df.empty:
            print(f"WARNING: Insufficient valid data for {plant_id}. Skipping.")
            continue

        print(f"STATUS: Successfully ingested continuous array for {plant_id}.")

        raw_signal = df['metric_value'].values
        raw_times = df['timestamp'].values

        # Execute Unilateral Signal Filtering Sequence
        processed_signal = _filter_signal(raw_signal, threshold_std, savgol_window, savgol_poly, apply_norm)

        if plot_full:
            plot_df = pd.DataFrame({'timestamp': raw_times, 'metric_value': processed_signal})
            plot_registry[plant_id] = plot_df

        # Execute chronological segmentation
        n_windows = len(processed_signal) // window_size

        for i in range(n_windows):
            start = i * window_size
            end = start + window_size

            X_list.append(processed_signal[start:end])
            time_list.append(raw_times[start])
            id_list.append(plant_id)

    if not X_list:
        print("FATAL ERROR: Zero valid windows generated. Terminating pipeline.")
        sys.exit(1)

    X_arr = np.array(X_list).reshape(len(X_list), 1, window_size)
    time_series = pd.Series(time_list)
    id_arr = np.array(id_list)

    print("-" * 50)
    print("INGESTION SUMMARY:")
    print(f"Total Temporal Windows Extracted: {X_arr.shape[0]}")
    print(f"Matrix Dimension: {X_arr.shape}")
    print(f"Unique Subjects Processed: {len(np.unique(id_arr))}")
    print("-" * 50)

    if plot_full and plot_registry:
        _render_full_timeseries(plot_registry, base_path)

    return X_arr, time_series, id_arr


if __name__ == "__main__":
    TEST_DIR = "/Users/erenuzun/Desktop/Thesis/ML/DATA/test_data/vivent46/1sec"
    TEST_PLANTS = ["25072203-1"]

    # Configuration Parameters
    WINDOW_SIZE = 500
    SPIKE_THRESHOLD_STD = 4.0
    SAVGOL_WINDOW = 201
    SAVGOL_POLY = 1
    APPLY_NORM = True

    try:
        X, t, ids = load_unilateral_data(
            base_directory=TEST_DIR,
            plant_id_list=TEST_PLANTS,
            window_size=WINDOW_SIZE,
            threshold_std=SPIKE_THRESHOLD_STD,
            savgol_window=SAVGOL_WINDOW,
            savgol_poly=SAVGOL_POLY,
            apply_norm=APPLY_NORM,
            preview_lines=False,
            plot_full=True
        )
    except FileNotFoundError:
        print("Execution test aborted: Target directory not mounted.")