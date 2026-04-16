import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import zscore


def remove_spikes(signal, threshold_std):
    mean_val = signal.mean()
    std_val = signal.std()
    if std_val == 0:
        return signal

    z_scores = np.abs((signal - mean_val) / std_val)
    cleaned = signal.copy()
    spike_mask = z_scores > threshold_std

    if spike_mask.any():
        x_indices = np.arange(len(signal))
        # Interpolate across the spike
        cleaned[spike_mask] = np.interp(x_indices[spike_mask], x_indices[~spike_mask], signal[~spike_mask])
        print(f"STATUS: Removed {spike_mask.sum()} spikes using Z-score threshold {threshold_std}.")
    else:
        print("STATUS: No spikes detected above threshold.")
    return cleaned


def filter_signal(signal, threshold_std, savgol_window, savgol_poly):
    """
    Executes the unilateral math pipeline: Despike -> Normalize -> Smooth
    """
    # 1. Remove extreme biological artifacts
    despiked = remove_spikes(signal, threshold_std)

    # 2. Individual Plant Normalization (Z-Score)
    print("STATUS: Applying Z-Score Normalization to the despiked signal...")
    # This transforms the signal to have a mean of 0 and std of 1
    normalized = zscore(despiked)

    # 3. Apply Local Polynomial Smoothing (Savitzky-Golay)
    if len(normalized) > savgol_window:
        print(f"STATUS: Applying Savitzky-Golay filter (Window={savgol_window}, Poly={savgol_poly})...")
        smooth = savgol_filter(normalized, savgol_window, savgol_poly)
    else:
        print("WARNING: Signal too short for Savitzky-Golay window. Skipping smoothing.")
        smooth = normalized

    return smooth


def get_file_paths(base_dir, plant_id):
    plant_dir = base_dir / plant_id

    # Prefix resolution protocol
    if not plant_dir.exists():
        if not str(plant_id).startswith("plant_id="):
            plant_dir = base_dir / f"plant_id={plant_id}"

    if not plant_dir.exists():
        print(f"ERROR: Target directory not found: {plant_dir}")
        return []

    files = sorted([f for f in plant_dir.glob("*chunk*.csv")])
    print(f"STATUS: Found {len(files)} chunk files in {plant_dir.name}")
    return files, plant_dir


def process_data(file_paths, threshold_std, savgol_window, savgol_poly):
    if not file_paths:
        print("FATAL: No file paths provided.")
        return None, None
    print(f"STATUS: Loading {len(file_paths)} raw files...")
    all_dfs = []
    for f in file_paths:
        try:
            df = pd.read_csv(f, usecols=['timestamp', 'metric_value'])
            all_dfs.append(df)
        except Exception as e:
            print(f"ERROR: Failed to read {f.name}: {e}")

    if not all_dfs:
        print("FATAL: No dataframes could be loaded.")
        return None, None

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
    full_df['metric_value'] = pd.to_numeric(full_df['metric_value'], errors='coerce')
    full_df = full_df.dropna(subset=['metric_value', 'timestamp'])
    full_df = full_df.sort_values('timestamp').drop_duplicates('timestamp')

    print(f"STATUS: Raw data count: {len(full_df)} rows.")

    signal_raw = full_df['metric_value'].values
    # Execute the updated processing pipeline
    signal_processed = filter_signal(signal_raw, threshold_std, savgol_window, savgol_poly)

    processed_df = full_df.copy()
    processed_df['metric_value'] = signal_processed

    return full_df, processed_df


def plot_data_span(raw_df, processed_df, output_html, plot_days_span):
    print(f"STATUS: Generating interactive plot showing the last {plot_days_span} days...")
    end_time = processed_df['timestamp'].max()
    start_time = end_time - pd.Timedelta(days=plot_days_span)

    plot_processed = processed_df[processed_df['timestamp'] >= start_time]
    plot_raw = raw_df[raw_df['timestamp'] >= start_time]

    fig = go.Figure()

    # Optional visual fix: Since the processed signal is now normalized (centered at 0)
    # and the raw signal is still in original microvolts, they will look separated on the Y-axis.
    # This is expected behavior and proves the normalization worked.
    fig.add_trace(go.Scattergl(
        x=plot_raw['timestamp'], y=plot_raw['metric_value'],
        mode='lines', name='Raw Signal',
        line=dict(color='gray', width=1), opacity=0.5
    ))

    # We map this to a secondary Y-axis to easily visualize both scales together
    fig.add_trace(go.Scattergl(
        x=plot_processed['timestamp'], y=plot_processed['metric_value'],
        mode='lines', name='Processed (Normalized)',
        line=dict(color='blue', width=2), opacity=0.9,
        yaxis="y2"
    ))

    fig.update_layout(
        title=f"Signal Processing: Z-Score Normalized & Smoothed (Last {plot_days_span} Days)",
        xaxis_title="Timestamp",
        yaxis_title="Raw EP Reading (µV)",
        yaxis2=dict(
            title="Normalized Amplitude (Z-Score)",
            overlaying="y",
            side="right"
        ),
        template="plotly_white", hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    print(f"STATUS: Saving interactive plot to {output_html}...")
    fig.write_html(str(output_html))
    # fig.show()  # Uncomment if you want it to automatically open in your browser


def execute_processing(base_dir, plant_identifier, threshold_std, savgol_window, savgol_poly, plot_days_span):
    base_dir_path = Path(base_dir)

    input_files, resolved_plant_dir = get_file_paths(base_dir_path, plant_identifier)
    if not input_files:
        print("STOP: No files found. Terminating processing phase.")
        sys.exit(1)

    output_csv = resolved_plant_dir / "processed_training_data.csv"
    output_html = resolved_plant_dir / "interactive_plot.html"

    raw_data, processed_data = process_data(input_files, threshold_std, savgol_window, savgol_poly)
    if processed_data is not None:
        processed_data.to_csv(output_csv, index=False)
        print(f"SUCCESS: Processed data saved to: {output_csv}")
        plot_data_span(raw_data, processed_data, output_html, plot_days_span)
    else:
        print("FAILURE: Processing returned None.")


if __name__ == "__main__":
    pass