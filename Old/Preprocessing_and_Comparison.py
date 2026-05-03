import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import numpy as np
from scipy.signal import savgol_filter


def remove_spikes(signal, threshold_std):
    """
    Identifies and interpolates anomalous unilateral voltage spikes based on internal variance.
    """
    mean_val = signal.mean()
    std_val = signal.std()

    if std_val == 0:
        return signal

    z_scores = np.abs((signal - mean_val) / std_val)
    cleaned = signal.copy()
    spike_mask = z_scores > threshold_std

    if spike_mask.any():
        x_indices = np.arange(len(signal))
        cleaned[spike_mask] = np.interp(x_indices[spike_mask], x_indices[~spike_mask], signal[~spike_mask])
        print(f"STATUS: Removed {spike_mask.sum()} spikes using unilateral Z-score threshold {threshold_std}.")
    else:
        print("STATUS: No spikes detected above threshold.")

    return cleaned


def filter_signal(signal, threshold_std, savgol_window, savgol_poly, apply_norm):
    """
    Executes the unilateral mathematical pipeline: Despike -> [Median-Center] -> Smooth.
    """
    # Phase 1: Despiking
    processed = remove_spikes(signal, threshold_std)

    # Phase 2: Baseline Correction (Median-Centering)
    if apply_norm:
        print("STATUS: Applying Median-Centering to correct unilateral baseline drift...")
        processed = processed - np.median(processed)

    # Phase 3: Polynomial Smoothing
    if len(processed) > savgol_window:
        print(f"STATUS: Applying Savitzky-Golay filter (Window={savgol_window}, Poly={savgol_poly})...")
        smooth = savgol_filter(processed, savgol_window, savgol_poly)
    else:
        print("WARNING: Signal too short for Savitzky-Golay window. Skipping smoothing phase.")
        smooth = processed

    return smooth


def get_file_paths(base_dir, plant_id):
    """
    Locates and indexes the chunked CSV files for a specific biological subject.
    """
    plant_dir = base_dir / plant_id

    if not plant_dir.exists():
        if not str(plant_id).startswith("plant_id="):
            plant_dir = base_dir / f"plant_id={plant_id}"

    if not plant_dir.exists():
        print(f"ERROR: Target directory not found: {plant_dir}")
        return []

    files = sorted([f for f in plant_dir.glob("*chunk*.csv")])
    print(f"STATUS: Found {len(files)} chunk files in {plant_dir.name}")

    return files, plant_dir


def process_data(file_paths, threshold_std, savgol_window, savgol_poly, apply_norm):
    """
    Aggregates chunk files and applies the unilateral filtering sequence.
    """
    if not file_paths:
        print("FATAL: No file paths provided. Terminating process.")
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
        print("FATAL: No dataframes could be loaded into memory.")
        return None, None

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
    full_df['metric_value'] = pd.to_numeric(full_df['metric_value'], errors='coerce')
    full_df = full_df.dropna(subset=['metric_value', 'timestamp'])
    full_df = full_df.sort_values('timestamp').drop_duplicates('timestamp')

    print(f"STATUS: Raw unilateral data count: {len(full_df)} rows.")

    signal_raw = full_df['metric_value'].values
    signal_processed = filter_signal(signal_raw, threshold_std, savgol_window, savgol_poly, apply_norm)

    processed_df = full_df.copy()
    processed_df['metric_value'] = signal_processed

    return full_df, processed_df


def plot_data_span(raw_df, processed_df, output_html, plot_days_span, is_normalized):
    """
    Generates an interactive HTML plot. Dynamically configures dual Y-axes if baseline correction is active.
    """
    print(f"STATUS: Generating interactive visual interface spanning the final {plot_days_span} days...")

    end_time = processed_df['timestamp'].max()
    start_time = end_time - pd.Timedelta(days=plot_days_span)

    plot_processed = processed_df[processed_df['timestamp'] >= start_time]
    plot_raw = raw_df[raw_df['timestamp'] >= start_time]

    fig = make_subplots(specs=[[{"secondary_y": is_normalized}]])

    fig.add_trace(
        go.Scattergl(
            x=plot_raw['timestamp'], y=plot_raw['metric_value'],
            mode='lines', name='Raw Signal',
            line=dict(color='gray', width=1), opacity=0.5
        ),
        secondary_y=False
    )

    trace_name = 'Processed (Median-Centered + SavGol)' if is_normalized else 'Processed (Despiked + SavGol)'

    fig.add_trace(
        go.Scattergl(
            x=plot_processed['timestamp'], y=plot_processed['metric_value'],
            mode='lines', name=trace_name,
            line=dict(color='blue', width=2), opacity=0.9
        ),
        secondary_y=is_normalized
    )

    fig.update_layout(
        title=f"Unilateral Signal Processing Span: {plot_days_span} Days (Centered: {is_normalized})",
        xaxis_title="Timestamp",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig.update_yaxes(title_text="Raw EP Reading (µV)", secondary_y=False)
    if is_normalized:
        fig.update_yaxes(title_text="Centered Amplitude (µV)", secondary_y=True)

    print(f"STATUS: Writing interactive file to storage block: {output_html}...")
    fig.write_html(str(output_html))


def execute_processing(base_dir, plant_identifier, threshold_std, savgol_window, savgol_poly, plot_days_span,
                       apply_norm=False):
    """
    Primary execution function for external pipeline calls.
    """
    base_dir_path = Path(base_dir)

    input_files, resolved_plant_dir = get_file_paths(base_dir_path, plant_identifier)

    if not input_files:
        print("STOP: Required chunk files absent. Terminating preprocessing phase for current subject.")
        sys.exit(1)

    output_csv = resolved_plant_dir / "processed_training_data.csv"
    output_html = resolved_plant_dir / "interactive_plot.html"

    raw_data, processed_data = process_data(input_files, threshold_std, savgol_window, savgol_poly, apply_norm)

    if processed_data is not None:
        processed_data.to_csv(output_csv, index=False)
        print(f"SUCCESS: Processed unilateral dataset committed to disk: {output_csv}")
        plot_data_span(raw_data, processed_data, output_html, plot_days_span, apply_norm)
    else:
        print("FAILURE: Signal processing sequence returned Null parameters.")


if __name__ == "__main__":
    pass