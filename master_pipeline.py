import parquet_to_csv
import Preprocessing_and_Comparison

# ==========================================
# --- GLOBAL CONFIGURATION PARAMETERS ---
# ==========================================

# Phase 1 Parameters
RAW_DATA_ROOT = "/Users/erenuzun/Desktop/Thesis/ML/DATA/test_data/vivent46/1sec"
DAYS_PER_CHUNK = 12

# Phase 2 Parameters
PLANT_IDENTIFIER = "25072203-1"
SPIKE_THRESHOLD_STD = 4.0
SAVGOL_WINDOW = 201
SAVGOL_POLY = 1
PLOT_DAYS_SPAN = 2.0

# ==========================================
# --- EXECUTION SEQUENCE ---
# ==========================================

if __name__ == "__main__":
    print("Initiating Phase 1: Parquet to CSV Chunking...")

    # Execute Phase 1
    generated_csv_dir = parquet_to_csv.execute_chunking(
        raw_data_path=RAW_DATA_ROOT,
        days_per_chunk=DAYS_PER_CHUNK
    )

    print("-" * 50)
    print("Initiating Phase 2: Signal Processing and Visualization...")

    # Execute Phase 2
    Preprocessing_and_Comparison.execute_processing(
        base_dir=generated_csv_dir,
        plant_identifier=PLANT_IDENTIFIER,
        threshold_std=SPIKE_THRESHOLD_STD,
        savgol_window=SAVGOL_WINDOW,
        savgol_poly=SAVGOL_POLY,
        plot_days_span=PLOT_DAYS_SPAN
    )

    print("-" * 50)
    print("Pipeline execution sequence terminated.")