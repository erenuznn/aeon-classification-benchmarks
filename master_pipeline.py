import parquet_to_csv
import Preprocessing_and_Comparison
import csv_to_ucr  # Requires the unilateral transformation script to be saved as csv_to_ts.py

## Configuration Parameter
RAW_DATA_ROOT = "/Users/erenuzun/Desktop/Thesis/ML/DATA/test_data/vivent46/1sec"
DAYS_PER_CHUNK = 12 # 1 million point max limit, divided manually csv files in 12
PLANT_IDENTIFIER = "25072203-1" # Put the option to choose between all channels or some specific channels
SPIKE_THRESHOLD_STD = 4.0
SAVGOL_WINDOW = 201
SAVGOL_POLY = 1
PLOT_DAYS_SPAN = 2.0

# Phase 3 Configuration Parameters
TS_OUTPUT_DIR = "/Users/erenuzun/Desktop/Thesis/ML/DATA/processed_ts"
DATASET_IDENTIFIER = f"Vivent_{PLANT_IDENTIFIER}_Unilateral"
STATE_LABEL = 1
POINTS_PER_WINDOW = 3600

print("1. Conversion from parquet files (raw data) into 4 csv files containing 12 days")

generated_csv_dir = parquet_to_csv.execute_chunking(
    raw_data_path=RAW_DATA_ROOT,
    days_per_chunk=DAYS_PER_CHUNK
)

print("-" * 50)
print("2. Signal Processing")

Preprocessing_and_Comparison.execute_processing(
    base_dir=generated_csv_dir,
    plant_identifier=PLANT_IDENTIFIER,
    threshold_std=SPIKE_THRESHOLD_STD,
    savgol_window=SAVGOL_WINDOW,
    savgol_poly=SAVGOL_POLY,
    plot_days_span=PLOT_DAYS_SPAN
)

print("-" * 50)
print("3. Unilateral TS Format Generation")

# Target the processed CSV output from Phase 2 for unilateral transformation
processed_csv_target = f"{generated_csv_dir}/processed_{PLANT_IDENTIFIER}.csv"

generated_ts_file = csv_to_ucr.generate_unilateral_ts(
    csv_path=processed_csv_target,
    output_dir=TS_OUTPUT_DIR,
    dataset_name=DATASET_IDENTIFIER,
    state_label=STATE_LABEL,
    points_per_window=POINTS_PER_WINDOW
)

print(f"Unilateral file generated at: {generated_ts_file}")

print("-" * 50)
print("Pipeline execution sequence terminated.")