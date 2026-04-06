import sys
import argparse
import os
from pathlib import Path
import parquet_to_csv
import Preprocessing_and_Comparison
import csv_to_ucr


from aeon.datasets import load_from_ts_file
from sklearn.model_selection import train_test_split
from aeon.classification.convolution_based import RocketClassifier
from sklearn.metrics import accuracy_score


def execute_rocket_classification(ts_file_path, test_fraction=0.2):
    """
    Executes ingestion, training, and evaluation of ROCKET on a unilateral .ts file.
    """
    print("-" * 50)
    print("Initiating unilateral data ingestion sequence.")
    # Standard aeon ingestion for .ts files
    X, y = load_from_ts_file(ts_file_path)

    print(f"Total unilateral instances extracted: {X.shape[0]}")
    print(f"Temporal dimension per instance: {X.shape[2]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=42,
        stratify=y
    )

    print("Initializing unilateral ROCKET classifier.")
    classifier = RocketClassifier(n_kernels=10000, random_state=42)

    print("Execution of training protocol.")
    classifier.fit(X_train, y_train)

    print("Execution of testing protocol.")
    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Unilateral classification accuracy: {accuracy:.4f}")

    return classifier


# ==========================================
# --- CONFIGURATION PARAMETERS ---
# ==========================================
# Mapping configuration for the 24 unilateral plant channels
# 0 = unstressed, 1 = stressed
PLANT_DICTIONARY = {
    "25072203-1": 0,    # Waterlogged
    "25072205-1": 1,    # Stress
    "25072219-1": 0,    # Adequate
    "25072221-1": 1,    # Stress
    "25072233-1": 0,    # Adequate
    "25072235-1": 0,    # Waterlogged
    "25072236-1": 0,    # Adequate
    "25072237-1": 0,    # Waterlogged
    "25072238-1": 0,    # Waterlogged
    "25072240-1": 0,    # Adequate
    "25072245-1": 0,    # Waterlogged
    "25072247-1": 0,    # Waterlogged
    "25072249-1": 0,    # Adequate
    "25072252-1": 0,    # Waterlogged
    "25072269-1": 0,    # Adequate
    "25072277-1": 0,    # Waterlogged
    "25072279-1": 1,    # Stress
    "25072280-1": 1,    # Stress
    "25072283-1": 1,    # Stress
    "25072288-1": 0,    # Adequate
    "25072290-1": 1,    # Stress
    "25072294-1": 1,    # Stress
    "25072300-1": 1,    # Stress
    "25072361-1": 0,    # Adequate
}

RAW_DATA_ROOT = "/Users/erenuzun/Desktop/Thesis/ML/DATA/test_data/vivent46/1sec"
DAYS_PER_CHUNK = 12 # 1 million point max limit, divided manually csv files in 12
SPIKE_THRESHOLD_STD = 4.0
SAVGOL_WINDOW = 201
SAVGOL_POLY = 1
PLOT_DAYS_SPAN = 2.0

TS_OUTPUT_FILE = "/Users/erenuzun/Desktop/Thesis/ML/DATA/processed_ts/Vivent_Master_Unilateral.ts"
DATASET_IDENTIFIER = "Vivent_Master_Unilateral"
POINTS_PER_WINDOW = 3600

# ==========================================
# --- ARGUMENT PARSING ---
# ==========================================
parser = argparse.ArgumentParser(description="Unilateral Biosignal Pipeline Control")
parser.add_argument("--data", action="store_true", help="Run parquet_to_csv conversion")
parser.add_argument("--filter", action="store_true", help="Run signal preprocessing and filtering")
parser.add_argument("--rocket", action="store_true", help="Run ROCKET classification")

args = parser.parse_args()

# ==========================================
# --- PHASE 1: DATA CONVERSION ---
# ==========================================
# Define target path used by parquet_to_csv
generated_csv_dir = Path(RAW_DATA_ROOT) / f"combined_csv_{DAYS_PER_CHUNK}day_chunks"

if args.data:
    print("-" * 50)
    print("1. Conversion from parquet files (raw data) into 4 csv files containing 12 days")
    # Call it ONCE globally. Internally handles the month/day/plant structure for all 24 plants.
    generated_csv_dir = parquet_to_csv.execute_chunking(
        raw_data_path=RAW_DATA_ROOT,
        days_per_chunk=DAYS_PER_CHUNK
    )
else:
    print("Phase 1: Skipped.")

# ==========================================
# --- PHASE 2: FILTERING & TS GEN ---
# ==========================================
if args.filter:
    print("-" * 50)
    print("2. Signal Processing")
    file_label_mapping = {}

    for plant_id, label in PLANT_DICTIONARY.items():
        # Match the directory structure created by Phase 1
        plant_csv_folder = Path(generated_csv_dir) / f"plant_id={plant_id}"

        if not plant_csv_folder.exists():
            print(f"SKIPPING: Data for {plant_id} not found at {plant_csv_folder}")
            continue

        print(f"Filtering unilateral data for: {plant_id}")
        Preprocessing_and_Comparison.execute_processing(
            base_dir=str(generated_csv_dir),
            plant_identifier=plant_id,
            threshold_std=SPIKE_THRESHOLD_STD,
            savgol_window=SAVGOL_WINDOW,
            savgol_poly=SAVGOL_POLY,
            plot_days_span=PLOT_DAYS_SPAN
        )

        # Path where Preprocessing_and_Comparison saves its file per plant directory
        processed_csv_target = plant_csv_folder / "processed_training_data.csv"
        file_label_mapping[str(processed_csv_target)] = label

    print("-" * 50)
    print("Initiating Master Unilateral TS Generation")
    csv_to_ucr.generate_unified_unilateral_ts(
        file_label_mapping=file_label_mapping,
        output_path=TS_OUTPUT_FILE,
        dataset_name=DATASET_IDENTIFIER,
        points_per_window=POINTS_PER_WINDOW
    )
else:
    print("Phase 2: Skipped.")

# ==========================================
# --- PHASE 3: ROCKET ---
# ==========================================
if args.rocket:
    if not os.path.exists(TS_OUTPUT_FILE):
        print(f"ERROR: .ts file not found at {TS_OUTPUT_FILE}. Run with --filter first.")
        sys.exit(1)

    print("-" * 50)
    print("3. Executing ROCKET Classification")
    trained_model = execute_rocket_classification(
        ts_file_path=TS_OUTPUT_FILE,
        test_fraction=0.2
    )
else:
    print("Phase 3: Skipped.")

print("-" * 50)
print("Pipeline execution sequence terminated.")