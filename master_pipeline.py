import sys                                                                              # System-specific parameters and functions for CLI arguments
import argparse                                                                         # Parser for command-line options, arguments, and sub-commands
import os                                                                               # Miscellaneous operating system interfaces for path handling
import time                                                                             # Time access and conversions to measure ROCKET execution duration
import psutil                                                                           # Library for retrieving information on running processes and system utilization
from pathlib import Path                                                                # Object-oriented filesystem paths for cleaner directory handling
import matplotlib.pyplot as plt                                                         # Plotting library for creating the confusion matrix visual
import seaborn as sns                                                                   # Statistical data visualization based on matplotlib for heatmaps
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score    # ML metrics for evaluation
import parquet_to_csv                                                                   # Custom module: Converts raw Parquet month/day tree to aggregated CSVs
import Preprocessing_and_Comparison                                                     # Custom module: Applies Savitzky-Golay and spike filtering
import csv_to_ucr                                                                       # Custom module: Segments CSV into 3600-point windows and creates .ts file
from aeon.datasets import load_from_ts_file                                             # Specialized loader for time-series standard formats
from sklearn.model_selection import train_test_split                                    # Utility to split data into Train/Test sets
from aeon.classification.convolution_based import RocketClassifier                      # The ROCKET algorithm core
import joblib


def execute_rocket_classification(ts_file_path, test_fraction=0.2):  # Define function for Model Training
    """
    Executes ingestion, training, and evaluation of ROCKET on a unilateral .ts file.
    """
    print("-" * 50)
    print("Loading the dataset.")

    start_time = time.time()                                # Record the exact start timestamp in seconds
    process = psutil.Process(os.getpid())                   # Get the current OS process ID to track memory
    start_mem = process.memory_info().rss / (1024 * 1024)   # Capture initial RAM usage in Megabytes

    X, y = load_from_ts_file(ts_file_path)                  # Convert .ts file into X (3D array) and y (labels)

    print(f"Total number of time windows: {X.shape[0]}")
    print(f"Data points per instance: {X.shape[2]}")

    # Split data: 80% for learning kernels, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,                 # Feature matrix
        y,                        # Label vector
        test_size=test_fraction,  # Fraction of data for testing
        random_state=41,          # Set seed for random splitting
        stratify=y                # Ensure Train and Test sets have identical Stress/Adequate ratios
    )

    print("Initializing ROCKET classifier.")
    classifier = RocketClassifier(n_kernels=10000, random_state=42)  # Create 10k random convolutional kernels
    print("Execution of training protocol.")
    classifier.fit(X_train, y_train)                                 # Map kernels to training data to find features
    print("Execution of testing protocol.")
    predictions = classifier.predict(X_test)                         # Apply trained model to the test data

    end_time = time.time()                               # Record the finish timestamp
    end_mem = process.memory_info().rss / (1024 * 1024)  # Capture final RAM usage in Megabytes
    elapsed_time = end_time - start_time                 # Calculate total time spent in Phase 3
    mem_used = end_mem - start_mem                       # Calculate how much RAM the ROCKET model consumed
    accuracy = accuracy_score(y_test, predictions)       # Calculate percentage of correct predictions

    print("-" * 50)
    print("ROCKET PERFORMANCE SUMMARY")
    print(f"Total Run Time: {elapsed_time:.2f} seconds")
    print(f"Peak Memory Increment: {max(0, mem_used):.2f} MB")
    print(f"Current System Memory Usage: {end_mem:.2f} MB")
    print(f"Unilateral Classification Accuracy: {accuracy:.4f}")
    print("-" * 50)

    cm = confusion_matrix(y_test, predictions)                  # Generate count of TP, TN, FP, FN
    plt.figure(figsize=(8, 6))                                  # Initialize plot size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',          # Create heatmap with integer counts
                xticklabels=['Class 0', 'Class 1'],             # Label X-axis as Predicted
                yticklabels=['Class 0', 'Class 1'])             # Label Y-axis as Actual
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')    # Add title with accuracy score
    plt.ylabel('Actual Label')                                  # Y-axis label
    plt.xlabel('Predicted Label')                               # X-axis label

    matrix_path = Path(ts_file_path).parent / "rocket_confusion_matrix.png"
    plt.savefig(matrix_path)
    print(f"Confusion matrix saved to: {matrix_path}")
    plt.show()

    return classifier                                           # Return the trained model object for potential future use


# CONFIGURATION PARAMETERS
PLANT_DICTIONARY = {  # Map of all plant IDs to their binary health state (0=Safe, 1=Stress)
    "25072203-1": 0,  # Waterlogged
    "25072205-1": 1,  # Stress
    "25072219-1": 0,  # Adequate
    "25072221-1": 1,  # Stress
    "25072233-1": 0,  # Adequate
    "25072235-1": 0,  # Waterlogged
    "25072236-1": 0,  # Adequate
    "25072237-1": 0,  # Waterlogged
    "25072238-1": 0,  # Waterlogged
    "25072240-1": 0,  # Adequate
    "25072245-1": 0,  # Waterlogged
    "25072247-1": 0,  # Waterlogged
    "25072249-1": 0,  # Adequate
    "25072252-1": 0,  # Waterlogged
    "25072269-1": 0,  # Adequate
    "25072277-1": 0,  # Waterlogged
    "25072279-1": 1,  # Stress
    "25072280-1": 1,  # Stress
    "25072283-1": 1,  # Stress
    "25072288-1": 0,  # Adequate
    "25072290-1": 1,  # Stress
    "25072294-1": 1,  # Stress
    "25072300-1": 1,  # Stress
    "25072361-1": 0,  # Adequate
}

RAW_DATA_ROOT = "/Users/erenuzun/Desktop/Thesis/ML/DATA/test_data/vivent46/1sec"  # Root directory of Parquet files
DAYS_PER_CHUNK = 12                                                               # Limit set to stay under 1M rows per file (approx 12 days at 1Hz)
SPIKE_THRESHOLD_STD = 4.0                                                         # Filtering: Ignore points > 4 standard deviations from mean
SAVGOL_WINDOW = 201                                                               # Smoothing: Number of points used for local polynomial fit
SAVGOL_POLY = 1                                                                   # Smoothing: Degree of the polynomial (1 = linear)
PLOT_DAYS_SPAN = 2.0                                                              # Visualization: Amount of data to show in plots

TS_OUTPUT_FILE = "/Users/erenuzun/Desktop/Thesis/ML/DATA/processed_ts/Vivent_Master_Unilateral.ts"  # Target .ts file
DATASET_IDENTIFIER = "Vivent_Master_Unilateral"                                                     # Header name inside the .ts file
POINTS_PER_WINDOW = 3600                                                                            # Segmentation: 1 hour (3600 seconds)

# ARGUMENT PARSING
parser = argparse.ArgumentParser(description="Unilateral Biosignal Pipeline Control")                   # Create CLI parser
parser.add_argument("--data", action="store_true", help="Run parquet_to_csv conversion")   # --data runs the data loading
parser.add_argument("--filter", action="store_true",
                    help="Run signal preprocessing and filtering")                                      # --fitler runs the preprocessing
parser.add_argument("--rocket", action="store_true",
                    help="Run ROCKET classification")                                                   # --rocket runs the model training
args = parser.parse_args()                                                                              # Extract arguments passed via terminal

# DATA CONVERSION
generated_csv_dir = Path(RAW_DATA_ROOT) / f"combined_csv_{DAYS_PER_CHUNK}day_chunks"  # Predicted output path

if args.data:                                                   # Check if --data was provided
    print("-" * 50)
    print("1. Conversion from parquet files into csv files")
    generated_csv_dir = parquet_to_csv.execute_chunking(        # Run conversion globally across month/day folders
        raw_data_path=RAW_DATA_ROOT,
        days_per_chunk=DAYS_PER_CHUNK
    )
else:
    print("Data conversion is skipped.")

# 2: FILTERING & .TS GENERATION
if args.filter:                         # Check if --filter was provided
    print("-" * 50)
    print("2. Signal Processing")
    file_label_mapping = {}             # Initialize dictionary to link processed files to Stress labels

    for plant_id, label in PLANT_DICTIONARY.items():                         # Iterate through the 24 plants
        plant_csv_folder = Path(generated_csv_dir) / f"plant_id={plant_id}"  # Locate specific plant CSV folder

        if not plant_csv_folder.exists():                                            # Safety check for missing data
            print(f"SKIPPING: Data for {plant_id} not found at {plant_csv_folder}")  # Log missing folder
            continue                                                                 # Move to next plant

        print(f"Filtering data for the plant: {plant_id}")
        Preprocessing_and_Comparison.execute_processing(  # Apply SavGol and Spiking filters
            base_dir=str(generated_csv_dir),
            plant_identifier=plant_id,
            threshold_std=SPIKE_THRESHOLD_STD,
            savgol_window=SAVGOL_WINDOW,
            savgol_poly=SAVGOL_POLY,
            plot_days_span=PLOT_DAYS_SPAN
        )

        processed_csv_target = plant_csv_folder / "processed_training_data.csv"  # Locate output of filter
        file_label_mapping[str(processed_csv_target)] = label                    # Store file-label pair for TS generation

    print("-" * 50)
    print("Beginning .TS file generation")
    csv_to_ucr.generate_unified_unilateral_ts(  # Segment all 24 plants into 1 hour windows and save .ts
        file_label_mapping=file_label_mapping,
        output_path=TS_OUTPUT_FILE,
        dataset_name=DATASET_IDENTIFIER,
        points_per_window=POINTS_PER_WINDOW
    )
else:
    print("Filtering and .ts generation skipped.")

# Model training with ROCKET
if args.rocket:                                                                                 # Check if --rocket was provided
    if not os.path.exists(TS_OUTPUT_FILE):                                                      # Safety check: Cannot train without the .ts file
        print(f"ERROR: .ts file not found at {TS_OUTPUT_FILE}. Run with --filter first.")
        sys.exit(1)                                                                             # Terminate script with error code

    print("-" * 50)
    print("3. Executing ROCKET Classification")
    trained_model = execute_rocket_classification(  # Run the ML training and evaluation function
        ts_file_path=TS_OUTPUT_FILE,
        test_fraction=0.2
    )
else:
    print("Model training Skipped.")

print("-" * 50)

# 3: SAVING THE TRAINED MODEL
model_filename = "/Users/erenuzun/Desktop/Thesis/ML/MODELS/rocket_unilateral_v1.joblib"     #Update the name
joblib.dump(classifier, model_filename)
print(f"Model saved successfully at: {model_filename}")

print("-" * 50)

print("Pipeline execution sequence terminated.")