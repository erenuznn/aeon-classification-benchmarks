import sys
import argparse
import os
import time
import psutil
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import parquet_to_csv
import Preprocessing_and_Comparison
import csv_to_ucr
from aeon.datasets import load_from_ts_file
from aeon.classification.convolution_based import RocketClassifier, MiniRocketClassifier
import joblib


def execute_unilateral_classification(train_ts_path, test_ts_path, classifier, model_name, num_kernels, window_size,
                                      days_per_chunk, spike_threshold, savgol_window, savgol_poly, is_normalized):
    """
    Executes ingestion, training, and evaluation using explicitly defined train and test .ts files.
    """
    print("-" * 50)
    print(f"Loading dedicated unilateral training dataset: {train_ts_path}")

    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)

    # 1. Direct Ingestion of Pre-partitioned Data
    X_train, y_train = load_from_ts_file(train_ts_path)
    print(f"Loading dedicated unilateral testing dataset: {test_ts_path}")
    X_test, y_test = load_from_ts_file(test_ts_path)

    print(f"Training instances: {X_train.shape[0]} | Testing instances: {X_test.shape[0]}")
    print(f"Unilateral data points per instance: {X_train.shape[2]}")

    # 2. Execution Protocol
    print(f"Execution of unilateral training protocol using {model_name}.")
    classifier.fit(X_train, y_train)
    print(f"Execution of testing protocol for {model_name}.")
    predictions = classifier.predict(X_test)

    end_time = time.time()
    end_mem = process.memory_info().rss / (1024 * 1024)
    elapsed_time = end_time - start_time
    mem_used = end_mem - start_mem
    accuracy = accuracy_score(y_test, predictions)

    print("-" * 50)
    print(f"{model_name.upper()} PERFORMANCE SUMMARY (SUBJECT-WISE CV)")
    print(f"Total Run Time: {elapsed_time:.2f} seconds")
    print(f"Peak Memory Increment: {max(0, mem_used):.2f} MB")
    print(f"Current System Memory Usage: {end_mem:.2f} MB")
    print(f"Unilateral Classification Accuracy: {accuracy:.4f}")
    print("-" * 50)

    # 3. Confusion Matrix Generation
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])

    title_metadata = (
        f"Unilateral Confusion Matrix (Subject-Wise {model_name})\n"
        f"Accuracy: {accuracy:.4f} | Kernels: {num_kernels} | Window: {window_size} pts\n"
        f"Chunks: {days_per_chunk}d | Spike STD: {spike_threshold} | SavGol: W{savgol_window}, P{savgol_poly}\n"
        f"Median-Centered: {is_normalized}"
    )
    plt.title(title_metadata, fontsize=10, pad=15)
    plt.ylabel('Actual Label (Unseen Plants)')
    plt.xlabel('Predicted Label')

    norm_str = "NormTrue" if is_normalized else "NormFalse"
    matrix_filename = f"{model_name.lower()}_subjectwise_{norm_str}_K{num_kernels}_W{window_size}.png"
    matrix_path = Path(train_ts_path).parent / matrix_filename

    plt.tight_layout()
    plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {matrix_path}")

    return classifier


# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================
PLANT_DICTIONARY = {
    "25072205-1": 1, "25072219-1": 0, "25072221-1": 1, "25072233-1": 0,
    "25072236-1": 0, "25072240-1": 0, "25072249-1": 0, "25072269-1": 0,
    "25072279-1": 1, "25072280-1": 1, "25072283-1": 1, "25072288-1": 0,
    "25072290-1": 1, "25072294-1": 1, "25072300-1": 1, "25072361-1": 0,
}

# EXPLICIT HOLD-OUT TARGETS
TEST_PLANTS = ["25072219-1", "25072205-1"]  # Target 1 Adequate (Class 0), 1 Stress (Class 1)

RAW_DATA_ROOT = "/Users/erenuzun/Desktop/Thesis/ML/DATA/test_data/vivent46/1sec"
DAYS_PER_CHUNK = 12
SPIKE_THRESHOLD_STD = 4.0
SAVGOL_WINDOW = 201
SAVGOL_POLY = 1
PLOT_DAYS_SPAN = 2.0

TS_TRAIN_FILE = "/Users/erenuzun/Desktop/Thesis/ML/DATA/processed_ts/Vivent_Master_Unilateral_Train.ts"
TS_TEST_FILE = "/Users/erenuzun/Desktop/Thesis/ML/DATA/processed_ts/Vivent_Master_Unilateral_Test.ts"
POINTS_PER_WINDOW = 450
NUM_KERNELS = 10000

# ==========================================
# ARGUMENT PARSING
# ==========================================
parser = argparse.ArgumentParser(description="Unilateral Biosignal Pipeline Control")
parser.add_argument("--data", action="store_true", help="Run parquet_to_csv conversion")
parser.add_argument("--filter", action="store_true", help="Run signal preprocessing and dual TS generation")
parser.add_argument("--normalize", action="store_true", help="Apply Z-score normalization to signals")

classifier_group = parser.add_mutually_exclusive_group()
classifier_group.add_argument('--rocket', action='store_true', help="Execute standard unilateral ROCKET")
classifier_group.add_argument('--minirocket', action='store_true', help="Execute memory-optimized MiniRocket")

args = parser.parse_args()

generated_csv_dir = Path(RAW_DATA_ROOT) / f"combined_csv_{DAYS_PER_CHUNK}day_chunks"

# ==========================================
# PHASE 1: DATA CONVERSION
# ==========================================
if args.data:
    print("-" * 50)
    print("1. Conversion from parquet files into csv files")
    generated_csv_dir = parquet_to_csv.execute_chunking(
        raw_data_path=RAW_DATA_ROOT,
        days_per_chunk=DAYS_PER_CHUNK
    )

# ==========================================
# PHASE 2: FILTERING & DUAL TS GENERATION
# ==========================================
if args.filter:
    print("-" * 50)
    print("2. Signal Processing and Separation")

    train_label_mapping = {}
    test_label_mapping = {}

    for plant_id, label in PLANT_DICTIONARY.items():
        plant_csv_folder = Path(generated_csv_dir) / f"plant_id={plant_id}"

        if not plant_csv_folder.exists():
            continue

        print(f"Filtering unilateral data for plant: {plant_id}")
        Preprocessing_and_Comparison.execute_processing(
            base_dir=str(generated_csv_dir),
            plant_identifier=plant_id,
            threshold_std=SPIKE_THRESHOLD_STD,
            savgol_window=SAVGOL_WINDOW,
            savgol_poly=SAVGOL_POLY,
            plot_days_span=PLOT_DAYS_SPAN,
            apply_norm=args.normalize
        )

        processed_csv_target = plant_csv_folder / "processed_training_data.csv"

        if plant_id in TEST_PLANTS:
            print(f"ROUTING: {plant_id} designated for TEST matrix.")
            test_label_mapping[str(processed_csv_target)] = label
        else:
            train_label_mapping[str(processed_csv_target)] = label

    print("-" * 50)
    print("Beginning Generation of DUAL .TS files")

    csv_to_ucr.generate_unified_unilateral_ts(
        file_label_mapping=train_label_mapping,
        output_path=TS_TRAIN_FILE,
        dataset_name="Vivent_Unilateral_Train",
        points_per_window=POINTS_PER_WINDOW
    )

    csv_to_ucr.generate_unified_unilateral_ts(
        file_label_mapping=test_label_mapping,
        output_path=TS_TEST_FILE,
        dataset_name="Vivent_Unilateral_Test",
        points_per_window=POINTS_PER_WINDOW
    )

# ==========================================
# PHASE 3: MODEL TRAINING
# ==========================================
trained_model = None
model_type_name = None

if args.rocket or args.minirocket:
    if not os.path.exists(TS_TRAIN_FILE) or not os.path.exists(TS_TEST_FILE):
        print("ERROR: Target .ts files not found. Run with --filter first to generate Train/Test tensors.")
        sys.exit(1)

    print("-" * 50)
    print("3. Executing Subject-Wise Classification")

    if args.minirocket:
        classifier_instance = MiniRocketClassifier(n_kernels=NUM_KERNELS, random_state=42)
        model_type_name = "MiniRocket"
    elif args.rocket:
        classifier_instance = RocketClassifier(n_kernels=NUM_KERNELS, random_state=42)
        model_type_name = "ROCKET"

    trained_model = execute_unilateral_classification(
        train_ts_path=TS_TRAIN_FILE,
        test_ts_path=TS_TEST_FILE,
        classifier=classifier_instance,
        model_name=model_type_name,
        num_kernels=NUM_KERNELS,
        window_size=POINTS_PER_WINDOW,
        days_per_chunk=DAYS_PER_CHUNK,
        spike_threshold=SPIKE_THRESHOLD_STD,
        savgol_window=SAVGOL_WINDOW,
        savgol_poly=SAVGOL_POLY,
        is_normalized=args.normalize
    )

    print("-" * 50)
    norm_str = "NormTrue" if args.normalize else "NormFalse"
    model_filename = f"/Users/erenuzun/Desktop/Thesis/ML/MODELS/{model_type_name.lower()}_unilateral_subjectwise_{norm_str}.joblib"
    joblib.dump(trained_model, model_filename)
    print(f"Model saved successfully at: {model_filename}")

print("-" * 50)
print("Pipeline execution sequence terminated.")