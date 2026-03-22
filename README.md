# aeon-classification-benchmarks

## Objective
This repository contains execution pipelines for comparing the classification performance, computational time, and memory allocation of feature-based (Catch22) and convolution-based (ROCKET) machine learning algorithms. It also includes an automated data preprocessing pipeline for formatting continuous raw sensor data into discrete arrays suitable for time-series classification.

## Execution Scripts and Datasets

### Benchmarking Modules
* **`benchmark_small_scale.py`:** Utilizes the ArrowHead dataset (3 classes, 211 total instances, univariate). Executes baseline accuracy and training speed comparisons between Catch22 and ROCKET models.
* **`benchmark_large_scale.py`:** Utilizes the Crop dataset (24 classes, 24,000 total instances, univariate). Integrates the `tracemalloc` library to profile peak RAM allocation alongside accuracy and computational speed. Outputs a dual-axis performance chart and a 24-class confusion matrix.

### Data Processing Pipeline
The data processing architecture is controlled via a centralized master script.

* **`master_pipeline.py`:** The primary execution node. Centralizes all global configuration parameters (e.g., chunk duration, target plant ID, signal smoothing windows) and sequentially executes Phase 1 and Phase 2.
* **`parquet_to_csv.py` (Phase 1):** Ingests raw `.parquet` biosignal files. Fixes epoch timestamp scaling, applies dynamic temporal chunking based on the `DAYS_PER_CHUNK` parameter, verifies existing files to prevent redundant processing, and exports partitioned `.csv` arrays.
* **`Preprocessing_and_Comparison.py` (Phase 2):** Dynamically resolves target directories (including `plant_id=` prefixes), concatenates CSV chunks, applies Z-score despiking, and executes Savitzky-Golay signal smoothing. Outputs the final `processed_training_data.csv` and an interactive Plotly HTML visualization.

## Requirements
Execution of all modules requires the following dependencies installed in the local Python environment:
* `aeon`
* `scikit-learn`
* `matplotlib`
* `pandas`
* `pyarrow`
* `plotly`
* `scipy`

## Execution Protocol
To process raw sensor data, configure the parameters within `master_pipeline.py` and execute the script via terminal:
```bash
python master_pipeline.py