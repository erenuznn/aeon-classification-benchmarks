# aeon-classification-benchmarks

## Objective
This repository contains execution pipelines comparing the classification performance, computational time, and memory allocation of feature-based (Catch22) and convolution-based (ROCKET) algorithms across different dataset scales.

## Execution Scripts and Datasets
* **`benchmark_small_scale.py`:** Utilizes the ArrowHead dataset (3 classes, 211 total instances, univariate). Focuses on baseline accuracy and training speed comparison.
* **`benchmark_large_scale.py`:** Utilizes the Crop dataset (24 classes, 24,000 total instances, univariate). Integrates `tracemalloc` to profile peak RAM allocation alongside accuracy and speed.