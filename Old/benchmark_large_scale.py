import ssl
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from aeon.datasets import load_classification
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.feature_based import Catch22Classifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def execute_large_scale_benchmark_and_visualize():
    # 1. High-Volume Dataset Acquisition
    print("Initiating dataset load sequence: Crop (24 classes, high volume)...")
    X_train, y_train = load_classification("Crop", split="train")
    X_test, y_test = load_classification("Crop", split="test")

    print(f"Data acquired. X_train dimensions: {X_train.shape}")
    print(f"Test instances: {X_test.shape[0]}")
    print("-" * 40)

    # 2. Model Initialization
    models = {
        "Catch22\n(Feature-Based)": Catch22Classifier(random_state=42),
        "ROCKET\n(Convolution-Based)": RocketClassifier(n_kernels=10000, random_state=42)
    }

    # Data structures for metrics
    algorithms = []
    accuracies = []
    training_times = []
    peak_memories = []
    confusion_matrices = []
    class_labels = np.unique(y_test)

    # 3. Execution and Memory Profiling Loop
    for name, model in models.items():
        print(f"Evaluating Model: {name.replace('\n', ' ')}")

        tracemalloc.start()
        start_time = time.perf_counter()

        model.fit(X_train, y_train)

        training_duration = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mem_mb = peak_mem / (1024 * 1024)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)

        print(f"  Training Duration: {training_duration:.4f} seconds")
        print(f"  Peak Memory Usage: {peak_mem_mb:.2f} MB")
        print(f"  Classification Accuracy: {accuracy:.4f}")
        print("-" * 40)

        algorithms.append(name)
        accuracies.append(accuracy)
        training_times.append(training_duration)
        peak_memories.append(peak_mem_mb)
        confusion_matrices.append(cm)

    # 4. Performance Visualization Generation
    print("Generating large-scale performance visualization...")

    x = np.arange(len(algorithms))
    width = 0.3

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    bars1 = ax1.bar(x - width / 2, accuracies, width, label='Accuracy', color='#1f77b4')
    ax1.set_ylabel('Accuracy', color='#1f77b4', fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontweight='bold')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, peak_memories, width, label='Peak Memory (MB)', color='#ff7f0e')
    line1 = ax2.plot(x, training_times, color='#d62728', marker='o', linestyle='dashed', linewidth=2, markersize=8,
                     label='Training Time (s)')

    ax2.set_ylabel('Memory (MB) / Time (s)', color='#333333', fontweight='bold')
    ax2.set_ylim(bottom=0)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Large-Scale Benchmark: Crop Dataset\nAccuracy vs. Computation Time vs. RAM Allocation',
              fontweight='bold')
    fig1.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 5. Confusion Matrix Visualization Generation
    print("Generating 24-class confusion matrix visualization...")
    # Increased figsize to accommodate 24 classes
    fig2, axes = plt.subplots(1, 2, figsize=(16, 8))

    for i, ax in enumerate(axes):
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrices[i], display_labels=class_labels)
        # include_values=False prevents text overlapping in a 24x24 grid
        disp.plot(ax=ax, cmap='Blues', colorbar=True, include_values=False, xticks_rotation='vertical')
        ax.set_title(f"{algorithms[i].replace('\n', ' ')} Matrix", fontweight='bold')

    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    execute_large_scale_benchmark_and_visualize()