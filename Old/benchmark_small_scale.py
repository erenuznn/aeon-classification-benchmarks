import time
import numpy as np
import matplotlib.pyplot as plt
from aeon.datasets import load_arrow_head
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.feature_based import Catch22Classifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def execute_benchmark_and_visualize():
    # 1. Dataset Acquisition
    print("Initiating dataset load sequence: ArrowHead...")
    X_train, y_train = load_arrow_head(split="train")
    X_test, y_test = load_arrow_head(split="test")
    print(f"Data acquired. X_train dimensions: {X_train.shape}")
    print("-" * 30)

    # 2. Model Initialization (with fixed random_state for reproducibility, delete the random state options if not required)
    models = {
        "Catch22\n(Feature-Based)": Catch22Classifier(random_state=42),
        "ROCKET\n(Convolution-Based)": RocketClassifier(n_kernels=10000, random_state=42)
    }

    # Data structures for visualization
    algorithms = []
    accuracies = []
    training_times = []
    confusion_matrices = []
    class_labels = np.unique(y_test)

    # 3. Execution Loop
    for name, model in models.items():
        print(f"Evaluating Model: {name.replace('\n', ' ')}")

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_duration = time.perf_counter() - start_time

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)

        print(f"  Training Duration: {training_duration:.4f} seconds")
        print(f"  Classification Accuracy: {accuracy:.4f}")
        print("-" * 30)

        # Store metrics
        algorithms.append(name)
        accuracies.append(accuracy)
        training_times.append(training_duration)
        confusion_matrices.append(cm)

    # 4. Performance Visualization Generation
    print("Generating performance visualization...")
    x = np.arange(len(algorithms))
    width = 0.4

    fig1, ax1 = plt.subplots(figsize=(8, 6))

    bars = ax1.bar(x, accuracies, width, label='Accuracy', color='#1f77b4')
    ax1.set_ylabel('Classification Accuracy (0.0 to 1.0)', color='#1f77b4', fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontweight='bold')

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom',
                 fontweight='bold')

    ax2 = ax1.twinx()
    line = ax2.plot(x, training_times, color='#d62728', marker='o', linestyle='dashed', linewidth=2, markersize=10,
                    label='Training Time (s)')
    ax2.set_ylabel('Training Time (seconds)', color='#d62728', fontweight='bold')
    ax2.set_ylim(bottom=0, top=max(training_times) * 1.2)

    for i, txt in enumerate(training_times):
        ax2.annotate(f'{txt:.2f}s', (x[i], training_times[i]), textcoords="offset points", xytext=(0, 10), ha='center',
                     color='#d62728', fontweight='bold')

    plt.title('Algorithm Benchmark: Performance vs. Computational Cost', fontweight='bold')
    fig1.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 5. Confusion Matrix Visualization Generation
    print("Generating confusion matrix visualization...")
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, ax in enumerate(axes):
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrices[i], display_labels=class_labels)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f"{algorithms[i].replace('\n', ' ')} Matrix", fontweight='bold')

    fig2.tight_layout()

    # Render Plots
    plt.show()


if __name__ == "__main__":
    execute_benchmark_and_visualize()