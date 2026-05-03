import time
import numpy as np
import matplotlib.pyplot as plt
from aeon.datasets import load_classification
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.feature_based import Catch22Classifier
from sklearn.metrics import accuracy_score


def inject_gaussian_noise(X, noise_factor):
    """Generates and applies Additive White Gaussian Noise to a 3D array."""
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=X.shape)
    return X + noise


def execute_multi_stage_noise_benchmark():
    print("Initiating dataset load sequence: ArrowHead...")
    X_train, y_train = load_classification("ArrowHead", split="train")
    X_test, y_test = load_classification("ArrowHead", split="test")

    # Define sequential noise factors (0.0 to 1.0 in increments of 0.1)
    noise_factors = np.arange(0.0, 1.1, 0.1)

    models = {
        "Catch22": Catch22Classifier(random_state=42),
        "ROCKET": RocketClassifier(n_kernels=10000, random_state=42)
    }

    # Initialize results matrix
    results = {name: [] for name in models.keys()}

    for factor in noise_factors:
        print(f"\n--- Testing Noise Factor: {factor:.1f} ---")

        # Bypass noise injection for the baseline 0.0 iteration
        if factor == 0.0:
            X_train_current = X_train
            X_test_current = X_test
        else:
            X_train_current = inject_gaussian_noise(X_train, factor)
            X_test_current = inject_gaussian_noise(X_test, factor)

        for name in models.keys():
            # Re-instantiate models per iteration to prevent state retention
            if name == "Catch22":
                model = Catch22Classifier(random_state=42)
            else:
                model = RocketClassifier(n_kernels=10000, random_state=42)

            model.fit(X_train_current, y_train)
            y_pred = model.predict(X_test_current)
            acc = accuracy_score(y_test, y_pred)

            results[name].append(acc)
            print(f"Algorithm: {name} | Accuracy: {acc:.4f}")

    # Pass trajectory data to visualization module
    generate_sequence_chart(noise_factors, results)


def generate_sequence_chart(noise_factors, results):
    """Generates a line plot visualizing accuracy degradation over noise intervals."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Catch22": "#1f77b4", "ROCKET": "#d62728"}
    markers = {"Catch22": "o", "ROCKET": "s"}

    for name, accuracies in results.items():
        ax.plot(noise_factors, accuracies, label=name, color=colors[name],
                marker=markers[name], linewidth=2, markersize=8)

    ax.set_xlabel('AWGN Noise Factor')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Algorithmic Robustness to Sequential Noise Injection')
    ax.set_xticks(noise_factors)
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    execute_multi_stage_noise_benchmark()